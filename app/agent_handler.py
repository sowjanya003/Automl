"""Azure AI Agent operations"""
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import (
    CodeInterpreterTool, FilePurpose, MessageAttachment,
    CodeInterpreterToolDefinition, ToolResources
)
from .utils import extract_json_from_text, looks_like_access_excuse
from .cosmos_respo import CosmosDBResponseHandler
from .cosmos_authe import CosmosDBAuthHandler  
from azure.storage.blob import BlobServiceClient
from fastapi import UploadFile, HTTPException
import logging
from app.config import FUN1_TIMEOUT
import tempfile 
import os, json, re, random
import pandas as pd, numpy as np
import io
import requests
from typing import Optional, Dict, Any, List
from .forecasting_template import SmartForecastAnswer
from .config import (
    BLOB_CONNECTION_STRING, UPLOAD_CONTAINER, FUN1_URL,FUN2_URL,FUN3_URL,
    AZURE_ENDPOINT, AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID,
    AZURE_PROJECT_NAME, AGENT_MODEL
)

from .system_prompt import SYSTEM_PROMPT
from .timeseries_transformer import TimeseriesTransformer
from dataset_intelligence.thread_metadata_handler import ThreadMetadataHandler

logger = logging.getLogger(__name__)


# AGENT HANDLER
class AgentHandler:
    def __init__(self):
        try:
            endpoint = os.getenv("AZURE_ENDPOINT")
            resource_group = os.getenv("AZURE_RESOURCE_GROUP")
            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            project_name = os.getenv("AZURE_PROJECT_NAME")

            self.client = AIProjectClient(
                endpoint=endpoint,
                credential=DefaultAzureCredential(),
                resource_group_name=resource_group,
                subscription_id=subscription_id,
                project_name=project_name
            )
        except Exception as init_err:
            logger.error(f"Client init failed: {init_err}")
            raise
        self.blob_service = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        self.fun1_url = os.getenv("FUN1_URL")
        if not self.fun1_url:
            logger.warning("FUN1_URL not set")

        self.ts_transformer = TimeseriesTransformer()
        self.forecast_answer = SmartForecastAnswer(self.blob_service)

        # Used by get_dynamic_suggestion to read recent chat context. Must be
        # a real instance - get_thread_history is a regular instance method,
        # not static, so calling it on the bare class (as this code used to)
        # always raised TypeError and silently forced the deterministic
        # fallback suggestion on every single call.
        try:
            self.response_handler = CosmosDBResponseHandler()
        except Exception as rh_err:
            logger.warning(f"CosmosDBResponseHandler unavailable in AgentHandler: {rh_err}")
            self.response_handler = None

        # Used to remember, per thread, which Azure AI file_ids have already
        # been (re-)attached for Code Interpreter access - see
        # _ensure_file_attached_to_thread. Cosmos-backed so this survives
        # restarts and works correctly across multiple worker processes.
        try:
            self.thread_metadata_handler = ThreadMetadataHandler()
        except Exception as tmh_err:
            logger.warning(f"ThreadMetadataHandler unavailable in AgentHandler: {tmh_err}")
            self.thread_metadata_handler = None


    def _trigger_automl_internal(
        self, 
        blob_file: str, 
        query: str, 
        time_budget: int, 
        user_id: str, 
        task_type: str,
        target: str,
        horizon: int = None,
        models: list = None,
        source_type: str = None
    ) -> str:
        try:
            payload = {
                "blob_file": blob_file,
                "query": query,
                "time_budget": time_budget,
                "task_type": task_type,
                "target": target
            }
            
            if task_type == "multistep_forecasting":
                effective_horizon = horizon or 12
                payload["horizon"] = effective_horizon
                logger.info(f"Using horizon {effective_horizon} for multistep_forecasting")
            
            if models:
                payload["models"] = models

            if source_type:
                payload["source_type"] = source_type
            
            logger.info(f"AutoML payload: {payload}")
            response = requests.post(
                self.fun1_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                # WAS 180. FUN1 legitimately runs up to the 10-minute
                # function timeout; 180 abandoned successful runs.
                timeout=FUN1_TIMEOUT
            )
            response.raise_for_status()
            logger.info("Function succeeded")

            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            blobs = list(container_client.list_blobs(name_starts_with=f"{user_id}/runs/"))
            results_blobs = [b for b in blobs if b.name.endswith('/results.json')]
            if not results_blobs:
                raise ValueError("No results.json found")
            latest_blob = max(results_blobs, key=lambda b: b.last_modified)
            logger.info(f"Results: {latest_blob.name}")
            return latest_blob.name
        except Exception as e:
            logger.error(f"AutoML internal failed: {e}")
            raise e

    # def _test_connectivity(self):
    #     try:
    #         agents = list(self.client.agents.list_agents(limit=1))
    #         logger.info(f"API connectivity test passed. Agents: {len(agents)}")
    #     except Exception as conn_err:
    #         logger.error(f"API connectivity failed: {conn_err}")
    #         raise

    def get_latest_blob_file(self, user_id: str, agent_name: str) -> Optional[str]:
        try:
            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            prefix = f"{user_id}/{agent_name}/"
            blobs = list(container_client.list_blobs(name_starts_with=prefix))
            data_blobs = [b for b in blobs if b.name.lower().endswith(('.csv', '.xlsx', '.json'))]
            if not data_blobs:
                return None
            latest_blob = max(data_blobs, key=lambda b: b.last_modified)
            return latest_blob.name
        except Exception as e:
            logger.error(f"Failed to fetch latest blob: {e}")
            return None

    def get_latest_file_id_from_agent(self, agent_id: str) -> Optional[str]:
        try:
            agent = self.client.agents.get_agent(agent_id=agent_id)
            tool_resources = agent.tool_resources
            if tool_resources and tool_resources.code_interpreter:
                file_ids = tool_resources.code_interpreter.file_ids or []
                if file_ids:
                    return file_ids[-1]
            return None
        except Exception as e:
            logger.error(f"Failed to get file_id: {e}")
            return None

    def _get_file_id_for_filename(self, agent_id: str, filename: Optional[str]) -> Optional[str]:
        """
        Resolve the Azure AI file_id that actually corresponds to `filename`,
        instead of blindly assuming the LAST file ever uploaded to this
        agent is the one for the CURRENT dataset. get_latest_file_id_from_agent
        breaks the moment more than one file has ever been uploaded to this
        agent - even in a different thread/session, on a completely
        different dataset - by attaching the wrong file to a thread that
        still genuinely needs the one it's actually querying.
        """
        try:
            agent = self.client.agents.get_agent(agent_id=agent_id)
            tool_resources = getattr(agent, "tool_resources", None)
            ci = getattr(tool_resources, "code_interpreter", None) if tool_resources else None
            file_ids = list(getattr(ci, "file_ids", None) or []) if ci else []
            if not file_ids:
                return None

            if filename:
                # Newest-first, so a re-uploaded/updated version of the same
                # filename resolves to its most recent file_id.
                for fid in reversed(file_ids):
                    try:
                        info = self.client.agents.files.get(file_id=fid)
                        if getattr(info, "filename", None) == filename:
                            return fid
                    except Exception:
                        continue
                logger.warning(
                    f"No agent file matched filename '{filename}'; "
                    f"falling back to the most recently uploaded file."
                )

            # No filename to match against, or no exact match found - fall
            # back to the most recent upload rather than attaching nothing.
            return file_ids[-1]
        except Exception as e:
            logger.warning(f"Could not resolve file_id for filename {filename!r}: {e}")
            return None

    def _ensure_file_attached_to_thread(
        self, thread_id: str, user_id: str, agent_id: str, filename: Optional[str] = None
    ) -> None:
        """
        Guarantee the CURRENT dataset (identified by `filename`) is
        explicitly attached to `thread_id` as a Code Interpreter file
        attachment, not just relying on the agent-level tool_resources.

        A file gets registered two ways at upload time: agent-level
        (tool_resources.code_interpreter.file_ids - meant to apply to every
        thread that agent ever uses) and thread-level (a message attachment
        created only on the specific thread used during that upload call).
        In practice this beta SDK (azure-ai-agents 1.2.0b5) doesn't always
        reliably surface the agent-level registration into a *different*
        thread's Code Interpreter sandbox, so a query landing on any thread
        other than the original upload thread can genuinely fail to see the
        file - the model then reports something like "I couldn't read the
        dataset file in this session." This re-attaches the file to the
        current thread the first time it's queried, so every thread gets
        guaranteed access regardless of where the dataset was originally
        uploaded.

        Resolves the file_id by matching `filename` (via
        _get_file_id_for_filename) rather than just grabbing whichever file
        was uploaded to this agent most recently - a user who has ever
        uploaded more than one dataset (even in a different thread) would
        otherwise risk this attaching the wrong file to the thread.

        Idempotent: which file_ids have already been attached to a thread is
        tracked in Cosmos via ThreadMetadataHandler, so this is a one-time
        cost per thread rather than happening on every single query.
        """
        if not thread_id or not user_id or not agent_id:
            return
        try:
            file_id = self._get_file_id_for_filename(agent_id, filename)
            if not file_id:
                return

            attached_ids = []
            if self.thread_metadata_handler:
                try:
                    attached_ids = self.thread_metadata_handler.get_metadata(
                        thread_id, user_id, "attached_file_ids"
                    ) or []
                except Exception:
                    attached_ids = []

            if file_id in attached_ids:
                return  # already attached to this thread - nothing to do

            attachment = MessageAttachment(
                file_id=file_id,
                tools=CodeInterpreterTool().definitions
            )
            self.client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content="(Dataset file re-attached for this session's Code Interpreter access.)",
                attachments=[attachment]
            )
            logger.info(f"Re-attached file {file_id} ({filename}) to thread {thread_id}")

            if self.thread_metadata_handler:
                try:
                    self.thread_metadata_handler.add_metadata(
                        thread_id, user_id, "attached_file_ids", attached_ids + [file_id]
                    )
                except Exception as meta_err:
                    logger.warning(
                        f"Could not persist attached_file_ids for thread {thread_id}: {meta_err}"
                    )
        except Exception as e:
            logger.warning(f"Could not ensure file attachment for thread {thread_id}: {e}")

    def create_agent(self, name: str):
        try:
            model_deployment = AGENT_MODEL
            if not model_deployment:
                raise ValueError("MODEL_DEPLOYMENT_NAME not set")

            instructions = SYSTEM_PROMPT

            code_interpreter = CodeInterpreterTool()
            assistant = self.client.agents.create_agent(
                model=model_deployment,
                name=name,
                instructions=instructions,
                tools=code_interpreter.definitions,
            )
            logger.info(f"Created agent: {assistant.id}")
            return assistant.id
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def create_thread(self, agent_id: str = None):
        try:
            thread = self.client.agents.threads.create()
            logger.info(f"Created thread: {thread.id}")
            return thread.id, agent_id
        except Exception as e:
            logger.error(f"Thread creation failed: {e}")
            raise

    def upload_file_to_blob_and_agent(
        self,
        upload_file: UploadFile,
        agent_id: str,
        user_id: str,
        thread_id: str = None,
    ):
        try:
            agent = self.client.agents.get_agent(agent_id=agent_id)
            agent_name = agent.name

            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            try:
                container_client.create_container()
            except:
                pass

            blob_name = f"{user_id}/{agent_name}/{upload_file.filename}"
            blob_client = container_client.get_blob_client(blob=blob_name)
            upload_file.file.seek(0)
            file_content = upload_file.file.read()
            blob_client.upload_blob(file_content, overwrite=True)
            logger.info(f"Uploaded to Blob: {blob_name}")

            tmp_file_path = os.path.join(tempfile.gettempdir(), upload_file.filename)
            with open(tmp_file_path, 'wb') as f:
                f.write(file_content)

            try:
                uploaded_file = self.client.agents.files.upload_and_poll(
                    file_path=tmp_file_path,
                    purpose=FilePurpose.AGENTS
                )
                logger.info(f"Uploaded file to Azure AI: {uploaded_file.id}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

            def _existing_file_ids(target):
                # A newly created agent has no code-interpreter resource yet, so
                # tool_resources (or .code_interpreter) can be None. Walk the
                # chain safely - this is the fix for
                #   'NoneType' object has no attribute 'file_ids'
                tr = getattr(target, "tool_resources", None)
                if not tr:
                    return []
                ci = getattr(tr, "code_interpreter", None)
                if not ci:
                    return []
                return list(getattr(ci, "file_ids", None) or [])

            def _update_agent_files(aid: str):
                target = self.client.agents.get_agent(agent_id=aid)
                existing = _existing_file_ids(target)
                new_ids = (existing + [uploaded_file.id]
                           if uploaded_file.id not in existing else existing)
                code_int = CodeInterpreterTool(file_ids=new_ids)
                # Preserve existing tools; ensure code-interpreter is present.
                tools = getattr(target, "tools", None) or code_int.definitions
                self.client.agents.update_agent(
                    agent_id=aid,
                    tools=tools,
                    tool_resources=code_int.resources
                )
            _update_agent_files(agent_id)

            if thread_id:                
                attachment = MessageAttachment(
                    file_id=uploaded_file.id,
                    tools=CodeInterpreterTool().definitions
                )
                self.client.agents.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=f"File uploaded: {upload_file.filename}",
                    attachments=[attachment]
                )
                logger.info(f"File {uploaded_file.id} attached to thread {thread_id}")

            return uploaded_file.id, blob_name, agent_id

        except Exception as e:
            logger.error(f"File upload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    def get_dataset_columns(self, blob_path: str, user_id: str) -> list:
        try:
            if not blob_path.startswith(f"{user_id}/"):
                raise ValueError("Access denied")
            
            client = self.blob_service.get_blob_client(
                container=UPLOAD_CONTAINER, 
                blob=blob_path
            )
            
            if not client.exists():
                return []
            
            data = client.download_blob().readall()
            
            if blob_path.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(data), nrows=0)  
            elif blob_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(data), nrows=0)
            else:
                return []
            
            return df.columns.tolist()
        
        except Exception as e:
            logger.error(f"Failed to get columns from {blob_path}: {e}")
            return []
        
    def extract_targets_from_query(self, query: str, available_columns: list) -> list:
        query_lower = query.lower()
        mentioned_columns = []
        
        for col in available_columns:
            col_lower = col.lower()
            if col_lower in query_lower:
                mentioned_columns.append(col)
        
        return mentioned_columns
        
    @staticmethod
    def _find_similar_columns(target: str, columns: list, limit: int = 5) -> list:
        """
        Rank dataset columns by similarity to a requested target.

        Uses three signals so a hallucinated name like 'downtime_cause' still
        surfaces 'downtime_hours', 'cause_code', 'cause_description' instead of
        an empty suggestion list:
          1. shared underscore/space tokens
          2. substring containment either way
          3. difflib ratio as a fallback
        """
        import difflib

        tl = target.lower().strip()
        t_tokens = set(re.split(r"[_\s]+", tl))
        scored = []
        for col in columns:
            cl = col.lower().strip()
            c_tokens = set(re.split(r"[_\s]+", cl))

            token_overlap = len(t_tokens & c_tokens)
            substring = 1 if (tl in cl or cl in tl) else 0
            ratio = difflib.SequenceMatcher(None, tl, cl).ratio()

            score = token_overlap * 2 + substring + ratio
            if token_overlap or substring or ratio >= 0.6:
                scored.append((score, col))

        scored.sort(reverse=True)
        return [col for _score, col in scored[:limit]]

    def validate_target_column(self, target: str, blob_file: str, user_id: str) -> dict:
        try:            
            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            blob_client = container_client.get_blob_client(blob_file)
            
            if not blob_client.exists():
                return {"valid": False, "message": "Dataset not found", "available_columns": []}
            
            data = blob_client.download_blob().readall()
            
            if blob_file.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(data))
            elif blob_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(data))
            else:
                return {"valid": False, "message": "Unsupported file format", "available_columns": []}
            
            columns = df.columns.tolist()
            
            # Case-insensitive matching
            target_lower = target.lower().strip()
            matching_cols = [col for col in columns if col.lower().strip() == target_lower]
            
            if matching_cols:
                return {
                    "valid": True,
                    "message": "Target column found",
                    "available_columns": columns,
                    "matched_column": matching_cols[0]
                }
            else:
                similar = self._find_similar_columns(target, columns)
                return {
                    "valid": False,
                    "message": f"Column '{target}' not found in dataset",
                    "available_columns": columns,
                    "similar_columns": similar
                }
        
        except Exception as e:
            logger.error(f"Error validating target: {e}")
            return {"valid": False, "message": str(e), "available_columns": []}
        
    def parse_and_validate_targets(self, targets_str: str, blob_file: str, user_id: str) -> dict:
        try:
            if not targets_str or targets_str.strip().lower() == 'none':
                return {
                    "valid": False,
                    "targets": [],
                    "message": "No target specified",
                    "available_columns": []
                }
            
            targets_list = [t.strip() for t in targets_str.split(',')]
            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            blob_client = container_client.get_blob_client(blob_file)
            
            if not blob_client.exists():
                return {
                    "valid": False,
                    "targets": targets_list,
                    "message": "Dataset not found",
                    "available_columns": []
                }
            
            data = blob_client.download_blob().readall()
            
            if blob_file.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(data))
            elif blob_file.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(data))
            else:
                return {
                    "valid": False,
                    "targets": targets_list,
                    "message": "Unsupported file format",
                    "available_columns": []
                }
            
            columns = df.columns.tolist()
            
            # Validate each target
            validated_targets = []
            missing_targets = []
            
            for target in targets_list:
                target_lower = target.lower().strip()
                matching_cols = [col for col in columns if col.lower().strip() == target_lower]
                
                if matching_cols:
                    validated_targets.append(matching_cols[0])
                else:
                    missing_targets.append(target)
            
            if missing_targets:
                similar = []
                for missing in missing_targets:
                    similar.extend(self._find_similar_columns(missing, columns))
                # de-dupe while preserving rank order
                seen = set()
                similar = [c for c in similar if not (c in seen or seen.add(c))]

                return {
                    "valid": False,
                    "targets": validated_targets,
                    "message": f"Targets not found: {', '.join(missing_targets)}",
                    "available_columns": columns,
                    "missing_targets": missing_targets,
                    "similar_columns": similar
                }
            
            return {
                "valid": True,
                "targets": validated_targets,
                "message": "All targets validated",
                "available_columns": columns
            }
        
        except Exception as e:
            logger.error(f"Error validating targets: {e}")
            return {
                "valid": False,
                "targets": [],
                "message": str(e),
                "available_columns": []
            }
        
    def ask_user_to_clarify_target(self, possible_targets: list, available_columns: list) -> str:
        if len(possible_targets) > 1:
            targets_str = "', '".join(possible_targets)
            return (
                f"I found multiple potential target columns: '{targets_str}'. "
                f"Please specify which one you want to predict. For example: "
                f"'Build a classification model to predict {possible_targets[0]}'"
            )
        elif len(possible_targets) == 0:
            cols_str = "', '".join(available_columns[:10])
            return (
                f"I couldn't identify a target column from your query. "
                f"Available columns: '{cols_str}'. "
                f"Please specify which column you want to predict."
            )
        else:
            return ""
    
    # Terminal Azure OpenAI errors that a retry cannot fix - fail immediately.
    _NON_RETRYABLE = (
        "content_filter", "invalid_request_error", "context_length_exceeded",
    )

    def _run_agent_with_retry(
        self,
        thread_id: str,
        agent_id: str,
        max_retries: int = 4,
        base_delay: float = 3.0,
    ):
        """
        Run an agent, retrying transient rate-limit / server errors.

        Azure OpenAI returns rate_limit_exceeded under load. It is transient:
        a short backoff usually clears it. Without this, one throttled call
        failed the whole job even though nothing was wrong with the request.

        Honours a Retry-After hint in the error message when present, otherwise
        uses exponential backoff (3s, 6s, 12s, 24s).
        """
        import re as _re
        import time as _time

        last_error = None
        for attempt in range(max_retries):
            run = self.client.agents.runs.create_and_process(
                thread_id=thread_id,
                agent_id=agent_id,
            )

            if run.status == "completed":
                return run

            last_error = getattr(run, "last_error", None)
            code = ""
            message = ""
            if last_error is not None:
                if isinstance(last_error, dict):
                    code = str(last_error.get("code", ""))
                    message = str(last_error.get("message", ""))
                else:
                    code = str(getattr(last_error, "code", ""))
                    message = str(getattr(last_error, "message", ""))

            # Give up straight away on errors a retry cannot fix
            if any(term in code.lower() for term in self._NON_RETRYABLE):
                break

            is_rate_limit = "rate_limit" in code.lower() or "429" in code
            is_server = code in ("server_error", "500", "503") or run.status != "failed"

            if not (is_rate_limit or is_server) or attempt == max_retries - 1:
                break

            # Prefer the server's Retry-After hint if it gave one
            delay = base_delay * (2 ** attempt)
            m = _re.search(r"retry.?after[\"':\s]*(\d+)", message, _re.I)
            if m:
                delay = max(delay, float(m.group(1)))

            logger.warning(
                f"Agent run throttled ({code}); retry "
                f"{attempt + 1}/{max_retries - 1} in {delay:.0f}s"
            )
            _time.sleep(delay)

        # Out of retries - raise with the real reason
        raise Exception(f"Run failed: {last_error}")

    @staticmethod
    def _looks_like_code(text: str) -> bool:
        """
        True if the agent returned a code snippet rather than a human answer.

        The agent is only supposed to route/answer in prose, but it sometimes
        emits pandas like df.groupby('Shift')['ProducedQuantity'].sum().idxmax().
        That must never be shown to the user - it means "the agent tried to
        compute instead of answering", so we compute it ourselves.
        """
        if not text:
            return False
        t = text.strip()
        # fenced code block
        if t.startswith("```") or t.startswith("`"):
            return True
        code_signals = (
            "df.", "df[", ".groupby(", ".idxmax(", ".idxmin(", ".sum()",
            ".mean()", ".value_counts(", "pd.", "np.", "import ",
            ".agg(", ".loc[", ".iloc[", "print(", "lambda ",
        )
        # Short, single-line, and full of code tokens -> it is code
        hits = sum(1 for s in code_signals if s in t)
        single_line = "\n" not in t
        if hits >= 1 and single_line and len(t) < 200:
            return True
        if hits >= 2:
            return True
        return False

    @staticmethod
    def _is_control_json(text: str, parsed) -> bool:
        """
        True if the agent's reply is only a routing/control JSON (e.g.
        {"is_ml": false}) rather than a human-readable answer.
        """
        if not isinstance(parsed, dict):
            return False
        # A real answer has prose; a control payload is basically just the JSON.
        stripped = (text or "").strip()
        looks_like_only_json = stripped.startswith("{") and stripped.endswith("}")
        control_keys = {"is_ml", "task_type", "target", "metric", "models", "horizon"}
        only_control = set(parsed.keys()) <= control_keys
        return looks_like_only_json and only_control

    @staticmethod
    def _looks_like_ml_task_prose(text: str) -> bool:
        """
        True if the agent DESCRIBED an ML task in prose instead of emitting the
        is_ml control JSON that actually triggers model building - e.g.
        "Run KMeans on quantityproduced, unitcost, productioncost; select k by
        silhouette." or "Forecast 12 months of quantityproduced using
        calendar_year and calendar_month."

        This is a ROUTING failure (the request really was an ML request), so the
        caller re-classifies rather than showing this fragment to the user.

        Deliberately narrow so it never swallows a genuine analysis answer:
        the text must be short, single-line, contain no computed result
        numbers, and open with an ML action or name a model/algorithm.
        """
        if not text:
            return False
        t = text.strip()
        if len(t) > 220 or "\n" in t:
            return False  # real analysis answers are longer / multi-line

        tl = t.lower()
        ml_openers = (
            "run ", "forecast", "predict", "classify", "cluster",
            "detect anomal", "build a model", "train a model", "fit ",
            "apply ",
        )
        model_names = (
            "kmeans", "k-means", "dbscan", "gmm", "gaussian mixture",
            "isolation forest", "one-class svm", "local outlier factor",
            "elliptic envelope", "arima", "prophet", "xgboost", "lightgbm",
            "catboost", "random forest", "gradient boosting", "ridge",
            "logistic regression",
        )
        opens_with_ml = any(tl.startswith(v) for v in ml_openers)
        names_model = any(m in tl for m in model_names)
        if not (opens_with_ml or names_model):
            return False

        # A real answer reports computed figures. Strip horizon/parameter
        # counts ("12 months", "k=5") first, then require no remaining digits.
        stripped = re.sub(
            r'\d+\s*(day|days|week|weeks|month|months|quarter|quarters|'
            r'year|years|step|steps)\b',
            '', t, flags=re.IGNORECASE
        )
        stripped = re.sub(r'\bk\s*=\s*\d+', '', stripped, flags=re.IGNORECASE)
        return not bool(re.search(r'\d', stripped))

    def _reclassify_as_ml_json(self, raw_query: str, agent_id: str, columns: list):
        """
        Second-pass classifier: re-asks the agent with a strict JSON-only
        prompt when it described an ML task in prose instead of emitting the
        is_ml control JSON. Also validates the returned target is a real
        column - if not (e.g. user said 'risk level' which doesn't exist),
        returns None so the caller asks the user to clarify rather than
        silently picking the wrong column.
        """
        try:
            col_block = f"\nThe dataset has EXACTLY these columns: {columns}\n" if columns else ""
            prompt = (
                "Classify this user request for an AutoML system.\n"
                f'Request: "{raw_query}"\n'
                f"{col_block}\n"
                "Reply with ONLY a JSON object, no prose, no markdown fences.\n"
                "If it IS a model-building request:\n"
                '{"is_ml": true, "task_type": "<classification|regression|'
                'forecasting|multistep_forecasting|clustering|anomaly_detection>", '
                '"target": "<real column name, or \\"None\\" for clustering/'
                'anomaly_detection>", "metric": "<metric>", '
                '"horizon": <int, only for multistep_forecasting>, '
                '"models": ["<model>"]}\n'
                'If NOT a model-building request, reply exactly {"is_ml": false}\n'
                "CRITICAL RULES:\n"
                "- Copy target column character-for-character from the list above.\n"
                "- NEVER invent a column name not in the list.\n"
                "- If the concept asked (e.g. 'risk level', 'quality score') does NOT "
                "map to any real column, reply {\"is_ml\": false} so the user is asked "
                "to clarify.\n"
                "- Defaults: classification=f1; regression=rmse; forecasting=rmse; "
                "multistep_forecasting=rmse (horizon 12 if unspecified); "
                "clustering=silhouette_score; anomaly_detection=anomaly_score.\n"
                "- For clustering and anomaly_detection set target to None."
            )
            text = self._ask_agent_for_text(agent_id, prompt)
            parsed = extract_json_from_text(text or "")
            if not isinstance(parsed, dict):
                return None
            # Validate target for supervised tasks
            task_type = (parsed.get("task_type") or "").lower()
            if parsed.get("is_ml") and task_type not in ("clustering", "anomaly_detection"):
                target = (parsed.get("target") or "").strip()
                if target and target.lower() != "none":
                    col_set_lower = {c.lower() for c in (columns or [])}
                    if target.lower() not in col_set_lower:
                        logger.warning(
                            f"Re-classification returned non-existent target '{target}'; "
                            f"returning None so user is asked to clarify."
                        )
                        return None
            return parsed
        except Exception as e:
            logger.warning(f"ML re-classification failed: {e}")
            return None

    def _answer_non_ml_query(self, query: str, agent_id: str = None,
                             thread_id: str = None, user_id: str = None,
                             agent_name: str = None) -> str:
        """
        Answer a non-ML question about the dataset in plain language.

        Handles two kinds:
          1. Greetings / small talk -> a friendly canned reply.
          2. Simple data questions (total / average / count / min / max of a
             column) -> computed directly from the uploaded CSV so the user
             gets a real number instead of a control JSON.

        Falls back to a helpful prompt if the question can't be resolved.
        """
        import io

        q = (query or "").strip()
        ql = q.lower()

        # ---- 1. Greetings / small talk ----
        greetings = ("hi", "hello", "hey", "how are you", "good morning",
                     "good afternoon", "good evening", "thanks", "thank you")
        if any(ql == g or ql.startswith(g) for g in greetings):
            return (
                "Hello! I'm your AutoML assistant. I can analyze your uploaded "
                "dataset, answer questions about it, and build models "
                "(classification, regression, forecasting, clustering, anomaly "
                "detection). What would you like to do?"
            )

        # ---- 2. Simple aggregate questions over the dataset ----
        try:
            blob_path = (
                self.get_latest_blob_file(user_id, agent_name)
                if user_id and agent_name else None
            )
            if not blob_path:
                raise ValueError("no dataset")

            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            blob_client = container_client.get_blob_client(blob_path)
            data = blob_client.download_blob().readall()
            if blob_path.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(data))
            elif blob_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(data))
            else:
                raise ValueError("unsupported format")

            # 2a. Fast path: rule-based engine for common aggregate shapes.
            answer = self._answer_data_question(q, df)
            if answer:
                return answer

            # 2b. General path: compute a SAFE statistical profile of the
            # dataset in Python, hand it to the LLM as context, and let it
            # reason and phrase the answer. No generated code is executed.
            # Handles arbitrary phrasings ("what percentage of production is
            # defective?") without a hand-written rule for each one.
            if agent_id:
                answer = self._answer_with_llm_pandas(
                    q, df, agent_id, thread_id
                )
                if answer:
                    return answer

            # Could not resolve - offer guidance grounded in real columns
            cols = ", ".join(list(df.columns)[:15])
            return (
                "I can answer questions about your dataset or build a model. "
                f"Your dataset has these columns: {cols}. "
                "Try asking e.g. \"total ProducedQuantity by Shift\", "
                "\"which shift is most efficient?\", or \"classify ProductionStatus\"."
            )

        except Exception as e:
            logger.warning(f"Could not answer non-ML query from data: {e}")
            return (
                "I can help you analyze your dataset or build a model. Could you "
                "rephrase your question, or tell me which column you're asking "
                "about?"
            )

    # ------------------------------------------------------------------
    # General data-QA: LLM writes a pandas expression, we execute it safely
    # ------------------------------------------------------------------
    def _answer_with_llm_pandas(self, query, df, agent_id, thread_id=None):
        """
        Answer ANY data question by asking the agent to write a single pandas
        expression against `df`, then executing it in a restricted sandbox.

        This replaces per-question keyword rules: "what percentage of
        production is defective?", "average cost for completed orders", etc. all
        work because the LLM writes the pandas, not us.

        Safety: the expression is run with no builtins, a whitelist of names
        (df, pd, np), and a static check that blocks imports, dunder access,
        and attribute names known to touch the filesystem or process. If
        anything looks off, we skip execution and return "".
        """
        try:
            schema = self._describe_schema(df)

            prompt = (
                "You are a data assistant. The user asked:\n"
                f'"{query}"\n\n'
                "You have a pandas DataFrame named df with this schema:\n"
                f"{schema}\n\n"
                "Write ONE Python expression (not statements) using only df, pd "
                "and np that computes the answer. Rules:\n"
                "- Return ONLY the expression, no explanation, no code fences.\n"
                "- No imports, no assignments, no loops, no file or network access.\n"
                "- If the question cannot be answered from these columns, reply "
                "exactly: CANNOT_ANSWER\n\n"
                "Examples:\n"
                "Q: total produced quantity -> df['ProducedQuantity'].sum()\n"
                "Q: what percentage of production is defective? -> "
                "df['DefectiveQuantity'].sum() / df['ProducedQuantity'].sum() * 100\n"
                "Q: average cost per shift -> "
                "df.groupby('Shift')['ProductionCost'].mean().to_dict()\n"
            )

            expr = self._ask_agent_for_text(agent_id, prompt, thread_id)
            if not expr:
                return ""
            expr = self._clean_expression(expr)
            if not expr or expr.strip() == "CANNOT_ANSWER":
                return ""

            if not self._expression_is_safe(expr):
                logger.warning(f"Unsafe pandas expression rejected: {expr!r}")
                return ""

            value = self._safe_eval_pandas(expr, df)
            if value is None:
                return ""

            phrased = self._phrase_result(query, expr, value)
            return phrased

        except Exception as e:
            logger.warning(f"LLM-pandas answer failed: {e}")
            return ""

    @staticmethod
    def _describe_schema(df) -> str:
        """Compact per-column schema: name, dtype, a couple of sample values."""
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            try:
                samples = df[col].dropna().unique()[:3]
                sample_str = ", ".join(str(s) for s in samples)
            except Exception:
                sample_str = ""
            lines.append(f"- {col} ({dtype}) e.g. {sample_str}")
        return "\n".join(lines)

    def _ask_agent_for_text(self, agent_id, prompt, thread_id=None):
        """Send a one-shot prompt to the agent and return its text reply."""
        if thread_id is None:
            thread = self.client.agents.threads.create()
            thread_id = thread.id
        self.client.agents.messages.create(
            thread_id=thread_id, role="user", content=prompt
        )
        run = self._run_agent_with_retry(thread_id, agent_id)
        if getattr(run, "status", None) != "completed":
            return ""
        # Filter to THIS run's own messages - without it, a run that
        # produces no fresh text can silently fall through to an older
        # message from a previous turn instead of returning "".
        for msg in self.client.agents.messages.list(thread_id=thread_id, run_id=run.id):
            if msg.role == "assistant" and msg.content:
                text = ""
                for c in msg.content:
                    if c.type == "text":
                        text += c.text.value
                if text.strip():
                    return text.strip()
        return ""

    @staticmethod
    def _clean_expression(text: str) -> str:
        """Strip code fences / prose the LLM may wrap around the expression."""
        import re as _re
        t = text.strip()
        # pull out of ```...``` if present
        fence = _re.search(r"```(?:python)?\s*(.+?)```", t, _re.DOTALL)
        if fence:
            t = fence.group(1).strip()
        # take the first non-empty line that looks like an expression
        for line in t.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line
        return t

    @staticmethod
    def _expression_is_safe(expr: str) -> bool:
        """
        Allow only a DataFrame expression. Reject anything that could escape:
        imports, dunders, attribute names that touch the system, statements.
        """
        import re as _re
        if len(expr) > 500:
            return False
        banned = (
            "__", "import", "eval", "exec", "open(", "compile(", "globals",
            "locals", "getattr", "setattr", "delattr", "os.", "sys.",
            "subprocess", "socket", "shutil", "Path(", "read_csv", "to_csv",
            "read_", "to_pickle", "system(", "popen", "=",  # no assignment
        )
        # allow == and != and >=,<= but not bare =
        cleaned = expr.replace("==", "").replace("!=", "").replace(">=", "").replace("<=", "")
        for tok in banned:
            if tok == "=":
                if "=" in cleaned:
                    return False
            elif tok in expr:
                return False
        # must reference df
        if "df" not in expr:
            return False
        return True

    @staticmethod
    def _safe_eval_pandas(expr: str, df):
        """Evaluate the expression with no builtins and a tiny namespace."""
        import numpy as _np
        safe_globals = {"__builtins__": {}}
        safe_locals = {"df": df, "pd": pd, "np": _np}
        try:
            return eval(expr, safe_globals, safe_locals)  # noqa: S307 - sandboxed
        except Exception as e:
            logger.warning(f"pandas expression failed to evaluate: {e}")
            return None

    @classmethod
    def _phrase_result(cls, query, expr, value) -> str:
        """Turn the computed value into a readable sentence / small table."""
        import numbers

        # Scalar number
        if isinstance(value, numbers.Number) and not isinstance(value, bool):
            v = float(value)
            pretty = f"{v:,.2f}" if abs(v) >= 1 or v == 0 else f"{v:.4f}"
            ql = query.lower()
            if "percent" in ql or "percentage" in ql or "%" in ql or "rate" in ql:
                # value may already be a percentage or a fraction
                if abs(v) <= 1:
                    return f"{v*100:.2f}%"
                return f"{pretty}%"
            return f"The answer is {pretty}."

        # dict (e.g. groupby().to_dict())
        if isinstance(value, dict):
            items = list(value.items())
            if not items:
                return "No results."
            lines = ["Here's the breakdown:", ""]
            for k, v in items[:30]:
                if isinstance(v, numbers.Number) and not isinstance(v, bool):
                    lines.append(f"- {k}: {float(v):,.2f}")
                else:
                    lines.append(f"- {k}: {v}")
            return "\n".join(lines)

        # pandas Series
        if isinstance(value, pd.Series):
            s = value.head(30)
            lines = ["Here's the breakdown:", ""]
            for k, v in s.items():
                try:
                    lines.append(f"- {k}: {float(v):,.2f}")
                except (TypeError, ValueError):
                    lines.append(f"- {k}: {v}")
            return "\n".join(lines)

        # pandas DataFrame -> markdown-ish
        if isinstance(value, pd.DataFrame):
            return "Here's the result:\n\n" + value.head(15).to_string()

        # strings / everything else
        text = str(value)
        return text[:1500] if text else ""

    # ------------------------------------------------------------------
    # Lightweight natural-language data-QA engine
    # ------------------------------------------------------------------
    # Word -> column-name-hint synonyms, so "production" finds ProducedQuantity,
    # "cost" finds ProductionCost, etc.
    COLUMN_SYNONYMS = {
        "production": ("produced", "output"),
        "produced": ("produced",),
        "planned": ("planned",),
        "output": ("produced", "output"),
        "cost": ("cost",),
        "defects": ("defective", "defect"),
        "defective": ("defective",),
        "downtime": ("downtime",),
        "quantity": ("quantity", "qty"),
    }

    AGG_WORDS = {
        "sum": ("total", "sum", "combined"),
        "mean": ("average", "avg", "mean", "typical"),
        "count": ("count", "number of", "how many"),
        "max": ("maximum", "max", "highest", "largest", "most", "top", "best"),
        "min": ("minimum", "min", "lowest", "smallest", "least", "worst"),
    }

    # Derived metrics keyed by name. Each: keywords that trigger it, numerator
    # column hints, denominator column hints, label, and whether a higher value
    # is better (drives "which is best/most X" ranking direction).
    DERIVED_METRICS = {
        "efficiency": {
            "keywords": ("efficiency", "efficient"),
            "num": ("produced", "output"), "den": ("time", "hour"),
            "label": "efficiency (produced per hour)", "higher_is_better": True,
        },
        "defect_rate": {
            "keywords": ("defect rate", "defective rate", "defect ratio"),
            "num": ("defect",), "den": ("produced", "planned"),
            "label": "defect rate", "higher_is_better": False,
        },
        "yield": {
            "keywords": ("yield", "fulfil", "fulfill", "completion rate"),
            "num": ("produced",), "den": ("planned",),
            "label": "yield (produced vs planned)", "higher_is_better": True,
        },
        "utilization": {
            "keywords": ("utilization", "utilisation", "throughput"),
            "num": ("produced",), "den": ("time", "hour"),
            "label": "utilization", "higher_is_better": True,
        },
    }

    # ------------------------------------------------------------------
    # Business KPI layer (maintenance / manufacturing)
    # ------------------------------------------------------------------
    # Column-name hints, so KPIs work whatever the exact SAP field names are.
    KPI_COLUMN_HINTS = {
        "cost_total":     ("actual_cost", "total_cost", "productioncost", "cost_inr", "cost"),
        "cost_material":  ("material_cost", "material"),
        "cost_labour":    ("labour_cost", "labor_cost", "labour", "labor"),
        "cost_external":  ("external_service_cost", "external_cost", "external", "service_cost"),
        "downtime":       ("downtime_hours", "downtime", "breakdown_hours"),
        "plant":          ("plant", "plant_id", "site", "location", "unit"),
        "category":       ("work_order_type", "order_type", "maintenance_type", "notification_type", "category", "wo_type"),
        "machine":        ("equipment_id", "machine_id", "machineid", "equipment", "asset_id"),
        "date":           ("date", "created_date", "notification_date", "datekey", "posting_date"),
        "produced":       ("producedquantity", "produced", "output"),
        "planned":        ("plannedquantity", "planned"),
        "defective":      ("defectivequantity", "defective", "defects"),
        "time_hours":     ("productiontimehours", "time_hours", "hours"),
    }

    @staticmethod
    def _norm(s):
        return s.lower().replace(" ", "").replace("_", "").replace("-", "")

    @classmethod
    def _kpi_col(cls, columns, key):
        """Resolve a KPI role to an actual column name via hints."""
        for hint in cls.KPI_COLUMN_HINTS.get(key, ()):  # ordered: best first
            nh = cls._norm(hint)
            for col in columns:
                if nh in cls._norm(col):
                    return col
        return None

    @classmethod
    def compute_maintenance_kpis(cls, df) -> dict:
        """
        Compute business KPIs from a maintenance / production dataframe.

        Everything is optional and guarded: whichever columns exist produce
        their KPIs, the rest are simply skipped. Returns a JSON-safe dict that
        the API appends to the response as `kpis` - it never removes or
        replaces any existing field.
        """
        import numpy as np
        cols = list(df.columns)
        kpis: dict = {"available": True, "row_count": int(len(df))}

        def num(colname):
            return pd.to_numeric(df[colname], errors="coerce") if colname else None

        cost_col   = cls._kpi_col(cols, "cost_total")
        mat_col    = cls._kpi_col(cols, "cost_material")
        lab_col    = cls._kpi_col(cols, "cost_labour")
        ext_col    = cls._kpi_col(cols, "cost_external")
        down_col   = cls._kpi_col(cols, "downtime")
        plant_col  = cls._kpi_col(cols, "plant")
        cat_col    = cls._kpi_col(cols, "category")
        machine_col= cls._kpi_col(cols, "machine")
        date_col   = cls._kpi_col(cols, "date")
        prod_col   = cls._kpi_col(cols, "produced")
        plan_col   = cls._kpi_col(cols, "planned")
        def_col    = cls._kpi_col(cols, "defective")
        time_col   = cls._kpi_col(cols, "time_hours")

        kpis["columns_detected"] = {
            k: v for k, v in {
                "cost": cost_col, "material_cost": mat_col, "labour_cost": lab_col,
                "external_cost": ext_col, "downtime": down_col, "plant": plant_col,
                "category": cat_col, "machine": machine_col, "date": date_col,
            }.items() if v
        }

        # A single summary the frontend can use to show what is / isn't
        # available and prompt the user to add missing columns. Nothing is
        # silently dropped - each group reports present True/False plus what it
        # needs.
        _has_date = False
        if date_col:
            _has_date = pd.to_datetime(df[date_col], errors="coerce").notna().mean() >= 0.5
        kpis["data_readiness"] = {
            "cost_kpis":        {"available": bool(cost_col),  "needs": ["cost"]},
            "downtime_kpis":    {"available": bool(down_col),  "needs": ["downtime"]},
            "by_plant":         {"available": bool(plant_col), "needs": ["plant"]},
            "by_category":      {"available": bool(cat_col),   "needs": ["category (work_order_type)"]},
            "by_machine":       {"available": bool(machine_col), "needs": ["machine/equipment id"]},
            "time_trends":      {"available": bool(_has_date), "needs": ["a usable date column"]},
            "multi_dimensional":{"available": bool(plant_col and (cat_col or _has_date)),
                                 "needs": ["plant + category, or plant + date"]},
        }

        # ---- Cost KPIs ----
        if cost_col:
            c = num(cost_col).dropna()
            if not c.empty:
                kpis["total_cost"] = round(float(c.sum()), 2)
                kpis["average_cost_per_order"] = round(float(c.mean()), 2)

        breakdown = {}
        for label, col in (("material", mat_col), ("labour", lab_col), ("external", ext_col)):
            if col:
                s = num(col).dropna()
                if not s.empty:
                    breakdown[label] = round(float(s.sum()), 2)
        if breakdown:
            kpis["cost_breakdown"] = breakdown

        # ---- Downtime KPIs ----
        if down_col:
            d = num(down_col).dropna()
            if not d.empty:
                kpis["total_downtime_hours"] = round(float(d.sum()), 2)
                kpis["average_downtime_hours"] = round(float(d.mean()), 2)

        # ---- Grouped rollups (cost & downtime by plant / category / machine) ----
        def rollup(group_col, value_col, agg="sum", top=None):
            if not group_col or not value_col:
                return None
            w = df[[group_col]].copy()
            w["_v"] = num(value_col)
            w = w.dropna(subset=["_v"])
            if w.empty:
                return None
            g = getattr(w.groupby(group_col)["_v"], agg)().sort_values(ascending=False)
            g = g.round(2)
            if top:
                g = g.head(top)
            return {str(k): float(v) for k, v in g.items()}

        if cost_col and plant_col:
            r = rollup(plant_col, cost_col)
            if r: kpis["cost_by_plant"] = r
        if cost_col and cat_col:
            r = rollup(cat_col, cost_col)
            if r: kpis["cost_by_category"] = r
        if down_col and plant_col:
            r = rollup(plant_col, down_col)
            if r: kpis["downtime_by_plant"] = r
        if down_col and cat_col:
            r = rollup(cat_col, down_col)
            if r: kpis["downtime_by_category"] = r

        # ---- Work-order counts by category ----
        if cat_col:
            counts = df[cat_col].astype(str).value_counts()
            kpis["work_orders_by_category"] = {str(k): int(v) for k, v in counts.items()}
            kpis["total_work_orders"] = int(counts.sum())

        # ---- Top failing / most expensive machines ----
        if machine_col:
            if down_col:
                r = rollup(machine_col, down_col, "sum", top=5)
                if r: kpis["top_machines_by_downtime"] = r
            if cost_col:
                r = rollup(machine_col, cost_col, "sum", top=5)
                if r: kpis["top_machines_by_cost"] = r
            # breakdown frequency
            freq = df[machine_col].astype(str).value_counts().head(5)
            kpis["top_machines_by_work_order_count"] = {str(k): int(v) for k, v in freq.items()}

        # ---- Monthly trend (needs a usable date) ----
        if date_col:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            if dt.notna().mean() >= 0.5:  # at least half parse as dates
                tmp = pd.DataFrame({"_month": dt.dt.to_period("M").astype(str)})
                if cost_col:
                    tmp["_cost"] = num(cost_col)
                    m = tmp.dropna(subset=["_cost"]).groupby("_month")["_cost"].sum().round(2)
                    if not m.empty:
                        kpis["monthly_cost_trend"] = {str(k): float(v) for k, v in m.items()}
                if down_col:
                    tmp["_down"] = num(down_col)
                    m = tmp.dropna(subset=["_down"]).groupby("_month")["_down"].sum().round(2)
                    if not m.empty:
                        kpis["monthly_downtime_trend"] = {str(k): float(v) for k, v in m.items()}
            else:
                kpis["date_warning"] = (
                    f"Column '{date_col}' could not be parsed as dates, so time "
                    f"trends were skipped."
                )

        # ---- Production efficiency KPIs ----
        if prod_col and plan_col:
            p = num(prod_col); pl = num(plan_col)
            mask = pl.notna() & (pl != 0) & p.notna()
            if mask.any():
                kpis["overall_yield"] = round(float(p[mask].sum() / pl[mask].sum()), 4)
        if def_col and prod_col:
            d = num(def_col); p = num(prod_col)
            mask = p.notna() & (p != 0) & d.notna()
            if mask.any():
                kpis["overall_defect_rate"] = round(float(d[mask].sum() / p[mask].sum()), 4)
        if prod_col and time_col:
            p = num(prod_col); t = num(time_col)
            mask = t.notna() & (t != 0) & p.notna()
            if mask.any():
                kpis["overall_efficiency_per_hour"] = round(float(p[mask].sum() / t[mask].sum()), 2)

        # ---- Multi-dimensional rollups (plant x category x month) ----
        # These give the frontend chart-ready structures for slicing cost and
        # downtime across more than one dimension at once.
        month_series = None
        if date_col:
            _dt = pd.to_datetime(df[date_col], errors="coerce")
            if _dt.notna().mean() >= 0.5:
                month_series = _dt.dt.to_period("M").astype(str)

        def crosstab(value_col, row_col, col_col, agg="sum"):
            """Return {row: {col: value}} for a two-dimension breakdown."""
            if not value_col or row_col is None or col_col is None:
                return None
            work = pd.DataFrame({
                "_r": df[row_col] if isinstance(row_col, str) else row_col,
                "_c": df[col_col] if isinstance(col_col, str) else col_col,
                "_v": num(value_col),
            }).dropna(subset=["_v"])
            if work.empty:
                return None
            pivot = work.pivot_table(index="_r", columns="_c", values="_v",
                                     aggfunc=agg, fill_value=0)
            pivot = pivot.round(2)
            return {str(r): {str(c): float(v) for c, v in row.items()}
                    for r, row in pivot.iterrows()}

        multi = {}

        # cost: plant x category
        if cost_col and plant_col and cat_col:
            ct = crosstab(cost_col, plant_col, cat_col)
            if ct: multi["cost_by_plant_and_category"] = ct

        # cost: plant x month
        if cost_col and plant_col and month_series is not None:
            ct = crosstab(cost_col, plant_col, month_series)
            if ct: multi["cost_by_plant_and_month"] = ct

        # cost: category x month
        if cost_col and cat_col and month_series is not None:
            ct = crosstab(cost_col, cat_col, month_series)
            if ct: multi["cost_by_category_and_month"] = ct

        # downtime: plant x category
        if down_col and plant_col and cat_col:
            ct = crosstab(down_col, plant_col, cat_col)
            if ct: multi["downtime_by_plant_and_category"] = ct

        # downtime: plant x month
        if down_col and plant_col and month_series is not None:
            ct = crosstab(down_col, plant_col, month_series)
            if ct: multi["downtime_by_plant_and_month"] = ct

        # Always include multi_dimensional. If the required columns are absent,
        # say so explicitly (with what's missing) instead of omitting it, so the
        # frontend can render a clear "add these columns" message.
        if multi:
            kpis["multi_dimensional"] = multi
        else:
            missing = []
            if not plant_col:
                missing.append("plant")
            if not cat_col:
                missing.append("category (work_order_type)")
            if month_series is None:
                missing.append("date")
            if not cost_col and not down_col:
                missing.append("cost or downtime")
            kpis["multi_dimensional"] = {
                "available": False,
                "reason": (
                    "Cross-tab KPIs need at least two of: plant, category, date - "
                    "plus a cost or downtime column."
                ),
                "missing_columns": missing,
            }

        return kpis

    @classmethod
    def assess_data_readiness(cls, df) -> dict:
        """
        One upfront verdict on what this dataset can and cannot support, so the
        user is told "forecasting won't work, no usable dates" ONCE, instead of
        discovering it through a per-task failure.

        Returns capability flags + human-readable messages.
        """
        cols = list(df.columns)
        n = len(df)
        report = {
            "row_count": int(n),
            "column_count": len(cols),
            "can_forecast": False,
            "can_trend": False,
            "can_predict_downtime": False,
            "can_cost_analysis": False,
            "issues": [],
            "ready_for": [],
        }

        # ---- Date usability (drives forecasting + trends) ----
        date_col = cls._kpi_col(cols, "date")
        usable_date = False
        if date_col:
            dt = pd.to_datetime(df[date_col], errors="coerce")
            frac = float(dt.notna().mean())
            if frac >= 0.5:
                usable_date = True
                report["date_column"] = date_col
            else:
                report["issues"].append(
                    f"Date column '{date_col}' is only {frac:.0%} parseable, so "
                    f"forecasting and time trends are unavailable. Populate it "
                    f"with real dates to enable them."
                )
        else:
            report["issues"].append(
                "No date column found. Forecasting and monthly trends need a "
                "real date column (e.g. work-order or posting date)."
            )

        report["can_forecast"] = usable_date and n >= 30
        report["can_trend"] = usable_date

        # ---- Cost analysis ----
        cost_col = cls._kpi_col(cols, "cost_total")
        if cost_col:
            report["can_cost_analysis"] = True
            report["cost_column"] = cost_col
        else:
            report["issues"].append(
                "No cost column found - cost KPIs (spend by plant/category) "
                "need a cost field such as actual_cost / material_cost."
            )

        # ---- Downtime prediction ----
        down_col = cls._kpi_col(cols, "downtime")
        machine_col = cls._kpi_col(cols, "machine")
        if down_col:
            report["can_predict_downtime"] = True
            report["downtime_column"] = down_col
        else:
            report["issues"].append(
                "No downtime column found - downtime prediction needs a "
                "downtime_hours (or equivalent) field."
            )

        # ---- Summary of what IS possible ----
        if report["can_cost_analysis"]:
            report["ready_for"].append("cost analysis")
        if cls._kpi_col(cols, "plant"):
            report["ready_for"].append("plant comparison")
        if cls._kpi_col(cols, "category"):
            report["ready_for"].append("category breakdown")
        if report["can_predict_downtime"]:
            report["ready_for"].append("downtime prediction")
        if report["can_forecast"]:
            report["ready_for"].append("forecasting")
        if report["can_trend"]:
            report["ready_for"].append("time trends")

        # A short plain-language headline
        if report["ready_for"]:
            report["summary"] = (
                "This dataset supports: " + ", ".join(report["ready_for"]) + "."
            )
        else:
            report["summary"] = (
                "This dataset is missing the key columns (dates, cost, "
                "downtime) needed for maintenance analytics."
            )
        if report["issues"]:
            report["summary"] += " " + str(len(report["issues"])) + " limitation(s) found."

        return report

    def _load_df_from_blob(self, blob_file: str):
        """Shared blob->DataFrame loader. Returns None on failure."""
        import io
        try:
            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            blob_client = container_client.get_blob_client(blob_file)
            data = blob_client.download_blob().readall()
            if blob_file.endswith(".csv"):
                return pd.read_csv(io.BytesIO(data))
            if blob_file.endswith((".xlsx", ".xls")):
                return pd.read_excel(io.BytesIO(data))
        except Exception as e:
            logger.warning(f"Could not load dataframe from blob: {e}")
        return None

    def assess_readiness_from_blob(self, blob_file: str) -> dict:
        """Load a dataset and assess readiness. Never raises."""
        df = self._load_df_from_blob(blob_file)
        if df is None:
            return {"available": False, "reason": "could not load dataset"}
        try:
            r = self.assess_data_readiness(df)
            r["available"] = True
            return r
        except Exception as e:
            logger.warning(f"Readiness check failed: {e}")
            return {"available": False, "reason": str(e)}

    def compute_kpis_from_blob(self, blob_file: str) -> dict:
        """Load a dataset from blob and compute KPIs. Never raises."""
        import io
        try:
            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            blob_client = container_client.get_blob_client(blob_file)
            data = blob_client.download_blob().readall()
            if blob_file.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(data))
            elif blob_file.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(data))
            else:
                return {"available": False, "reason": "unsupported file format"}
            return self.compute_maintenance_kpis(df)
        except Exception as e:
            logger.warning(f"Could not compute KPIs: {e}")
            return {"available": False, "reason": str(e)}

    @classmethod
    def _answer_data_question(cls, query: str, df) -> str:
        """
        Resolve a natural-language question against the dataframe.

        Handles, in order:
          1. row counts
          2. derived-metric questions, optionally "by <group>"
             (e.g. "which shift is most efficient")
          3. plain aggregate "by <group>" (e.g. "total ProducedQuantity by Shift")
          4. plain single-column aggregate (e.g. "total ProducedQuantity")
        Returns "" if it cannot answer, so the caller can fall back.
        """
        import re as _re
        ql = query.lower()
        columns = list(df.columns)

        # ---- 1. Row count ----
        if _re.search(r"how many (rows|records|entries)", ql) or ql.strip() in (
            "how many rows", "number of rows", "row count"
        ):
            return f"The dataset has {len(df):,} rows."

        op = cls._detect_operation(ql)
        explicit_group = cls._detect_group_by(query, columns)  # from "by/per <col>"

        # "which <group> has the highest <value>" - the group comes right after
        # "which"; the value is a different column mentioned later.
        which_group = None
        is_which = ql.strip().startswith("which") or " which " in ql
        if is_which:
            which_group = cls._detect_which_group(query, columns)

        group_col = explicit_group or which_group

        # When comparing groups ("which shift has the HIGHEST production"),
        # highest/most/lowest describe the RANK across group TOTALS, not a
        # per-row max/min. So default the aggregate to sum for these.
        if group_col and op in ("max", "min", None):
            # keep an explicit average/count if the user said so
            if any(w in ql for w in ("average", "avg", "mean")):
                op = "mean"
            elif "count" in ql or "how many" in ql:
                op = "count"
            else:
                op = "sum"

        # ---- 2. Derived metric (efficiency, yield, defect rate, ...) ----
        metric_key = None
        for name, spec in cls.DERIVED_METRICS.items():
            if any(kw in ql for kw in spec["keywords"]):
                metric_key = name
                break
        if metric_key:
            # If no explicit group was named but the question is comparative
            # ("which shift is most efficient"), infer the group from a
            # "which <col>" phrase.
            if not group_col:
                group_col = cls._detect_which_group(query, columns)
            # Respect an explicit direction word in the question ("highest",
            # "lowest") over the metric's natural better-direction.
            want_high = None
            if any(w in ql for w in ("highest", "most", "largest", "top", "best", "maximum", "max")):
                want_high = True
            elif any(w in ql for w in ("lowest", "least", "smallest", "worst", "minimum", "min")):
                want_high = False
            ans = cls._answer_derived_metric(df, metric_key, group_col, want_high)
            if ans:
                return ans

        # ---- 3 & 4. Plain aggregate (optionally grouped) ----
        has_group_signal = bool(explicit_group) or is_which
        if op or has_group_signal:
            agg = op or "sum"
            value_col = cls._resolve_value_column(query, columns, exclude=group_col)

            if has_group_signal and group_col and value_col and value_col != group_col:
                want_high = not any(w in ql for w in
                    ("lowest", "least", "smallest", "worst", "minimum", "min"))
                return cls._answer_grouped_aggregate(df, agg, value_col, group_col, want_high)
            # No grouping -> a plain single-column total/average/etc.
            if value_col and (not group_col or not has_group_signal):
                # if value_col accidentally resolved to the group, re-resolve
                if value_col == group_col:
                    value_col = cls._resolve_value_column(query, columns, exclude=group_col)
                if value_col and value_col != group_col:
                    return cls._answer_simple_aggregate(df, agg, value_col)

        return ""

    @classmethod
    def _detect_operation(cls, ql: str):
        for op, words in cls.AGG_WORDS.items():
            if any(w in ql for w in words):
                return op
        return None

    @staticmethod
    def _detect_which_group(query: str, columns: list):
        """
        For comparative questions with no explicit 'by', infer the grouping
        column from a 'which <col>' phrase, e.g. 'which shift is most efficient'
        -> Shift. Falls back to any low-cardinality categorical column named in
        the query.
        """
        import re as _re
        ql = query.lower()
        m = _re.search(r"which\s+([a-z_ ]+?)\s+(?:is|has|had|are|were|gives|produces)", ql)
        if m:
            col = AgentHandler._match_column_in_query(m.group(1).strip(), columns)
            if col:
                return col
        # otherwise: any column name mentioned in the query
        return AgentHandler._match_column_in_query(query, columns)

    @staticmethod
    def _detect_group_by(query: str, columns: list):
        """Find the column after 'by'/'per'/'for each', or a categorical mention."""
        import re as _re
        ql = query.lower()

        m = _re.search(r"\b(?:by|per|for each|grouped by)\s+([a-z_ ]+)", ql)
        if m:
            phrase = m.group(1).strip()
            col = AgentHandler._match_column_in_query(phrase, columns)
            if col:
                return col
        return None

    @staticmethod
    def _answer_simple_aggregate(df, op, col) -> str:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty and op != "count":
            return f"Column '{col}' has no numeric values to compute a {op}."
        label = {"sum": "total", "mean": "average", "count": "count",
                 "max": "maximum", "min": "minimum"}[op]
        if op == "count":
            return f"The {label} of non-empty '{col}' values is {int(series.count()):,}."
        val = getattr(series, op)()
        return f"The {label} {col} is {val:,.2f}."

    @staticmethod
    def _answer_grouped_aggregate(df, op, target_col, group_col, want_high=True) -> str:
        work = df[[group_col, target_col]].copy()
        work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
        work = work.dropna(subset=[target_col])
        if work.empty:
            return f"No numeric '{target_col}' values to aggregate by {group_col}."

        grouped = getattr(work.groupby(group_col)[target_col], op)()
        total_groups = len(grouped)
        winner = grouped.sort_values(ascending=not want_high)
        direction = "highest" if want_high else "lowest"
        top_key = winner.index[0]
        top_val = winner.iloc[0]

        label = {"sum": "Total", "mean": "Average", "count": "Count",
                 "max": "Maximum", "min": "Minimum"}[op]

        # Show only the top 5 — dumping all groups (40+ employees, 15
        # products, etc.) is never a chatbot answer.
        top_n = winner.head(5)
        lines = [f"**{label} {target_col} by {group_col} — Top 5:**", ""]
        for key, val in top_n.items():
            lines.append(f"- {key}: {val:,.2f}")
        if total_groups > 5:
            lines.append(f"  *(and {total_groups - 5} more)*")
        lines.append("")
        lines.append(
            f"**{top_key}** has the {direction} {label.lower()} "
            f"{target_col} ({top_val:,.2f})."
        )
        return "\n".join(lines)

    @classmethod
    def _answer_derived_metric(cls, df, metric_key, group_col, want_high=None) -> str:
        spec = cls.DERIVED_METRICS[metric_key]
        num_col = cls._find_col_by_hints(df.columns, spec["num"])
        den_col = cls._find_col_by_hints(df.columns, spec["den"])
        label = spec["label"]
        higher_is_better = spec["higher_is_better"]
        if not num_col or not den_col:
            return ""

        work = df.copy()
        work[num_col] = pd.to_numeric(work[num_col], errors="coerce")
        work[den_col] = pd.to_numeric(work[den_col], errors="coerce")
        work = work.dropna(subset=[num_col, den_col])
        work = work[work[den_col] != 0]
        if work.empty:
            return ""

        if group_col:
            num_sum = work.groupby(group_col)[num_col].sum()
            den_sum = work.groupby(group_col)[den_col].sum()
            ratio = num_sum / den_sum

            # Which end the user actually asked about
            pick_high = want_high if want_high is not None else higher_is_better
            ranked = ratio.sort_values(ascending=not pick_high)

            lines = [f"**{label.title()} by {group_col}** "
                     f"({num_col} / {den_col}):", ""]
            for key, val in ratio.sort_values(ascending=False).items():
                lines.append(f"- {key}: {val:,.4f}")

            winner = ranked.index[0]
            winner_val = ranked.iloc[0]
            direction = "highest" if pick_high else "lowest"
            # Only call it "best" when the chosen direction matches what is
            # genuinely better for this metric.
            quality = "best" if pick_high == higher_is_better else "notable"
            lines.append("")
            lines.append(
                f"**{winner}** has the {direction} {label} "
                f"({winner_val:,.4f})."
            )
            return "\n".join(lines)

        overall = work[num_col].sum() / work[den_col].sum()
        return (f"The overall {label} ({num_col} / {den_col}) is "
                f"{overall:,.4f}.")

    @staticmethod
    def _find_col_by_hints(columns, hints):
        """First column whose lowercased name contains any hint substring."""
        for col in columns:
            cl = col.lower()
            if any(h in cl for h in hints):
                return col
        return None

    @classmethod
    def _resolve_value_column(cls, query, columns, exclude=None):
        """
        Find the numeric column a question refers to as its VALUE, excluding the
        grouping column. Tries synonyms ("production" -> ProducedQuantity)
        before a direct name match.
        """
        import re as _re
        ql = query.lower()
        cand = [c for c in columns if c != exclude]

        # 1. synonym hits - score every candidate, prefer the most specific.
        #    "production cost" must resolve to ProductionCost, not
        #    ProducedQuantity, even though both "production" and "cost" appear.
        #    Longer matched column names win (ProductionCost > ProducedQuantity
        #    only when "cost" is present), and an exact word like "cost" that is
        #    itself a column-name token is preferred.
        priority_words = [w for w in ("cost", "downtime", "defective", "defects",
                                      "planned", "produced", "production", "output",
                                      "quantity") if w in ql]
        for word in priority_words:
            hints = cls.COLUMN_SYNONYMS.get(word, (word,))
            for h in hints:
                for c in cand:
                    if cls._norm(h) in cls._norm(c):
                        return c

        # 2. direct name match (ignoring the group column)
        col = cls._match_column_in_query(query, cand)
        if col:
            return col
        return None

    @staticmethod
    def _match_column_in_query(query: str, columns: list):
        """Find which dataset column a question is about (fuzzy, token-based)."""
        import re as _re
        ql = query.lower()

        # 1. exact / substring: column name appears in the query
        best = None
        best_len = 0
        for col in columns:
            cl = col.lower()
            # compare ignoring spaces/underscores
            norm_col = _re.sub(r"[_\s]+", "", cl)
            norm_q = _re.sub(r"[_\s]+", "", ql)
            if norm_col in norm_q and len(norm_col) > best_len:
                best, best_len = col, len(norm_col)
        if best:
            return best

        # 2. token overlap
        q_tokens = set(_re.split(r"[_\s]+", ql))
        best, best_score = None, 0
        for col in columns:
            c_tokens = set(_re.split(r"[_\s]+", col.lower()))
            overlap = len(q_tokens & c_tokens)
            if overlap > best_score:
                best, best_score = col, overlap
        return best if best_score > 0 else None

    def process_query_on_main_agent(self, agent_id: str, query: str, thread_id: str = None, user_id: str = None, agent_name: str = None, raw_query: str = None):
        try:
            if thread_id is None:
                thread = self.client.agents.threads.create()
                thread_id = thread.id

            blob_path = self.get_latest_blob_file(user_id, agent_name) if user_id and agent_name else None
            filename = blob_path.split('/')[-1] if blob_path else None

            # Make sure THIS thread actually has the CURRENT dataset attached
            # for Code Interpreter before asking anything - closes the gap
            # where a query on a thread other than the original upload
            # thread can't see the file (see _ensure_file_attached_to_thread).
            # Resolved by filename, not just "whatever was uploaded to this
            # agent last" - a user who has ever uploaded more than one file
            # (even in a different thread) would otherwise risk attaching
            # the wrong file to this thread.
            self._ensure_file_attached_to_thread(thread_id, user_id, agent_id, filename)

            # Give the agent the REAL column names so it never invents a target
            # like "sales" or "downtime_cause" that is not in the dataset. This
            # is the fix for the agent picking columns that do not exist.
            columns = []
            if blob_path and user_id:
                try:
                    columns = self.get_dataset_columns(blob_path, user_id)
                except Exception as col_err:
                    logger.warning(f"Could not read columns for prompt: {col_err}")

            if columns:
                col_block = (
                    "\n\nThe uploaded dataset has EXACTLY these columns "
                    "(you MUST choose 'target' only from this list, copied "
                    "character-for-character; NEVER invent or guess a column "
                    f"name that is not here):\n{columns}\n"
                    "If the user's requested target is not clearly one of these "
                    "columns, pick the closest matching real column."
                )
            else:
                col_block = ""

            if filename:
                enhanced_query = f"{query} (using uploaded file: {filename}){col_block}"
            else:
                enhanced_query = f"{query}{col_block}"

            self.client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=enhanced_query
            )

            # Retries transient rate_limit_exceeded instead of failing the job.
            run = self._run_agent_with_retry(thread_id, agent_id)

            # CRITICAL: filter to messages produced by THIS run (run_id=run.id).
            # Without this filter, if this run's own reply has no usable text
            # content for any reason, this loop would silently fall through
            # to an older assistant message from a PREVIOUS turn and return
            # that instead - answering a completely different question than
            # the one just asked. Also skip past any assistant message that
            # has no real text (e.g. a tool-call-only step) instead of
            # breaking on the first role match regardless of content.
            messages = self.client.agents.messages.list(thread_id=thread_id, run_id=run.id)
            latest_assistant_msg = None
            for msg in messages:
                if msg.role == "assistant" and msg.content and any(
                    getattr(c, "type", None) == "text" and getattr(c.text, "value", "").strip()
                    for c in msg.content
                ):
                    latest_assistant_msg = msg
                    break

            if not latest_assistant_msg or not latest_assistant_msg.content:
                raise Exception("No assistant response")

            full_response = ""
            for content in latest_assistant_msg.content:
                if content.type == "text":
                    full_response += content.text.value

            if not full_response.strip():
                raise Exception("Empty response")

            ml_json = extract_json_from_text(full_response)

            if ml_json and isinstance(ml_json, dict):
                if ml_json.get("is_ml") and "task_type" in ml_json and "target" in ml_json:
                    # Validate that the target column actually exists in the
                    # dataset for supervised tasks. Without this, a thread
                    # biased by previous ML runs (e.g. forecasting) can return
                    # the WRONG task/target for a completely different query
                    # (e.g. "Predict production cost" returning a forecasting
                    # JSON for quantityproduced because that was last time).
                    task_t = ml_json.get("task_type", "").lower()
                    target_v = (ml_json.get("target") or "").strip()
                    if task_t not in ("clustering", "anomaly_detection") and target_v and target_v.lower() != "none":
                        if columns and target_v.lower() not in {c.lower() for c in columns}:
                            logger.warning(
                                f"Agent returned non-existent target '{target_v}' for task "
                                f"'{task_t}'; re-classifying from raw query to avoid wrong ML run."
                            )
                            reclassified = self._reclassify_as_ml_json(raw_query or query, agent_id, columns)
                            if reclassified and reclassified.get("is_ml"):
                                logger.info(f"Re-classification corrected JSON: {reclassified}")
                                return thread_id, reclassified, None
                            # Re-classification also failed to find a real target -
                            # fall through and answer from the data instead.
                            logger.warning("Re-classification found no valid target; falling through to data answer.")
                            ml_json = None
                        else:
                            logger.info(f"ML JSON detected: {ml_json}")
                            return thread_id, ml_json, None
                    else:
                        logger.info(f"ML JSON detected: {ml_json}")
                        return thread_id, ml_json, None
                elif ml_json.get("is_ml"):
                    logger.warning(f"Incomplete ML JSON: {ml_json}")
                    return thread_id, ml_json, None

            # NON-ML.
            #
            # The classifier agent replies with a control JSON such as
            # {"is_ml": false}. That is a routing signal, NOT an answer - it must
            # never be shown to the user (previously it was returned verbatim, so
            # "What is the total planned quantity?" got "{ \"is_ml\": false }").
            #
            # If the whole response is just that control JSON, produce a real
            # answer: greetings get a friendly reply; data questions are answered
            # from the dataset itself.
            logger.info("Analysis response (non-ML)")

            answer = full_response.strip()

            # ROUTING RESCUE: the agent sometimes DESCRIBES an ML task in prose
            # ("Run KMeans on quantityproduced, unitcost, productioncost;
            # select k by silhouette.") instead of emitting the is_ml control
            # JSON that actually triggers model building. That is a routing
            # failure, not an answer - the request WAS an ML request. Rather
            # than showing the user that fragment (or a "rephrase it" message),
            # re-ask the agent to classify the ORIGINAL question as JSON only.
            # If that succeeds, the request routes to AutoML as it should have.
            if self._looks_like_ml_task_prose(answer):
                logger.info("ML task described in prose; re-classifying for is_ml JSON")
                reclassified = self._reclassify_as_ml_json(
                    raw_query or query, agent_id, columns
                )
                if reclassified and reclassified.get("is_ml"):
                    logger.info(f"Re-classification recovered ML JSON: {reclassified}")
                    return thread_id, reclassified, None
                # Not actually an ML request after all - fall through and
                # answer it from the data below rather than echoing the prose.
                answer = ""

            # The agent sometimes returns a control JSON ({"is_ml": false}) or a
            # raw code snippet (df.groupby(...)) instead of a real answer. Never
            # show those to the user - answer the question ourselves from the
            # data. Only keep the agent's text if it is a genuine prose answer.
            if (self._is_control_json(answer, ml_json)
                    or self._looks_like_code(answer)
                    or not answer):
                answer = self._answer_non_ml_query(
                    raw_query or query, agent_id, thread_id, user_id, agent_name
                )

            return thread_id, None, answer
        
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise

    @staticmethod
    def _compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collapse per-horizon metrics (h1_rmse ... h30_mape) into summaries.

        A 30-horizon multistep run produces 120 per-horizon keys per model.
        Dumping them all made the analysis prompt ~32,000 characters (~8,000
        tokens) of noise, which is what made the agent run fail.
        """
        if not isinstance(metrics, dict):
            return metrics

        compact: Dict[str, Any] = {}
        per_horizon: Dict[str, list] = {}

        for key, value in metrics.items():
            m = re.match(r"^h(\d+)_(.+)$", str(key))
            if m:
                per_horizon.setdefault(m.group(2), []).append(value)
            else:
                compact[key] = value

        for metric_name, values in per_horizon.items():
            nums = [v for v in values
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not nums:
                continue
            compact[f"{metric_name}_across_horizons"] = {
                "n_horizons": len(nums),
                "best": round(min(nums), 4),
                "worst": round(max(nums), 4),
                "mean": round(sum(nums) / len(nums), 4),
            }

        return compact

    @classmethod
    def _compact_results_for_analysis(cls, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Shrink results.json to what an analyst actually needs."""
        if not isinstance(results_data, dict):
            return results_data

        compact = {}
        for key, value in results_data.items():
            if key in ("test_metrics", "train_metrics"):
                compact[key] = cls._compact_metrics(value)
            elif key in ("all_models", "all_test_results") and isinstance(value, dict):
                compact[key] = {}
                for model_name, model_info in value.items():
                    if isinstance(model_info, dict):
                        entry = {k: v for k, v in model_info.items()
                                 if k not in ("train_metrics", "test_metrics")}
                        for mk in ("train_metrics", "test_metrics"):
                            if mk in model_info:
                                entry[mk] = cls._compact_metrics(model_info[mk])
                        compact[key][model_name] = entry
                    else:
                        compact[key][model_name] = model_info
            else:
                compact[key] = value
        return compact

    @staticmethod
    def _fallback_summary(results_data: Dict[str, Any]) -> str:
        """
        Deterministic report used when the LLM analysis is unavailable.

        The model itself trained fine; a summariser outage must not turn a
        successful run into a failed job.
        """
        best = results_data.get("best_model", "the selected model")
        task = results_data.get("task", "model")
        target = results_data.get("target", "the target")
        metric = results_data.get("metric", "score")
        test = results_data.get("test_metrics", {}) or {}

        def g(*names):
            for n in names:
                if n in test and isinstance(test[n], (int, float)):
                    return test[n]
            return None

        r2 = g("avg_r2", "r2")
        rmse = g("avg_rmse", "rmse")
        mae = g("avg_mae", "mae")

        lines = [
            "## Model Summary",
            "",
            f"- **Task:** {task}",
            f"- **Target:** {target}",
            f"- **Best model:** {best}",
            f"- **Primary metric:** {metric}",
            "",
            "### Test performance",
            "",
        ]
        for label, val in (("R2", r2), ("RMSE", rmse), ("MAE", mae)):
            if val is not None:
                lines.append(f"- {label}: {round(val, 4)}")

        if isinstance(r2, (int, float)) and r2 < 0:
            lines += [
                "",
                "### Warning",
                "",
                "A negative R2 means the model performs worse on unseen data "
                "than simply predicting the average. This usually indicates the "
                "dataset does not support this task - for forecasting, check "
                "that a usable date column exists and that rows are ordered in "
                "time.",
            ]

        lines += ["", "_(Automated summary: the AI analyst was unavailable.)_"]
        return "\n".join(lines)

    def generate_query_response(
        self,
        agent_id: str,
        query: str,
        results_data: Dict[str, Any],
        task_type: str,
        thread_id: str = None,
    ) -> str:
        """
        Answer the user's actual QUESTION in business terms, using the real
        predicted values (from results_data["prediction_summary"]) plus the
        headline metrics.

        This is different from `analysis`:
          - analysis        = a report card on how the MODELS performed
          - query_response  = the answer to what the USER asked, e.g. the
                              forecast itself: expected value, peak, trend,
                              a recommendation.

        Free-form markdown; NOT the metrics schema. Always returns something
        (deterministic fallback) so it can never fail the job.
        """
        pred = (results_data or {}).get("prediction_summary", {}) or {}
        headline = self._headline_facts(results_data, task_type)

        try:
            if thread_id is None:
                thread = self.client.agents.threads.create()
                thread_id = thread.id

            prompt = (
                "A user asked this question about their data:\n"
                f'"{query}"\n\n'
                "An AutoML pipeline has already run and produced these VERIFIED "
                "results. Use ONLY these numbers - do not invent any.\n\n"
                f"Prediction summary (the actual outputs):\n{json.dumps(pred, indent=2, default=str)}\n\n"
                f"Model quality (context):\n{json.dumps(headline, indent=2, default=str)}\n\n"
                "Write a direct, well-formatted answer to the user's question "
                "as a business analyst would - lead with a short descriptive "
                "title of the FINDING itself (e.g. 'Defect Quantity Forecast'), "
                "give the concrete predicted figures (expected value, average, "
                "peak/low, trend where relevant), then one 'Insight' line and "
                "one 'Recommendation' line. Use markdown. Do NOT restate, "
                "quote, or repeat the user's question verbatim anywhere in "
                "your answer, including as the title - go straight to the "
                "finding. Do NOT include model accuracy tables or a "
                "methodology report - that is handled separately. If the "
                "model quality is poor (e.g. negative R2), say so honestly "
                "in the insight."
            )

            self.client.agents.messages.create(
                thread_id=thread_id, role="user", content=prompt
            )
            run = self.client.agents.runs.create_and_process(
                thread_id=thread_id, agent_id=agent_id
            )

            if run.status != "completed":
                last_error = getattr(run, "last_error", None)
                logger.warning(
                    f"Query-response run did not complete "
                    f"(status={run.status}, error={last_error}); using fallback"
                )
                return self._fallback_query_response(query, results_data, task_type)

            messages = self.client.agents.messages.list(thread_id=thread_id, run_id=run.id)
            for msg in messages:
                if msg.role == "assistant" and msg.content:
                    text = ""
                    for c in msg.content:
                        if c.type == "text":
                            text += c.text.value
                    if text.strip():
                        return text.strip()

            return self._fallback_query_response(query, results_data, task_type)

        except Exception as e:
            logger.warning(f"Query response unavailable, using fallback: {e}")
            return self._fallback_query_response(query, results_data, task_type)

    @staticmethod
    def _headline_facts(results_data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Pull just the model-quality numbers a one-paragraph answer needs."""
        test = results_data.get("test_metrics", {}) or {}
        facts = {
            "task": task_type,
            "target": results_data.get("target"),
            "best_model": results_data.get("best_model"),
        }
        t = (task_type or "").lower()
        if "classification" in t:
            for k in ("f1", "accuracy", "precision", "recall", "roc_auc"):
                if k in test and isinstance(test[k], (int, float)):
                    facts[k] = round(test[k], 4)
        elif "regression" in t or "forecast" in t:
            for k in ("avg_r2", "r2", "avg_rmse", "rmse", "avg_mae", "mae"):
                if k in test and isinstance(test[k], (int, float)):
                    facts[k] = round(test[k], 4)
        elif "clustering" in t:
            bm = results_data.get("best_model")
            am = (results_data.get("all_models", {}) or {}).get(bm, {})
            info = am.get("test", am) if isinstance(am, dict) else {}
            for k in ("n_clusters", "silhouette_score"):
                if isinstance(info, dict) and k in info:
                    facts[k] = round(info[k], 4) if isinstance(info[k], (int, float)) else info[k]
        elif "anomaly" in t:
            bm = results_data.get("best_model")
            am = (results_data.get("all_models", {}) or {}).get(bm, {})
            info = am.get("test", am) if isinstance(am, dict) else {}
            for k in ("n_anomalies", "anomaly_percentage"):
                if isinstance(info, dict) and k in info:
                    facts[k] = round(info[k], 4) if isinstance(info[k], (int, float)) else info[k]
        return facts

    @classmethod
    def _fallback_query_response(
        cls, query: str, results_data: Dict[str, Any], task_type: str
    ) -> str:
        """
        Deterministic, business-facing answer built from prediction_summary.
        Used when the LLM is unavailable - still formatted, still useful.
        """
        pred = (results_data or {}).get("prediction_summary", {}) or {}
        headline = cls._headline_facts(results_data, task_type)
        target = results_data.get("target") or "the target"
        best = results_data.get("best_model") or "the model"
        t = (task_type or "").lower()

        def fmt(x):
            return f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)

        # ---- Forecast ----
        if "forecast" in t and pred.get("kind") == "forecast":
            h = pred.get("horizon")
            lines = [
                f"## {target} Forecast — Next {h} Steps",
                "",
                f"Based on historical patterns in your dataset, the model "
                f"forecasts **{target}** for the next {h} steps.",
                "",
                "**Forecast Summary**",
                "",
                f"- Forecast period: next {h} steps",
                f"- Total expected {target}: {fmt(pred.get('total_predicted'))}",
                f"- Average per step: {fmt(pred.get('average_per_step'))}",
                f"- Expected trend: {str(pred.get('trend','')).title()}",
                f"- Peak: {fmt(pred.get('peak_value'))} at step {pred.get('peak_step')}",
                f"- Lowest: {fmt(pred.get('lowest_value'))} at step {pred.get('lowest_step')}",
                "",
            ]
            r2 = headline.get("avg_r2", headline.get("r2"))
            if isinstance(r2, (int, float)) and r2 < 0:
                lines.append(
                    f"**Insight:** The model could not find a reliable pattern "
                    f"(R\u00b2 {r2}); these figures are unreliable. Your data likely "
                    f"lacks a usable time order for forecasting."
                )
            else:
                _trend = pred.get("trend", "stable")
                _verb = {"increasing": "increase",
                         "decreasing": "decrease",
                         "stable": "remain stable"}.get(_trend, "remain stable")
                lines.append(
                    f"**Insight:** {target} is expected to {_verb} "
                    f"over the forecast window compared with the recent trend."
                )
            lines.append(
                "**Recommendation:** Consider adjusting inventory, production, "
                "and supply planning based on the predicted demand."
            )
            return "\n".join(lines)

        # ---- Regression ----
        if "regression" in t and pred.get("kind") == "regression":
            lines = [
                f"## Predicted {target}",
                "",
                f"- Predictions generated: {pred.get('n_predictions')}",
                f"- Expected range: {fmt(pred.get('predicted_min'))} to {fmt(pred.get('predicted_max'))}",
                f"- Average predicted {target}: {fmt(pred.get('predicted_mean'))}",
                f"- Median predicted {target}: {fmt(pred.get('predicted_median'))}",
            ]
            r2 = headline.get("r2", headline.get("avg_r2"))
            if isinstance(r2, (int, float)) and r2 < 0:
                lines += ["", f"**Insight:** Model reliability is low (R\u00b2 {r2}); "
                              f"treat these predictions with caution."]
            return "\n".join(lines)

        # ---- Classification ----
        if "classification" in t and pred.get("kind") == "classification":
            dist = pred.get("predicted_class_distribution", {})
            top = pred.get("most_common_prediction")
            lines = [
                f"## {target} — Predicted Classes",
                "",
                f"The model predicts **{target}** across {pred.get('n_predictions')} records.",
                "",
                "**Predicted distribution:**",
                "",
            ]
            for cls_name, cnt in sorted(dist.items(), key=lambda kv: -kv[1]):
                lines.append(f"- {cls_name}: {cnt}")
            lines += ["", f"**Insight:** Most records are predicted as "
                          f"**{top}**. Best model: {best} "
                          f"(F1 {headline.get('f1','N/A')})."]
            return "\n".join(lines)

        # ---- Clustering ----
        if "clustering" in t and pred.get("kind") == "clustering":
            sizes = pred.get("cluster_sizes", {})
            lines = [
                f"## Data Segmentation Result",
                "",
                f"Your data was grouped into **{pred.get('n_clusters')} clusters** "
                f"using {best}.",
                "",
                "**Cluster sizes:**",
                "",
            ]
            for c, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
                label = "noise" if c == "-1" else f"cluster {c}"
                lines.append(f"- {label}: {n} records")
            lines += ["", f"**Insight:** The largest group is "
                          f"cluster {pred.get('largest_cluster')}. "
                          f"Profile each cluster to understand what defines it."]
            return "\n".join(lines)

        # ---- Anomaly ----
        if "anomaly" in t and pred.get("kind") == "anomaly_detection":
            return "\n".join([
                f"## Anomaly Detection Result",
                "",
                f"The model scanned **{pred.get('n_records')} records** and flagged "
                f"**{pred.get('n_anomalies')}** as anomalies "
                f"({pred.get('anomaly_percentage')}% of the data) using {best}.",
                "",
                "**Recommendation:** Review the flagged records for root-cause "
                "analysis; these are the points that deviate most from normal.",
            ])

        # ---- Generic fallback (no prediction_summary available) ----
        return (
            f"I completed the {task_type} task on **{target}** using {best}. "
            f"See the detailed report below for full results."
        )

    def generate_ml_analysis(self, agent_id: str, results_data: Dict[str, Any], thread_id: str = None) -> str:
        try:
            if thread_id is None:
                thread = self.client.agents.threads.create()
                thread_id = thread.id

            # Shrink the payload before sending. The raw results.json for a
            # 30-horizon run is ~32k chars, almost all per-horizon metrics.
            compact_results = self._compact_results_for_analysis(results_data)
            # Compact separators: indent=2 wastes ~40% of the payload on
            # whitespace, and an LLM reads minified JSON just as well.
            payload = json.dumps(
                compact_results, separators=(",", ":"), default=str
            )

            # Hard cap as a last resort
            MAX_PAYLOAD_CHARS = 12000
            if len(payload) > MAX_PAYLOAD_CHARS:
                logger.warning(
                    f"Analysis payload {len(payload)} chars - truncating to "
                    f"{MAX_PAYLOAD_CHARS}"
                )
                payload = payload[:MAX_PAYLOAD_CHARS] + "\n... (truncated)"

            logger.info(f"Analysis payload size: {len(payload)} chars")

            analysis_prompt = (
                "You are an expert ML analyst. Analyze this AutoML results.json and output a structured report (<600 words):\n"
                "1. Task Summary\n2. Performance Metrics (table)\n3. Feature Insights (top 3-5)\n"
                "4. Recommendations\n5. Next Steps\n6. Overall Verdict\n\n"
                f"JSON:\n{payload}\n\n"
                "Use markdown. No extras."
            )

            self.client.agents.messages.create(thread_id=thread_id, role="user", content=analysis_prompt)
            # Retries transient throttling; raises with the real reason if
            # it still cannot complete (caller falls back to a summary).
            run = self._run_agent_with_retry(thread_id, agent_id)

            messages = self.client.agents.messages.list(thread_id=thread_id, run_id=run.id)
            latest_msg = None
            for msg in messages:
                if msg.role == "assistant" and msg.content and any(
                    getattr(c, "type", None) == "text" and getattr(c.text, "value", "").strip()
                    for c in msg.content
                ):
                    latest_msg = msg
                    break

            if not latest_msg or not latest_msg.content:
                raise Exception("No analysis generated")

            response = ""
            for c in latest_msg.content:
                if c.type == "text":
                    response += c.text.value
            return response.strip()

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    
    @staticmethod
    def _is_no_data_found_reply(text: str) -> bool:
        """True if `text` is the honest 'nothing in this dataset answers that'
        fallback reply. When the previous turn already told the user their
        question isn't answerable from this data, asking the LLM to invent a
        follow-up tends to produce a plausible-looking but non-existent
        column (e.g. 'plant_plantname' guessed from 'employee_employeename'),
        which just repeats the same failure one step later."""
        if not text:
            return False
        t = text.lower()
        markers = (
            "couldn't find data",
            "could not find data",
            "doesn't answer that question",
            "does not answer that question",
            "answers that question directly",
            "cannot_answer",
        )
        return any(m in t for m in markers)

    @staticmethod
    def _suggestion_is_grounded(suggestion: str, columns: list) -> bool:
        """Reject a suggestion that references an identifier-shaped token
        (snake_case or camelCase, e.g. 'plant_plantname') which isn't actually
        one of the dataset's real columns. Plain English words are left alone
        so this only catches invented column/field names, not normal prose.
        """
        if not columns:
            return True
        norm_cols = {re.sub(r'[^a-z0-9]', '', c.lower()) for c in columns}
        tokens = re.findall(r'[A-Za-z][A-Za-z0-9_]{3,}', suggestion)
        for tok in tokens:
            looks_like_identifier = "_" in tok or bool(re.search(r'[a-z][A-Z]', tok))
            if not looks_like_identifier:
                continue
            norm_tok = re.sub(r'[^a-z0-9]', '', tok.lower())
            if len(norm_tok) < 4:
                continue
            if any(norm_tok in nc or nc in norm_tok for nc in norm_cols):
                continue
            return False
        return True

    @staticmethod
    def _fallback_suggestion_from_columns(columns: list) -> str:
        """
        A specific, answerable query built directly from the real schema
        instead of asked of the LLM - it can never reference data that
        doesn't exist. Used only as a genuine last resort (right after a
        "no data found" reply, or if the LLM path fails).

        Picks randomly among every valid measure/dimension pair (and
        id-style columns are included as legitimate, if less
        business-friendly, group-bys - not excluded outright) so repeated
        fallbacks in the same thread don't always surface the identical
        suggestion just because there's only one "friendly" dimension in
        this particular dataset.
        """
        if not columns:
            return "Ask: what's next?"

        id_like = re.compile(r'(_?id$|code$)', re.IGNORECASE)
        date_like = re.compile(r'(date|year|month|day|time|period)', re.IGNORECASE)
        measure_like = re.compile(
            r'(cost|price|amount|quantity|qty|revenue|sales|value|count|'
            r'rate|score|hours|duration|total)',
            re.IGNORECASE
        )

        measures = [c for c in columns if measure_like.search(c) and not id_like.search(c)]
        date_cols = [c for c in columns if date_like.search(c)]

        # Business-readable dimensions (names, categories) are preferred,
        # but a real id column is still a valid, real, answerable group-by
        # - just less friendly - so keep both pools instead of excluding
        # ids outright whenever a friendlier column happens to exist.
        friendly_dims = [
            c for c in columns
            if not measure_like.search(c) and not date_like.search(c) and not id_like.search(c)
        ]
        id_dims = [
            c for c in columns
            if not measure_like.search(c) and not date_like.search(c) and id_like.search(c)
        ]
        # Weight friendly dimensions more heavily by listing them twice,
        # without making id columns unreachable.
        dimensions = friendly_dims * 2 + id_dims

        candidates = []
        if measures and dimensions:
            for m in measures:
                for d in dimensions:
                    candidates.append(f"Compare total {m} by {d}")
        if measures and date_cols:
            for m in measures:
                candidates.append(f"Show the trend of {m} over time")

        if candidates:
            return random.choice(candidates)
        if measures:
            return f"Show the trend of {random.choice(measures)} over time"

        # Nothing that looks like a measure - naming real columns is safer
        # than guessing a query shape that might not make sense here.
        cols_preview = ", ".join(columns[:5])
        return f"Ask about: {cols_preview}"

    def _pick_verified_suggestion(self, blob_path: Optional[str], user_id: Optional[str], columns: list) -> str:
        """
        Build a next-query suggestion that is verified against the REAL
        data, not just plausible-looking column names. A suggestion the
        user can click should mean "this will work" - so this actually
        loads the dataset and checks that:
          - the measure column has real, non-constant numeric values, and
          - the dimension column has more than one real group (an id-like
            column that's actually just a unique key per row, or a column
            that's constant across every row, would make "compare X by Y"
            come back empty or meaningless even though both names are real).

        Only column pairs that pass both checks are ever suggested. Falls
        back to the name-only heuristic (_fallback_suggestion_from_columns)
        only if the data itself can't be loaded at all - naming real
        columns is still safer than suggesting nothing.
        """
        if not blob_path or not user_id or not columns:
            return self._fallback_suggestion_from_columns(columns)

        try:
            container_client = self.blob_service.get_container_client(UPLOAD_CONTAINER)
            blob_client = container_client.get_blob_client(blob_path)
            if not blob_client.exists():
                return self._fallback_suggestion_from_columns(columns)
            data = blob_client.download_blob().readall()
            if blob_path.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(data))
            elif blob_path.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(data))
            else:
                return self._fallback_suggestion_from_columns(columns)
        except Exception as e:
            logger.warning(f"Could not load data to verify suggestion, using name-only fallback: {e}")
            return self._fallback_suggestion_from_columns(columns)

        date_like = re.compile(r'(date|year|month|day|time|period)', re.IGNORECASE)
        measure_like = re.compile(
            r'(cost|price|amount|quantity|qty|revenue|sales|value|count|'
            r'rate|score|hours|duration|total)',
            re.IGNORECASE
        )

        # A verified measure: real numeric values, more than a handful of
        # them, and not just the same number repeated every row.
        verified_measures = []
        for col in columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) >= 5 and series.nunique() > 1:
                verified_measures.append(col)
        name_matched = [c for c in verified_measures if measure_like.search(c)]
        ordered_measures = name_matched + [c for c in verified_measures if c not in name_matched]

        # A verified dimension: more than one real group, but not so many
        # unique values that it's effectively a per-row identifier (that
        # would make a "compare by X" answer come back as a huge,
        # meaningless list rather than a real comparison).
        verified_dimensions = []
        for col in columns:
            if col not in df.columns or measure_like.search(col) or date_like.search(col):
                continue
            nunique = df[col].nunique(dropna=True)
            row_count = max(len(df), 1)
            if 1 < nunique <= max(20, int(row_count * 0.5)):
                verified_dimensions.append(col)

        # Build natural question templates from verified pairs, phrased as
        # business questions not technical commands.
        measure_labels = {
            "productioncost": "production cost",
            "defectquantity": "defect quantity",
            "quantityproduced": "quantity produced",
            "unitcost": "unit cost",
        }
        dimension_labels = {
            "plant_plantname": "plant",
            "product_productname": "product",
            "product_category": "product category",
            "supplier_suppliername": "supplier",
            "employee_role": "employee role",
            "employee_employeename": "employee",
        }
        date_cols = [c for c in columns if date_like.search(c) and c in df.columns]
        candidates = []
        if ordered_measures and verified_dimensions:
            for m in ordered_measures:
                for d in verified_dimensions:
                    ml = measure_labels.get(m, m.replace("_", " "))
                    dl = dimension_labels.get(d, d.replace("_", " "))
                    candidates.append(f"Which {dl} has the highest {ml}?")
        if ordered_measures and date_cols:
            for m in ordered_measures:
                ml = measure_labels.get(m, m.replace("_", " "))
                candidates.append(f"How did {ml} change month by month?")

        if candidates:
            return random.choice(candidates)
        if ordered_measures:
            ml = measure_labels.get(ordered_measures[0], ordered_measures[0].replace("_", " "))
            return f"What is the total {ml} across all orders?"

        cols_preview = ", ".join(columns[:5])
        return f"Ask about: {cols_preview}"

    def get_dynamic_suggestion(self,
        thread_id: str,
        user_email: str,
        agent_info: dict,
        analysis_text: str = "",
        context_limit: int = 3,
        user_id: str = None,
        agent_name: str = None,
    ) -> str:
        """Ask the agent for ONE ultra-short next step, grounded in the
        dataset's real columns so it can never suggest a query about a
        column/field that doesn't actually exist in the data."""
        columns = []
        blob_path = None
        try:
            if user_id and agent_name:
                blob_path = self.get_latest_blob_file(user_id, agent_name)
                if blob_path:
                    columns = self.get_dataset_columns(blob_path, user_id)
        except Exception as col_err:
            logger.warning(f"Could not load columns for suggestion grounding: {col_err}")

        # If the last turn already told the user their question isn't
        # answerable from this dataset, don't ask the LLM to invent a
        # follow-up on top of it - build a guaranteed-safe one from the
        # real schema instead.
        if self._is_no_data_found_reply(analysis_text) or not (analysis_text or "").strip():
            return self._pick_verified_suggestion(blob_path, user_id, columns)

        try:
            # Must be called on a real instance - get_thread_history is a
            # regular instance method, not static.
            messages = (
                self.response_handler.get_thread_history(user_email, thread_id)
                if self.response_handler else []
            )
            recent = messages[-context_limit:] if len(messages) >= context_limit else messages
            context = "\n".join([f"{m['role']}: {m['content']}" for m in recent])

            col_block = (
                f"\n\nThe dataset has EXACTLY these columns (copy names "
                f"character-for-character; NEVER invent, combine, or guess "
                f"a column name that is not in this list): {columns}\n"
                if columns else ""
            )

            prompt = (
                f"Conversation so far:\n{context}\n\n"
                f"Latest answer (truncated):\n{analysis_text[-1200:]}\n"
                f"{col_block}\n"
                "Suggest ONE natural follow-up question a business user would want to ask NEXT.\n\n"
                "RULES - read every rule before writing:\n"
                "1. Write a QUESTION in plain business language - not a technical command.\n"
                "   BAD: 'Compute total defectquantity per plant_plantname'\n"
                "   GOOD: 'Which plant has the highest defect rate?'\n"
                "   BAD: 'Filter by calendar_year = max; group by plant_plantname'\n"
                "   GOOD: 'How did production costs change month by month?'\n"
                "2. The question must be GENUINELY DIFFERENT from what was just answered.\n"
                "   Never suggest the same question again or a trivial rephrasing of it.\n"
                "3. If the latest answer was about defects -> ask about costs or production volume next.\n"
                "   If it was about costs -> ask about defects, employees, or products next.\n"
                "   If it was an ML model result -> ask what drives that outcome or compare categories.\n"
                "   Always move the conversation FORWARD, not sideways or backward.\n"
                "4. Use real column values where helpful (e.g. plant names: Pune Manufacturing Unit,\n"
                "   Hyderabad Manufacturing Unit, Chennai Manufacturing Unit), not raw column names.\n"
                "5. ≤ 12 words. No markdown. No prefix like 'Suggestion:' or 'Next step:'.\n"
                "6. Only suggest something this dataset can actually answer.\n\n"
                "Write the question now:"
            )

            self.client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt
            )
            try:
                run = self._run_agent_with_retry(thread_id, agent_info["agent_id"])
            except Exception as e:
                logger.warning(f"Suggestion agent unavailable: {e}")
                return self._pick_verified_suggestion(blob_path, user_id, columns)

            # Filter to THIS run's own messages via run_id - not just newest-
            # first ordering. Without this, a run whose reply has no text
            # (e.g. a tool-call-only step) could silently fall through to an
            # older suggestion from a previous turn instead of a fresh one.
            msgs = self.client.agents.messages.list(thread_id=thread_id, run_id=run.id)
            suggestion = None
            for msg in msgs:
                if msg.role == "assistant" and msg.content:
                    text = ""
                    for c in msg.content:
                        if getattr(c, "type", None) == "text":
                            text += c.text.value
                    if text.strip():
                        suggestion = text.strip().split("\n")[0].strip('."\'')
                        break

            if not suggestion:
                return self._pick_verified_suggestion(blob_path, user_id, columns)

            # Defense-in-depth: don't trust the LLM's output blindly - reject
            # it if it still references a column-shaped name that isn't real.
            if not self._suggestion_is_grounded(suggestion, columns):
                logger.warning(f"Discarding ungrounded suggestion: {suggestion!r}")
                return self._pick_verified_suggestion(blob_path, user_id, columns)

            return suggestion
        except Exception as e:
            # Log the real reason instead of swallowing it silently - this
            # exact silent-swallow is what hid the reversed()-on-a-pager bug
            # (and the earlier unbound-instance-method bug) for so long.
            logger.warning(f"get_dynamic_suggestion LLM path failed, using fallback: {e}")
            return self._pick_verified_suggestion(blob_path, user_id, columns)

    def _extract_suggestions(self, text: str) -> list:
        lines = [l.strip() for l in text.split("\n") if l.strip().startswith("-")]
        return [l[1:].strip() for l in lines][:3]

    def process_query(
        self,
        agent_id: str,
        thread_id: str,
        message: str,
        user_email: str,
        agent_name: str = None
    ) -> dict:
        try:
            thread_id, ml_json, text_response = self.process_query_on_main_agent(
                agent_id=agent_id,
                query=message,
                thread_id=thread_id,
                user_id=user_email,
                agent_name=agent_name
            )

            blob_path = self.get_latest_blob_file(user_email, agent_name)
            dataset_id = blob_path.split('/')[-1] if blob_path else None

            direct_answer = "Processing..."
            results_data = None
            task = None
            run_path = None
            analysis = ""

            if ml_json and ml_json.get("is_ml"):
                results_data = ml_json["results"]
                task = results_data.get("task", "")
                run_path = results_data.get("run_path", "")

                if task == "forecasting" and run_path:
                    direct_answer = self.forecast_answer.make_answer(run_path, message)
                else:
                    direct_answer = "Model is running..."

                try:
                    analysis = self.generate_ml_analysis(agent_id, results_data, thread_id)
                except Exception as analysis_error:
                    logger.warning(
                        f"AI analysis unavailable, using fallback: {analysis_error}"
                    )
                    analysis = self._fallback_summary(results_data)

            else:
                direct_answer = text_response.split("\n", 1)[0] if text_response else "Done."
                analysis = text_response

            return {
                "direct_answer": direct_answer,
                "session_id": thread_id,
                "task_type": task,
                "blob_file_used": blob_path,
                "results_filename": f"{run_path}/results.json" if run_path else None,
                "results": results_data,
                "analysis": analysis,
                "response": "Done",
                "dataset_id": dataset_id,
                "suggestions": self._extract_suggestions(analysis)
            }

        except Exception as e:
            logger.error(f"process_query failed: {e}")
            return {"direct_answer": "Error", "error": str(e)}

    def get_preview(self, blob_path: str, user_id: str, max_rows: int = 5) -> Optional[Dict[str, Any]]:
        if not blob_path.startswith(f"{user_id}/"):
            raise ValueError("Access denied")

        try:
            client = self.blob_service.get_blob_client(container=UPLOAD_CONTAINER, blob=blob_path)
            if not client.exists():
                return None

            data = client.download_blob().readall()

            if blob_path.lower().endswith('.csv'):
                df = pd.read_csv(io.BytesIO(data), nrows=max_rows + 5, low_memory=False)
            elif blob_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(data), nrows=max_rows + 5, engine='openpyxl')
            else:
                return None

            dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
            df = df.head(max_rows)

            def make_json_friendly(val):
                if pd.isna(val):
                    return None
                if isinstance(val, (np.integer, np.int64)):
                    return int(val)
                if isinstance(val, (np.floating, np.float64)):
                    if np.isnan(val) or np.isinf(val):
                        return None
                    return float(val)
                if isinstance(val, (pd.Timestamp, np.datetime64)):
                    return val.isoformat() 
                return str(val)

            # Apply to entire dataframe
            clean_data = df.map(make_json_friendly)

            return {
                "columns": df.columns.tolist(),
                "dtypes": dtypes_dict,
                "rows": clean_data.to_dict(orient='records')  
            }

        except Exception as e:
            logger.error(f"Preview failed for {blob_path}: {str(e)}")
            return None
        
    def get_features_for_task(
        self,
        blob_path: str,
        task: str,
        user_id: str,
        max_preview_rows: int = 300
    ) -> Dict[str, Any]:

        if not blob_path.startswith(f"{user_id}/"):
            raise PermissionError("Access denied: blob path does not belong to this user")

        # Download blob 
        try:
            client = self.blob_service.get_blob_client(container=UPLOAD_CONTAINER, blob=blob_path)
            data = client.download_blob().readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download blob {blob_path}: {str(e)}")

        # Read preview 
        try:
            if blob_path.lower().endswith('.csv'):
                df = pd.read_csv(
                    io.BytesIO(data),
                    nrows=max_preview_rows,
                    low_memory=False,
                    skip_blank_lines=True
                )
            elif blob_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                df = pd.read_excel(io.BytesIO(data), nrows=max_preview_rows)
            else:
                raise ValueError("Only .csv, .xlsx, .xls, .xlsm files are supported")
        except Exception as e:
            raise ValueError(f"Could not parse file as DataFrame: {str(e)}")

        if df.empty:
            raise ValueError("File is empty or contains no data")

        # CLEANING 
        df.columns = df.columns.str.strip()
        df = df.dropna(how="all")
        df = df[df.notna().sum(axis=1) > len(df.columns) // 2]

        # Try datetime parsing
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower() or "year" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                except:
                    pass

        columns = list(df.columns)
        numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        # Auto detect for multistep_forecasting
        dimensions = []
        measures = []
        time_column = None
        month_names = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}

        for col in columns:
            col_lower = col.lower().strip()
            if col_lower in month_names or any(m in col_lower for m in month_names):
                measures.append(col)
            elif "year" in col_lower or col in datetime_cols:
                time_column = col
            elif df[col].nunique() <= 50 and not pd.api.types.is_numeric_dtype(df[col]) and col not in datetime_cols:
                dimensions.append(col)

        if not measures:
            measures = numeric_cols[:12] if numeric_cols else columns[:12]
        if not time_column and datetime_cols:
            time_column = datetime_cols[0]

        features_list = dimensions + measures
        if time_column and time_column not in features_list:
            features_list.append(time_column)

        task = task.lower()

        if task == "classification":
            low_card_cols = [c for c in columns if 2 <= df[c].nunique() <= 12]
            very_discrete_num = [c for c in numeric_cols if 2 <= df[c].nunique() <= 9]
            features = list(dict.fromkeys(very_discrete_num + low_card_cols))
            if not features:
                features = numeric_cols[:6] if numeric_cols else columns[:6]
            reason = "Low-cardinality features suitable for categorical prediction"

        elif task == "regression":
            features = [c for c in numeric_cols if df[c].nunique() / len(df) < 0.9]
            if not features:
                features = columns[:6]
            reason = "Numeric features suitable for continuous prediction"

        elif task == "forecasting":
            features = list(dict.fromkeys(numeric_cols + datetime_cols))
            if not features:
                features = columns[:6]
            reason = "Numeric predictors + datetime index"

        elif task == "multistep_forecasting":
            features = features_list
            return {
                "task": task,
                "dimensions": dimensions,
                "measures": measures,
                "time_column": time_column,
                "features": features,
                "count": len(features)
            }

        elif task == "clustering":
            medium_card_cols = [c for c in columns if 2 <= df[c].nunique() <= 25]
            features = list(dict.fromkeys(numeric_cols + medium_card_cols))
            if not features:
                features = columns[:8]
            reason = "Numeric + medium-cardinality features suitable for segmentation"

        elif task == "anomaly_detection":
            features = [c for c in numeric_cols if df[c].nunique() / len(df) < 0.9]
            if not features:
                features = columns[:6]
            reason = "Primarily numeric features"

        else:
            raise ValueError(f"Invalid task: {task}")

        return {
            "task": task,
            "features": features,
            "count": len(features),
            "reason": reason
        }

    def get_all_task_feature_suggestions(
        self,
        blob_path: str,
        user_id: str,
        max_preview_rows: int = 300
    ) -> Dict[str, Any]:

        if not blob_path.startswith(f"{user_id}/"):
            raise PermissionError("Access denied: blob path does not belong to this user")

        #  Download blob 
        try:
            client = self.blob_service.get_blob_client(container=UPLOAD_CONTAINER, blob=blob_path)
            data = client.download_blob().readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download blob {blob_path}: {str(e)}")

        #  Read preview 
        try:
            if blob_path.lower().endswith('.csv'):
                df = pd.read_csv(
                    io.BytesIO(data),
                    nrows=max_preview_rows,
                    low_memory=False,
                    skip_blank_lines=True
                )
            elif blob_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                df = pd.read_excel(io.BytesIO(data), nrows=max_preview_rows)
            else:
                raise ValueError("Only .csv, .xlsx, .xls, .xlsm files are supported")
        except Exception as e:
            raise ValueError(f"Could not parse file as DataFrame: {str(e)}")

        if df.empty:
            raise ValueError("File is empty or contains no data")

        #  CLEANING 
        df.columns = df.columns.str.strip()
        df = df.dropna(how="all")

        # Remove rows missing too many values (like malformed last row)
        df = df[df.notna().sum(axis=1) > len(df.columns) // 2]

        # Try datetime parsing
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                except:
                    pass

        columns = list(df.columns)

        # Identify numeric + datetime columns
        numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        #  Identifier detection 
        identifier_keywords = {'id', 'key', 'uuid', 'code', 'number', 'email', 'phone', 'name', 'address'}

        def is_likely_identifier(c: str) -> bool:
            c_lower = c.lower()
            if any(kw in c_lower for kw in identifier_keywords):
                return True
            if len(df) > 10 and df[c].nunique() / len(df) > 0.75:
                return True
            return False

        #  CARDINALITY-BASED DETECTION 

        low_card_cols = [
            c for c in columns
            if 2 <= df[c].nunique() <= 12
            and not is_likely_identifier(c)
            and c not in datetime_cols
        ]

        medium_card_cols = [
            c for c in columns
            if 2 <= df[c].nunique() <= 25
            and not is_likely_identifier(c)
            and c not in datetime_cols
        ]

        very_discrete_num = [
            c for c in numeric_cols
            if 2 <= df[c].nunique() <= 9
            and not is_likely_identifier(c)
        ]

        tasks: Dict[str, Dict[str, Any]] = {}

        #  Classification 
        cls_features = list(dict.fromkeys(very_discrete_num + low_card_cols))
        if not cls_features:
            cls_features = numeric_cols[:6] if numeric_cols else columns[:6]

        tasks["classification"] = {
            "features": cls_features,
            "count": len(cls_features),
            "reason": "Low-cardinality features suitable for categorical prediction"
        }

        #  Regression 
        reg_features = [c for c in numeric_cols if not is_likely_identifier(c)]
        if not reg_features:
            reg_features = columns[:6]

        tasks["regression"] = {
            "features": reg_features,
            "count": len(reg_features),
            "reason": "Numeric features suitable for continuous prediction"
        }

        #  Forecasting 
        ts_features = list(dict.fromkeys(numeric_cols + datetime_cols))
        if not ts_features:
            ts_features = columns[:6]

        tasks["forecasting"] = {
            "features": ts_features,
            "count": len(ts_features),
            "reason": "Numeric predictors + datetime index"
        }

        tasks["multistep_forecasting"] = {
            "features": ts_features.copy(),
            "count": len(ts_features),
            "reason": "Numeric predictors + datetime index (multi-step)"
        }

        #  Clustering 
        clus_features = list(dict.fromkeys(numeric_cols + medium_card_cols))
        if not clus_features:
            clus_features = columns[:8]

        tasks["clustering"] = {
            "features": clus_features,
            "count": len(clus_features),
            "reason": "Numeric + medium-cardinality features suitable for segmentation"
        }

        #  Anomaly Detection
        ano_features = [c for c in numeric_cols if not is_likely_identifier(c)]
        if not ano_features:
            ano_features = columns[:6]

        tasks["anomaly_detection"] = {
            "features": ano_features,
            "count": len(ano_features),
            "reason": "Primarily numeric features"
        }

        return {"tasks": tasks}
    
    def generate_build_model_text_summary(
        self,
        user_email: str, 
        task: str,
        best_model: str,
        primary_metric: str,
        primary_score: float,
        all_models: dict,
        dataset_name: str = "your dataset",
        top_features: list | None = None,
        result_details: dict | None = None
    ) -> str:

        try:
            # Get user's personal agent_id using only email
            auth_handler = CosmosDBAuthHandler()
            user = auth_handler.get_user(user_email)
            if not user or not user.get("agents"):
                raise ValueError(f"No agent found for user: {user_email}")
            
            user_agent_id = user["agents"][0]["agent_id"]
            logger.info(f"Using user's personal agent {user_agent_id} for summary")

            # Formatting
            task = task.lower().strip()
            task_display = task.replace("_", " ").title()
            metric_display = primary_metric.upper() if primary_metric in ["rmse", "mae", "mape", "r2"] else primary_metric.replace("_", " ").title()
            model_count = len(all_models)

            if isinstance(primary_score, (float, int)):
                if primary_metric in ["f1", "accuracy", "precision", "recall", "roc_auc"]:
                    score_str = f"{primary_score:.1%}"
                elif primary_metric in ["rmse", "mae", "mape"]:
                    score_str = f"{primary_score:.3f}"
                elif primary_metric == "r2":
                    score_str = f"{primary_score:.3f}"
                else:
                    score_str = f"{primary_score:.3f}"
            else:
                score_str = str(primary_score)

            features_context = ""
            if top_features and len(top_features) >= 3:
                top_5 = top_features[:5]
                features_context = f"\nTop important features: {', '.join([f'`{f}`' for f in top_5])}"

            extra_insights = ""
            if result_details:
                if task == "classification":
                    class_dist = result_details.get("class_distribution") or all_models.get(best_model, {}).get("train", {}).get("class_distribution")
                    if class_dist and len(class_dist) == 2:
                        counts = list(class_dist.values())
                        minority, majority = min(counts), max(counts)
                        if minority / majority < 0.5:
                            extra_insights += f"\nNote: Significant class imbalance ({minority}:{majority})"

                elif task == "clustering":
                    n_clusters = result_details.get("n_clusters")
                    if n_clusters:
                        extra_insights += f"\nDetected {n_clusters} distinct clusters"

                elif task == "anomaly_detection":
                    n_anomalies = result_details.get("n_anomalies", 0)
                    total = result_details.get("total_rows", 1)
                    perc = (n_anomalies / total) * 100 if total > 0 else 0
                    extra_insights += f"\nFound {n_anomalies} anomalies (~{perc:.1f}% of data)"

                elif task == "forecasting":
                    horizon = result_details.get("forecast_horizon")
                    if horizon:
                        extra_insights += f"\nForecast horizon: {horizon} steps"

            # Task-specific guidance for LLM
            task_guidance = {
                "classification": "Focus on non-linear patterns, feature interactions, handling imbalance, precision/recall balance.",
                "regression": "Explain linear vs non-linear fit, error magnitude, and how well trends were captured.",
                "forecasting": "Highlight trend, seasonality, and cycle capture; mention short vs long-term accuracy.",
                "multistep_forecasting": "Evaluate accuracy across multiple horizons, error propagation, uncertainty handling, and long-range dependency capture.",
                "clustering": "Describe cluster separation, meaningful groupings, and interpretability.",
                "anomaly_detection": "Emphasize isolation of rare patterns and effectiveness in noisy/high-dimensional data."
            }.get(task, "Explain why this model achieved the strongest performance.")

            # Create isolated temp thread
            temp_thread = self.client.agents.threads.create()
            thread_id = temp_thread.id

            # Final prompt
            prompt = f"""\
    You are a professional machine learning report writer.

    YOUR **ONLY** TASK is to write a short, human-readable summary in markdown.

    STRICT RULES — YOU **MUST** FOLLOW EVERY SINGLE ONE OR YOUR RESPONSE WILL BE INVALID:
    1. ALWAYS begin **EXACTLY** with this exact sentence: "**Model trained successfully!**"
    2. Write only 4–7 lines total — no more, no less
    3. Write in natural, professional English — like you're talking to a colleague
    4. **NEVER** return JSON, raw code, or any task detection output
    5. **NEVER** return anything that starts with {{ or looks like {{ "task_type": ...
    6. **NEVER** return anything that includes the words "task_type", "target", "metric" in a JSON-like structure
    7. ONLY write the summary — no explanations, no thinking, no extra text
    8. Must contain exactly these sections in this order:
    - **Best model:** ...
    - **Performance:** ...
    - **Why this worked best:** followed by 2–4 short bullet points
    9. Do NOT describe the dataset or its contents — ONLY talk about the model and performance
    10. Do NOT repeat the context info — weave it naturally into the sections

    Context:
    Dataset: **{dataset_name}**
    Task: **{task_display}**
    Best model: **{best_model.replace('_', ' ').title()}**
    Performance: **{metric_display} = {score_str}** (best result){features_context}{extra_insights}

    Start writing **NOW**. Begin with the required first sentence. Do **NOT** return JSON.
    Remember: If your response doesn't start with "**Model trained successfully!**" or has more than 7 lines, it's invalid.
    """

            self.client.agents.messages.create(
                thread_id=thread_id,
                role="user",
                content=prompt.strip()
            )

            # Run using USER'S PERSONAL AGENT
            run = self._run_agent_with_retry(thread_id, user_agent_id)

            # Extract response - filtered to THIS run so a reply with no
            # text can't silently fall through to an older message instead.
            messages = self.client.agents.messages.list(thread_id=thread_id, run_id=run.id)
            response = ""
            for msg in messages:
                if msg.role == "assistant" and msg.content:
                    text = ""
                    for block in msg.content:
                        if block.type == "text":
                            text += block.text.value
                    if text.strip():
                        response = text
                        break

            summary = response.strip()
            if not summary:
                raise Exception("Empty summary generated")

            logger.info(f"Successfully generated summary for {user_email} using their agent")
            return summary

        except Exception as e:
            logger.warning(f"Failed to generate LLM summary for {user_email}: {e}")
            # Safe fallback
            return (
                f"**Model trained successfully!**\n\n"
                f"**Best model:** {best_model.replace('_', ' ').title()}\n"
                f"**Performance:** {metric_display} = {score_str} (highest among {model_count} models)"
            )
        finally:
            # Clean up temp thread
            try:
                if 'thread_id' in locals():
                    self.client.agents.threads.delete(thread_id=thread_id)
            except:
                pass
    
    def _create_data_profile(self, df: pd.DataFrame, max_rows: int = 5) -> Dict[str, Any]:
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": []
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "nunique": int(df[col].nunique()),
                "missing_pct": float(df[col].isnull().sum() / len(df) * 100),
                "sample_values": df[col].dropna().head(max_rows).tolist()
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["numeric_stats"] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean())
                }
            
            profile["columns"].append(col_info)
        return profile
    
    def _get_analysis_agent_id(self) -> str:
        if not hasattr(self, '_analysis_agent_id'):
            try:
                self._analysis_agent_id = self.create_agent("Dataset Analyzer")
                logger.info(f"Created analysis agent: {self._analysis_agent_id}")
            except Exception as e:
                logger.error(f"Failed to create analysis agent: {e}")
                raise
        return self._analysis_agent_id

    def _llm_analyze_structure(self, df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
        try:
            profile = self._create_data_profile(df)
            
            prompt = f"""
You are a data schema analyzer for time-series datasets in an AutoML system.
Analyze this dataset profile and return ONLY valid JSON (no markdown, no explanations):

Dataset Profile:
{json.dumps(profile, indent=2)}
Return exactly this JSON structure:
{{
"dataset_format": "wide" | "long" | "unknown",
"time_definition": {{
    "type": "year_month" | "datetime" | "period" | "unknown",
    "year_column": "column_name" or null,
    "month_columns": ["col1", "col2"],
    "datetime_column": "column_name" or null,
    "frequency": "monthly" | "quarterly" | "weekly" | "unknown"
}},
"dimensions": ["dim1", "dim2"],
"measures": ["measure1", "measure2"],
"needs_transformation": true | false
}}

Rules:
- Use EXACT column names from the profile
- If months are column names → "wide"
- Dimensions = categorical identifiers with 2-50% unique values
- Measures = numeric columns to forecast (exclude IDs)
- If unclear → use "unknown"
- Output ONLY the JSON object
"""
            
            temp_thread = self.client.agents.threads.create()
            thread_id = temp_thread.id
            
            try:
                self.client.agents.messages.create(thread_id=thread_id,role="user",content=prompt)
                run = self._run_agent_with_retry(thread_id, self._get_analysis_agent_id())
                
                # Extract response - filtered to THIS run
                messages = self.client.agents.messages.list(thread_id=thread_id, run_id=run.id)
                response_text = ""
                for msg in messages:
                    if msg.role == "assistant" and msg.content:
                        text = ""
                        for block in msg.content:
                            if block.type == "text":
                                text += block.text.value
                        if text.strip():
                            response_text = text
                            break
                    
                llm_json = extract_json_from_text(response_text)
                
                if not llm_json:
                    raise Exception("LLM returned invalid JSON")
                
                logger.info(f"LLM analysis successful: format={llm_json.get('dataset_format')}")
                return llm_json
                
            finally:
                pass
                # try:
                #     self.client.agents.threads.delete(thread_id=thread_id)
                # except:
                #     pass
            
        except Exception as e:
            logger.error(f"LLM structure analysis failed: {e}")
            raise

    def analyze_dataset_structure(self, file_path: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(file_path)
            
            # Python-based structural analysis
            python_analysis = self.ts_transformer.analyze_dataset_structure(df)
            
            # Try LLM semantic analysis
            llm_analysis = None
            try:
                llm_analysis = self._llm_analyze_structure(df, file_path)
            except Exception as e:
                logger.warning(f"LLM analysis failed, using Python fallback: {e}")
            
            # Merge analyses (LLM semantic + Python structural)
            if llm_analysis:
                final_analysis = self.ts_transformer.merge_analyses(
                    python_analysis=python_analysis,
                    llm_analysis=llm_analysis,
                    df=df  
                )
                logger.info("Successfully merged LLM and Python analyses")
            else:
                final_analysis = python_analysis
                logger.info("Using Python-only analysis (LLM unavailable)")
            
            return final_analysis
            
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            raise

    def analyze_structure_with_llm(self, file_path: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(file_path)
            profile = self._create_data_profile(df)
            
            system_prompt = """
You are a data schema analyzer for time-series datasets in an AutoML system.
Analyze the dataset profile and return ONLY valid JSON (no markdown, no explanations).

Return exactly this JSON structure:
{
"dataset_format": "wide" | "long" | "unknown",
"time_definition": {
    "type": "year_month" | "datetime" | "period" | "unknown",
    "year_column": "column_name" or null,
    "month_columns": ["col1", "col2"],
    "datetime_column": "column_name" or null,
    "frequency": "monthly" | "quarterly" | "weekly" | "unknown"
},
"dimensions": ["dim1", "dim2"],
"measures": ["measure1", "measure2"],
"needs_transformation": true | false
}

Rules:
- Use EXACT column names from the profile
- If months are spread across columns (Jan, Feb, etc.) → "wide"
- If there's a datetime/date column → "long"
- Dimensions = categorical columns with 2-50% unique values (for grouping)
- Measures = numeric columns to forecast (exclude year/ID columns)
- year_column = column with years (1990-2100)
- month_columns = columns named like months (Jan, Feb, etc.)
- If unclear → use "unknown"
- Output ONLY the JSON object, no other text
"""

            user_prompt = f"""Analyze this dataset profile:
                        {json.dumps(profile, indent=2)}
                        Return the JSON analysis now."""

            temp_thread = self.client.agents.threads.create()
            thread_id = temp_thread.id
            
            try:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                self.client.agents.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=full_prompt
                )
                
                agent_id = self._get_or_create_analysis_agent()
                
                run = self._run_agent_with_retry(thread_id, agent_id)
                
                messages = self.client.agents.messages.list(thread_id=thread_id, run_id=run.id)
                response_text = ""
                
                for msg in messages:
                    if msg.role == "assistant" and msg.content:
                        text = ""
                        for content_block in msg.content:
                            if content_block.type == "text":
                                text += content_block.text.value
                        if text.strip():
                            response_text = text
                            break
                
                if not response_text:
                    raise Exception("Empty response from LLM")
                
                # Parse JSON from response
                llm_json = extract_json_from_text(response_text)
                
                if not llm_json:
                    logger.error(f"LLM returned non-JSON response: {response_text[:200]}")
                    raise Exception("LLM did not return valid JSON")
                
                # Validate required keys
                required_keys = ["dataset_format", "time_definition", "dimensions", "measures", "needs_transformation"]
                missing_keys = [k for k in required_keys if k not in llm_json]
                
                if missing_keys:
                    logger.warning(f"LLM response missing keys: {missing_keys}")
                    # Fill in missing keys with defaults
                    defaults = {
                        "dataset_format": "unknown",
                        "time_definition": {
                            "type": "unknown",
                            "year_column": None,
                            "month_columns": [],
                            "datetime_column": None,
                            "frequency": "unknown"
                        },
                        "dimensions": [],
                        "measures": [],
                        "needs_transformation": False
                    }
                    for key in missing_keys:
                        llm_json[key] = defaults.get(key)
                
                logger.info(f"LLM structure analysis complete: format={llm_json.get('dataset_format')}")
                return llm_json
                
            finally:
                pass
                # try:
                #     self.client.agents.threads.delete(thread_id=thread_id)
                #     logger.debug(f"Deleted temp analysis thread: {thread_id}")
                # except Exception as cleanup_error:
                #     logger.warning(f"Failed to cleanup thread {thread_id}: {cleanup_error}")
            
        except Exception as e:
            logger.error(f"LLM structure analysis failed: {e}", exc_info=True)
            raise

    def _get_or_create_analysis_agent(self) -> str:
        # Check if we already have an analysis agent cached
        if hasattr(self, '_cached_analysis_agent_id'):
            return self._cached_analysis_agent_id
        
        try:
            # Create a simple agent for analysis (no code interpreter needed for JSON tasks)
            analysis_agent = self.client.agents.create_agent(
                model=AGENT_MODEL,
                name="Dataset Structure Analyzer",
                instructions="You are a data analyst. Analyze dataset profiles and return JSON analysis.",
                tools=[]
            )
            
            self._cached_analysis_agent_id = analysis_agent.id
            logger.info(f"Created dedicated analysis agent: {analysis_agent.id}")
            return analysis_agent.id
            
        except Exception as e:
            logger.error(f"Failed to create analysis agent: {e}")
            raise

    def _create_data_profile(self, df: pd.DataFrame, max_samples: int = 5) -> Dict[str, Any]:
        profile = {
            "shape": {
                "rows": int(len(df)),
                "columns": int(len(df.columns))
            },
            "columns": []
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "nunique": int(df[col].nunique()),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct": round(float(df[col].isnull().sum() / len(df) * 100), 2)
            }
            
            sample_vals = df[col].dropna().head(max_samples).tolist()
            # Convert numpy types to native Python for JSON serialization
            col_info["sample_values"] = [
                int(v) if isinstance(v, np.integer) else
                float(v) if isinstance(v, np.floating) else
                str(v)
                for v in sample_vals
            ]
            
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                col_info["stats"] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": round(float(df[col].mean()), 2)
                }
            
            profile["columns"].append(col_info)
        
        return profile
        
    def create_transformation_config_from_analysis(
    self,
    analysis: Dict,
    user_selection: Dict
) -> Dict:
        return self.ts_transformer.create_transformation_config(analysis, user_selection)