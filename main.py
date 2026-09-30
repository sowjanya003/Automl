"""Main FastAPI application"""
#hello
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Path, BackgroundTasks
from fastapi.responses import StreamingResponse
from datetime import datetime
import json, os, logging, sys, tempfile, traceback, statistics, requests, time, io, asyncio, threading
import pandas as pd
import numpy as np
import hashlib, uuid
from typing import Optional, Dict, Any
from collections import defaultdict, Counter
from scipy.stats import ks_2samp
# Local imports
from app.config import UPLOAD_CONTAINER, COSMOS_CONNECTION_STRING, DATABASE_NAME, FUN3_URL, FUN2_URL,FUN1_URL, FUN1_TIMEOUT, FUN2_TIMEOUT, FUN3_TIMEOUT
from app.cosmos_respo import CosmosDBResponseHandler
from app.cosmos_authe import CosmosDBAuthHandler
from app.agent_handler import AgentHandler
from app.model_registry_handler import ModelRegistryHandler
from app.databricks_handler import DatabricksHandler, DatabricksFileNotFoundError
from app.source_fetch import (
    fetch_source_file,
    fetch_train_and_optional_test_file,
    write_back_to_databricks,
    write_back_to_snowflake,
    write_back_to_onelake,
    SourceFileNotFoundError,
    SourceCredentialsMissingError,
)
from app.snowflake_handler import get_snowflake_handler
from app.test_metrics_handler import TestMetricsHandler
from app.utils import extract_json_from_text
# Dataset Intelligence imports
from dataset_intelligence import DatasetService
from dataset_intelligence.thread_metadata_handler import ThreadMetadataHandler
# from app.onelake_handler import OneLake_handler
from app.timeseries_transformer import TimeseriesTransformer
from app.llm_transformer_handler import get_llm_transformer
from app.utils import to_csv_if_needed , to_python_types, split_next_steps, looks_like_access_excuse, looks_like_leaked_ml_intent, looks_like_agent_asking_question
from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                mean_squared_error, mean_absolute_error, r2_score,
                silhouette_score, davies_bouldin_score, calinski_harabasz_score
            )

from dotenv import load_dotenv
load_dotenv()

# Azure Cosmos DB
COSMOS_CONNECTION_STRING = os.getenv("COSMOS_CONNECTION_STRING")
DATABASE_NAME = os.getenv('DATABASE_NAME')

CONTAINER_NAME = os.getenv('USER_TABLE', 'UserData')
CONTAINER_CHAT_HISTORY = os.getenv('CHAT_HISTORY', 'ChatHistory')
THREAD_METADATA = os.getenv('THREAD_METADATA', 'ThreadMetadata')
DATASET_METADATA = os.getenv('DATASET_METADATA', 'DatasetMetadata')
MODEL_REGISTRY = os.getenv('MODEL_REGISTRY', 'ModelRegistry')
TEST_METRICS = os.getenv('TEST_METRICS', 'TestMetrics')

# Azure Blob Storage
BLOB_CONNECTION_STRING = os.getenv("BLOB_CONNECTION_STRING")
UPLOAD_CONTAINER = os.getenv("UPLOAD_CONTAINER", "uploaded-file")

# Azure AI
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_PROJECT_NAME = os.getenv("AZURE_PROJECT_NAME")
AGENT_MODEL = os.getenv("MODEL_DEPLOYMENT_NAME")

# Azure Functions
FUN1_URL = os.getenv("FUN1_URL")
FUN2_URL = os.getenv("FUN2_URL")
FUN3_URL = os.getenv("FUN3_URL")

ANALYSIS_AGENT_ID = os.getenv("ANALYSIS_AGENT_ID") 
TRANSFORMER_AGENT_ID = os.getenv("TRANSFORMER_AGENT_ID")

# OneLake
FABRIC_TOKEN = os.getenv("FABRIC_TOKEN")

# ML Task & Model Configurations
TASKS = ["classification", "regression", "forecasting", "multistep_forecasting", "clustering", "anomaly_detection"]

MODELS = {
    "classification": ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"],
    "regression": ["ridge", "random_forest", "gradient_boosting", "xgboost"],
    "forecasting": ["arima", "prophet", "xgboost", "lightgbm", "catboost"],
    "multistep_forecasting": ["xgboost", "lightgbm", "catboost"],
    "clustering": ["kmeans", "kmeans_plusplus", "dbscan", "gmm"],
    "anomaly_detection": ["isolation_forest_fast", "isolation_forest_precise", "one_class_svm", 
                          "local_outlier_factor", "elliptic_envelope"]
}

METRICS = {
    "classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
    "regression": ["rmse", "mae", "r2", "mape"],
    "forecasting": ["rmse", "mae", "r2"],
    "multistep_forecasting": ["rmse", "mae", "r2", "mape"],
    "clustering": ["silhouette_score", "davies_bouldin_score", "calinski_harabasz"],
    "anomaly_detection": ["anomaly_score", "precision", "recall", "f1"]
}

DEFAULT_METRICS = {
    "classification": "f1",
    "regression": "rmse",
    "forecasting": "rmse",
    "multistep_forecasting": "rmse",
    "clustering": "silhouette_score",
    "anomaly_detection": "anomaly_score"
}

# Logging setup
# log_folder = "logs"
# os.makedirs(log_folder, exist_ok=True)
# log_filename = os.path.join(log_folder, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Silence all noisy Azure/OpenAI SDK logs
for noisy in ['azure', 'urllib3', 'openai', 'httpx', 'httpcore', 'AIProjectClient']:
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").disabled = True
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()], #only in terminal
    # handlers=[
    #     logging.FileHandler(log_filename, encoding="utf-8"),
    #     logging.StreamHandler(sys.stdout)
    # ],
    force=True
)

logger = logging.getLogger("app")
logger.info("AutoML API started")
app = FastAPI(title="AutoML API", version="4.0",root_path="/api/service3")

# Initialize handlers
logger.info("Initializing authentication handler...")
auth_handler = CosmosDBAuthHandler()

logger.info("Initializing response handler...")
response_handler = CosmosDBResponseHandler()

logger.info("Initializing agent handler...")
handler = AgentHandler()

logger.info("Initializing model registry...")
model_registry = ModelRegistryHandler()

logger.info("Initializing dataset service...")
dataset_service = DatasetService()

logger.info("Initializing thread metadata handler...")
thread_metadata_handler = ThreadMetadataHandler()

logger.info("Initializing test results handler...")
test_metrics_handler = TestMetricsHandler()

# lake_handler = OneLake_handler()
logger.info("All handlers initialized successfully")


@app.post("/automl_register_login")
def automl_register_login(
    email: str = Form(...),
    full_name: str = Form(None),
    user_id: str = Form(None)
):
    try:
        # ---------- CASE 1: USER ALREADY EXISTS ----------
        if auth_handler.user_exists(email):

            # Fetch user using existing internal mechanism
            # We bypass password check because parent already authenticated
            user = auth_handler.get_user(email)

            if not user or not user.get("agents"):
                raise HTTPException(
                    status_code=500,
                    detail="User exists but agent not initialized"
                )

            agent = user["agents"][0]

            return {
                "message": "Login successful",
                "user": {
                    "user_id": user["user_id"],
                    "email": user["email"],
                    "full_name": user.get("full_name") or user["email"].split("@")[0],
                    "is_active": user.get("is_active", True)
                },
                "agent_id": agent["agent_id"],
                "agent_name": agent.get("agent_name", "My Data Assistant"),
                "session_id": agent["threads"][-1],
                "total_chats": len(agent["threads"])
            }

        # ---------- CASE 2: NEW USER ) ----------

        # Create internal placeholder password (not used for real auth)
        internal_password = f"EXTERNAL_USER_{email.lower()}"

        try:
            user = auth_handler.create_user(
                email=email,
                password=internal_password,
                full_name=full_name,
                user_id=user_id  # reuse caller-supplied id if given, else auto-generate
            )
        except ValueError as ve:
            # Most likely: the supplied user_id is already assigned to a
            # different email. Surface this clearly instead of a generic 500.
            raise HTTPException(status_code=400, detail=str(ve))

        user_id = user["user_id"]

        # Create agent & thread exactly like original register
        agent_name = full_name or email.split("@")[0] or "Data Assistant"
        agent_id = handler.create_agent(name=agent_name)

        thread_id, _ = handler.create_thread(agent_id)

        auth_handler.add_agent_to_user(
            email=email,
            agent_id=agent_id,
            agent_name=agent_name,
            thread_id=thread_id
        )

        # Save welcome message
        response_handler.save_response(
            email=email,
            user_id=user_id,
            agent_id=agent_id,
            thread_id=thread_id,
            role="assistant",
            content=(
                "Hi! I'm your personal AI data scientist. "
                "Upload a CSV or Excel file and ask me anything — "
                "I'll analyze, visualize and build models for you."
            )
        )

        logger.info(f"Auto-created agent {agent_id} and thread {thread_id} for {email}")

        return {
            "message": "Account created successfully",
            "user": {
                "user_id": user_id,
                "email": email,
                "full_name": full_name or email.split("@")[0],
                "is_active": True
            },
            "agent_id": agent_id,
            "agent_name": agent_name,
            "session_id": thread_id,
            "total_chats": 1
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Automl register/login failed for {email}: {e}")
        raise HTTPException(status_code=500, detail="Automl onboarding failed")


@app.post("/register")
def register_user(
    email: str = Form(...), 
    password: str = Form(...), 
    full_name: str = Form(None),
    user_id: str = Form(None)
):
    """
    If `user_id` is provided, it is reused instead of generating a new one
    (must not already be assigned to a different email). If omitted, a new
    user_id is generated automatically, same as before.
    """
    try:
        if auth_handler.user_exists(email):
            raise HTTPException(status_code=400, detail="User already exists")

        try:
            user = auth_handler.create_user(
                email=email,
                password=password,
                full_name=full_name,
                user_id=user_id
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        user_id = user["user_id"]

        agent_name = full_name or email.split("@")[0] or "Data Assistant"
        agent_id = handler.create_agent(name=agent_name)
        
        thread_id, _ = handler.create_thread(agent_id)

        auth_handler.add_agent_to_user(
            email=email,
            agent_id=agent_id,
            agent_name=agent_name,
            thread_id=thread_id
        )

        response_handler.save_response(
            email=email,
            user_id=user_id,
            agent_id=agent_id,
            thread_id=thread_id,
            role="assistant",
            content=f"Hi! I'm your personal AI data scientist. Upload a CSV or Excel file and ask me anything — I'll analyze, visualize and build models for you."
        )

        logger.info(f"Auto-created agent {agent_id} and thread {thread_id} for {email}")
        user.pop("password", None)
        
        return {
            "message": "Account created successfully",
            "user": {
                "user_id": user_id,
                "email": email,
                "full_name": full_name or email.split("@")[0],
                "is_active": True
            },
            "agent_id": agent_id,
            "agent_name": agent_name,
            "session_id": thread_id,
            "total_chats": 1
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/login")
def login_user(email: str = Form(...), password: str = Form(...)):
    try:
        user = auth_handler.authenticate_user(email=email, password=password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        agent = user["agents"][0]

        return {
            "message": "Login successful",
            "user": {
                "user_id": user["user_id"],
                "email": user["email"],
                "full_name": user.get("full_name") or user["email"].split("@")[0],
                "is_active": user.get("is_active", True)
            },
            "agent_id": agent["agent_id"],
            "agent_name": agent.get("agent_name", "My Data Assistant"),
            "session_id": agent["threads"][-1],
            "total_chats": len(agent["threads"])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed for {email}: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/user/{email}")
def get_user_details(email: str):
    user = auth_handler.get_user(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user": user}

@app.post("/create_session")
def create_thread_endpoint(user_email: str = Form(...)):
    user = auth_handler.get_user(user_email)
    if not user or len(user.get("agents", [])) == 0:
        raise HTTPException(status_code=400, detail="No agent")
    agent_id = user["agents"][0]["agent_id"]
    thread_id, _ = handler.create_thread(agent_id)
    auth_handler.add_thread_to_user_agent(user_email, thread_id)
    return {"session_id": thread_id, "agent_id": agent_id}

@app.delete("/delete_session")
def delete_session(
    session_id: str = Form(...),
    user_email: str = Form(...)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")

        agent_info = user.get("agents", [{}])[0]
        if not agent_info:
            raise HTTPException(status_code=400, detail="No agent found")

        if session_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Session not found or access denied")

        try:
            handler.client.agents.threads.delete(thread_id=session_id)
            logger.info(f"Deleted thread {session_id} from Azure AI")
        except Exception as e:
            logger.warning(f"Thread {session_id} already deleted or not found: {e}")

        auth_handler.remove_thread_from_user(user_email, session_id)
        response_handler.delete_thread(user_email, session_id)

        user_id = user["user_id"]
        try:
            thread_metadata_handler.delete_thread_metadata(session_id, user_id)
            logger.info(f"Deleted thread metadata for session: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to delete thread metadata: {e}")

        return {
            "message": "Session deleted successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session deletion failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to delete session")
    
@app.post("/upload_file")
async def upload_file_endpoint(  
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_email: str = Form(...),
    query: Optional[str] = Form(None)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        agent_info = user["agents"][0]
        if session_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Invalid thread")
        user_id = user["user_id"]
        agent_id = agent_info["agent_id"]
        agent_name = agent_info["agent_name"]
        thread_id = session_id
        
        processed_file, processed_filename, was_converted = await to_csv_if_needed(file)
        display_name = file.filename  # Original for user
        if was_converted:
            display_name += " (converted to CSV)"
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        try:
            processed_file.file.seek(0)
            content = processed_file.file.read()
            temp_file.write(content)
            temp_file.close()
            
            file_id, blob_name, _ = handler.upload_file_to_blob_and_agent(
                processed_file, agent_id, user_id, thread_id=session_id
            )
            
            intelligence_result = dataset_service.process_new_upload(
                file_path=temp_file.name,
                filename=processed_filename,
                user_id=user_id,
                user_email=user_email,
                agent_id=agent_id,
                blob_path=blob_name,
                min_similarity_threshold=60.0
            )
            
            thread_metadata_handler.create_or_update_thread_metadata(
                thread_id=session_id,
                user_id=user_id,
                user_email=user_email,
                agent_id=agent_id,
                dataset_id=intelligence_result["dataset_id"],
                blob_path=blob_name,
                filename=processed_filename,
                # FIXED: 'file_path' was never defined in this function (line 439
                # is a keyword argument, not an assignment), so every call to
                # /upload_file raised NameError. Plain uploads have no lakehouse
                # path - that is what /upload_file_V is for.
                veriton_file_path=None
            )
            
            run_auto_analysis = str(query or "").lower() == "true"
            # CASE 1 – NO AUTO-ANALYSIS
            if not run_auto_analysis:
                suggestion = handler.get_dynamic_suggestion(
                    thread_id=thread_id,
                    user_email=user_email,
                    agent_info=agent_info,
                    analysis_text=f"Uploaded {display_name}",
                    user_id=user_id,
                    agent_name=agent_name,
                )
                return {
                    "filename": display_name,
                    "fileid": file_id,
                    "threadid": session_id,
                    "response": f"File '{display_name}' uploaded successfully.",
                    "ml_ready": False,
                    "auto_analysis": False,
                    "suggestions": _suggestions_list(suggestion)
                }
           
            # ------------------------------------------------------------
            # Pull the model's internal "NEXT_STEPS: a | b | c" line out of
            # the visible text so the user never sees the marker, and turn
            # it into a real list of actionable suggestions instead of one
            # weak "Ask: what's next?" line. Shared implementation lives in
            # app/utils.py (it also handles "NEXT STEPS:" with a space and a
            # payload on the following line, which this local alias used to
            # miss - that gap is exactly what let a leaked marker reach the
            # user on the query endpoints).
            # ------------------------------------------------------------
            _split_next_steps = split_next_steps

            # ------------------------------------------------------------
            # Classify the upload: NEW / CHANGED (meaningful) / UNCHANGED
            # ------------------------------------------------------------
            has_matches = intelligence_result.get("has_matches", False)
            similar_datasets = intelligence_result.get("similar_datasets") or []
            top_match = similar_datasets[0] if similar_datasets else None
            match_type = (top_match or {}).get("match_type")

            upload_case = "new"
            if has_matches and top_match:
                # "exact_match" = same data, nothing meaningful changed.
                # evolved_dataset / similar_evolved / similar_dataset = a
                # changed version of a dataset we've seen before.
                upload_case = "unchanged" if match_type == "exact_match" else "changed"

            # Shared reasoning framework injected into both the NEW and
            # CHANGED prompts below - this is the core fix: don't just
            # describe columns, drill into every important finding.
            DRILLDOWN_RULES = (
                "NO FIXED TEMPLATE RULE (read this first):\n"
                "Do not follow a fixed response template or predefined section structure. "
                "Analyze the actual dataset and dynamically organize your answer around "
                "whatever is most relevant for THIS data - the real business insights, "
                "patterns, risks, opportunities, data-quality issues, and analytical "
                "possibilities it actually contains. Use headings only when they genuinely "
                "improve readability, and derive any heading from this dataset's own domain "
                "and findings - never from a predefined checklist. Never reuse the same "
                "section titles, opening line, or overall shape response after response; two "
                "different uploads (or two looks at the same file) should read like two "
                "different pieces of analysis, not the same form filled in twice. Skip "
                "anything - a section, a heading, a whole topic - that isn't genuinely "
                "relevant here. Be concise when the data only supports a simple answer, and "
                "go deeper only where the data genuinely warrants it.\n\n"
                "CRITICAL REASONING RULE - follow this for every important finding:\n"
                "Do not merely describe columns or repeat dataset statistics. For every "
                "important finding, investigate whether the data provides a meaningful "
                "comparison, concentration, relationship, trend, anomaly, or business "
                "impact. When you identify a potentially important metric or issue, drill "
                "down into it to determine WHERE, WHEN, or FOR WHOM it occurs, whenever the "
                "columns support that analysis. Follow this progression: "
                "DATA -> FINDING -> DRILL-DOWN -> IMPACT -> ACTION.\n"
                "Weak (never do this): 'Downtime is tracked.' or 'About 30 of 80 orders are "
                "flagged as breakdown risk.' and then stop.\n"
                "Good (always aim for this): 'Downtime is concentrated in 3 of the 8 "
                "machines, with Machine M04 contributing the largest share, and those same "
                "orders also show elevated defect rates - worth investigating first.'\n"
                "Whenever you mention a count, a rate, or a flagged group (e.g. 'X of Y "
                "orders are flagged'), immediately break it down further using whatever "
                "columns exist - which machine, shift, product, plant, supplier, or time "
                "period it's concentrated in - and check whether it coincides with other "
                "problems (higher downtime, more defects, higher cost, more delays). Never "
                "leave a striking number unexplored.\n\n"
                "PRIORITIZATION RULE:\n"
                "Focus on findings that are genuinely meaningful to the business rather than "
                "trying to touch every column, metric, or observation in the file. Cover "
                "however many findings actually deserve attention for this dataset - not a "
                "fixed count - each with: what it is, why it matters, the evidence (real "
                "numbers/names), and what to do about it. Show what matters most through the "
                "order you present things, the depth of evidence you give it, and your "
                "wording - never through explicit labels like 'Priority 1', 'High Priority', "
                "or numbered rankings. Let the most important thing simply come first and get "
                "the most substance; let minor observations stay brief or get left out.\n\n"
                "WHAT WE COULD DO WITH THIS DATA (only when genuinely useful - be selective):\n"
                "Weave in, wherever it reads naturally (a short labeled section is fine, but "
                "don't force the exact same heading and shape every time), which realistic "
                "analytical or ML approaches genuinely fit THIS dataset - not a checklist of "
                "everything the system happens to support. Judge this using the real columns "
                "available, whether a sensible target/outcome column exists, data quality, "
                "sample size, and the actual business problem. Recommend only the ones that "
                "provide real apparent business value; if several are technically possible, "
                "lead with the one(s) that matter most rather than listing all of them with "
                "equal weight. The possible kinds are:\n"
                "- Classification -> predicting a category or label for each record (e.g. "
                "'yes/no', 'high/medium/low').\n"
                "- Regression -> predicting a number (like a cost, quantity, or score).\n"
                "- Forecasting -> predicting future values over time (like next month's "
                "demand or cost) - needs a real date/time column with enough history.\n"
                "- Clustering -> grouping similar records together to find natural patterns "
                "or segments - useful when there's no single obvious outcome column.\n"
                "- Anomaly detection -> automatically flagging unusual records that don't fit "
                "the normal pattern.\n"
                "Do not recommend a kind just because it's technically possible - if the data "
                "lacks a suitable target, enough history, the right features, or good enough "
                "quality for something a user might expect (e.g. forecasting with no usable "
                "date column), say so plainly and explain why it isn't a good fit right now, "
                "rather than silently omitting it or forcing it in anyway. If genuinely "
                "nothing fits well, say that honestly instead of padding the list.\n\n"
                "HOW TO WRITE EACH OPPORTUNITY YOU DO RECOMMEND:\n"
                "For each one, cover: what could be predicted or found (naming the REAL "
                "column from this dataset - copied exactly, never invented or guessed), what "
                "real columns would be used to do it, and why it matters for the business. "
                "Example shape: 'ProductionCost could be predicted using PlannedQuantity, "
                "ProducedQuantity, and ProductionTimeHours - tree-based methods like Random "
                "Forest or XGBoost could work well here since cost likely has a nonlinear "
                "relationship with these operational variables.'\n\n"
                "MODEL NAMING RULE (strict):\n"
                "You may name 1-2 model families per opportunity when they add real value - "
                "never an exhaustive list, and never bare (always paired with a brief reason "
                "tied to this dataset's actual characteristics: size, feature types, "
                "linearity, target type). Do NOT use your own general ML knowledge to name "
                "models - use ONLY the exact models this system actually has available for "
                "that task, listed below. Never mention any model not on this list for that "
                "task (for example, never say SVM, KNN, hierarchical clustering, linear "
                "regression, neural networks, or deep learning - this system does not offer "
                "them):\n"
                "- Classification: Logistic Regression, Random Forest, Gradient Boosting, "
                "XGBoost\n"
                "- Regression: Ridge Regression, Random Forest, Gradient Boosting, XGBoost\n"
                "- Forecasting: ARIMA, Prophet, XGBoost, LightGBM, CatBoost\n"
                "- Clustering: K-Means, K-Means++, DBSCAN, Gaussian Mixture Model\n"
                "- Anomaly Detection: Isolation Forest, One-Class SVM, Local Outlier Factor, "
                "Elliptic Envelope\n"
                "Pick only the 1-2 from the relevant list above that genuinely suit this "
                "dataset - e.g. tree-based models (Random Forest, XGBoost, Gradient Boosting, "
                "LightGBM, CatBoost) for data with nonlinear relationships between mixed "
                "numeric/categorical features; Isolation Forest as a reasonable default "
                "starting point for anomaly detection; K-Means for clustering when features "
                "are numeric and reasonably scaled. Do not recommend a specific model without "
                "real evidence from the data that it fits - if unsure, it's fine to mention "
                "the opportunity without naming a specific model at all.\n\n"
                "CLOSING RULE:\n"
                "End your visible answer naturally. Then, on its own final line, include "
                "an internal marker the user will never see (it gets stripped before "
                "display): NEXT_STEPS: <action 1> | <action 2> | <action 3> - each action "
                "under 12 words, phrased as a plain-language OUTCOME a non-technical person "
                "would understand, using real machine/shift/product/group names where "
                "relevant but NEVER raw column names or technical terms. Say 'Find which "
                "machines or shifts have the highest defect rates' not 'analyze "
                "defective_quantity by machine_id'. Say 'Investigate how downtime relates to "
                "production delays' not 'review downtime_hours and production_status'. This "
                "line is a control signal, not part of the message.\n\n"
                "LANGUAGE RULES:\n"
                "- Outside the ML-possibilities part of your answer, never use the words "
                "model, target, classification, regression, clustering, algorithm, dataframe, "
                "feature engineering, null ratio, or any ML/stats jargon in the visible text.\n"
                "- Inside that part, you may name a task type or technique, but only ever "
                "right next to a plain-language explanation of what it means, never bare.\n"
                "- Use real column names and real numbers throughout, never placeholders.\n"
                "- Write like a sharp analyst talking to a manager - flowing paragraphs, a "
                "few bullets where useful, never a rigid numbered report with headers.\n"
                "- Vary your structure, opening line, and phrasing every time - never reuse "
                "the same section titles or shape from one response to the next.\n"
            )


            FILE_LOAD_HINT = (
                f"\n\nFILE ACCESS (Code Interpreter): The file '{processed_filename}' has been "
                "uploaded and is available to Code Interpreter. To load it, use:\n"
                "  import pandas as pd, glob, os\n"
                f"  matches = glob.glob('/mnt/data/**/{processed_filename}', recursive=True) "
                f"or glob.glob('/mnt/data/{processed_filename}')\n"
                "  df = pd.read_csv(matches[0]) if matches else pd.read_csv"
                f"('/mnt/data/{processed_filename}')\n"
                "Always use this glob approach - never hard-code a path that may not exist.\n"
            )

            INTERNAL_ML_TRIGGER = (
                "\n\nINTERNAL ONLY - never shown to the user, only used by the system:\n"
                "If, in a LATER message, the user replies with agreement or continuation "
                "('yes', 'ok', 'go', 'do it', 'sounds good', 'predict [something]', 'start', "
                "etc.), respond to THAT message with ONLY this JSON (no markdown, no "
                "explanation - this is a control signal, not something the user reads):\n"
                "{\n"
                '  "is_ml": true,\n'
                '  "task_type": "<task>",\n'
                '  "target": "<column_name>",\n'
                '  "metric": "<metric_name>",\n'
                ' "models": ["<model1>", "<model2>",""], \n'
                "}\n"
                "Do NOT output this JSON unless the user clearly wants to run the suggested "
                "task; otherwise keep responding normally in plain language.\n"
            )

            # ============================================================
            # CASE: NEW DATASET - full domain-aware, drill-down analysis
            # ============================================================
            if upload_case == "new":
                analysis_prompt = (
                    f"You are an experienced data analyst explaining a file **{display_name}** to "
                    f"a business person with ZERO technical background - no user question yet, "
                    f"this is the first look at the file.\n\n"
                    "Use Code Interpreter to actually explore the real data first - real columns, "
                    "real numbers, real groups.\n\n"
                    "Cover, weaving in only what the data genuinely supports (skip anything that "
                    "doesn't apply, never force it):\n"
                    "- what this data is and why it matters to the business\n"
                    "- what timeframe it covers and whether there's enough history for trends\n"
                    "- whichever things genuinely need attention, each drilled down to "
                    "where/when/for whom it occurs and why it matters - however many that "
                    "turns out to be for this dataset, shown through order and depth, not "
                    "labels\n"
                    "- what's going well, not just problems\n"
                    "- any data-quality gaps that limit how far you can trust a conclusion\n"
                    "- concrete recommendations and what to investigate next\n"
                    "- anything you genuinely cannot conclude because information is missing\n"
                    "- which of the tasks in the What we could do with this data section genuinely fit, with real target columns named\n\n"
                    f"{FILE_LOAD_HINT}"
                    f"{DRILLDOWN_RULES}"
                    "Keep the whole answer focused - roughly 200-350 words depending on how much "
                    "the data supports; don't pad it out."
                    f"{INTERNAL_ML_TRIGGER}"
                )

            # ============================================================
            # CASE: CHANGED - explain what's different, drill into new/
            # resolved issues, still fully analyze the current file
            # ============================================================
            elif upload_case == "changed":
                prev_history = dataset_service.get_dataset_history(top_match["dataset_id"], user_id)
                prev_tasks = (prev_history or {}).get("tasks_performed", [])
                prev_semantic = (prev_history or {}).get("semantic_analysis", {})
                details = top_match.get("details", {})
                task_lines = "\n".join(
                    f"- {t.get('task_type')}: {t.get('query','')[:80]} -> {t.get('status')}"
                    for t in prev_tasks[-5:]
                ) or "nothing was run on it yet"

                analysis_prompt = (
                    f"You are an experienced data analyst explaining a file **{display_name}** to "
                    f"a business person with ZERO technical background. This is a CHANGED version "
                    f"of a dataset seen before: **{top_match['similarity_score']:.0f}% similar** to "
                    f"one from {str(top_match.get('upload_date',''))[:10]}. Row count changed by "
                    f"{details.get('row_difference', 'an unknown amount')} and column count changed "
                    f"by {details.get('column_difference', 'an unknown amount')} vs that version. "
                    f"Previous structural summary (for your own context): "
                    f"{json.dumps(prev_semantic)[:1200]}. Previously run on it: {task_lines}.\n\n"
                    "Use Code Interpreter to actually explore the CURRENT data yourself.\n\n"
                    "Open by clearly stating, in plain words, what concretely changed since last "
                    "time (more/fewer records, new or missing columns, shifted numbers) - be "
                    "specific, not vague. Then cover:\n"
                    "- whether previously known issues are resolved, worse, or unchanged now\n"
                    "- whichever things genuinely need attention right now, each drilled "
                    "down to where/when/for whom and why it matters - shown through order "
                    "and depth, not labels\n"
                    "- new findings that weren't true before\n"
                    "- what's going well\n"
                    "- whether what was tried before still makes sense given the change\n"
                    "- concrete recommendations and what to investigate next\n"
                    "- which of the tasks in the What we could do with this data section genuinely fit, with real target columns named\n\n"
                    f"{FILE_LOAD_HINT}"
                    f"{DRILLDOWN_RULES}"
                    "Give a complete analysis of the current file - not just a diff - "
                    "while making the meaningful changes and how they affect the "
                    "previous understanding, findings, data quality, or analytical "
                    "possibilities a clear part of that analysis, not a separate bolted-on "
                    "section. Roughly 250-400 words depending on what the data supports."
                    f"{INTERNAL_ML_TRIGGER}"
                )

            # ============================================================
            # ============================================================
            # CASE: UNCHANGED - short REFRESH analysis, not a full report
            # and not a "nothing to see here" skip. Still uses Code
            # Interpreter to verify current findings hold, but stays brief.
            # ============================================================
            else:
                prev_history = dataset_service.get_dataset_history(top_match["dataset_id"], user_id)
                prev_tasks = (prev_history or {}).get("tasks_performed", [])
                task_count = len(prev_tasks)
                task_lines = "\n".join(
                    f"- {t.get('task_type')}: {t.get('query','')[:80]} -> {t.get('status')}"
                    for t in prev_tasks[-5:]
                ) or "nothing was run on it yet"

                analysis_prompt = (
                    f"You are an experienced data analyst explaining a file **{display_name}** to "
                    f"a business person with ZERO technical background. This file is materially "
                    f"UNCHANGED from a previous upload - {top_match['similarity_score']:.0f}% "
                    f"similar to one from {str(top_match.get('upload_date',''))[:10]}, no "
                    f"meaningful differences in structure or content. Previously run on it: "
                    f"{task_lines} (times run: {task_count}).\n\n"
                    "Use Code Interpreter to actually check the CURRENT data - don't skip "
                    "analysis just because it's unchanged, but keep this SHORT: a refresh, not a "
                    "full report.\n\n"
                    "SIMILAR DATASET WITHOUT MEANINGFUL CHANGES - do exactly this:\n"
                    "Open by clearly stating this is materially unchanged from before, so most of "
                    "the earlier analysis still applies. Then briefly cover, only where genuinely "
                    "true, in flowing plain-language prose (not headers, not a numbered report):\n"
                    "- the most important current findings, verified against the current file\n"
                    "- 1-3 key insights or patterns worth restating\n"
                    "- any notable anomalies or unusual observations, if present\n"
                    "- issues that still need attention (naming specifics, not generic)\n"
                    "- unresolved risks or data-quality concerns still relevant\n"
                    "- which previous findings remain relevant right now\n"
                    "- 1-2 practical recommendations or useful next investigations\n"
                    "- which of the tasks in the What we could do with this data section still genuinely fit, with real target columns named\n\n"
                    "Use the CURRENT dataset to verify these still hold - do not simply copy the "
                    "previous analysis text or invent findings that aren't really there. If there "
                    "genuinely isn't anything new or noteworthy beyond what's already known, say "
                    "so plainly instead of manufacturing insights. Do not repeat basic dataset "
                    "descriptions or full column explanations - the user already knows this "
                    "dataset. The goal is a short refresh of what matters NOW, not a skipped "
                    "analysis and not a full repeat.\n"
                    "Answer this question in your own words somewhere in the reply: 'Okay, I "
                    "uploaded it again - what's important in it right now?'\n\n"
                    "Keep the whole thing to roughly 100-180 words.\n\n"
                    f"{FILE_LOAD_HINT}"
                    f"{DRILLDOWN_RULES}"
                    f"{INTERNAL_ML_TRIGGER}"
                )

                _, ml_json_unchanged, refresh_response_raw = handler.process_query_on_main_agent(
                    agent_id=agent_id,
                    query=analysis_prompt,
                    thread_id=session_id,
                    user_id=user_id,
                    agent_name=agent_name
                )
                fallback_suggestion = "Ask: what's next?"
                try:
                    fallback_suggestion = handler.get_dynamic_suggestion(
                        thread_id=thread_id,
                        user_email=user_email,
                        agent_info=agent_info,
                        analysis_text=refresh_response_raw or "",
                        user_id=user_id,
                        agent_name=agent_name,
                    )
                except Exception:
                    pass
                clean_text, steps = _split_next_steps(refresh_response_raw, fallback_suggestion)
                if not clean_text:
                    clean_text = (
                        f"This looks materially unchanged from {top_match['filename']} "
                        f"({top_match['similarity_score']:.0f}% match), so the earlier findings "
                        f"still apply. You've run {task_count} task(s) on it before."
                    )

                response_handler.save_response(user_email, user_id, agent_id, session_id, "user", f"Uploaded file: {display_name}")
                response_handler.save_response(user_email, user_id, agent_id, session_id, "assistant", clean_text)
                suggestion = handler.get_dynamic_suggestion(
                    thread_id=thread_id,
                    user_email=user_email,
                    agent_info=agent_info,
                    analysis_text=clean_text,
                    user_id=user_id,
                    agent_name=agent_name,
                )
                return {
                    "filename": display_name,
                    "fileid": file_id,
                    "threadid": session_id,
                    "overview_response": clean_text,
                    "query_response": clean_text,
                    "ml_ready": True,
                    "auto_analysis": True,
                    "upload_case": upload_case,
                    "suggestions": steps or _suggestions_list(suggestion),
                    "message": "Quick refresh on your familiar dataset - here's what still matters."
                }
            # Shared execution path for NEW and CHANGED cases
            # ------------------------------------------------------------
            _, ml_json, overview_response_raw = handler.process_query_on_main_agent(
                agent_id=agent_id,
                query=analysis_prompt,
                thread_id=session_id,
                user_id=user_id,
                agent_name=agent_name
            )
            fallback_suggestion = "Ask: what's next?"
            try:
                fallback_suggestion = handler.get_dynamic_suggestion(
                    thread_id=thread_id,
                    user_email=user_email,
                    agent_info=agent_info,
                    analysis_text=overview_response_raw or "",
                    user_id=user_id,
                    agent_name=agent_name,
                )
            except Exception:
                pass
            overview_response, next_steps = _split_next_steps(overview_response_raw, fallback_suggestion)

            response_handler.save_response(user_email, user_id, agent_id, session_id,
                                          "user", f"Uploaded file: {display_name}")
            if overview_response:
                response_handler.save_response(user_email, user_id, agent_id, session_id,
                                              "assistant", overview_response)
            ml_ready = any(k in (overview_response or "").lower()
                           for k in ["predict", "breakdown", "likely to", "flagged", "risk"])
            return {
                "filename": display_name,
                "fileid": file_id,
                "threadid": session_id,
                "overview_response": overview_response.strip() if overview_response else "Analysis completed",
                "query_response": overview_response.strip() if overview_response else "Analysis completed",
                "ml_ready": ml_ready,
                "auto_analysis": True,
                "upload_case": upload_case,
                "suggestions": next_steps or [fallback_suggestion],
                "message": "New dataset analyzed! Ready for action." if upload_case == "new"
                            else "Dataset compared to previous version! Ready for action."
            }
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)
    except Exception as e:
        logger.error(f"Upload error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Upload failed")

def _suggestions_list(suggestion: Optional[str]) -> list:
    """
    Wrap a single suggestion string as the suggestions list the API returns,
    or an empty list when there's nothing to suggest. get_dynamic_suggestion
    returns None on purpose whenever the main response itself was a failure
    (no matching data, or a "couldn't read the file" excuse) - showing an
    unrelated "try this query" suggestion right under an error message is
    confusing, so in that case we show no suggestion at all instead of
    substituting a generic fallback.
    """
    return [suggestion] if suggestion else []


def _build_query_analysis_rules(filename: Optional[str]) -> str:
    """
    Shared prompt wrapper for a plain chatbot query (non-ML) about the
    uploaded dataset, used by both /process_task_query and
    /process_task_query_v so the two endpoints behave identically:

    - Tells the model how to actually load the file (the glob hint), so it
      doesn't fall back to a hypothetical "here's what I'd do once I can
      read the file" answer for lack of knowing where the file is - this was
      previously only given to the model on the initial upload turn.
    - Requires answering from the closest REAL related data when the exact
      thing asked isn't present, instead of jumping straight to "no data".
    - Makes "I don't have data related to that" a last resort, only once
      every reasonably related column has genuinely been checked - not a
      default/repeated answer.
    - Forbids inventing facts/numbers/categories not verified by code run
      against the real data this turn.
    - Matches response length to the question: one line if one line fully
      answers it; longer prose only when the question genuinely needs it.
      No forced report sections, no bullet headers, for a plain query.
    - Forbids hypothetical "I will do X once I can access the file" replies;
      if the file genuinely can't be read this turn, say so in one short,
      honest sentence and stop.
    - Forbids a visible NEXT_STEPS marker on a query turn - suggestions are
      generated separately.
    """
    file_hint = (
        (
            f"\n\nFILE ACCESS (Code Interpreter): The file '{filename}' has "
            "already been uploaded and is available to Code Interpreter. To "
            "load it, use:\n"
            "  import pandas as pd, glob, os\n"
            f"  matches = glob.glob('/mnt/data/**/{filename}', recursive=True) "
            f"or glob.glob('/mnt/data/{filename}')\n"
            "  df = pd.read_csv(matches[0]) if matches else pd.read_csv"
            f"('/mnt/data/{filename}')\n"
            "Always use this glob approach - never hard-code a path that may "
            "not exist. If, after actually trying this, the file truly "
            "cannot be read, say so in ONE short honest sentence and stop - "
            "never write a hypothetical 'here's what I would do once I can "
            "read the file' plan.\n"
        )
        if filename else ""
    )

    return (
        "\n\n---\n"
        "RESPONSE QUALITY RULES for this question (only if this is not an ML "
        "model-building request - if it is, emit ONLY the is_ml JSON control "
        "block as usual - never a plain-language description of the task "
        "instead of that JSON, e.g. never reply with something like "
        "'Forecast 12 months of X using Y' as prose - either output the real "
        "JSON so the system can actually run it, or don't mention it here at "
        "all):\n"
        f"{file_hint}"
        "- GROUNDING (MOST IMPORTANT): Use Code Interpreter to actually load "
        "and query the real uploaded dataset before answering. Every fact, "
        "number, ranking, category name, or 'best/worst' claim in your reply "
        "must come from code you actually ran against the real data this "
        "turn - never state something because it sounds plausible for this "
        "kind of dataset. Do not invent reasons or explanations (e.g. "
        "'better maintenance practices', 'operational conditions') unless "
        "the dataset has a column that actually supports that specific "
        "claim - if it doesn't, leave that reasoning out entirely, and "
        "never state a category name unless that exact value was verified "
        "in the real data by code you ran.\n"
        "- Actually answer the specific question asked. If asked for root "
        "causes, reasons, or 'why', identify and name the specific real "
        "column, category, or group actually responsible, using real data - "
        "never substitute an unrelated statistic (e.g. never answer a "
        "root-cause question with an unrelated max/min of some other "
        "column).\n"
        "- PARTIAL / RELATED MATCH FIRST: if the exact thing asked for isn't "
        "a column in this dataset but something clearly related is, answer "
        "using that closest real data instead of saying there's no data - "
        "just say plainly that the exact field isn't present and name the "
        "real column you used instead, so the user knows it's the nearest "
        "available proxy, not a direct hit.\n"
        "- 'NO DATA' IS A LAST RESORT: only say you couldn't find data "
        "related to the question after genuinely checking whether any "
        "column in this dataset is related - never as a routine or default "
        "answer, and never for a question that a real or related column "
        "here can actually answer. If truly nothing matches, reply in one "
        "honest, plain-language sentence and name a few of the real columns "
        "that ARE available - never reply with just a bare placeholder "
        "token like CANNOT_ANSWER.\n"
        "- If a model has already been built on this dataset and its results "
        "answer the question, use those real results rather than "
        "re-deriving or guessing.\n"
        "- Never dump a long unsorted list of raw numbers as the entire "
        "answer. If many groups/values are involved, summarize only the "
        "top few that matter (e.g. top 3-5) with real numbers, then state "
        "what that implies - not a flat table of everything.\n"
        "- Never give a single bare value with no context (e.g. just "
        "'P001'). Say what it is, WHY it's the answer (the real numbers "
        "that make it best/worst/most relevant), and what that means.\n"
        "- When asked 'which X should be investigated/prioritized', don't "
        "just sort one column - briefly explain why the top result(s) "
        "stand out, using at least one other related column for context "
        "if the data supports it.\n"
        "- LENGTH MATCHES THE QUESTION: if a single sentence fully answers "
        "it, give exactly that - one line, no padding. Only write more when "
        "the question genuinely needs explanation. This is a chat reply, "
        "not a report: no headers, no bullet sections, no numbered plan.\n"
        "- Do not include a NEXT_STEPS line or any control marker in your "
        "answer for this question - suggestions are handled separately.\n"
    )


@app.post("/process_task_query")
def process_task_query_endpoint(
    session_id: str = Form(...), 
    query: str = Form(...), 
    user_email: str = Form(...)
):
    
    def get_dynamic_suggestion():
        # Delegates to the shared, schema-grounded suggestion generator so a
        # failed/"no data found" answer never gets followed by a suggestion
        # that references a column or field that doesn't actually exist in
        # this dataset (it used to hallucinate names like 'plant_plantname'
        # from an unrelated column such as 'employee_employeename').
        try:
            return handler.get_dynamic_suggestion(
                thread_id=thread_id,
                user_email=user_email,
                agent_info=agent_info,
                analysis_text=analysis_text or "",
                user_id=user_id,
                agent_name=agent_name,
            )
        except Exception:
            return "Ask: what's next?"

    try:
        user = auth_handler.get_user(user_email)
        if not user or len(user.get("agents", [])) == 0:
            raise HTTPException(status_code=400, detail="No agent")
        
        agent_info = user["agents"][0]
        if session_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Invalid thread")

        user_id = user["user_id"]
        agent_name = agent_info["agent_name"]
        thread_id = session_id
        blob_file = handler.get_latest_blob_file(user_id, agent_name)

        dataset_id = thread_metadata_handler.get_current_dataset_id(
            thread_id=thread_id,
            user_id=user_id
        )
        
        if dataset_id:
            logger.info(f"Using dataset: {dataset_id} for thread: {thread_id}")
        else:
            logger.warning(f"No dataset found for thread: {thread_id}")

        # STEP 1: Run agent to detect intent
        # Wrap the raw question with reasoning-quality rules so plain-language
        # answers actually drill into the data instead of returning an
        # unrelated stat, a raw unsorted dump, or a bare value with no
        # context. This does not affect ML-task detection - the model still
        # sees the user's real question first and can still emit the is_ml
        # control JSON when appropriate.
        filename = blob_file.split('/')[-1] if blob_file else None
        QUERY_ANALYSIS_RULES = _build_query_analysis_rules(filename)
        thread_id, ml_json, analysis_text = handler.process_query_on_main_agent(
            agent_id=agent_info["agent_id"],
            query=f"{query}{QUERY_ANALYSIS_RULES}",
            raw_query=query,
            thread_id=thread_id,
            user_id=user_id,
            agent_name=agent_name
        )

        # Strip any leaked "NEXT_STEPS: a | b | c" control marker before this
        # ever reaches the user - suggestions for a chat query are generated
        # separately via get_dynamic_suggestion(), so this line was never
        # meant to be visible here. Prefer the model's own steps as
        # suggestions when it does supply them.
        extracted_steps = []
        if not (ml_json and ml_json.get("is_ml")) and analysis_text:
            analysis_text, extracted_steps = split_next_steps(analysis_text, None)

        # Defensive net: if the agent still leaks a bare control token instead
        # of a real sentence (e.g. a literal "CANNOT_ANSWER" reaching the
        # user), replace it with an honest, grounded message that names the
        # real available columns instead of silently showing the placeholder.
        if analysis_text and analysis_text.strip().strip(".").upper() in (
            "CANNOT_ANSWER", "CANNOT ANSWER", "I CANNOT ANSWER", "N/A", "NONE"
        ):
            try:
                available_cols = handler.get_dataset_columns(blob_file, user_id) if blob_file else []
            except Exception:
                available_cols = []
            if available_cols:
                col_preview = ", ".join(available_cols[:15])
                analysis_text = (
                    "I couldn't find data in this dataset that answers that "
                    f"question directly. The columns available are: {col_preview}. "
                    "Could you rephrase your question using one of these?"
                )
            else:
                analysis_text = (
                    "I couldn't find data in this dataset that answers that "
                    "question directly. Could you rephrase it or tell me which "
                    "column you mean?"
                )

        # Defensive net: if the model excused itself out of actually reading
        # the dataset ("I can't access the file directly from here...") and
        # answered with a hypothetical plan instead of a real computed
        # answer. Force a fresh re-attachment and retry the query ONCE -
        # the intermittent failure is usually because the thread's Code
        # Interpreter session lost its file reference between queries.
        # Only fall back to the "trouble reading" message if the retry
        # also fails.
        if analysis_text and looks_like_access_excuse(analysis_text):
            logger.warning(f"Access excuse detected for {filename}; forcing re-attachment and retrying once")
            try:
                # Force re-attachment by clearing the Cosmos record of
                # attached_file_ids for this thread so _ensure_file_attached
                # unconditionally re-attaches on the retry.
                if hasattr(handler, 'thread_metadata_handler') and handler.thread_metadata_handler and user_id:
                    try:
                        handler.thread_metadata_handler.add_metadata(thread_id, user_id, "attached_file_ids", [])
                    except Exception:
                        pass
                handler._ensure_file_attached_to_thread(thread_id, user_id, agent_info["agent_id"], filename)
                _, ml_json_retry, analysis_text_retry = handler.process_query_on_main_agent(
                    agent_id=agent_info["agent_id"],
                    query=f"{query}{QUERY_ANALYSIS_RULES}",
                    raw_query=query,
                    thread_id=thread_id,
                    user_id=user_id,
                    agent_name=agent_name
                )
                if analysis_text_retry and not looks_like_access_excuse(analysis_text_retry):
                    analysis_text = analysis_text_retry
                    if ml_json_retry:
                        ml_json = ml_json_retry
                else:
                    analysis_text = (
                        f"I had trouble reading {filename or 'your dataset'} just now "
                        "and couldn't compute a real answer. Please try asking again "
                        "in a moment."
                    )
            except Exception as retry_err:
                logger.warning(f"Retry after access excuse failed: {retry_err}")
                analysis_text = (
                    f"I had trouble reading {filename or 'your dataset'} just now "
                    "and couldn't compute a real answer. Please try asking again "
                    "in a moment."
                )

        # Defensive net: if the agent returned a leaked task description
        # (e.g. "Compute sum(productioncost) across all orders." instead of
        # the actual total) OR asked the user a question back instead of
        # answering (e.g. "What price per unit should be used for revenue?"),
        # try to answer from the dataset directly via the rule-based engine
        # before falling back to a retry message.
        if not (ml_json and ml_json.get("is_ml")) and analysis_text and (
            looks_like_leaked_ml_intent(analysis_text)
            or looks_like_agent_asking_question(analysis_text)
        ):
            logger.warning(f"Bad agent response detected (leaked task or question-back); trying rule-based answer.")
            try:
                rule_answer = handler._answer_non_ml_query(
                    query, agent_info["agent_id"], thread_id, user_id, agent_name
                )
                if rule_answer and not looks_like_agent_asking_question(rule_answer):
                    analysis_text = rule_answer
                else:
                    analysis_text = (
                        "I wasn't able to compute a real answer for that just now. "
                        "Please try asking again in a moment."
                    )
            except Exception:
                analysis_text = (
                    "I wasn't able to compute a real answer for that just now. "
                    "Please try asking again in a moment."
                )

        # STEP 2: Handle ML Task
        if ml_json and ml_json.get("is_ml"):
            
            if not blob_file:
                error_msg = "ML task requires a dataset. Please upload a file first."
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", error_msg)
                return {
                    "session_id": thread_id,
                    "task_type": ml_json.get("task_type"),
                    "response": error_msg,
                    "analysis": None,
                    "results": None,
                    "dataset_id": dataset_id,
                    "suggestions": ["Upload a CSV or Excel file"]
                }
            
            task_type = ml_json.get("task_type", "").lower().strip()
            target_raw = ml_json.get("target", "").strip()
            metric = ml_json.get("metric")
            models_requested = ml_json.get("models", [])
            
            # MULTISTEP FORECASTING
            if task_type == "multistep_forecasting":
                horizon = ml_json.get("horizon", 12)
                
                # Validate horizon
                if not isinstance(horizon, int) or horizon < 2:
                    clarification_msg = (
                        f"Invalid horizon value: {horizon}. "
                        f"For multi-step forecasting, horizon must be ≥ 2. "
                        f"Example: 'predict sales for next 12 months' (horizon=12)"
                    )
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", clarification_msg)
                    return {
                        "session_id": thread_id,
                        "task_type": task_type,
                        "response": clarification_msg,
                        "needs_clarification": True,
                        "dataset_id": dataset_id,
                        "suggestions": ["Specify horizon ≥ 2"]
                    }
                
                validation = handler.parse_and_validate_targets(target_raw, blob_file, user_id)
                
                if not validation["valid"]:
                    clarification_msg = validation["message"]
                    if validation.get("similar_columns"):
                        similar_str = "', '".join(validation["similar_columns"][:5])
                        clarification_msg += f"\n\nDid you mean: '{similar_str}'?"
                    
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", clarification_msg)
                    
                    return {
                        "session_id": thread_id,
                        "task_type": task_type,
                        "response": clarification_msg,
                        "needs_clarification": True,
                        "available_columns": validation["available_columns"][:20],
                        "dataset_id": dataset_id,
                        "suggestions": ["Specify valid target column(s)"]
                    }
                
                # Convert validated targets list back to comma-separated string
                target = ", ".join(validation["targets"])
                logger.info(f"Validated targets: {target} (horizon={horizon})")
                
                available_models = ["xgboost", "lightgbm", "catboost"]
                models_to_train = []
                invalid_models = []
                
                if models_requested:
                    for model in models_requested:
                        if model in available_models:
                            models_to_train.append(model)
                        else:
                            invalid_models.append(model)
                
                model_warning = ""
                if invalid_models:
                    model_warning = (
                        f"Note: Models {invalid_models} are not available for multi-step forecasting. "
                        f"Building with: {models_to_train or available_models}"
                    )
                    logger.warning(model_warning)
                
                if not models_to_train:
                    models_to_train = None  
            
            # HANDLE OTHER SUPERVISED TASKS
            elif task_type not in ['clustering', 'anomaly_detection']:
                validation = handler.validate_target_column(target_raw, blob_file, user_id)
            
                if not validation["valid"]:
                    possible_targets = handler.extract_targets_from_query(
                        query, 
                        validation["available_columns"]
                    )
                    
                    clarification_msg = handler.ask_user_to_clarify_target(
                        possible_targets, 
                        validation["available_columns"]
                    )
                    
                    if not clarification_msg:
                        clarification_msg = validation["message"]
                        if validation.get("similar_columns"):
                            similar_str = "', '".join(validation["similar_columns"])
                            clarification_msg += f"\n\nDid you mean one of these? '{similar_str}'"
                    
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", clarification_msg)
                    
                    return {
                        "session_id": thread_id,
                        "task_type": task_type,
                        "response": clarification_msg,
                        "analysis": None,
                        "results": None,
                        "dataset_id": dataset_id,
                        "needs_clarification": True,
                        "available_columns": validation["available_columns"][:20],
                        "suggestions": ["Specify target column clearly"]
                    }
                
                target = validation["matched_column"]
                horizon = None
                models_to_train = models_requested or None
                model_warning = ""
                logger.info(f"Validated target: '{target}' for task: {task_type}")
            
            # UNSUPERVISED TASKS
            else:
                target = target_raw if target_raw and target_raw.lower() != 'none' else None
                horizon = None
                models_to_train = models_requested or None
                model_warning = ""
                logger.info(f"Unsupervised task ({task_type}) - target: {target}")
                # logger.info(f"Unsupervised task ({task_type}) - target: {target} or 'None'")

            #  TRIGGER AUTOML
            try:
                # Look up how this dataset originally got into Blob (e.g.
                # "databricks", "onelake", "upload") so run artifacts can
                # also be mirrored back to the right source, same as the
                # explicit /build_ml_model* endpoints already do.
                source_type_for_run = None
                if dataset_id:
                    try:
                        dataset_doc = dataset_service.metadata_handler.get_dataset_by_id(dataset_id, user_id)
                        if dataset_doc:
                            source_type_for_run = dataset_doc.get("source_type")
                    except Exception as e:
                        logger.warning(f"Could not look up dataset source_type: {e}")

                # Fallback: a new session has no thread-linked dataset_id
                # yet, but blob_file (looked up per user+agent, not per
                # session) may still be the same physical file uploaded
                # from Databricks/OneLake in an earlier session. Match on
                # blob_path directly so write-back doesn't silently regress
                # just because this is a fresh session.
                if not source_type_for_run and blob_file:
                    try:
                        dataset_doc = dataset_service.metadata_handler.get_dataset_by_blob_path(user_id, blob_file)
                        if dataset_doc:
                            source_type_for_run = dataset_doc.get("source_type")
                    except Exception as e:
                        logger.warning(f"Could not look up dataset source_type by blob_path: {e}")

                results_filename = handler._trigger_automl_internal(
                    blob_file=blob_file,
                    query=query,
                    time_budget=300,
                    user_id=user_id,
                    task_type=task_type,
                    target=target,
                    horizon=horizon,
                    models=models_to_train,
                    source_type=source_type_for_run
                )

                container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
                blob_client = container_client.get_blob_client(results_filename)
                results_data = json.loads(blob_client.download_blob().readall().decode('utf-8'))

                # Analysis is cosmetic; never fail a trained model over it.
                try:
                    analysis_text = handler.generate_ml_analysis(
                        agent_id=agent_info["agent_id"],
                        results_data=results_data,
                        thread_id=thread_id
                    )
                except Exception as analysis_error:
                    logger.warning(
                        f"AI analysis unavailable, using fallback summary: "
                        f"{analysis_error}"
                    )
                    analysis_text = AgentHandler._fallback_summary(results_data)

                try:
                    query_response = handler.generate_query_response(
                        agent_id=agent_info["agent_id"],
                        query=query,
                        results_data=results_data,
                        task_type=task_type,
                        thread_id=thread_id,
                    )
                except Exception as qr_error:
                    logger.warning(f"Query response unavailable: {qr_error}")
                    query_response = AgentHandler._fallback_query_response(
                        query, results_data, task_type
                    )
                
                if model_warning:
                    analysis_text = f"{model_warning}\n\n{analysis_text}"

                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", analysis_text)
                
                best_model = results_data.get('best_model')
                train_metrics = results_data.get('train_metrics', {})
                test_metrics = results_data.get('test_metrics', {})
                all_models = results_data.get('all_models', {})
                
                clean_all_models = ModelRegistryHandler.normalize_all_models(all_models, task=task_type)

                if task_type == "multistep_forecasting":
                    primary_metric = f"avg_{metric}" if not metric.startswith("avg_") else metric
                    primary_score = test_metrics.get(primary_metric) or train_metrics.get(primary_metric)
                else:
                    primary_score = test_metrics.get(metric) or train_metrics.get(metric)
                
                if primary_score is None:
                    primary_score = "N/A"
                
                model_id = None
                if results_data:
                    try:
                        run_id = results_data.get('run_id')
                        run_path = results_data.get('run_path', '')
                        
                        model_id = model_registry.register_model(
                            user_id=user_id,
                            user_email=user_email,
                            run_id=run_id,
                            task=task_type,
                            target=target,
                            model_name=best_model,
                            metric=metric,
                            train_metrics=train_metrics,
                            test_metrics=test_metrics,
                            all_models=clean_all_models,
                            dataset_id=dataset_id,
                            blob_path=run_path,
                            horizon=horizon if task_type == "multistep_forecasting" else None,
                            targets_list=target.split(", ") if task_type == "multistep_forecasting" else None,
                            source_type=source_type_for_run,
                            databricks_path=results_data.get("databricks_run_path"),
                            snowflake_path=results_data.get("snowflake_run_path"),
                            onelake_path=results_data.get("onelake_run_path")
                        )
                    except Exception as e:
                        logger.error(f"Failed to register model: {e}", exc_info=True)

                if dataset_id:
                    try:
                        dataset_service.record_task_completion(
                            dataset_id=dataset_id,
                            user_id=user_id,
                            task_type=task_type,
                            query=query,
                            thread_id=thread_id,
                            status="completed",
                            model_id=model_id,
                            model_name=best_model,
                            results_path=results_filename
                        )
                        logger.info(f"Recorded AutoML task in dataset history")
                    except Exception as e:
                        logger.warning(f"Failed to record task: {e}")

                return {
                    "status": "success",
                    "message": "AutoML completed successfully!",
                    "session_id": thread_id,
                    "model_id": model_id,
                    "task_type": task_type.replace("_", " ").title(),
                    "target": target,
                    "best_model": best_model,
                    "primary_metric": metric,
                    "primary_score": primary_score,
                    "all_models": clean_all_models,
                    "analysis": analysis_text,
                    "query_response": query_response,
                    "kpis": handler.compute_kpis_from_blob(blob_file) if blob_file else {"available": False},
                    "blob_file_used": blob_file,
                    "results_filename": results_filename,
                    "dataset_id": dataset_id,
                    "suggestions": _suggestions_list(get_dynamic_suggestion())
                }

            except Exception as e:
                logger.error(f"AutoML failed: {e}")
                error_msg = f"AutoML failed: {str(e)}"
                
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", error_msg)

                return {
                    "status": "error",
                    "session_id": thread_id,
                    "task_type": task_type,
                    "target": target,
                    "blob_file_used": blob_file,
                    "results_filename": None,
                    "results": None,
                    "analysis": error_msg,
                    "response": "ML task failed during processing.",
                    "dataset_id": dataset_id,
                    "suggestions": ["Check your data format", "Try a different task"]
                }
            
        # NON-ML: Just return analysis
        else:
            if not analysis_text:
                raise Exception("No response from agent")
            response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
            response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", analysis_text)
            
            if dataset_id:
                try:
                    dataset_service.record_task_completion(
                        dataset_id=dataset_id,
                        user_id=user_id,
                        task_type="analysis",
                        query=query,
                        thread_id=thread_id,
                        status="completed"
                    )
                    logger.info(f"Recorded analysis in dataset history")
                except Exception as e:
                    logger.warning(f"Failed to record analysis: {e}")
            
            return {
                "status": "success",
                "session_id": thread_id,
                "task_type": "analysis",
                "blob_file_used": blob_file,
                "results": None,
                "response": analysis_text,
                "dataset_id": dataset_id,
                # Prefer steps the model itself supplied (via the now-stripped
                # NEXT_STEPS marker) over asking a fresh LLM call for one.
                "suggestions": extracted_steps or _suggestions_list(get_dynamic_suggestion())
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {traceback.format_exc()}")
        return {
            "status": "error",
            "session_id": session_id,
            "task_type": None,
            "target": None,
            "response": "Sorry, something went wrong.",
            "analysis": None,
            "results": None,
            "dataset_id": None,
            "error": str(e),
            "suggestions": ["Try rephrasing your query"]
        }

@app.get("/get_run_results")
def get_run_results(filename: str = Query(...), user_email: str = Query(...)):
    user = auth_handler.get_user(user_email)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not filename.startswith(f"{user['user_id']}/"):
        raise HTTPException(status_code=403, detail="Access denied")

    container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
    blob_client = container_client.get_blob_client(filename)
    results_data = json.loads(blob_client.download_blob().readall().decode('utf-8'))

    agent_id = user["agents"][0]["agent_id"]
    analysis_thread_id = user["agents"][0]["threads"][0]
    try:
        analysis_text = handler.generate_ml_analysis(agent_id, results_data, analysis_thread_id)
    except Exception as analysis_error:
        logger.warning(f"AI analysis unavailable, using fallback: {analysis_error}")
        analysis_text = AgentHandler._fallback_summary(results_data)

    return {
        "status": "success",
        "filename": filename,
        "results": results_data,
        "analysis": analysis_text
    }

@app.get("/list_files")
def list_user_files(user_email: str = Query(...), agent_name: str = Query(None)):
    user = auth_handler.get_user(user_email)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    user_id = user["user_id"]
    container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
    
    prefix = f"{user_id}/"
    if agent_name:
        prefix += f"{agent_name}/"
    
    blobs = list(container_client.list_blobs(name_starts_with=prefix))
    files = [{"filename": b.name.split("/")[-1], "blob_name": b.name} for b in blobs]

    build_blobs = list(container_client.list_blobs(name_starts_with=f"{user_id}/build_model/"))
    build_model_files = [
        {"filename": b.name.split("/")[-1], "blob_name": b.name}
        for b in build_blobs
    ]

    return {
        "files": files,
        "total_count": len(files),
        "build_model_files": build_model_files,
        "build_model_count": len(build_model_files)
    }

@app.get("/data_preview")
def get_preview(blob_path: str, user_email: str = Query(...)):
    user = auth_handler.get_user(user_email)
    if not user:
        raise HTTPException(401, "Unauthorized")
    
    preview = handler.get_preview(blob_path, user["user_id"])
    if not preview:
        raise HTTPException(404, "No preview")
    
    return {"preview": preview}

@app.get("/task_features")
def get_task_features(
    blob_path: str = Query(..., description="Full path to your file in storage"),
    task: str = Query(..., description="One of: classification, regression, forecasting, clustering, anomaly_detection"),
    user_email: str = Query(..., description="Your email")
):
    
    user = auth_handler.get_user(user_email)
    if not user:
        raise HTTPException(401, "Unauthorized")

    try:
        result = handler.get_features_for_task(
            blob_path=blob_path,
            task=task.lower(),
            user_id=user["user_id"]
        )
        return result
    except PermissionError:
        raise HTTPException(403, "Not your file")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Task features error: {e}")
        raise HTTPException(500, "Failed to analyze file")

@app.get("/models")
def list_user_models(
    user_email: str = Query(...),
    limit: int = Query(20, ge=1, le=100)
):
    user = auth_handler.get_user(user_email)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
 
    user_id = user["user_id"]
    container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
 
    blobs = list(container_client.list_blobs(name_starts_with=f"{user_id}/runs/"))
    results_blobs = [b for b in blobs if b.name.endswith("/results.json")]
    results_blobs.sort(key=lambda b: b.last_modified or datetime.min, reverse=True)
    task_runs = defaultdict(list)
 
    for blob in results_blobs:
        try:
            data = json.loads(
                container_client.get_blob_client(blob.name)
                .download_blob().readall().decode("utf-8")
            )
 
            raw_task = (data.get("task") or data.get("task_type") or "").lower().strip()
            task = {
                "classif": "classification",
                "regress": "regression",
                "forecast": "forecasting",
                "anomaly": "anomaly_detection",
                "cluster": "clustering",
            }.get(raw_task[:8], raw_task)
 
            model = data.get("best_model")
            if not model or not task:
                continue
 
            test_metrics = data.get("test_metrics", {})
 
            # Smart primary metric
            if task == "classification":
                score = test_metrics.get("f1") or test_metrics.get("accuracy") or 0
            elif task == "regression":
                score = test_metrics.get("r2") or -test_metrics.get("rmse", 999)
            elif task == "forecasting":
                score = -test_metrics.get("mape", 999)
            else:
                score = 0
 
            # Human-readable metrics
            pretty = {}
            for key, fmt in [
                ("f1", "{:.1%}"), ("accuracy", "{:.1%}"), ("roc_auc", "{:.1%}"),
                ("r2", "{:.3f}"), ("mape", "{:.1f}%"), ("rmse", "{:.2f}")
            ]:
                if key in test_metrics:
                    pretty[key.replace("_", " ").title()] = fmt.format(test_metrics[key])
 
            task_runs[task].append({
                "model": model,
                "score": float(score),
                "run_time": blob.last_modified,
                "path": blob.name,
                "pretty": pretty or {"Score": f"{score:.3f}"},
            })
        except Exception as e:
            logger.warning(f"Parse failed {blob.name}: {e}")
 
    final_models = []
 
    for task, runs in task_runs.items():
        if not runs:
            continue
 
        counts = Counter(r["model"] for r in runs)
        max_count = max(counts.values())
        frequent = [m for m, c in counts.items() if c == max_count]
 
        def avg_score(m):
            return statistics.mean([r["score"] for r in runs if r["model"] == m])
 
        recommended_model = max(frequent, key=avg_score)
        best_run = max((r for r in runs if r["model"] == recommended_model), key=lambda x: x["run_time"])
 
        final_models.append({
            "task": task.capitalize(),
            "model": recommended_model,
            "used_times": max_count,
            "total_runs": len(runs),
            "metrics": best_run["pretty"],
            "run_time": best_run["run_time"].strftime("%b %d, %I:%M %p"),
            "load_results": f"/get_run_results?filename={best_run['path']}&user_email={user_email}",
        })
 
    final_models.sort(key=lambda x: (x["used_times"], list(x["metrics"].values())[0] if x["metrics"] else "0"), reverse=True)
 
    return {
        "total_tasks": len(final_models),
        "models": final_models[:limit]
    }

# CHAT HISTORY ENDPOINTS
@app.get("/conversation_history/{thread_id}")
def get_conversation_history(thread_id: str, user_email: str = Query(...)):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_agents = user.get("agents", [])
        if len(user_agents) == 0:
            raise HTTPException(status_code=400, detail="User has no agent")
        
        agent_info = user_agents[0]
        if thread_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Thread does not belong to user")
        
        messages = response_handler.get_thread_history(user_email, thread_id)
        
        return {
            "thread_id": thread_id,
            "message_count": len(messages),
            "messages": messages
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch conversation history")

@app.get("/user_threads")
def get_user_threads(user_email: str = Query(...)):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        threads = response_handler.get_all_threads(user_email)
        
        return {
            "user_email": user_email,
            "thread_count": len(threads),
            "threads": threads
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user threads: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch user threads")


@app.delete("/conversation_history/{thread_id}")
def delete_conversation_history(thread_id: str, user_email: str = Query(...)):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_agents = user.get("agents", [])
        if len(user_agents) == 0:
            raise HTTPException(status_code=400, detail="User has no agent")
        
        agent_info = user_agents[0]
        if thread_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Thread does not belong to user")
        
        success = response_handler.delete_thread(user_email, thread_id)
        
        if success:
            return {"message": "Thread history deleted successfully", "thread_id": thread_id}
        else:
            raise HTTPException(status_code=404, detail="Thread not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation history")


@app.delete("/all_conversations")
def delete_all_conversations(user_email: str = Query(...)):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        deleted_count = response_handler.delete_all_threads(user_email)
        
        return {
            "message": "All conversation history deleted successfully",
            "threads_deleted": deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting all conversations: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete all conversations")
    
    
@app.delete("/delete_account")
def delete_account(
    user_email: str = Form(...),
    confirm: str = Form(...)
):
    if confirm != "DELETE":
        raise HTTPException(status_code=400, detail="You must type 'DELETE' to confirm")

    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user_id = user["user_id"]
        agent_info = user.get("agents", [{}])[0]
        agent_id = agent_info.get("agent_id")

        # 1. Delete Agent
        if agent_id:
            try:
                handler.client.agents.delete_agent(agent_id=agent_id)
                logger.info(f"Deleted agent {agent_id}")
            except Exception as e:
                logger.warning(f"Agent {agent_id} already deleted: {e}")

        # 2. Delete all threads
        if agent_info.get("threads"):
            for thread_id in agent_info["threads"]:
                try:
                    handler.client.agents.threads.delete(thread_id=thread_id)
                except:
                    pass
            logger.info(f"Deleted {len(agent_info['threads'])} threads")

        # 3. Delete all blobs
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        prefix = f"{user_id}/"
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        for blob in blobs:
            try:
                container_client.delete_blob(blob.name)
                logger.info(f"Deleted blob: {blob.name}")
            except Exception as e:
                logger.warning(f"Failed to delete blob {blob.name}: {e}")

        # 4. Delete chat history
        response_handler.delete_user_history(user_email)

        try:
            deleted_count = dataset_service.cleanup_user_data(user_id)
            logger.info(f"Deleted {deleted_count} dataset metadata records")
        except Exception as e:
            logger.warning(f"Failed to delete dataset metadata: {e}")

        try:
            thread_count = thread_metadata_handler.delete_user_threads(user_id)
            logger.info(f"Deleted {thread_count} thread metadata records")
        except Exception as e:
            logger.warning(f"Failed to delete thread metadata: {e}")

        # 5. Delete user
        auth_handler.container.delete_item(item=user_email, partition_key=user_email)
        logger.info(f"Deleted user document: {user_email}")

        return {
            "message": "Account and all data permanently deleted",
            "user_email": user_email
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to delete account")
    

@app.get("/dataset_history/{dataset_id}")
def get_dataset_history(dataset_id: str, user_email: str = Query(...)):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_id = user["user_id"]
        history = dataset_service.get_dataset_history(dataset_id, user_id)
        
        if not history:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        return {
            "dataset_id": dataset_id,
            "filename": history["filename"],
            "upload_date": history["upload_timestamp"],
            "blob_path": history["blob_path"],
            "tasks_performed": history.get("tasks_performed", []),
            "task_count": len(history.get("tasks_performed", []))
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_datasets")
def get_user_datasets(user_email: str = Query(...)):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_id = user["user_id"]
        datasets = dataset_service.metadata_handler.get_user_datasets(user_id)
        
        dataset_list = [
            {
                "dataset_id": ds["dataset_id"],
                "filename": ds["filename"],
                "upload_date": ds["upload_timestamp"],
                "task_count": len(ds.get("tasks_performed", [])),
                "last_accessed": ds.get("last_accessed"),
                "row_count": ds["fingerprint"]["structural_signature"]["num_rows"],
                "column_count": ds["fingerprint"]["structural_signature"]["num_columns"]
            }
            for ds in datasets
        ]
        
        return {
            "user_email": user_email,
            "dataset_count": len(dataset_list),
            "datasets": dataset_list
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare_datasets")
def compare_datasets(
    dataset_id_1: str = Form(...),
    dataset_id_2: str = Form(...),
    user_email: str = Form(...)
):
    """Compare two datasets"""
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_id = user["user_id"]
        
        comparison = dataset_service.compare_two_datasets(
            dataset_id_1,
            dataset_id_2,
            user_id
        )
        
        if not comparison:
            raise HTTPException(status_code=404, detail="One or both datasets not found")
        
        return comparison
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/thread_datasets/{session_id}")
def get_thread_datasets(session_id: str, user_email: str = Query(...)):
    """Get all datasets associated with a thread"""
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        user_id = user["user_id"]
        
        # Verify user owns this thread
        agent_info = user.get("agents", [{}])[0]
        if session_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Thread does not belong to user")
        
        dataset_ids = thread_metadata_handler.get_thread_datasets(session_id, user_id)
        
        # Get full dataset information
        datasets = []
        for dataset_id in dataset_ids:
            dataset = dataset_service.get_dataset_history(dataset_id, user_id)
            if dataset:
                datasets.append({
                    "dataset_id": dataset_id,
                    "filename": dataset["filename"],
                    "upload_date": dataset["upload_timestamp"],
                    "task_count": len(dataset.get("tasks_performed", []))
                })
        
        return {
            "session_id": session_id,
            "dataset_count": len(datasets),
            "datasets": datasets
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thread datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/predict")
def predict_on_new_data(
    run_id: str = Form(..., description="AutoML run ID, e.g., classification-churn-20251118_123456"),
    data_blob: str = Form(..., description="Full blob path, e.g., user_id/agent_name/new_data.csv"),
    user_email: str = Form(..., description="Logged-in user email"),
    session_id: str = Form(..., description="Active chat session/thread ID"),
    return_proba: bool = Form(False, description="Return probabilities (classification only)")
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")

        user_id = user["user_id"]
        agent_info = user["agents"][0]

        # Validate session belongs to user
        if session_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Invalid or unauthorized session_id")

        # Security: user can only access their own files
        if not data_blob.startswith(f"{user_id}/"):
            raise HTTPException(status_code=403, detail="Access denied: not your file")

        if not FUN2_URL:
            raise HTTPException(status_code=500, detail="Inference service not configured")

        payload = {
            "run_id": run_id,
            "data_blob": data_blob,
            "return_proba": return_proba
        }

        logger.info(f"Predict | User: {user_email} | Session: {session_id} | Run: {run_id} | File: {data_blob}")

        response = requests.post(FUN2_URL, json=payload, timeout=FUN2_TIMEOUT)

        if response.status_code != 200:
            error_msg = response.text[:500]
            logger.error(f"Inference failed: {response.status_code} {error_msg}")
            raise HTTPException(status_code=502, detail=f"Prediction failed: {error_msg}")

        result = response.json()

        # Extract clean filename
        filename = data_blob.split("/")[-1]

        # Save to chat history as assistant message
        prediction_summary = (
            f"**Prediction Results** (Model: `{result['model_info'].get('model_name', 'Unknown')}`)\n\n"
            f"• **Run ID:** `{run_id}`\n"
            f"• **File:** `{filename}`\n"
            f"• **Rows predicted:** {len(result['predictions']['predictions'])}\n"
            f"• **Task:** {result['model_info'].get('task', 'unknown').title()}\n"
            f"• **Target:** `{result['model_info'].get('target', 'N/A')}`\n"
        )

        if "top_features" in result and result["top_features"]:
            top_feats = ", ".join(list(result["top_features"].keys())[:5])
            prediction_summary += f"• **Top features:** {top_feats}\n"

        prediction_summary += "\nPredictions ready! Ask me to explain, export, or visualize."

        # Save to conversation
        response_handler.save_response(
            email=user_email,
            user_id=user_id,
            agent_id=agent_info["agent_id"],
            thread_id=session_id,
            role="assistant",
            content=prediction_summary
        )

        try:
            dataset_id = thread_metadata_handler.get_current_dataset_id(session_id, user_id)
            if dataset_id:
                dataset_service.record_task_completion(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    task_type="inference",
                    query=f"Predict using {run_id}",
                    thread_id=session_id,
                    status="completed",
                    results_path=data_blob
                )
        except Exception as e:
            logger.warning(f"Failed to record inference task: {e}")

        return {
            "status": "success",
            "session_id": session_id,
            "run_id": run_id,
            "filename": filename,
            "rows_predicted": len(result["predictions"]["predictions"]),
            "predictions": result["predictions"],
            "model_info": result["model_info"],
            "top_features": result.get("top_features"),
            "timestamp": result["timestamp"],
            "chat_message": prediction_summary,
            "message": "Prediction completed and added to chat!"
        }

    except requests.Timeout:
        error_msg = "Prediction timed out after 5 minutes"
        response_handler.save_response(user_email, user_id, agent_info["agent_id"], session_id, "assistant", error_msg)
        raise HTTPException(status_code=504, detail=error_msg)

    except requests.RequestException as e:
        error_msg = "Failed to reach prediction service"
        response_handler.save_response(user_email, user_id, agent_info["agent_id"], session_id, "assistant", error_msg)
        logger.error(f"Inference connection error: {e}")
        raise HTTPException(status_code=502, detail=error_msg)

    except Exception as e:
        error_msg = "Prediction failed"
        response_handler.save_response(user_email, user_id, agent_info["agent_id"], session_id, "assistant", error_msg)
        logger.error(f"Predict endpoint error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal error during prediction")


@app.post("/api/predict/json")
def predict_json(
    user_id: str = Form(..., description="Your user_id from Cosmos DB"),
    run_id: str = Form(..., description="AutoML run ID, e.g., classification-species-20251114_125502"),
    data: str = Form(..., description="JSON string of one or more records"),
    return_proba: bool = Form(False),
    return_explanation: bool = Form(False)
):
    try:
        # Parse the incoming JSON string
        import json
        input_data = json.loads(data)

        payload = {
            "user_id": user_id,
            "run_id": run_id,
            "data": input_data,
            "return_proba": return_proba,
            "return_explanation": return_explanation
        }

        # Trigger your Azure Function
        response = requests.post(
            FUN3_URL,
            json=payload,
            timeout=FUN3_TIMEOUT
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )

        return response.json()

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in 'data' field")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models_registry")
def get_models_registry(
    user_email: str = Query(...),
    task: str = Query(None)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        user_id = user["user_id"]
        
        # Get models from registry
        models = model_registry.get_user_models(user_id, task=task)
        
        # Format response
        models_list = []
        for model in models:
            models_list.append({
                "model_id": model["model_id"],
                "run_id": model["run_id"],
                "task": model["task"],
                "target": model["target"],
                "model_name": model["model_name"],
                "metric": model["metric"],
                "train_metrics": model["train_metrics"],
                "test_metrics": model["test_metrics"],
                "created_at": model["created_at"],
                "usage_count": model.get("usage_count", 0),
                "dataset_id": model.get("dataset_id")
            })
        
        return {
            "total_models": len(models_list),
            "models": models_list
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/best_models")
def get_best_models(user_email: str = Query(...)):
    """Get best performing model for each task"""
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        user_id = user["user_id"]
        best_models = model_registry.get_best_models_per_task(user_id)
        
        return {
            "best_models": [
                {
                    "task": m["task"],
                    "model_name": m["model_name"],
                    "target": m["target"],
                    "metric": m["metric"],
                    "test_metrics": m["test_metrics"],
                    "run_id": m["run_id"],
                    "created_at": m["created_at"]
                }
                for m in best_models
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get best models: {e}")
        raise HTTPException(status_code=500, detail=str(e))
           
# # ============== ONELAKE ENDPOINTS ==============
# @app.get("/workspaces")
# def get_workspaces_endpoint():
#     """ List all workspaces """
#     return {"workspaces": lake_handler.get_workspaces()}

# @app.get("/workspaces/{workspace_name}/lakehouses")
# def get_lakehouses_endpoint(workspace_name: str = Path(..., description="Workspace name ('My workspace')")):
#     """ List lakehouses in workspace """
#     # Backend: Resolve name to ID
#     workspace_id = lake_handler.get_workspace_id_by_name(workspace_name)
#     return {"lakehouses": lake_handler.get_lakehouses(workspace_id)}

# @app.get("/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/contents")
# def get_folder_contents_endpoint(
#     workspace_name: str = Path(..., description="Workspace name ('My workspace')"),
#     lakehouse_name: str = Path(..., description="Lakehouse name ('automonelake')"),
#     path: str = Query("Files", description="Folder path, e.g., Files or Files/bronze")
# ):
#     """ Get folder contents (Files & subfolders) """
#     # Backend: Resolve names to IDs
#     workspace_id = lake_handler.get_workspace_id_by_name(workspace_name)
#     lakehouse_id = lake_handler.get_lakehouse_id_by_name(workspace_name, lakehouse_name)
#     all_items = lake_handler.list_folder_contents(workspace_id, lakehouse_id, path)
#     folders = [{"name": item["name"]} for item in all_items if item["is_directory"]]
#     files = [{"name": item["name"], "size": item["size"]} for item in all_items if not item["is_directory"]]
#     return {
#         # "folders": folders,
#         "files": files
#         # "currentPath": path
#     }

# @app.get("/debug/identity")
# def debug_identity():
#     from azure.identity import DefaultAzureCredential
#     try:
#         credential = DefaultAzureCredential()
#         token = credential.get_token("https://api.fabric.microsoft.com/.default")
#         return {
#             "token_acquired": True,
#             "expires_in": token.expires_on,
#             "hint": "If you see this → Managed Identity works!"
#         }
#     except Exception as e:
#         return {"error": str(e)}


@app.get("/user_models_summary")
def get_user_models_summary(
    user_email: str = Query(...),
    start: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=50)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(401, "Unauthorized")

        user_id = user["user_id"]
        success_models = model_registry.get_user_models(user_id)
        failed_models = model_registry.get_user_models(user_id, include_failed=True)

        seen_runs = set()
        rows = []
        failed_rows = []  

        model_id_for_run = {}
        target_for_run = {}

        # ------------- SUCCESS MODELS  --------------
        for model in sorted(success_models, key=lambda x: x.get("created_at", ""), reverse=True):
            run_id = model["run_id"]

            if run_id not in model_id_for_run:
                model_id_for_run[run_id] = model["model_id"]
                target_for_run[run_id] = model.get("target")

            if run_id in seen_runs:
                continue

            seen_runs.add(run_id)

            dataset_name = model.get("original_filename", "Untitled Dataset")
            task = model["task"].lower()
            train_m = model.get("train_metrics", {})
            test_m = model.get("test_metrics", {})

            if task == "classification":
                train_acc = train_m.get("accuracy") or train_m.get("f1")
                test_acc = test_m.get("accuracy") or test_m.get("f1")
            elif task == "regression":
                train_acc = train_m.get("r2")
                test_acc = test_m.get("r2")
            elif "forecast" in task:
                train_acc = train_m.get("avg_rmse") or train_m.get("rmse") or train_m.get("mape")
                test_acc = test_m.get("avg_rmse") or test_m.get("rmse") or test_m.get("mape")
            elif task == "clustering":
                train_acc = train_m.get("silhouette_score") or train_m.get("silhouette")
                test_acc = train_m.get("silhouette_score") or train_m.get("silhouette")
            elif "anomaly" in task:
                train_acc = train_m.get("anomaly_rate_train") or train_m.get("anomaly_percentage") or train_m.get("avg_anomaly_score")
                test_acc = test_m.get("anomaly_percentage") or train_m.get("anomaly_rate") or test_m.get("avg_anomaly_score") or train_m.get("anomaly_score")
            else:
                train_acc = test_acc = None

            if train_acc is not None and 0 <= train_acc <= 1:
                train_acc *= 100
            if test_acc is not None and 0 <= test_acc <= 1:
                test_acc *= 100

            data_platform = {
                "databricks": "Databricks",
                "snowflake": "Snowflake",
                "onelake": "OneLake",
                "upload": "Upload"
            }.get(model.get("source_type"), "Upload")

            rows.append({
                "dataset_name": dataset_name,
                "veriton_file_path": model.get("veriton_file_path", "N/A"),
                "data_platform": data_platform,
                "task_type": task.replace("anomaly detection", "Anomaly Detection")
                                   .replace("forecasting", "Forecasting")
                                   .title(),
                "best_model": model["model_name"],
                "train_accuracy": round(train_acc, 4) if isinstance(train_acc, (int, float)) else train_acc or "—",
                "test_accuracy": round(test_acc, 4) if isinstance(test_acc, (int, float)) else test_acc or "—",
                "model_id": model_id_for_run[run_id],
                "run_id": run_id,
                "created_at": model.get("created_at").split("T")[0],
                "last_run": model.get("trained_at").split("T")[0],
                "status": "success"  
            })

        # --------------- FAILED MODELS  ---------------------------
        for model in failed_models:
            run_id = model.get("run_id") or model.get("job_id")

            if run_id and run_id in seen_runs:
                continue

            data_platform = {
                "databricks": "Databricks",
                "snowflake": "Snowflake",
                "onelake": "OneLake",
                "upload": "Upload"
            }.get(model.get("source_type"), "N/A")

            failed_rows.append({
                "dataset_name": model.get("original_filename", "Unknown Dataset"),
                "veriton_file_path": model.get("veriton_file_path", "N/A"),
                "data_platform": data_platform,
                "task_type": (model.get("task") or "unknown").replace("_", " ").title(),
                "best_model": "FAILED",
                "train_accuracy": "—",
                "test_accuracy": "—",
                "model_id": model.get("model_id"),
                "run_id": run_id,
                "status": "failed",
                "created_at": model.get("created_at").split("T")[0],
                "last_run": model.get("updated_at").split("T")[0],
                "error_message": model.get("error_message", "Training failed")
            })

        
        paginated_success = rows[start:start + limit]
        paginated_failed = failed_rows[start:start + limit]
        return {
            "total_count": len(rows) + len(failed_rows),
            "success_count": len(rows),
            "failed_count": len(failed_rows),
            "success_models": paginated_success,
            "failed_models": paginated_failed
        }
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(500, "Failed")


@app.get("/model_detailed_metrics/{model_id}")
def get_detailed_metrics(model_id: str, user_email: str):
    try:
        model_id = model_id.strip()  # Remove any accidental whitespace

        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(401, "Unauthorized")
        user_id = user["user_id"]

        logger.info(f"Fetching detailed metrics for model_id={model_id} by user_id={user_id}")

        model_doc = model_registry.get_model_by_id(model_id, user_id)
        if not model_doc:
            logger.warning(f"Model not found or access denied: model_id={model_id}, user_id={user_id}")
            raise HTTPException(404, "Model not found or access denied")

        # Dataset and task info
        original_filename = model_doc.get("original_filename", "unknown.csv")
        dataset_name = original_filename.rsplit(".", 1)[0].replace("_", " ").title()

        task_raw = str(model_doc.get("task", "")).lower()
        task_display = (
            task_raw
            .replace("anomaly detection", "Anomaly Detection")
            .replace("anomaly_detection", "Anomaly Detection")
            .replace("forecasting", "Forecasting")
            .title()
        )

        winning_model_name = str(model_doc.get("model_name", "Unknown"))
        best_model_display = winning_model_name.replace("_", " ").title()

        # Target field (for supervised tasks)
        target_field = model_doc.get("target") or model_doc.get("target_column") or "—"

        # Handle all_models: support both old ("train"/"test") and new ("train_metrics"/"test_metrics") formats
        all_models_raw = model_doc.get("all_models", {})
        comparison = []

        for model_name, data in all_models_raw.items():
            # Prefer new format, fall back to old format
            train_metrics = data.get("train_metrics") or data.get("train") or {}
            test_metrics = data.get("test_metrics") or data.get("test") or {}

            comparison.append({
                "model_name": model_name.replace("_", " ").title(),
                "is_best": model_name == winning_model_name,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics
            })

        # Put the best model first
        comparison.sort(key=lambda x: not x["is_best"])

        return {
            "run_id": model_doc.get("run_id"),
            "model_id": model_id,
            "dataset_name": dataset_name,
            "task_type": task_display,
            "best_model": best_model_display,
            "target": target_field,  #  shows predicted column (e.g., "churn")
            "created_at": model_doc.get("created_at", ""),
            "all_models": comparison
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Detailed metrics failed :: model_id={model_id} user_email={user_email} :: {e}",
            exc_info=True
        )
        raise HTTPException(500, "Server error")

# @app.post("/analyze_dataset_structure")
# async def analyze_dataset_structure(
#     file: UploadFile = File(...),
#     user_email: str = Form(...)
# ):
#     try:
#         user = auth_handler.get_user(user_email)
#         if not user:
#             raise HTTPException(401, "Unauthorized")
        
#         user_id = user["user_id"]
        
#         # Convert to CSV if needed
#         processed_file, processed_filename, was_converted = await to_csv_if_needed(file)
        
#         # Create temp file for analysis
#         temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
#         try:
#             processed_file.file.seek(0)
#             content = processed_file.file.read()
#             temp_file.write(content)
#             temp_file.close()
            
#             # Analyze structure
#             analysis = handler.analyze_dataset_structure(temp_file.name)
            
#             return {
#                 "status": "success",
#                 "filename": file.filename,
#                 "analysis": analysis,
#                 "message": "Dataset structure analyzed successfully"
#             }
        
#         finally:
#             if os.path.exists(temp_file.name):
#                 os.remove(temp_file.name)
    
#     except Exception as e:
#         logger.error(f"Dataset analysis failed: {traceback.format_exc()}")
#         raise HTTPException(500, f"Analysis failed: {str(e)}")
    
@app.post("/build_ml_model")
async def build_ml_model(
    # Required
    file: UploadFile = File(...),
    upload_file_path: bool = Form(False),
    user_email: str = Form(...),
    task: Optional[str] = Form(None),
    target: Optional[str] = Form(None),
    models: Optional[str] = Form(None),
    metric: Optional[str] = Form(None),
    horizon: Optional[int] = Form(12),
    transformation_config: Optional[str] = Form(None),
    # Preprocessing
    preprocessing_mode: Optional[str] = Form("simple"),
    use_cleaning: Optional[bool] = Form(True),
    use_feature_selection: Optional[bool] = Form(False),
    # Hyperparameter Tuning
    use_optuna: Optional[bool] = Form(True),
    optuna_trials: Optional[int] = Form(2),
    # Training Config
    time_budget: Optional[int] = Form(300),
    test_size: Optional[float] = Form(0.2),
    # Test File
    test_file: Optional[UploadFile] = File(None)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
             
        user_id = user["user_id"]
       
        original_filename = file.filename
        processed_file, processed_filename, was_converted = await to_csv_if_needed(file)
        if was_converted:
            original_filename = processed_filename
            file_to_upload = processed_file
        else:
            file_to_upload = file

        train_blob = f"{user_id}/build_model/{original_filename}"
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        train_blob_client = container_client.get_blob_client(train_blob)
        
        await file_to_upload.seek(0)
        content = await file_to_upload.read()
        train_blob_client.upload_blob(content, overwrite=True)

        logger.info(f"File uploaded: {train_blob}")

        dataset_id = None
        analysis_metadata = None

        if upload_file_path:
            # DATASET INTELLIGENCE PROCESSING (matching upload_file_endpoint)
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name

                try:
                    # Run dataset intelligence processing 
                    intelligence_result = dataset_service.process_new_upload(
                        file_path=tmp_path,
                        filename=original_filename,
                        user_id=user_id,
                        user_email=user_email,
                        agent_id=user.get("agents", [{}])[0].get("agent_id"),
                        blob_path=train_blob,
                        min_similarity_threshold=60.0
                    )
                    dataset_id = intelligence_result.get("dataset_id")
                    analysis_metadata = intelligence_result.get("analysis_metadata", {})

                    logger.info(f"Dataset intelligence processed: {dataset_id}")
                    logger.info(f"Analysis metadata: format={analysis_metadata.get('format')}, dims={analysis_metadata.get('n_dimensions')}")
                    
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Dataset intelligence processing failed (non-blocking): {e}")

            feature_suggestions = None
            try:
                feature_suggestions = handler.get_all_task_feature_suggestions(
                    blob_path=train_blob,
                    user_id=user_id,
                    max_preview_rows=200   # or 300, depending on your preference
                )
                logger.info(f"Generated feature suggestions for all tasks - {len(feature_suggestions.get('columns', {}).get('all', []))} columns analyzed")
            except Exception as e:
                logger.warning(f"Feature suggestions generation failed (non-blocking): {str(e)}")
                feature_suggestions = {"status": "failed", "error": str(e)}

            return {
                "status": "file_saved",
                "message": "File uploaded successfully! Ready for model building later.",
                "blob_path": train_blob,
                "filename": original_filename,
                "ready_for_training": True,
                "dataset_id": dataset_id,
                "analysis_metadata": {
                    "dataset_structure": analysis_metadata,
                    "intelligence_processed": dataset_id is not None
                },
              "features" : feature_suggestions
            }
        
        # if not dataset_id:
        #     try:
        #         dataset_id = dataset_service.find_dataset_by_blob_path(train_blob, user_id)
        #         if dataset_id:
        #             logger.info(f"Found existing dataset_id: {dataset_id}")
        #     except Exception as e:
        #         logger.warning(f"Could not retrieve dataset_id: {e}")

        if not task or not task.strip():
            raise HTTPException(status_code=400, detail="Task is required for training")
        if not target or not target.strip():
            raise HTTPException(status_code=400, detail="Target is required for training")
        
        task = task.lower().strip()
        if task not in TASKS:
            raise HTTPException(status_code=400, detail=f"Invalid task. Must be one of: {TASKS}")

        transformation_config_dict = None

        if transformation_config:
            try:
                transformation_config_dict = json.loads(transformation_config)
                logger.info(f"User provided transformation_config: {transformation_config_dict}")

                transformation_config_dict.setdefault("generated_code", None)
                transformation_config_dict.setdefault("code_metadata", None)
                transformation_config_dict.setdefault("needs_transformation", True)

                # Load dataset for analysis
                temp_df = pd.read_csv(io.BytesIO(content))
                
                # Check what user provided
                has_year_column = bool(transformation_config_dict.get('year_column', '').strip())
                has_month_columns = bool(transformation_config_dict.get('month_columns'))
                
                # Auto-detect year column if missing
                if not has_year_column:
                    logger.info("year_column missing - auto-detecting from dataset...")
                    # Try to find year/date columns
                    year_col = None
                    # Check for columns with 'year' in name
                    year_candidates = [col for col in temp_df.columns if 'year' in col.lower()]
                    if year_candidates:
                        year_col = year_candidates[0]
                        logger.info(f"Found year column by name: {year_col}")
                    
                    # Check for datetime columns
                    if not year_col:
                        datetime_cols = temp_df.select_dtypes(include=['datetime64']).columns.tolist()
                        if datetime_cols:
                            year_col = datetime_cols[0]
                            logger.info(f"Found datetime column: {year_col}")
                        else:
                            # Check for columns that can be converted to datetime
                            for col in temp_df.columns:
                                try:
                                    pd.to_datetime(temp_df[col].dropna().head(5))
                                    year_col = col
                                    logger.info(f"Found parseable date column: {year_col}")
                                    break
                                except:
                                    continue
                    
                    if not year_col:
                        raise HTTPException(
                            status_code=400,
                            detail="Cannot find year/date column in dataset. Please add a column with year values or dates."
                        )
                    
                    transformation_config_dict['year_column'] = year_col
                    logger.info(f"Auto-detected year_column: {year_col}")
                
                # Auto-detect month columns if missing
                if not has_month_columns:
                    logger.info("month_columns missing - detecting from measures...")
                    
                    measures = transformation_config_dict.get('measures', [])
                    if not measures:
                        raise HTTPException(400, "measures are required in transformation_config")
                    
                    # The measures are the month columns in wide format
                    transformation_config_dict['month_columns'] = measures
                    logger.info(f"Month columns set to measures: {len(measures)} columns")
                
                # Validate configuration
                ts_transformer = TimeseriesTransformer()
                
                # Temporarily add needs_transformation for validation
                transformation_config_dict['needs_transformation'] = True
                
                is_valid, error_msg = ts_transformer.validate_transformation_config(
                    temp_df, 
                    transformation_config_dict
                )
                
                if not is_valid:
                    raise HTTPException(400, f"Invalid transformation config: {error_msg}")
                
                logger.info(f"Transformation config validated: {len(transformation_config_dict['measures'])} measures, "
                        f"{len(transformation_config_dict.get('group_by', []))} group_by columns, "
                        f"year_column='{transformation_config_dict['year_column']}'")
                
                # Generate transformation code using LLM
                try:                    
                    llm_transformer = get_llm_transformer()
                    
                    # Create temp file for LLM analysis
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_input:
                        tmp_input.write(content)
                        tmp_input_path = tmp_input.name
                    
                    try:
                        # Get dataset analysis
                        analysis = ts_transformer.analyze_dataset_structure(temp_df)
                        
                        # Generate transformation code with auto detected fields
                        generated_code, code_metadata = llm_transformer.generate_transformation_code(
                            file_path=tmp_input_path,
                            analysis=analysis,
                            transformation_config=transformation_config_dict
                        )
                        
                        # Store generated code
                        transformation_config_dict['generated_code'] = generated_code
                        transformation_config_dict['code_metadata'] = code_metadata
                        
                        logger.info(f"LLM generated transformation code ({len(generated_code)} chars)")
                        # logger.info(f"Code preview: {generated_code}")
                        
                    finally:
                        if os.path.exists(tmp_input_path):
                            os.remove(tmp_input_path)
                
                except Exception as llm_error:
                    logger.error(f"LLM transformation code generation failed: {llm_error}")
                    logger.warning("Falling back to rule-based transformation")
                    
                    # Fallback: just mark as needs transformation (use Python script)
                    transformation_config_dict['generated_code'] = ""
                    transformation_config_dict['needs_transformation'] = True
                
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid transformation_config JSON")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Transformation config error: {traceback.format_exc()}")
                raise HTTPException(400, f"Transformation config error: {str(e)}")
            
        if not metric:
            metric = DEFAULT_METRICS[task]
        elif metric not in METRICS[task]:
            raise HTTPException(status_code=400, detail=f"Invalid metric for {task}. Must be one of: {METRICS[task]}")

        if task == 'multistep_forecasting':
            if transformation_config_dict and transformation_config_dict.get('needs_transformation'):
                # For wide-format datasets, use horizon from transformation_config
                config_horizon = transformation_config_dict.get('horizon')
                if config_horizon and isinstance(config_horizon, int):
                    horizon = config_horizon
                    logger.info(f"Using horizon from transformation_config: {horizon}")
                else:
                    logger.warning("transformation_config present but horizon missing/invalid - using direct horizon parameter")

            # Validate horizon
            if horizon is None or horizon < 2 or horizon > 50:
                raise HTTPException(
                    status_code=400,
                    detail="Horizon must be between 2 and 50 steps for multistep forecasting (received: {horizon})"
                )
            
            targets_list = [t.strip() for t in target.split(',')]
            target_param = ", ".join(targets_list)
            
            logger.info(f"Multistep forecasting: {len(targets_list)} target(s), horizon={horizon}")
            
            available_models = ["xgboost", "lightgbm", "catboost"]
            models_to_train = None
            
            if models:
                models_list = [m.strip() for m in models.split(",")]
                invalid_models = [m for m in models_list if m not in available_models]
                valid_models = [m for m in models_list if m in available_models]
                
                if invalid_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid models for multistep forecasting: {invalid_models}. Valid models: {available_models}"
                    )
                
                models_to_train = valid_models if valid_models else None
        
        else:
            target_param = target
            horizon = None
            models_to_train = None
            
            if models:
                models_list = [m.strip() for m in models.split(",")]
                invalid_models = [m for m in models_list if m not in MODELS[task]]
                if invalid_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid models: {invalid_models}. Valid models for {task}: {MODELS[task]}"
                    )
                models_to_train = models_list
        
        if preprocessing_mode not in ["simple", "advanced"]:
            preprocessing_mode = "simple"
        
        if optuna_trials < 2 or optuna_trials > 50:
            optuna_trials = 5

        if test_size <= 0 or test_size >= 1:
            test_size = 0.2
        
        test_blob = None
        if test_file:
            processed_test_file, processed_test_filename, was_converted_test = await to_csv_if_needed(test_file)
            if was_converted_test:
                test_file = processed_test_file
                test_file.filename = processed_test_filename
            
            test_blob = f"{user_id}/build_model/{test_file.filename}"
            test_blob_client = container_client.get_blob_client(test_blob)
           
            await test_file.seek(0)
            test_content = await test_file.read()
            test_blob_client.upload_blob(test_content, overwrite=True)
            logger.info(f"Uploaded test file: {test_blob}")

            if transformation_config_dict and transformation_config_dict.get("needs_transformation"):
                logger.info("Wide-format dataset detected - will transform in function")

                if not transformation_config_dict.get("measures"):
                    raise HTTPException(400, "transformation_config.measures is required")

                if not transformation_config_dict.get("year_column"):
                    raise HTTPException(400, "transformation_config.year_column is required")

                if not transformation_config_dict.get("group_by"):
                    logger.warning("No group_by specified - treating as single series")
                    transformation_config_dict["group_by"] = []

                if task == "multistep_forecasting":
                    target_param = "target"
                    logger.info("Wide-format: target will be 'target' after transformation")

            elif task == 'multistep_forecasting':
                # Normal multistep (already in long format)
                if not target or ',' not in target:
                    targets = [target.strip()] if target else []
                else:
                    targets = [t.strip() for t in target.split(',')]
                
                target_param = target
                logger.info(f"Long-format multistep: targets={target_param}")
            else:
                target_param = target

        payload = {
            "blob_file": train_blob,
            "task": task,
            "target": target_param,
            "metric": metric,
            "models": models_to_train,
            "test_blob": test_blob,
            "test_size": test_size,
            "time_budget": time_budget,
            "horizon": horizon,
            "preprocessing_config": {
                "mode": preprocessing_mode,
                "use_cleaning": use_cleaning
            },
            "optuna_config": {
                "use_optuna": use_optuna,
                "optuna_trials": optuna_trials
            }
        }
        
        if task == "multistep_forecasting":
            payload["horizon"] = horizon
            logger.info(f"Added horizon={horizon} to payload")
        
        if transformation_config:
            payload["transformation_config"] = transformation_config_dict
            logger.info("Added transformation_config to payload")

        logger.info(f"Triggering AutoML with payload: {payload}")

        response = requests.post(
            FUN1_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            # host.json functionTimeout is 00:10:00. A 300s client
            # timeout abandoned runs FUN1 was still executing.
            timeout=FUN1_TIMEOUT
        )
        
        if response.status_code != 200:
            error_detail = response.text[:500]
            logger.error(f"AutoML failed: {response.status_code} - {error_detail}")
            raise HTTPException(
                status_code=502,
                detail=f"Model training failed: {error_detail}"
            )
        
        result = response.json()
        
        run_id = result.get("run_id")
        best_model = result.get("best_model")
        train_metrics = result.get("train_metrics", {})
        test_metrics = result.get("test_metrics", {})
        all_models = result.get("all_models", {})
        data_quality = result.get("data_quality", {})
        clean_all_models = ModelRegistryHandler.normalize_all_models(all_models, task=task)

        if data_quality.get("warnings"):
            logger.info(f"[DATA QUALITY] {data_quality['warnings']}")

        if task == "multistep_forecasting":
            primary_metric_key = f"avg_{metric}" if not metric.startswith("avg_") else metric
            primary_score = test_metrics.get(primary_metric_key) or train_metrics.get(primary_metric_key)
        else:
            primary_score = test_metrics.get(metric) or train_metrics.get(metric)
        
        if primary_score is None:
            primary_score = "N/A"
        
        try:
            model_id = model_registry.register_model(
                user_id=user_id,
                user_email=user_email,
                run_id=run_id,
                task=task,
                target=target_param,
                model_name=best_model,
                metric=metric,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                all_models=clean_all_models,
                dataset_id=dataset_id,
                blob_path=result.get("run_path"),
                original_filename=original_filename,
                original_blob_path=train_blob,
                horizon=horizon if task == "multistep_forecasting" else None,
                targets_list=target_param.split(", ") if task == "multistep_forecasting" else None
            )

            # Record task completion in dataset history
            if dataset_id:
                try:
                    dataset_service.record_task_completion(
                        dataset_id=dataset_id,
                        user_id=user_id,
                        task_type=task,
                        query=f"Built {task} model using {original_filename}",
                        thread_id=None,
                        status="completed",
                        model_id=model_id,
                        model_name=best_model,
                        results_path=result.get("run_path")
                    )
                    logger.info(f"Recorded model build task in dataset history")
                except Exception as e:
                    logger.warning(f"Failed to record task in dataset history: {e}")

            logger.info(f"Registered model: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
            model_id = None
        
        top_features = (
            result.get("feature_importance", {}).get("top_features") or
            result.get("top_features") or
            []
        )
        if isinstance(top_features, dict):
            top_features = list(top_features.keys())[:5]

        text_summary = handler.generate_build_model_text_summary(
            user_email=user_email,  
            task=task,
            best_model=best_model,
            primary_metric=metric,
            primary_score=primary_score,
            all_models=clean_all_models,
            dataset_name=original_filename,
            top_features=top_features,
            result_details=result
        )
        response_payload = {
            "status": "success",
            "message": "Model built successfully!",
            "dataset": original_filename,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "task_type": task.replace("_", " ").title(),
            "best_model": best_model,
            "primary_metric": metric,
            "primary_score": primary_score,
            "all_models": clean_all_models,
            "text_summary": text_summary
        }

        if task == "multistep_forecasting":
            response_payload["horizon"] = horizon
            response_payload["targets"] = target_param.split(", ")
            logger.info(f"Response includes horizon={horizon}, targets={response_payload['targets']}")

        return response_payload
    
    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Model training timed out after 5 minutes"
        )
    except Exception as e:
        logger.error(f"Build model failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build model: {str(e)}"
        )


@app.post("/test_model")
async def test_model(
    # Required
    test_file: UploadFile = File(...),
    model_id: str = Form(...),
    user_email: str = Form(...),
    
    # Optional
    return_predictions: bool = Form(True),
    return_probabilities: bool = Form(False),
    save_predictions: bool = Form(True)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
     
        user_id = user["user_id"]
        
        # Get model info from registry
        model_info = model_registry.get_model_by_id(model_id, user_id)
          
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )
       
        run_id = model_info["run_id"]
        task = model_info["task"]
        target = model_info["target"]
        model_name = model_info["model_name"]
        training_test_metrics = model_info.get("test_metrics", {})

        task_model_name = f"{task}_{model_name}"
        logger.info(f"Testing model: {model_name} (run_id: {run_id})")
        
        # ───── Convert test file if needed (Parquet → CSV) ─────
        processed_test_file, processed_test_filename, was_converted = await to_csv_if_needed(test_file)
        if was_converted:
            test_file = processed_test_file
            test_file.filename = processed_test_filename

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # test_blob = f"{user_id}/test_model/{timestamp}/{test_file.filename}"
        test_folder_name = f"{task}_{model_name}_test"
        test_blob = f"{user_id}/test_model/{test_folder_name}/{test_file.filename}"
        
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        test_blob_client = container_client.get_blob_client(test_blob)

        await test_file.seek(0)
        test_content = await test_file.read()
        test_blob_client.upload_blob(test_content, overwrite=True)

        logger.info(f"Uploaded test file: {test_blob}")
        
        # CHECK IF TEST FILE HAS TARGET COLUMN
        test_file_df = None
        has_ground_truth = False
        
        try:
            test_blob_client_read = container_client.get_blob_client(test_blob)
            data = test_blob_client_read.download_blob().readall()
           
            ext = os.path.splitext(test_blob)[1].lower()
            if ext == '.csv':
                test_file_df = pd.read_csv(io.BytesIO(data))
            elif ext in {'.xlsx', '.xls'}:
                test_file_df = pd.read_excel(io.BytesIO(data))
           
            if test_file_df is not None:
                if task not in ['clustering', 'anomaly_detection']:
                    has_ground_truth = target in test_file_df.columns
                    logger.info(f"Ground truth {'FOUND' if has_ground_truth else 'NOT FOUND'} in test file")
                else:
                    has_ground_truth = False
                    logger.info(f"Unsupervised task ({task}) - no ground truth check")
        
        except Exception as e:
            logger.warning(f"Could not read test file for ground truth check: {e}")
            has_ground_truth = False
        
        # === HANDLE MULTISTEP FORECASTING ===
        if task == "multistep_forecasting":
            logger.info("Multistep forecasting model - checking for history buffer")
            
            # Try to load history buffer
            history_buffer_path = f"{user_id}/runs/{run_id}/history_buffer.csv"
            history_blob_client = container_client.get_blob_client(history_buffer_path)
            
            has_history_buffer = False
            if history_blob_client.exists():
                logger.info(f"Found history buffer: {history_buffer_path}")
                has_history_buffer = True
            else:
                logger.warning(f"No history buffer found at {history_buffer_path}")
        
        test_result_id = None
        
        # SCENARIO 1: HAS GROUND TRUTH
        if has_ground_truth:
            logger.info("SCENARIO 1: Ground truth available - Computing fresh test metrics")

            # Separate features and target
            X_test = test_file_df.drop(columns=[target])
            y_test = test_file_df[target]
            
            # HANDLE FORECASTING AND MULTI-HORIZON SEPARATELY
            if task == 'multistep_forecasting':
                logger.info("Multistep forecasting - checking for history buffer")
                
                # Try to load history buffer
                history_buffer_path = f"{user_id}/runs/{run_id}/history_buffer.csv"
                history_blob_client = container_client.get_blob_client(history_buffer_path)
                
                if history_blob_client.exists():
                    logger.info(f"Found history buffer: {history_buffer_path}")
                    
                    # Download history buffer
                    history_data = history_blob_client.download_blob().readall()
                    history_df = pd.read_csv(io.BytesIO(history_data))
                    
                    logger.info(f"History buffer: {len(history_df)} rows")
                    
                    # Combine history + test data for feature engineering
                    df_combined = pd.concat([history_df, test_file_df], ignore_index=True)
                    logger.info(f"Combined: {len(history_df)} history + {len(test_file_df)} test = {len(df_combined)} total")
                    
                    # Use combined data for inference (function will handle feature engineering)
                    inference_blob = test_blob  # Original test file
                else:
                    logger.warning("No history buffer found - using test file directly")
                    inference_blob = test_blob

            elif task in ['forecasting']:
                inference_blob = test_blob
                logger.info(f"Forecasting task - using file: {inference_blob}")

            else:
                # For non-forecasting: create features-only file
                features_only_blob = f"{user_id}/test_model/{test_folder_name}/features_only_{test_file.filename}"
                features_blob_client = container_client.get_blob_client(features_only_blob)
                
                # Save as CSV
                features_buffer = io.BytesIO()
                X_test.to_csv(features_buffer, index=False)
                features_buffer.seek(0)
                features_blob_client.upload_blob(features_buffer, overwrite=True)
            
                logger.info(f"Saved features-only file: {features_only_blob}")
                inference_blob = features_only_blob
            
            payload = {
                "run_id": run_id,
                "data_blob": inference_blob,
                "return_proba": return_probabilities
            }
            logger.info(f"Triggering inference with payload: {payload}")
            
            response = requests.post(
                FUN2_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=FUN2_TIMEOUT
            )
            
            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"Inference failed: {response.status_code} - {error_detail}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Model testing failed: {error_detail}"
                )
            
            result = response.json()
            predictions = result["predictions"]["predictions"]

            # COMPUTE FRESH TEST METRICS
            fresh_test_metrics = {}
            try:
                if task == 'classification':
                    fresh_test_metrics['accuracy'] = float(accuracy_score(y_test, predictions))
                    fresh_test_metrics['f1'] = float(f1_score(y_test, predictions, average='weighted', zero_division=0))
                    fresh_test_metrics['precision'] = float(precision_score(y_test, predictions, average='weighted', zero_division=0))
                    fresh_test_metrics['recall'] = float(recall_score(y_test, predictions, average='weighted', zero_division=0))

                    if return_probabilities and "probabilities" in result:
                        try:
                            proba = np.array(result["probabilities"])
                            if proba.shape[1] == 2:
                                fresh_test_metrics['roc_auc'] = float(roc_auc_score(y_test, proba[:, 1]))
                            else:
                                fresh_test_metrics['roc_auc'] = float(roc_auc_score(y_test, proba, multi_class='ovr', average='weighted'))
                        except Exception as e:
                            logger.warning(f"Could not compute ROC-AUC: {e}")
               
                elif task == 'regression':
                    fresh_test_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
                    fresh_test_metrics['mae'] = float(mean_absolute_error(y_test, predictions))
                    fresh_test_metrics['r2'] = float(r2_score(y_test, predictions))
                    fresh_test_metrics['mse'] = float(mean_squared_error(y_test, predictions))
               
                elif task == 'forecasting':
                    fresh_test_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
                    fresh_test_metrics['mae'] = float(mean_absolute_error(y_test, predictions))
                    fresh_test_metrics['r2'] = float(r2_score(y_test, predictions))

                elif task == 'multistep_forecasting':
                    # For multistep: predictions are 2D array
                    predictions_array = np.array(predictions)
                    y_test_array = y_test.values if isinstance(y_test, pd.DataFrame) else y_test
                    
                    if len(predictions_array.shape) == 1:
                        predictions_array = predictions_array.reshape(-1, 1)
                    
                    if len(y_test_array.shape) == 1:
                        y_test_array = y_test_array.reshape(-1, 1)
                    
                    # Compute average metrics across horizons
                    n_horizons = predictions_array.shape[1]
                    rmse_values = []
                    mae_values = []
                    r2_values = []
                    
                    for h in range(n_horizons):
                        y_h = y_test_array[:, h]
                        p_h = predictions_array[:, h]
                        
                        rmse_values.append(np.sqrt(mean_squared_error(y_h, p_h)))
                        mae_values.append(mean_absolute_error(y_h, p_h))
                        r2_values.append(r2_score(y_h, p_h))
                    
                    fresh_test_metrics['avg_rmse'] = float(np.mean(rmse_values))
                    fresh_test_metrics['avg_mae'] = float(np.mean(mae_values))
                    fresh_test_metrics['avg_r2'] = float(np.mean(r2_values))
                    
                    logger.info(f"Multistep metrics: RMSE={fresh_test_metrics['avg_rmse']:.4f}, "
                               f"MAE={fresh_test_metrics['avg_mae']:.4f}, R²={fresh_test_metrics['avg_r2']:.4f}")
               
                elif task == 'clustering':
                    if len(np.unique(predictions)) > 1:
                        fresh_test_metrics['silhouette_score'] = float(silhouette_score(X_test, predictions))
                        fresh_test_metrics['davies_bouldin_score'] = float(davies_bouldin_score(X_test, predictions))
                        fresh_test_metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_test, predictions))
                    fresh_test_metrics['n_clusters'] = int(len(np.unique(predictions)))
               
                elif task == 'anomaly_detection':
                    pred_binary = (np.array(predictions) == -1).astype(int)
                    true_binary = (y_test == -1).astype(int) if y_test.dtype != bool else y_test.astype(int)
                    fresh_test_metrics['accuracy'] = float(accuracy_score(true_binary, pred_binary))
                    fresh_test_metrics['precision'] = float(precision_score(true_binary, pred_binary, zero_division=0))
                    fresh_test_metrics['recall'] = float(recall_score(true_binary, pred_binary, zero_division=0))
                    fresh_test_metrics['f1'] = float(f1_score(true_binary, pred_binary, zero_division=0))
                    fresh_test_metrics['n_anomalies'] = int(np.sum(pred_binary))
               
                logger.info(f"Computed fresh test metrics: {fresh_test_metrics}")
           
            except Exception as e:
                logger.error(f"Failed to compute metrics: {e}")
                fresh_test_metrics = {"error": "Could not compute metrics", "details": str(e)}
           
            # SAVE PREDICTIONS WITH GROUND TRUTH
            predictions_file_path = None
            predictions_filename = None
            if save_predictions:
                try:
                    predictions_df = test_file_df.copy()
                    predictions_df[f'{target}_predicted'] = predictions
                   
                    if return_probabilities and "probabilities" in result and task == 'classification':
                        proba = np.array(result["probabilities"])
                        unique_classes = sorted(predictions_df[target].unique())
                        for idx, cls in enumerate(unique_classes):
                            if idx < proba.shape[1]:
                                predictions_df[f'{target}_prob_class_{cls}'] = proba[:, idx]
                   
                    original_filename = os.path.splitext(test_file.filename)[0]
                    predictions_filename = f"{original_filename}_with_predictions_{timestamp}.csv"
                    predictions_blob_path = f"{user_id}/predictions/{task_model_name}/{predictions_filename}"
                    # predictions_blob_path = f"{user_id}/predictions/{model_name}/{predictions_filename}"
                   
                    predictions_blob_client = container_client.get_blob_client(predictions_blob_path)
                    csv_buffer = io.BytesIO()
                    predictions_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    predictions_blob_client.upload_blob(csv_buffer, overwrite=True)
                    predictions_file_path = predictions_blob_path
                    logger.info(f"Saved predictions to: {predictions_blob_path}")
               
                except Exception as e:
                    logger.error(f"Failed to save predictions: {e}", exc_info=True)
                    predictions_file_path = None
                    predictions_filename = None
                    
            # Save test result to DB
            try:
                test_result_id = test_metrics_handler.save_test_result(
                    model_id=model_id,
                    user_id=user_id,
                    user_email=user_email,
                    test_file_name=test_file.filename,
                    test_blob_path=test_blob,
                    has_ground_truth=True,
                    test_metrics=fresh_test_metrics,
                    predictions_blob_path=predictions_file_path,
                    rows_tested=len(predictions),
                    notes=f"Test with ground truth - {task} on {target}",
                    model_name=model_name,
                    task=task,
                    target=target,
                    run_id=run_id
                )
                logger.info(f"Saved test result: {test_result_id}")
            except Exception as e:
                logger.warning(f"Failed to save test result: {e}")
            
            try:
                model_registry.increment_usage(model_id, user_id)
            except Exception as e:
                logger.warning(f"Failed to increment usage: {e}")
            
            response_data = {
                "status": "success",
                "test_result_id": test_result_id,
                "message": "Model tested successfully with ground truth!",
                "scenario": "ground_truth_available",
                "model_info": {
                    "model_id": model_id,
                    "model_name": model_name,
                    "task": task,
                    "target": target,
                    "run_id": run_id
                },
                "test_file": test_file.filename,
                "rows_tested": len(predictions),
                "has_ground_truth": True,
                "test_metrics": fresh_test_metrics,
                "training_test_metrics": training_test_metrics,
                # "metrics_comparison": {
                #     "note": "Fresh metrics computed from this test file vs. metrics from training",
                #     "fresh": fresh_test_metrics,
                #     "training": training_test_metrics
                # },
                "timestamp": result["timestamp"]
            }
            
            if predictions_file_path:
                response_data["predictions_file"] = {
                    "saved": True,
                    "blob_path": predictions_file_path,
                    "filename": predictions_filename
                }
            else:
                response_data["predictions_file"] = {"saved": False}
           
            if return_predictions:
                response_data["predictions"] = {"predicted": predictions, "actual": y_test.tolist()}
           
            if return_probabilities and "probabilities" in result:
                response_data["probabilities"] = result["probabilities"]
           
            if "top_features" in result:
                response_data["top_features"] = result["top_features"]

        # NO GROUND TRUTH
        else:
            logger.info("SCENARIO 2: No ground truth - Using training test metrics")
           
            payload = {
                "run_id": run_id,
                "data_blob": test_blob,
                "return_proba": return_probabilities
            }
           
            response = requests.post(
                FUN2_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=FUN2_TIMEOUT
            )
           
            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"Inference failed: {response.status_code} - {error_detail}")
                raise HTTPException(status_code=502, detail=f"Model testing failed: {error_detail}")
           
            result = response.json()
            predictions = result["predictions"]["predictions"]
           
            predictions_file_path = None
            predictions_filename = None
            
            if save_predictions:
                try:
                    predictions_df = test_file_df.copy() if test_file_df is not None else None
                    if predictions_df is not None:
                        if task == 'clustering':
                            predictions_df['cluster_assigned'] = predictions
                        elif task == 'anomaly_detection':
                            predictions_df['is_anomaly'] = (np.array(predictions) == -1).astype(int)
                            predictions_df['anomaly_label'] = predictions
                        else:
                            predictions_df[f'{target}_predicted'] = predictions
                       
                        if return_probabilities and "probabilities" in result and task == 'classification':
                            proba = np.array(result["probabilities"])
                            n_classes = proba.shape[1]
                            for idx in range(n_classes):
                                predictions_df[f'{target}_prob_class_{idx}'] = proba[:, idx]
                       
                        original_filename = os.path.splitext(test_file.filename)[0]
                        predictions_filename = f"{original_filename}_with_predictions_{timestamp}.csv"
                        predictions_blob_path = f"{user_id}/predictions/{task_model_name}/{predictions_filename}"
                        # predictions_blob_path = f"{user_id}/predictions/{model_name}/{predictions_filename}"
                       
                        predictions_blob_client = container_client.get_blob_client(predictions_blob_path)
                        csv_buffer = io.BytesIO()
                        predictions_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        predictions_blob_client.upload_blob(csv_buffer, overwrite=True)
                        predictions_file_path = predictions_blob_path
                        logger.info(f"Saved predictions to: {predictions_blob_path}")
                
                except Exception as e:
                    logger.error(f"Failed to save predictions: {e}")
                    # predictions_file_path = None
            
            try:
                model_registry.increment_usage(model_id, user_id)
            except Exception as e:
                logger.warning(f"Failed to increment usage: {e}")
           
            try:
                test_result_id = test_metrics_handler.save_test_result(
                    model_id=model_id,
                    user_id=user_id,
                    user_email=user_email,
                    test_file_name=test_file.filename,
                    test_blob_path=test_blob,
                    has_ground_truth=False,
                    test_metrics=training_test_metrics,
                    predictions_blob_path=predictions_file_path,
                    rows_tested=len(predictions),
                    notes="Inference only - no ground truth available",
                    model_name=model_name,
                    task=task,
                    target=target,
                    run_id=run_id
                )
                logger.info(f"Saved test result (no ground truth): {test_result_id}")
            except Exception as e:
                logger.error(f"Failed to save test result: {e}")
                test_result_id = None


            response_data = {
                "status": "success",
                "test_result_id": test_result_id,
                "message": "Model predictions generated successfully!",
                "scenario": "no_ground_truth",
                "model_info": {
                    "model_id": model_id,
                    "model_name": model_name,
                    "task": task,
                    "target": target,
                    "run_id": run_id
                },
                "test_file": test_file.filename,
                "rows_tested": len(predictions),
                "has_ground_truth": False,
                "test_metrics": training_test_metrics,
                "metrics_note": "These are test metrics from the training phase (no ground truth in current test file)",
                "timestamp": result["timestamp"]
            }
            
            if predictions_file_path:
                response_data["predictions_file"] = {
                    "saved": True,
                    "blob_path": predictions_file_path,
                    "filename": predictions_filename
                }
            else:
                response_data["predictions_file"] = {"saved": False}
           
            if return_predictions:
                response_data["predictions"] = result["predictions"]
           
            if return_probabilities and "probabilities" in result:
                response_data["probabilities"] = result["probabilities"]
           
            if "top_features" in result:
                response_data["top_features"] = result["top_features"]
        
        return response_data
   
    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Model testing timed out after 3 minutes")
    except Exception as e:
        logger.error(f"Test model failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to test model: {str(e)}")
    
@app.get("/test_result/{test_result_id}")
def get_test_result(
    test_result_id: str,
    user_email: str = Query(...)
):
    """Get a specific test result by its ID"""
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_id = user["user_id"]
        test_result = test_metrics_handler.get_test_result(test_result_id, user_email)
        
        if not test_result:
            raise HTTPException(status_code=404, detail="Test result not found")
        
        if test_result.get("user_id") != user_id or test_result.get("user_email") != user_email:
            raise HTTPException(status_code=403, detail="Forbidden: You do not have access to this test result")

        return {
            "status": "success",
            "test_result": {
                "test_result_id": test_result["test_result_id"],
                "model_id": test_result["model_id"],
                "model_name": model_registry.get_model_by_id(test_result["model_id"], user_id).get("model_name", "Unknown"),
                "task": test_result.get("task"),  # You might want to store task in test result or fetch from model
                "target": test_result.get("target"),
                "run_id": test_result.get("run_id"),
                "test_file_name": test_result["test_file_name"],
                "tested_at": test_result["tested_at"],
                "has_ground_truth": test_result["has_ground_truth"],
                "test_metrics": test_result["test_metrics"],
                "rows_tested": test_result["rows_tested"],
                "predictions_file": test_result.get("predictions_blob_path"),
                "test_blob_path": test_result.get("test_blob_path"),
                "notes": test_result.get("notes")
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test result {test_result_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.get("/download_predictions")
def download_predictions(
    blob_path: str = Query(...),
    user_email: str = Query(...)
):
    try:        
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        user_id = user["user_id"]
        
        if not blob_path.startswith(f"{user_id}/"):
            raise HTTPException(
                status_code=403,
                detail="Access denied: You can only download your own files"
            )
        
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        blob_client = container_client.get_blob_client(blob_path)
        
        if not blob_client.exists():
            raise HTTPException(
                status_code=404,
                detail="Predictions file not found"
            )
        
        blob_data = blob_client.download_blob().readall()
        filename = blob_path.split("/")[-1]
        
        logger.info(f"Downloading predictions file: {filename}")
        
        return StreamingResponse(
            io.BytesIO(blob_data),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file: {str(e)}"
        )
    
@app.get("/model_test_history/{model_id}")
async def get_model_test_history(
    model_id: str,
    user_email: str = Query(...)
):
    """Get all test results for a specific model"""
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        user_id = user["user_id"]
        
        # Verify model belongs to user
        model = model_registry.get_model_by_id(model_id, user_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        test_history = test_metrics_handler.get_model_test_history(model_id, user_id)
        
        return {
            "model_id": model_id,
            "model_name": model["model_name"],
            "task": model["task"],
            "total_tests": len(test_history),
            "test_history": [
                {
                    "test_result_id": t["test_result_id"],
                    "test_file_name": t["test_file_name"],
                    # "tested_at": t["tested_at"],
                    "has_ground_truth": t["has_ground_truth"],
                    "test_metrics": t["test_metrics"],
                    # "rows_tested": t["rows_tested"],
                    "predictions_file": t.get("predictions_blob_path")
                }
                for t in test_history
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/drift/report")
async def get_drift_report(
    mode: str = Form(..., description="'build' or 'test'"),
    user_email: str = Form(..., description="User email"),
    model_id: Optional[str] = Form(None, description="Required for mode='build'"),
    test_result_id: Optional[str] = Form(None, description="Required for mode='test'"),
):
    """Generate a data drift report based on training or test"""
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_id = user["user_id"]
        
        if mode not in ["build", "test"]:
            raise HTTPException(status_code=400, detail="Invalid mode")
        if mode == "build" and not model_id:
            raise HTTPException(status_code=400, detail="model_id required for build")
        if mode == "test" and not test_result_id:
            raise HTTPException(status_code=400, detail="test_result_id required for test")

        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)

        # Get model and training blob
        if mode == "build":
            model = model_registry.get_model_by_id(model_id, user_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model not found or access denied: {model_id}")
            training_blob = model.get("original_blob_path")
        else:
            test_result = test_metrics_handler.get_test_result(test_result_id, user_email)
            if not test_result or test_result.get("user_id") != user_id:
                raise HTTPException(status_code=404, detail="Test result not found")
            model = model_registry.get_model_by_id(test_result["model_id"], user_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model not found or access denied: {test_result['model_id']}")
            training_blob = model.get("original_blob_path")

        if not training_blob:
            raise HTTPException(status_code=400, detail="Training blob path not found in model registry")

        task = model.get("task")
        target = model.get("target") or "None"
        if task in ["clustering", "anomaly_detection"]:
            target = "None"

        # Compute fingerprint from training data (used to identify the reference set)
        blob_client = container_client.get_blob_client(training_blob)
        if not blob_client.exists():
            raise HTTPException(status_code=404, detail="Training data not found")
        data = blob_client.download_blob().readall()
        try:
            df_train = pd.read_csv(io.BytesIO(data))
        except:
            df_train = pd.read_excel(io.BytesIO(data))
        df_sorted = df_train[sorted(df_train.columns)]
        csv_bytes = df_sorted.to_csv(index=False, encoding='utf-8').encode('utf-8')
        fingerprint = hashlib.sha256(csv_bytes).hexdigest()
        dataset_key = f"{fingerprint}_{task}_{target}"

        prefix = f"{user_id}/drift_references/{dataset_key}/"
        logger.info(f"Dataset key: {dataset_key}")
        logger.info(f"Searching blob prefix: {prefix}")

        # List blobs
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        version_blobs = [b for b in blobs if b.name.endswith("_drift_reference.json") and "v" in b.name]

        # Bootstrap fallback if no exact match
        if not version_blobs:
            broad_prefix = f"{user_id}/drift_references/"
            all_refs = list(container_client.list_blobs(name_starts_with=broad_prefix))
            if not all_refs:
                logger.info("No drift references exist yet for this user")
                return {"drift_report": {"status": "no_versions", "message": "No drift references found yet"}}
            all_refs.sort(key=lambda b: b.last_modified or datetime.min, reverse=True)
            latest_blob = all_refs[0]
            path_parts = latest_blob.name.split("/")
            if len(path_parts) >= 4:
                dataset_key = path_parts[-2] if "drift_reference.json" in path_parts[-1] else None
            if not dataset_key:
                raise ValueError("Could not extract dataset_key from reference path")
            prefix = f"{user_id}/drift_references/{dataset_key}/"
            logger.info(f"Bootstrapped from: {latest_blob.name}")
            blobs = list(container_client.list_blobs(name_starts_with=prefix))
            version_blobs = [b for b in blobs if b.name.endswith("_drift_reference.json") and "v" in b.name]
            if not version_blobs:
                version_blobs = [b for b in blobs if b.name.endswith("latest_drift_reference.json")]
                if not version_blobs:
                    logger.info("No references (versioned or latest) found")
                    return {"drift_report": {"status": "no_versions", "message": "No valid references found"}}

        # Helper to get version number
        def get_version_num(blob):
            filename = blob.name.split("/")[-1]
            if "latest" in filename:
                return 999999
            try:
                return int(filename[1:filename.index("_")])
            except:
                return 0

        # Sort descending → newest first
        version_blobs.sort(key=get_version_num, reverse=True)

        if not version_blobs:
            return {"drift_report": {"status": "no_versions", "message": "No valid references found"}}

        #  the LATEST reference as baseline
        latest_blob = version_blobs[0]
        ref_client = container_client.get_blob_client(latest_blob.name)
        ref_json = json.loads(ref_client.download_blob().readall())
        ref_stats = ref_json.get("data_stats", {})
        baseline_metric_val = ref_json.get("test_metrics", {}).get(model.get("metric"))

        latest_version_num = get_version_num(latest_blob)
        latest_version_str = "latest" if "latest" in latest_blob.name else f"v{latest_version_num}"
        total_versions = len(version_blobs)

        # build mode with only 1 version → keep original "activated" message
        if mode == "build" and total_versions == 1:
            return {"drift_report": to_python_types({
                "overall_status": "activated",
                "summary_message": "Drift monitoring activated!",
                "details": "This is the initial baseline version (v1). There is no previous version to compare.",
                "recommendation": "Future runs will enable drift detection",
                "checked_at": datetime.now().isoformat(),
                "version_compared": "v1 (baseline)",
                "total_versions": 1
            })}

        # For test mode (or build with ≥2 versions) → compare against latest
        comparison_note = f"Compared against latest build reference ({latest_version_str})"
        if mode == "test":
            comparison_note += " (test data vs most recent trained model)"
        if total_versions > 1:
            comparison_note += f" (out of {total_versions} versions)"
        else:
            comparison_note += " (initial baseline – monitoring active)"

        version_compared = latest_version_str

        #   Current metric (only meaningful in test mode with ground truth)
        current_metric_val = None
        if mode == "test" and test_result.get("has_ground_truth"):
            current_metric_val = test_result["test_metrics"].get(model.get("metric"))

        #   Load current data (build = training, test = test file)
        current_blob = training_blob if mode == "build" else test_result["test_blob_path"]
        blob_client = container_client.get_blob_client(current_blob)
        if not blob_client.exists():
            raise HTTPException(status_code=404, detail="Current data not found")
        data = blob_client.download_blob().readall()
        try:
            df_current = pd.read_csv(io.BytesIO(data))
        except:
            df_current = pd.read_excel(io.BytesIO(data))
        if df_current.empty:
            raise HTTPException(status_code=400, detail="Data is empty")

        #   Data Drift Detection (PSI)
        drifted_features = []
        psi_values = []
        for col, stats in ref_stats.items():
            if col not in df_current.columns:
                continue
            vals = pd.to_numeric(df_current[col], errors='coerce').dropna()
            if stats["type"] == "categorical":
                all_cats = set(stats["distribution"].keys()) | set(df_current[col].dropna().unique())
                all_cats = list(all_cats)
                ref_hist = np.array([stats["distribution"].get(c, 0) for c in all_cats])
                curr_hist = np.array([df_current[col].value_counts(normalize=True).get(c, 0) for c in all_cats])
            else:
                ref_hist = np.array(stats["hist"])
                edges = np.array(stats["bin_edges"])
                curr_hist, _ = np.histogram(vals, bins=edges, density=True)
            eps = 1e-8
            ref_hist = ref_hist / (ref_hist.sum() + eps)
            curr_hist = curr_hist / (curr_hist.sum() + eps)
            psi = np.sum((ref_hist - curr_hist) * np.log((ref_hist + eps) / (curr_hist + eps)))
            psi = np.clip(psi, 0, 10)
            psi_values.append(psi)
            if psi > 0.1:
                drifted_features.append(col)

        overall_psi = round(np.percentile(psi_values, 90) if psi_values else 0, 3)
        data_drift_detected = len(drifted_features) > 0 or overall_psi > 0.05

        #   Performance Drift
        perf_change = 0.0
        perf_degradation = False
        if current_metric_val is not None and baseline_metric_val is not None and baseline_metric_val > 0:
            perf_change = round((current_metric_val - baseline_metric_val) / baseline_metric_val * 100, 1)
            perf_degradation = perf_change < -8

      
        # ──────────────── Overall Status for both build and test ────────────────
        if data_drift_detected and perf_degradation:
            status = "critical" if mode == "build" else "warning"
            summary = (
                "Critical: Both data and performance drift detected"
                if mode == "build"
                else "Data & performance drift detected in test"
            )
            rec = "Retrain immediately with recent data" if mode == "build" else "Review test dataset before retraining"
        elif perf_degradation:
            status = "degraded"
            summary = "Performance degradation detected"
            rec = "Investigate recent predictions"
        elif data_drift_detected:
            status = "data_drift"
            summary = "Data drift detected"
            rec = "Monitor closely — performance may degrade soon"
        else:
            status = "stable"
            summary = "All stable"
            rec = "Model performing as expected"

        # ──────────────── Drift report  ────────────────
        drift_report = {
            "data_drift": {
                "detected": data_drift_detected,
                "overall_psi": overall_psi,
                "drifted_features_count": len(drifted_features),
                "drifted_features": drifted_features[:6],
                # "all_drifted_features": drifted_features,
                "status": "drift" if data_drift_detected else "stable"
            },
            "performance_drift": {
                "detected": perf_degradation,
                "change_percent": perf_change or None,
                "baseline_metric": baseline_metric_val or None,
                "current_metric": current_metric_val or None,
                "status": "degraded" if perf_degradation else "improved" if perf_change and perf_change > 5 else "stable"
            },
            "overall_status": status,
            "summary_message": summary,
            "details": f"{comparison_note}\nOverall PSI (90th percentile): {overall_psi} | Drifted features: {len(drifted_features)}",
            "recommendation": rec,
            "checked_at": datetime.now().isoformat(),
            "version_compared": version_compared,
            "total_versions": total_versions
        }
        return {"drift_report": to_python_types(drift_report)}

    except HTTPException:
        raise


@app.get("/drift/report")
async def get_drift_report(
    mode: str = Query(..., description="'build' or 'test'"),
    user_email: str = Query(..., description="User email"),
    model_id: Optional[str] = Query(None, description="Required when mode='build'"),
    test_result_id: Optional[str] = Query(None, description="Required when mode='test'"),
):

    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_id = user["user_id"]

        if mode not in ["build", "test"]:
            raise HTTPException(status_code=400, detail="Invalid mode: must be 'build' or 'test'")

        if mode == "build" and not model_id:
            raise HTTPException(status_code=400, detail="model_id is required when mode=build")
        if mode == "test" and not test_result_id:
            raise HTTPException(status_code=400, detail="test_result_id is required when mode=test")

        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)

        # ─── Get model and reference training blob ───────────────────────────────
        if mode == "build":
            model = model_registry.get_model_by_id(model_id, user_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model not found or access denied: {model_id}")
            training_blob = model.get("original_blob_path")
        else:  # mode == "test"
            test_result = test_metrics_handler.get_test_result(test_result_id, user_email)
            if not test_result or test_result.get("user_id") != user_id:
                raise HTTPException(status_code=404, detail="Test result not found or access denied")
            model = model_registry.get_model_by_id(test_result["model_id"], user_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model not found or access denied: {test_result['model_id']}")
            training_blob = model.get("original_blob_path")

        if not training_blob:
            raise HTTPException(status_code=400, detail="Training blob path not found in model registry")

        task = model.get("task")
        target = model.get("target") or "None"
        if task in ["clustering", "anomaly_detection"]:
            target = "None"

        # ─── Fingerprint of training data (reference) ────────────────────────────
        blob_client = container_client.get_blob_client(training_blob)
        if not blob_client.exists():
            raise HTTPException(status_code=404, detail="Training data blob not found")

        data = blob_client.download_blob().readall()
        try:
            df_train = pd.read_csv(io.BytesIO(data))
        except Exception:
            df_train = pd.read_excel(io.BytesIO(data))

        df_sorted = df_train[sorted(df_train.columns)]
        csv_bytes = df_sorted.to_csv(index=False, encoding='utf-8').encode('utf-8')
        fingerprint = hashlib.sha256(csv_bytes).hexdigest()
        dataset_key = f"{fingerprint}_{task}_{target}"

        prefix = f"{user_id}/drift_references/{dataset_key}/"
        logger.info(f"Dataset key: {dataset_key} | prefix: {prefix}")

        # ─── Find reference versions ─────────────────────────────────────────────
        blobs = list(container_client.list_blobs(name_starts_with=prefix))
        version_blobs = [b for b in blobs if b.name.endswith("_drift_reference.json") and "v" in b.name]

        # Fallback: bootstrap from latest reference if exact match not found
        if not version_blobs:
            broad_prefix = f"{user_id}/drift_references/"
            all_refs = list(container_client.list_blobs(name_starts_with=broad_prefix))
            if not all_refs:
                return {"drift_report": {
                    "status": "no_versions",
                    "message": "No drift references found yet. Monitoring not activated."
                }}

            all_refs.sort(key=lambda b: b.last_modified or datetime.min, reverse=True)
            latest_blob = all_refs[0]
            # Try to extract dataset_key from path
            parts = latest_blob.name.split("/")
            if len(parts) >= 4:
                dataset_key = parts[-2]
                prefix = f"{user_id}/drift_references/{dataset_key}/"
                blobs = list(container_client.list_blobs(name_starts_with=prefix))
                version_blobs = [b for b in blobs if b.name.endswith("_drift_reference.json") and "v" in b.name]

        if not version_blobs:
            return {"drift_report": {
                "status": "no_versions",
                "message": "No valid drift reference versions found"
            }}

        # Sort by version number (descending → newest first)
        def get_version_num(b):
            name = b.name.split("/")[-1]
            if "latest" in name:
                return 999999
            try:
                return int(name[1:name.index("_")])
            except:
                return 0

        version_blobs.sort(key=get_version_num, reverse=True)
        latest_blob = version_blobs[0]

        ref_client = container_client.get_blob_client(latest_blob.name)
        ref_json = json.loads(ref_client.download_blob().readall())
        ref_stats = ref_json.get("data_stats", {})
        baseline_metric_val = ref_json.get("test_metrics", {}).get(model.get("metric"))

        latest_version_num = get_version_num(latest_blob)
        latest_version_str = "latest" if "latest" in latest_blob.name else f"v{latest_version_num}"
        total_versions = len(version_blobs)

        # ─── Special case: first build, just activated ───────────────────────────
        if mode == "build" and total_versions == 1:
            return {"drift_report": to_python_types({
                "overall_status": "activated",
                "summary_message": "Drift monitoring activated!",
                "details": "This is the initial baseline (v1). No previous version to compare.",
                "recommendation": "Future evaluations will include drift checks.",
                "checked_at": datetime.now().isoformat(),
                "version_compared": "v1 (baseline)",
                "total_versions": 1
            })}

        # ─── Load current data ───────────────────────────────────────────────────
        current_blob_path = training_blob if mode == "build" else test_result["test_blob_path"]
        current_blob = container_client.get_blob_client(current_blob_path)
        if not current_blob.exists():
            raise HTTPException(status_code=404, detail="Current data file not found")

        data = current_blob.download_blob().readall()
        try:
            df_current = pd.read_csv(io.BytesIO(data))
        except Exception:
            df_current = pd.read_excel(io.BytesIO(data))

        if df_current.empty:
            raise HTTPException(status_code=400, detail="Current dataset is empty")

        # ─── PSI-based data drift detection ──────────────────────────────────────
        drifted_features = []
        psi_values = []

        for col, stats in ref_stats.items():
            if col not in df_current.columns:
                continue
            vals = pd.to_numeric(df_current[col], errors='coerce').dropna()

            if stats["type"] == "categorical":
                all_cats = set(stats["distribution"].keys()) | set(df_current[col].dropna().unique())
                all_cats = list(all_cats)
                ref_hist = np.array([stats["distribution"].get(c, 0) for c in all_cats])
                curr_hist = np.array([df_current[col].value_counts(normalize=True).get(c, 0) for c in all_cats])
            else:
                ref_hist = np.array(stats["hist"])
                edges = np.array(stats["bin_edges"])
                curr_hist, _ = np.histogram(vals, bins=edges, density=True)

            eps = 1e-10
            ref_hist = ref_hist / (ref_hist.sum() + eps)
            curr_hist = curr_hist / (curr_hist.sum() + eps)

            psi = np.sum((ref_hist - curr_hist) * np.log((ref_hist + eps) / (curr_hist + eps + eps)))
            psi = np.clip(psi, 0, 10)
            psi_values.append(psi)

            if psi > 0.1:
                drifted_features.append(col)

        overall_psi = round(np.percentile(psi_values, 90) if psi_values else 0, 3)
        data_drift_detected = len(drifted_features) > 0 or overall_psi > 0.05

        # ─── Performance drift (only meaningful in test mode with ground truth) ───
        current_metric_val = None
        if mode == "test" and test_result.get("has_ground_truth"):
            current_metric_val = test_result["test_metrics"].get(model.get("metric"))

        perf_change = None
        perf_degradation = False
        if current_metric_val is not None and baseline_metric_val is not None and baseline_metric_val > 0:
            perf_change = round((current_metric_val - baseline_metric_val) / baseline_metric_val * 100, 1)
            perf_degradation = perf_change < -8

        # ─── Determine overall status ────────────────────────────────────────────
        if data_drift_detected and perf_degradation:
            status = "critical" if mode == "build" else "warning"
            summary = "Critical: data + performance drift" if mode == "build" else "Data & performance drift detected"
            rec = "Retrain urgently" if mode == "build" else "Review test data quality"
        elif perf_degradation:
            status = "degraded"
            summary = "Performance degradation detected"
            rec = "Investigate model behavior on recent data"
        elif data_drift_detected:
            status = "data_drift"
            summary = "Data drift detected"
            rec = "Monitor — performance may degrade soon"
        else:
            status = "stable"
            summary = "No drift detected"
            rec = "Model appears stable"

        comparison_note = f"Compared to latest reference ({latest_version_str})"
        if mode == "test":
            comparison_note += " — test vs most recent model"
        if total_versions > 1:
            comparison_note += f" (of {total_versions} versions)"

        # ─── Final report ────────────────────────────────────────────────────────
        drift_report = {
            "data_drift": {
                "detected": data_drift_detected,
                "overall_psi": overall_psi,
                "drifted_features_count": len(drifted_features),
                "drifted_features": drifted_features[:6],  # limited output
                "status": "drift" if data_drift_detected else "stable"
            },
            "performance_drift": {
                "detected": perf_degradation,
                "change_percent": perf_change,
                "baseline_metric": baseline_metric_val,
                "current_metric": current_metric_val,
                "status": "degraded" if perf_degradation else "improved" if (perf_change or 0) > 5 else "stable"
            },
            "overall_status": status,
            "summary_message": summary,
            "details": f"{comparison_note}\nPSI (90th perc.): {overall_psi} | Drifted features: {len(drifted_features)}",
            "recommendation": rec,
            "checked_at": datetime.now().isoformat(),
            "version_compared": latest_version_str,
            "total_versions": total_versions
        }

        return {"drift_report": to_python_types(drift_report)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drift report failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal error generating drift report")
    except Exception as e:
        logger.error(f"Drift report failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to generate drift report")


#Onelake Endpoints

import os, io
from fastapi import FastAPI, Path, Query, HTTPException
from typing import List, Dict, Any, Optional
import requests
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import ClientSecretCredential
from deltalake import DeltaTable
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.csv as pc
import pyarrow.json as pj
from functools import lru_cache
from fastapi.responses import StreamingResponse


# === CONFIG: Set these in Azure App Service → Application settings ===
# TENANT_ID = os.getenv("AZURE_TENANT_ID")
# CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
# CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
    raise RuntimeError("Set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET in App Settings!")

credential = ClientSecretCredential(
    tenant_id=TENANT_ID,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
)

VERITAS_TENANT_ID     = os.getenv("VERITAS_TENANT_ID")
VERITAS_CLIENT_ID     = os.getenv("VERITAS_CLIENT_ID")
VERITAS_CLIENT_SECRET = os.getenv("VERITAS_CLIENT_SECRET")

has_veritas = all([VERITAS_TENANT_ID, VERITAS_CLIENT_ID, VERITAS_CLIENT_SECRET])

veritas_credential = None
if has_veritas:
    veritas_credential = ClientSecretCredential(
        tenant_id=VERITAS_TENANT_ID,
        client_id=VERITAS_CLIENT_ID,
        client_secret=VERITAS_CLIENT_SECRET
    )
else:
    print("STARTUP → VERITAS credentials NOT loaded (one or more variables missing)")

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_CLUSTER_ID = os.getenv("DATABRICKS_CLUSTER_ID")

has_databricks = all([DATABRICKS_HOST, DATABRICKS_TOKEN])

databricks_handler_global = None
if has_databricks:
    databricks_handler_global = DatabricksHandler(
        host=DATABRICKS_HOST,
        token=DATABRICKS_TOKEN,
        cluster_id=DATABRICKS_CLUSTER_ID
    )
else:
    print("STARTUP → Databricks credentials NOT loaded (DATABRICKS_HOST / DATABRICKS_TOKEN missing)")


class OneLakeHandler:
    def __init__(self):
        self.credential = veritas_credential

    def _token(self, scope: str = "https://api.fabric.microsoft.com/.default"):
        return self.credential.get_token(scope).token

    def _headers(self):
        return {"Authorization": f"Bearer {self._token()}"}

    def _get(self, url: str) -> dict:
        r = requests.get(url, headers=self._headers())
        r.raise_for_status()
        return r.json()

    # === Workspaces ===
    def get_workspaces(self) -> List[Dict[str, Any]]:
        data = self._get("https://api.fabric.microsoft.com/v1/workspaces")
        return [{"id": ws["id"], "name": ws["displayName"]} for ws in data.get("value", [])]

    def get_workspace_id_by_name(self, name: str) -> str:
        for ws in self.get_workspaces():
            if ws["name"].lower() == name.lower():
                return ws["id"]
        raise HTTPException(404, f"Workspace '{name}' not found")

    @lru_cache(maxsize=128)
    def get_workspace_name_by_id(self, workspace_id: str) -> str:
        """Reverse lookup: get displayName from workspace ID"""
        for ws in self.get_workspaces():
            if ws["id"] == workspace_id:
                return ws["name"]
        raise ValueError(f"Workspace ID {workspace_id} not found")

    @lru_cache(maxsize=128)
    def get_lakehouse_name_by_id(self, workspace_id: str, lakehouse_id: str) -> str:
        """Get lakehouse displayName from ID"""
        for lh in self.get_lakehouses(workspace_id):
            if lh["id"] == lakehouse_id:
                return lh["name"]
        raise ValueError(f"Lakehouse ID {lakehouse_id} not found in workspace {workspace_id}")

    # === Lakehouses ===
    def get_lakehouses(self, workspace_id: str) -> List[Dict[str, Any]]:
        data = self._get(f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/lakehouses")
        return [{"id": lh["id"], "name": lh["displayName"]} for lh in data.get("value", [])]

    def get_lakehouse_id_by_name(self, workspace_name: str, lakehouse_name: str) -> str:
        ws_id = self.get_workspace_id_by_name(workspace_name)
        for lh in self.get_lakehouses(ws_id):
            if lh["name"].lower() == lakehouse_name.lower():
                return lh["id"]
        raise HTTPException(404, f"Lakehouse '{lakehouse_name}' not found")

    # === Tables (Delta) ===
    def get_tables(self, workspace_id: str, lakehouse_id: str) -> List[Dict[str, Any]]:
        data = self._get(f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/tables")
        return [{"name": t["name"], "format": t.get("format", "Delta")} for t in data.get("value", [])]

    # === Generic Folder Listing (Files/, Tables/, or any path) ===
    def list_folder_contents(
        self,
        workspace_id: str,
        lakehouse_id: str,
        path: str = "Files"  # e.g. "Files", "Files/bronze", "Tables", "Tables/customer"
    ) -> List[Dict[str, Any]]:
        service_client = DataLakeServiceClient(
            account_url="https://onelake.dfs.fabric.microsoft.com",
            credential=self.credential
        )
        fs_client = service_client.get_file_system_client(workspace_id)

        # Build correct prefix
        if not path or path == "Files":
            prefix = f"{lakehouse_id}/Files/"
        elif path == "Tables":
            prefix = f"{lakehouse_id}/Tables/"
        else:
            # Custom path: Files/bronze, Tables/customer, etc.
            clean_path = path.strip("/")
            prefix = f"{lakehouse_id}/{clean_path}/"

        items = []
        try:
            for p in fs_client.get_paths(path=prefix, recursive=False):
                rel = p.name[len(prefix):].rstrip("/")
                name = rel.split("/")[-1] if rel else "(root)"
                items.append({
                    "name": name,
                    "full_path": p.name,
                    "is_directory": p.is_directory,
                    "size": p.content_length if not p.is_directory else None,
                    "last_modified": p.last_modified.isoformat() if p.last_modified else None,
                })
        except Exception as e:
            raise HTTPException(500, f"Error listing '{path}': {str(e)}")

        return items

    # ==================== UNIFIED DATA PREVIEW ====================
    def preview_path(
    self,
    workspace_id: str,
    lakehouse_id: str,
    path: str,
    max_rows: int = 100
) -> Dict[str, Any]:
    
        service_client = DataLakeServiceClient(
            account_url="https://onelake.dfs.fabric.microsoft.com",
            credential=self.credential
        )
        fs_client = service_client.get_file_system_client(workspace_id)

        # === STEP 1: Normalize and validate path ===
        if not path or path.strip("/") == "":
            raise HTTPException(400, "Path is required. Examples: Tables/orders, Files/data.csv")

        clean_path = path.strip("/")
        
        # Force correct root if user gave only table/file name without Tables/ or Files/
        if not clean_path.lower().startswith(("tables/", "files/", "tablemaintenance/")):
            # Heuristic: if it contains a dot → probably a file → assume Files/
            if "." in clean_path.split("/")[-1]:
                clean_path = "Files/" + clean_path
            else:
                # Otherwise assume it's a table
                clean_path = "Tables/" + clean_path

        full_adls_path = f"{lakehouse_id}/{clean_path}"

        # === STEP 2: Helper – is this a Delta table? ===
        def is_delta_table(adls_path: str) -> bool:
            try:
                # We only look one level deep for _delta_log
                for p in fs_client.get_paths(path=adls_path, recursive=False, max_results=10):
                    if p.name == f"{adls_path.rstrip('/')}/_delta_log" and p.is_directory:
                        return True
                return False
            except:
                return False

        # === CASE 1: Delta Table ===
        if is_delta_table(full_adls_path.rstrip("/")):
            try:
                table_uri = f"abfs://{workspace_id}/{full_adls_path}".rstrip("/")
                dt = DeltaTable(
                    table_uri,
                    storage_options={
                        "bearer_token": self._token("https://storage.azure.com/.default"),
                        "use_fabric_endpoint": "false"
                    }
                )
                pdf = dt.to_pyarrow_dataset().to_table().to_pandas().head(max_rows)
                return {
                    "type": "delta_table",
                    "path": path,
                    "rows_returned": len(pdf),
                    "columns": [{"name": c, "dtype": str(t)} for c, t in pdf.dtypes.items()],
                    "data": pdf.where(pdf.notnull(), None).to_dict(orient="records")
                }
            except Exception as e:
                raise HTTPException(500, f"Failed to read Delta table: {str(e)}")

        # === CASE 2: Regular file ===
        try:
            file_client = fs_client.get_file_client(full_adls_path)
            download = file_client.download_file()
            file_bytes = download.readall()

            if len(file_bytes) == 0:
                raise HTTPException(404, "File is empty or not found")

            file_lower = clean_path.lower()

            if file_lower.endswith(".parquet"):
                table = pq.read_table(io.BytesIO(file_bytes))
                file_type = "parquet"
            elif file_lower.endswith(".csv"):
                table = pc.read_csv(io.BytesIO(file_bytes))
                file_type = "csv"
            elif file_lower.endswith((".json", ".jsonl", ".ndjson")):
                table = pj.read_json(io.BytesIO(file_bytes))
                file_type = "json_lines"
            else:
                raise HTTPException(415, f"Unsupported file type. Only .parquet, .csv, .json/.jsonl allowed.")

            pdf = table.to_pandas().head(max_rows)

            return {
                "type": file_type,
                "path": path,
                "rows_returned": len(pdf),
                "columns": [{"name": c, "dtype": str(t)} for c, t in pdf.dtypes.items()],
                "data": pdf.where(pdf.notnull(), None).to_dict(orient="records")
            }

        except Exception as e:
            if "ResourceNotFound" in str(e):
                raise HTTPException(404, f"File or table not found: {path}")
            if "OperationNotAllowedOnThePath" in str(e):
                raise HTTPException(400, "Invalid path. Must be under Files/ or Tables/ folder.")
            raise HTTPException(500, f"Error reading path: {str(e)}")

    def tables_path(self, workspace_id: str, lakehouse_id: str) -> List[Dict[str, Any]]:
        data = self._get(f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/tables")
    
        # Get human-readable names once
        workspace_name = self.get_workspace_name_by_id(workspace_id)
        lakehouse_name = self.get_lakehouse_name_by_id(workspace_id, lakehouse_id)
    
        tables = []
        for t in data.get("data", []): # Official key is "data"
            abfss_location: str = t.get("location", "")
        
            # Default fallback values
            https_guid = abfss_location.replace("abfss://", "https://").replace("@onelake.dfs.fabric.microsoft.com/", "/") if abfss_location else ""
            https_name = ""
            relative_path = f"Tables/{t.get('name', '')}"
        
            if abfss_location.startswith("abfss://"):
                try:
                    # Parse: abfss://ws_guid@onelake.dfs.fabric.microsoft.com/lh_guid/Tables/table_name
                    remainder = abfss_location[len("abfss://"):]
                    ws_part, path_part = remainder.split("@onelake.dfs.fabric.microsoft.com/", 1)
                    # path_part = lh_guid/Tables/table_name
                    https_guid = f"https://onelake.dfs.fabric.microsoft.com/{ws_part}/{path_part}"
                
                    # Name-based HTTPS (only if names are URL-safe)
                    https_name = f"https://onelake.dfs.fabric.microsoft.com/{workspace_name}/{lakehouse_name}.Lakehouse/Tables/{t.get('name')}"
                except Exception:
                    pass # fallback to empty/guid version
        
            tables.append({
                "name": t.get("name"),
                "catalog_name": f"{lakehouse_id}.Lakehouse",
                "schema_name": "dbo", # Default in non-schema-enabled lakehouses
                "table_type": t.get("type"), # e.g. "Managed"
                "data_source_format": t.get("format", "Delta").upper(),
                "storage_location": https_guid, # Primary: GUID-based HTTPS (matches your example)
                "storage_location_https_name": https_name or None, # Name-based (may be empty if unsafe)
                "storage_location_abfss": abfss_location or None,
                "relative_path": relative_path
            })
    
        return tables


# Global handler
ohandler = OneLakeHandler()


# ==================== ENDPOINTS ====================

@app.get("/workspaces")
def get_workspaces_endpoint():
    """List all accessible workspaces"""
    return {"workspaces": ohandler.get_workspaces()}


@app.get("/workspaces/{workspace_name}/lakehouses")
def get_lakehouses_endpoint(workspace_name: str = Path(..., description="Exact workspace name")):
    """List all lakehouses in a workspace"""
    ws_id = ohandler.get_workspace_id_by_name(workspace_name)
    return {"lakehouses": ohandler.get_lakehouses(ws_id)}


@app.get("/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/tables")
def get_tables_endpoint(
    workspace_name: str = Path(...),
    lakehouse_name: str = Path(...)
):
    """List all Delta tables in the lakehouse"""
    ws_id = ohandler.get_workspace_id_by_name(workspace_name)
    lh_id = ohandler.get_lakehouse_id_by_name(workspace_name, lakehouse_name)
    return {"tables": ohandler.get_tables(ws_id, lh_id)}


@app.get("/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/contents")
def get_folder_contents_endpoint(
    workspace_name: str = Path(..., description="Workspace name"),
    lakehouse_name: str = Path(..., description="Lakehouse name"),
    path: str = Query("Files", description="Path like: Files, Files/bronze, Tables, Tables/customer")
):
    ws_id = ohandler.get_workspace_id_by_name(workspace_name)
    lh_id = ohandler.get_lakehouse_id_by_name(workspace_name, lakehouse_name)

    clean_path = path.strip().rstrip("/")

    # CASE 1: Inside a table → Tables/data, Tables/customer_churn, etc.
    if clean_path.startswith("Tables/"):
        table_name = clean_path[len("Tables/"):]
        table_metadata = None
        if table_name:
            all_tables = ohandler.tables_path(ws_id, lh_id)
            table_metadata = next((t for t in all_tables if t["name"] == table_name), None)

        # Get raw folder contents
        items = ohandler.list_folder_contents(ws_id, lh_id, path)
        folders_raw = [i for i in items if i["is_directory"]]
        files_raw = [i for i in items if not i["is_directory"]]

        # Build base paths for file URLs
        base_https_name = f"https://onelake.dfs.fabric.microsoft.com/{workspace_name}/{lakehouse_name}.Lakehouse"
        base_abfss = f"abfss://{workspace_name}@onelake.dfs.fabric.microsoft.com/{lakehouse_name}.Lakehouse"

        # Enrich files with full path variants
        files = []
        for f in files_raw:
            file_relative = f"{clean_path}/{f['name']}"  
            files.append({
                "name": f["name"],
                "size_bytes": f["size"],
                "last_modified": f["last_modified"],
                "full_path": f["full_path"],  
                "url_https": f"{base_https_name}/{file_relative}",
                "relative_path": file_relative,
                "url_abfss": f"{base_abfss}/{file_relative}",
                # "preview_url": f"/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/preview?path={file_relative}",
                "download_url": f"/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/download?path={file_relative}"
            })

        response = {
            "workspace": workspace_name,
            "lakehouse": lakehouse_name,
            "current_path": path,
            "folders": [{"name": folder["name"]} for folder in folders_raw],
            "files": files
        }

        # Add table-level metadata
        if table_metadata:
            response.update({
                "table_name": table_metadata["name"],
                "storage_location": table_metadata["storage_location"],
                "storage_location_https_name": table_metadata["storage_location_https_name"],
                "storage_location_abfss": table_metadata["storage_location_abfss"],
                "table_relative_path": table_metadata["relative_path"],
                "table_preview_url": f"/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/preview?path=Tables/{table_name}"
            })

        return response

    # CASE 2: Normal Files/ paths → unchanged
    items = ohandler.list_folder_contents(ws_id, lh_id, path)
    folders = [i for i in items if i["is_directory"]]
    files = [i for i in items if not i["is_directory"]]

    return {
        "workspace": workspace_name,
        "lakehouse": lakehouse_name,
        "current_path": path,
        "folders": [{"name": f["name"]} for f in folders],
        "files": [{
            "name": f["name"],
            "size_bytes": f["size"],
            "last_modified": f["last_modified"],
            "full_path": f["full_path"]
        } for f in files]
    }

# Data Preview Endpoint
@app.get("/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/preview")
def preview_endpoint(
    workspace_name: str = Path(..., description="Workspace name"),
    lakehouse_name: str = Path(..., description="Lakehouse name"),
    path: str = Query(..., description="e.g. Files/data.csv | Tables/orders | or full ADLS path from /contents"),
    rows: Optional[int]  = Query(100, ge=1, le=1000, description="rows to preview (1–1000)")
):
    ws_id = ohandler.get_workspace_id_by_name(workspace_name)
    lh_id = ohandler.get_lakehouse_id_by_name(workspace_name, lakehouse_name)

    # Smart path handling – accept both friendly and full ADLS paths
    if path.startswith(lh_id + "/"):
        relative_path = path[len(lh_id) + 1:]  # strip lakehouse GUID
    else:
        relative_path = path.strip("/")

    result = ohandler.preview_path(ws_id, lh_id, relative_path, max_rows=rows)

    return {
        "workspace": workspace_name,
        "lakehouse": lakehouse_name,
        "requested_path": path,
        "resolved_path": relative_path,
        "rows_requested": rows,
        "preview": result
    }

@app.get("/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/download")
def download_file(
    workspace_name: str,
    lakehouse_name: str,
    path: str = Query(..., description="Relative path")
):
    ws_id = ohandler.get_workspace_id_by_name(workspace_name)
    lh_id = ohandler.get_lakehouse_id_by_name(workspace_name, lakehouse_name)

    clean_path = path.strip("/")
    full_adls_path = f"{lh_id}/{clean_path}"

    service_client = DataLakeServiceClient(
        account_url="https://onelake.dfs.fabric.microsoft.com",
        credential=ohandler.credential
    )
    fs_client = service_client.get_file_system_client(ws_id)
    file_client = fs_client.get_file_client(full_adls_path)

    try:
        download = file_client.download_file()
        file_bytes = download.readall()

        if clean_path.startswith("Tables/"):
            parts = clean_path.split("/", 2)
            if len(parts) >= 2:
                table_name = parts[1]
                filename = f"{table_name}.parquet"
            else:
                filename = clean_path.split("/")[-1]
        else:
            filename = clean_path.split("/")[-1]

        return StreamingResponse(
            io.BytesIO(file_bytes),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Content-Length": str(len(file_bytes))
            }
        )
    except Exception as e:
        if "ResourceNotFound" in str(e):
            raise HTTPException(404, "File not found")
        raise HTTPException(500, f"Download failed: {str(e)}")



@app.get("/workspaces/{workspace_name}/lakehouses/{lakehouse_name}/download-veritas")
def download_veritas(
    workspace_name: str = Path(...),
    lakehouse_name: str = Path(...),
    path: str = Query(...)
):

    if not has_veritas or veritas_credential is None:
        raise HTTPException(
            status_code=503,
            detail="VERITAS credentials not configured. Check VERITAS_TENANT_ID, VERITAS_CLIENT_ID, VERITAS_CLIENT_SECRET env vars."
        )

    # Use VERITAS credential for name 
    veritas_local_handler = OneLakeHandler()
    veritas_local_handler.credential = veritas_credential

    try:
        ws_id = veritas_local_handler.get_workspace_id_by_name(workspace_name)
        # print(f"  → workspace ID (VERITAS) = {ws_id!r}")
    except HTTPException as exc:
        # print(f"  → Workspace lookup failed (VERITAS): {exc.status_code} {exc.detail}")
        raise

    try:
        lh_id = veritas_local_handler.get_lakehouse_id_by_name(workspace_name, lakehouse_name)
        # print(f"  → lakehouse ID (VERITAS) = {lh_id!r}")
    except HTTPException as exc:
        # print(f"  → Lakehouse lookup failed (VERITAS): {exc.status_code} {exc.detail}")
        raise

    clean_path = path.strip("/")
    full_adls_path = f"{lh_id}/{clean_path}"
    # print(f"  → full ADLS path = {full_adls_path!r}")

    try:
        service_client = DataLakeServiceClient(
            account_url="https://onelake.dfs.fabric.microsoft.com",
            credential=veritas_credential
        )
        # print("  → DataLakeServiceClient created (VERITAS)")

        fs_client = service_client.get_file_system_client(ws_id)
        file_client = fs_client.get_file_client(full_adls_path)

        download = file_client.download_file()
        file_bytes = download.readall()
        # print(f"  → file read → size = {len(file_bytes)} bytes")

        if clean_path.startswith("Tables/"):
            parts = clean_path.split("/", 2)
            filename = f"{parts[1]}.parquet" if len(parts) >= 2 else clean_path.split("/")[-1]
        else:
            filename = clean_path.split("/")[-1]

        return StreamingResponse(
            io.BytesIO(file_bytes),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(file_bytes))
            }
        )

    except Exception as e:
        err = str(e)
    
        if "ResourceNotFound" in err:
            raise HTTPException(404, f"File not found (VERITAS): {path}")
        if any(x in err for x in ["Unauthorized", "Authentication Failed", "Bearer token"]):
            raise HTTPException(401, f"VERITAS authentication failed: {err}")
        raise HTTPException(500, f"VERITAS download failed: {err}")

@app.get("/health")
def health_check():
    return {"status": "ok"}

from io import BytesIO
from azure.identity import ClientSecretCredential
from azure.storage.filedatalake import DataLakeServiceClient
UPLOAD_JOB_RESULTS: dict = {}


def full_background_upload_process(
    job_id: str,
    file_path: str,
    session_id: str,
    user_email: str,
    query: Optional[str]
):
    """
    Does everything upload_file_endpoint_v used to do synchronously:
    fetch from OneLake, run dataset intelligence, run the Code-Interpreter
    analysis, and produce the final response payload - but writes the
    result into UPLOAD_JOB_RESULTS instead of returning it directly, so the
    HTTP request can return instantly and the client polls for the result.
    This avoids gateway timeouts on the long-running analysis calls.
    """
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            UPLOAD_JOB_RESULTS[job_id] = {"status": "failed", "message": "Unauthorized"}
            return

        agent_info = user["agents"][0]
        if session_id not in agent_info.get("threads", []):
            UPLOAD_JOB_RESULTS[job_id] = {"status": "failed", "message": "Invalid thread"}
            return

        user_id = user["user_id"]
        agent_id = agent_info["agent_id"]
        agent_name = agent_info["agent_name"]
        thread_id = session_id

        original_filename = file_path.split("/")[-1] or "unnamed_file.csv"
        display_name = original_filename

        # Fetch file from whichever source it lives in (Databricks Volume or
        # OneLake) - dispatch logic lives in app/source_fetch.py.
        try:
            file_content, resolved_source_type = fetch_source_file(
                file_path=file_path,
                databricks_handler=databricks_handler_global,
            )
        except SourceFileNotFoundError as e:
            UPLOAD_JOB_RESULTS[job_id] = {"status": "failed", "message": str(e)}
            return
        except SourceCredentialsMissingError as e:
            UPLOAD_JOB_RESULTS[job_id] = {"status": "failed", "message": str(e)}
            return

        class FetchedFile:
            def __init__(self, content: BytesIO, filename: str):
                self.file = content
                self.filename = filename
                self.content_type = "application/octet-stream"

        fetched_file = FetchedFile(file_content, original_filename)

        processed_file, processed_filename, was_converted = asyncio.run(to_csv_if_needed(fetched_file))
        if was_converted:
            display_name += " (converted to CSV)"

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        try:
            processed_file.file.seek(0)
            content = processed_file.file.read()
            temp_file.write(content)
            temp_file.close()

            file_id, blob_name, _ = handler.upload_file_to_blob_and_agent(
                processed_file, agent_id, user_id, thread_id=session_id
            )

            intelligence_result = dataset_service.process_new_upload(
                file_path=temp_file.name,
                filename=processed_filename,
                user_id=user_id,
                user_email=user_email,
                agent_id=agent_id,
                blob_path=blob_name,
                min_similarity_threshold=60.0,
                source_type=resolved_source_type,
                source_path=file_path
            )

            thread_metadata_handler.create_or_update_thread_metadata(
                thread_id=session_id,
                user_id=user_id,
                user_email=user_email,
                agent_id=agent_id,
                dataset_id=intelligence_result["dataset_id"],
                blob_path=blob_name,
                filename=processed_filename,
                veriton_file_path=file_path
            )

            run_auto_analysis = str(query or "").lower() == "true"

            # CASE 1 – NO AUTO-ANALYSIS
            if not run_auto_analysis:
                suggestion = handler.get_dynamic_suggestion(
                    thread_id=thread_id,
                    user_email=user_email,
                    agent_info=agent_info,
                    analysis_text=f"Uploaded {display_name}",
                    user_id=user_id,
                    agent_name=agent_name,
                )
                UPLOAD_JOB_RESULTS[job_id] = {
                    "status": "success",
                    "filename": display_name,
                    "fileid": file_id,
                    "threadid": session_id,
                    "response": f"File '{display_name}' uploaded successfully.",
                    "ml_ready": False,
                    "auto_analysis": False,
                    "suggestions": _suggestions_list(suggestion)
                }
                return

            # Shared implementation lives in app/utils.py (also handles
            # "NEXT STEPS:" with a space and a payload on the following
            # line, which this local version used to miss).
            _split_next_steps = split_next_steps

            has_matches = intelligence_result.get("has_matches", False)
            similar_datasets = intelligence_result.get("similar_datasets") or []
            top_match = similar_datasets[0] if similar_datasets else None
            match_type = (top_match or {}).get("match_type")

            upload_case = "new"
            if has_matches and top_match:
                upload_case = "unchanged" if match_type == "exact_match" else "changed"

            DRILLDOWN_RULES = (
                "NO FIXED TEMPLATE RULE (read this first):\n"
                "Do not follow a fixed response template or predefined section structure. "
                "Analyze the actual dataset and dynamically organize your answer around "
                "whatever is most relevant for THIS data - the real business insights, "
                "patterns, risks, opportunities, data-quality issues, and analytical "
                "possibilities it actually contains. Use headings only when they genuinely "
                "improve readability, and derive any heading from this dataset's own domain "
                "and findings - never from a predefined checklist. Never reuse the same "
                "section titles, opening line, or overall shape response after response; two "
                "different uploads (or two looks at the same file) should read like two "
                "different pieces of analysis, not the same form filled in twice. Skip "
                "anything - a section, a heading, a whole topic - that isn't genuinely "
                "relevant here. Be concise when the data only supports a simple answer, and "
                "go deeper only where the data genuinely warrants it.\n\n"
                "CRITICAL REASONING RULE - follow this for every important finding:\n"
                "Do not merely describe columns or repeat dataset statistics. For every "
                "important finding, investigate whether the data provides a meaningful "
                "comparison, concentration, relationship, trend, anomaly, or business "
                "impact. When you identify a potentially important metric or issue, drill "
                "down into it to determine WHERE, WHEN, or FOR WHOM it occurs, whenever the "
                "columns support that analysis. Follow this progression: "
                "DATA -> FINDING -> DRILL-DOWN -> IMPACT -> ACTION.\n"
                "Weak (never do this): 'Downtime is tracked.' or 'About 30 of 80 orders are "
                "flagged as breakdown risk.' and then stop.\n"
                "Good (always aim for this): 'Downtime is concentrated in 3 of the 8 "
                "machines, with Machine M04 contributing the largest share, and those same "
                "orders also show elevated defect rates - worth investigating first.'\n"
                "Whenever you mention a count, a rate, or a flagged group (e.g. 'X of Y "
                "orders are flagged'), immediately break it down further using whatever "
                "columns exist - which machine, shift, product, plant, supplier, or time "
                "period it's concentrated in - and check whether it coincides with other "
                "problems (higher downtime, more defects, higher cost, more delays). Never "
                "leave a striking number unexplored.\n\n"
                "PRIORITIZATION RULE:\n"
                "Focus on findings that are genuinely meaningful to the business rather than "
                "trying to touch every column, metric, or observation in the file. Cover "
                "however many findings actually deserve attention for this dataset - not a "
                "fixed count - each with: what it is, why it matters, the evidence (real "
                "numbers/names), and what to do about it. Show what matters most through the "
                "order you present things, the depth of evidence you give it, and your "
                "wording - never through explicit labels like 'Priority 1', 'High Priority', "
                "or numbered rankings. Let the most important thing simply come first and get "
                "the most substance; let minor observations stay brief or get left out.\n\n"
                "WHAT WE COULD DO WITH THIS DATA (only when genuinely useful - be selective):\n"
                "Weave in, wherever it reads naturally (a short labeled section is fine, but "
                "don't force the exact same heading and shape every time), which realistic "
                "analytical or ML approaches genuinely fit THIS dataset - not a checklist of "
                "everything the system happens to support. Judge this using the real columns "
                "available, whether a sensible target/outcome column exists, data quality, "
                "sample size, and the actual business problem. Recommend only the ones that "
                "provide real apparent business value; if several are technically possible, "
                "lead with the one(s) that matter most rather than listing all of them with "
                "equal weight. The possible kinds are:\n"
                "- Classification -> predicting a category or label for each record (e.g. "
                "'yes/no', 'high/medium/low').\n"
                "- Regression -> predicting a number (like a cost, quantity, or score).\n"
                "- Forecasting -> predicting future values over time (like next month's "
                "demand or cost) - needs a real date/time column with enough history.\n"
                "- Clustering -> grouping similar records together to find natural patterns "
                "or segments - useful when there's no single obvious outcome column.\n"
                "- Anomaly detection -> automatically flagging unusual records that don't fit "
                "the normal pattern.\n"
                "Do not recommend a kind just because it's technically possible - if the data "
                "lacks a suitable target, enough history, the right features, or good enough "
                "quality for something a user might expect (e.g. forecasting with no usable "
                "date column), say so plainly and explain why it isn't a good fit right now, "
                "rather than silently omitting it or forcing it in anyway. If genuinely "
                "nothing fits well, say that honestly instead of padding the list.\n\n"
                "HOW TO WRITE EACH OPPORTUNITY YOU DO RECOMMEND:\n"
                "For each one, cover: what could be predicted or found (naming the REAL "
                "column from this dataset - copied exactly, never invented or guessed), what "
                "real columns would be used to do it, and why it matters for the business. "
                "Example shape: 'ProductionCost could be predicted using PlannedQuantity, "
                "ProducedQuantity, and ProductionTimeHours - tree-based methods like Random "
                "Forest or XGBoost could work well here since cost likely has a nonlinear "
                "relationship with these operational variables.'\n\n"
                "MODEL NAMING RULE (strict):\n"
                "You may name 1-2 model families per opportunity when they add real value - "
                "never an exhaustive list, and never bare (always paired with a brief reason "
                "tied to this dataset's actual characteristics: size, feature types, "
                "linearity, target type). Do NOT use your own general ML knowledge to name "
                "models - use ONLY the exact models this system actually has available for "
                "that task, listed below. Never mention any model not on this list for that "
                "task (for example, never say SVM, KNN, hierarchical clustering, linear "
                "regression, neural networks, or deep learning - this system does not offer "
                "them):\n"
                "- Classification: Logistic Regression, Random Forest, Gradient Boosting, "
                "XGBoost\n"
                "- Regression: Ridge Regression, Random Forest, Gradient Boosting, XGBoost\n"
                "- Forecasting: ARIMA, Prophet, XGBoost, LightGBM, CatBoost\n"
                "- Clustering: K-Means, K-Means++, DBSCAN, Gaussian Mixture Model\n"
                "- Anomaly Detection: Isolation Forest, One-Class SVM, Local Outlier Factor, "
                "Elliptic Envelope\n"
                "Pick only the 1-2 from the relevant list above that genuinely suit this "
                "dataset - e.g. tree-based models (Random Forest, XGBoost, Gradient Boosting, "
                "LightGBM, CatBoost) for data with nonlinear relationships between mixed "
                "numeric/categorical features; Isolation Forest as a reasonable default "
                "starting point for anomaly detection; K-Means for clustering when features "
                "are numeric and reasonably scaled. Do not recommend a specific model without "
                "real evidence from the data that it fits - if unsure, it's fine to mention "
                "the opportunity without naming a specific model at all.\n\n"
                "CLOSING RULE:\n"
                "End your visible answer naturally. Then, on its own final line, include "
                "an internal marker the user will never see (it gets stripped before "
                "display): NEXT_STEPS: <action 1> | <action 2> | <action 3> - each action "
                "under 12 words, phrased as a plain-language OUTCOME a non-technical person "
                "would understand, using real machine/shift/product/group names where "
                "relevant but NEVER raw column names or technical terms. Say 'Find which "
                "machines or shifts have the highest defect rates' not 'analyze "
                "defective_quantity by machine_id'. Say 'Investigate how downtime relates to "
                "production delays' not 'review downtime_hours and production_status'. This "
                "line is a control signal, not part of the message.\n\n"
                "LANGUAGE RULES:\n"
                "- Outside the ML-possibilities part of your answer, never use the words "
                "model, target, classification, regression, clustering, algorithm, dataframe, "
                "feature engineering, null ratio, or any ML/stats jargon in the visible text.\n"
                "- Inside that part, you may name a task type or technique, but only ever "
                "right next to a plain-language explanation of what it means, never bare.\n"
                "- Use real column names and real numbers throughout, never placeholders.\n"
                "- Write like a sharp analyst talking to a manager - flowing paragraphs, a "
                "few bullets where useful, never a rigid numbered report with headers.\n"
                "- Vary your structure, opening line, and phrasing every time - never reuse "
                "the same section titles or shape from one response to the next.\n"
            )

            FILE_LOAD_HINT = (
                f"\n\nFILE ACCESS (Code Interpreter): The file '{processed_filename}' has been "
                "uploaded and is available to Code Interpreter. To load it, use:\n"
                "  import pandas as pd, glob, os\n"
                f"  matches = glob.glob('/mnt/data/**/{processed_filename}', recursive=True) "
                f"or glob.glob('/mnt/data/{processed_filename}')\n"
                "  df = pd.read_csv(matches[0]) if matches else pd.read_csv"
                f"('/mnt/data/{processed_filename}')\n"
                "Always use this glob approach - never hard-code a path that may not exist.\n"
            )

            INTERNAL_ML_TRIGGER = (
                "\n\nINTERNAL ONLY - never shown to the user, only used by the system:\n"
                "If, in a LATER message, the user replies with agreement or continuation "
                "('yes', 'ok', 'go', 'do it', 'sounds good', 'predict [something]', 'start', "
                "etc.), respond to THAT message with ONLY this JSON (no markdown, no "
                "explanation - this is a control signal, not something the user reads):\n"
                "{\n"
                '  "is_ml": true,\n'
                '  "task_type": "<task>",\n'
                '  "target": "<column_name>",\n'
                '  "metric": "<metric_name>",\n'
                ' "models": ["<model1>", "<model2>",""], \n'
                "}\n"
                "Do NOT output this JSON unless the user clearly wants to run the suggested "
                "task; otherwise keep responding normally in plain language.\n"
            )

            if upload_case == "new":
                analysis_prompt = (
                    f"You are an experienced data analyst explaining a file **{display_name}** to "
                    f"a business person with ZERO technical background - no user question yet, "
                    f"this is the first look at the file.\n\n"
                    "Use Code Interpreter to actually explore the real data first - real columns, "
                    "real numbers, real groups.\n\n"
                    "Cover, weaving in only what the data genuinely supports (skip anything that "
                    "doesn't apply, never force it):\n"
                    "- what this data is and why it matters to the business\n"
                    "- what timeframe it covers and whether there's enough history for trends\n"
                    "- whichever things genuinely need attention, each drilled down to "
                    "where/when/for whom it occurs and why it matters - however many that "
                    "turns out to be for this dataset, shown through order and depth, not "
                    "labels\n"
                    "- what's going well, not just problems\n"
                    "- any data-quality gaps that limit how far you can trust a conclusion\n"
                    "- concrete recommendations and what to investigate next\n"
                    "- anything you genuinely cannot conclude because information is missing\n"
                    "- which of the tasks in the What we could do with this data section genuinely fit, with real target columns named\n\n"
                    f"{FILE_LOAD_HINT}"
                    f"{DRILLDOWN_RULES}"
                    "Keep the whole answer focused - roughly 200-350 words depending on how much "
                    "the data supports; don't pad it out."
                    f"{INTERNAL_ML_TRIGGER}"
                )
            elif upload_case == "changed":
                prev_history = dataset_service.get_dataset_history(top_match["dataset_id"], user_id)
                prev_tasks = (prev_history or {}).get("tasks_performed", [])
                prev_semantic = (prev_history or {}).get("semantic_analysis", {})
                details = top_match.get("details", {})
                task_lines = "\n".join(
                    f"- {t.get('task_type')}: {t.get('query','')[:80]} -> {t.get('status')}"
                    for t in prev_tasks[-5:]
                ) or "nothing was run on it yet"

                analysis_prompt = (
                    f"You are an experienced data analyst explaining a file **{display_name}** to "
                    f"a business person with ZERO technical background. This is a CHANGED version "
                    f"of a dataset seen before: **{top_match['similarity_score']:.0f}% similar** to "
                    f"one from {str(top_match.get('upload_date',''))[:10]}. Row count changed by "
                    f"{details.get('row_difference', 'an unknown amount')} and column count changed "
                    f"by {details.get('column_difference', 'an unknown amount')} vs that version. "
                    f"Previous structural summary (for your own context): "
                    f"{json.dumps(prev_semantic)[:1200]}. Previously run on it: {task_lines}.\n\n"
                    "Use Code Interpreter to actually explore the CURRENT data yourself.\n\n"
                    "Open by clearly stating, in plain words, what concretely changed since last "
                    "time (more/fewer records, new or missing columns, shifted numbers) - be "
                    "specific, not vague. Then cover:\n"
                    "- whether previously known issues are resolved, worse, or unchanged now\n"
                    "- whichever things genuinely need attention right now, each drilled "
                    "down to where/when/for whom and why it matters - shown through order "
                    "and depth, not labels\n"
                    "- new findings that weren't true before\n"
                    "- what's going well\n"
                    "- whether what was tried before still makes sense given the change\n"
                    "- concrete recommendations and what to investigate next\n"
                    "- which of the tasks in the What we could do with this data section genuinely fit, with real target columns named\n\n"
                    f"{FILE_LOAD_HINT}"
                    f"{DRILLDOWN_RULES}"
                    "Give a complete analysis of the current file - not just a diff - "
                    "while making the meaningful changes and how they affect the "
                    "previous understanding, findings, data quality, or analytical "
                    "possibilities a clear part of that analysis, not a separate bolted-on "
                    "section. Roughly 250-400 words depending on what the data supports."
                    f"{INTERNAL_ML_TRIGGER}"
                )
            else:
                prev_history = dataset_service.get_dataset_history(top_match["dataset_id"], user_id)
                prev_tasks = (prev_history or {}).get("tasks_performed", [])
                task_count = len(prev_tasks)
                task_lines = "\n".join(
                    f"- {t.get('task_type')}: {t.get('query','')[:80]} -> {t.get('status')}"
                    for t in prev_tasks[-5:]
                ) or "nothing was run on it yet"

                analysis_prompt = (
                    f"You are an experienced data analyst explaining a file **{display_name}** to "
                    f"a business person with ZERO technical background. This file is materially "
                    f"UNCHANGED from a previous upload - {top_match['similarity_score']:.0f}% "
                    f"similar to one from {str(top_match.get('upload_date',''))[:10]}, no "
                    f"meaningful differences in structure or content. Previously run on it: "
                    f"{task_lines} (times run: {task_count}).\n\n"
                    "Use Code Interpreter to actually check the CURRENT data - don't skip "
                    "analysis just because it's unchanged, but keep this SHORT: a refresh, not a "
                    "full report.\n\n"
                    "SIMILAR DATASET WITHOUT MEANINGFUL CHANGES - do exactly this:\n"
                    "Open by clearly stating this is materially unchanged from before, so most of "
                    "the earlier analysis still applies. Then briefly cover, only where genuinely "
                    "true, in flowing plain-language prose (not headers, not a numbered report):\n"
                    "- the most important current findings, verified against the current file\n"
                    "- 1-3 key insights or patterns worth restating\n"
                    "- any notable anomalies or unusual observations, if present\n"
                    "- issues that still need attention (naming specifics, not generic)\n"
                    "- unresolved risks or data-quality concerns still relevant\n"
                    "- which previous findings remain relevant right now\n"
                    "- 1-2 practical recommendations or useful next investigations\n"
                    "- which of the tasks in the What we could do with this data section still genuinely fit, with real target columns named\n\n"
                    "Use the CURRENT dataset to verify these still hold - do not simply copy the "
                    "previous analysis text or invent findings that aren't really there. If there "
                    "genuinely isn't anything new or noteworthy beyond what's already known, say "
                    "so plainly instead of manufacturing insights. Do not repeat basic dataset "
                    "descriptions or full column explanations - the user already knows this "
                    "dataset. The goal is a short refresh of what matters NOW, not a skipped "
                    "analysis and not a full repeat.\n"
                    "Answer this question in your own words somewhere in the reply: 'Okay, I "
                    "uploaded it again - what's important in it right now?'\n\n"
                    "Keep the whole thing to roughly 100-180 words.\n\n"
                    f"{FILE_LOAD_HINT}"
                    f"{DRILLDOWN_RULES}"
                    f"{INTERNAL_ML_TRIGGER}"
                )

                _, ml_json_unchanged, refresh_response_raw = handler.process_query_on_main_agent(
                    agent_id=agent_id,
                    query=analysis_prompt,
                    thread_id=session_id,
                    user_id=user_id,
                    agent_name=agent_name
                )
                fallback_suggestion = "Ask: what's next?"
                try:
                    fallback_suggestion = handler.get_dynamic_suggestion(
                        thread_id=thread_id,
                        user_email=user_email,
                        agent_info=agent_info,
                        analysis_text=refresh_response_raw or "",
                        user_id=user_id,
                        agent_name=agent_name,
                    )
                except Exception:
                    pass
                clean_text, steps = _split_next_steps(refresh_response_raw, fallback_suggestion)
                if not clean_text:
                    clean_text = (
                        f"This looks materially unchanged from {top_match['filename']} "
                        f"({top_match['similarity_score']:.0f}% match), so the earlier findings "
                        f"still apply. You've run {task_count} task(s) on it before."
                    )

                response_handler.save_response(user_email, user_id, agent_id, session_id, "user", f"Uploaded file: {display_name}")
                response_handler.save_response(user_email, user_id, agent_id, session_id, "assistant", clean_text)
                suggestion = handler.get_dynamic_suggestion(
                    thread_id=thread_id,
                    user_email=user_email,
                    agent_info=agent_info,
                    analysis_text=clean_text,
                    user_id=user_id,
                    agent_name=agent_name,
                )
                UPLOAD_JOB_RESULTS[job_id] = {
                    "status": "success",
                    "filename": display_name,
                    "fileid": file_id,
                    "threadid": session_id,
                    "overview_response": clean_text,
                    "ml_ready": True,
                    "auto_analysis": True,
                    "upload_case": upload_case,
                    "suggestions": steps or _suggestions_list(suggestion),
                    "message": "Quick refresh on your familiar dataset - here's what still matters."
                }
                return

            # Shared execution path for NEW and CHANGED cases
            _, ml_json, overview_response_raw = handler.process_query_on_main_agent(
                agent_id=agent_id,
                query=analysis_prompt,
                thread_id=session_id,
                user_id=user_id,
                agent_name=agent_name
            )
            fallback_suggestion = "Ask: what's next?"
            try:
                fallback_suggestion = handler.get_dynamic_suggestion(
                    thread_id=thread_id,
                    user_email=user_email,
                    agent_info=agent_info,
                    analysis_text=overview_response_raw or "",
                    user_id=user_id,
                    agent_name=agent_name,
                )
            except Exception:
                pass
            overview_response, next_steps = _split_next_steps(overview_response_raw, fallback_suggestion)

            response_handler.save_response(user_email, user_id, agent_id, session_id,
                                          "user", f"Uploaded file: {display_name}")
            if overview_response:
                response_handler.save_response(user_email, user_id, agent_id, session_id,
                                              "assistant", overview_response)
            ml_ready = any(k in (overview_response or "").lower()
                           for k in ["predict", "breakdown", "likely to", "flagged", "risk"])
            UPLOAD_JOB_RESULTS[job_id] = {
                "status": "success",
                "filename": display_name,
                "fileid": file_id,
                "threadid": session_id,
                "overview_response": overview_response.strip() if overview_response else "Analysis completed",
                "ml_ready": ml_ready,
                "auto_analysis": True,
                "upload_case": upload_case,
                "suggestions": next_steps or [fallback_suggestion],
                "message": "New dataset analyzed! Ready for action." if upload_case == "new"
                            else "Dataset compared to previous version! Ready for action."
            }
        finally:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

    except Exception as e:
        logger.error(f"[UPLOAD BACKGROUND] Job crashed - job_id={job_id}: {traceback.format_exc()}")
        UPLOAD_JOB_RESULTS[job_id] = {
            "status": "failed",
            "message": "Upload processing failed",
            "error": str(e)
        }


@app.post("/upload_file_V")
async def upload_file_endpoint_v(
    file_path: str = Form(..., description="Relative path inside lakehouse, e.g. Files/Datasets/.../file.csv"),
    session_id: str = Form(...),
    user_email: str = Form(...),
    query: Optional[str] = Form(None)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
        agent_info = user["agents"][0]
        if session_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Invalid thread")

        job_id = str(uuid.uuid4())
        logger.info(f"upload_file_V job started - job_id={job_id}")

        threading.Thread(
            target=full_background_upload_process,
            kwargs={
                "job_id": job_id,
                "file_path": file_path,
                "session_id": session_id,
                "user_email": user_email,
                "query": query
            },
            daemon=True
        ).start()

        return {
            "status": "started",
            "job_id": job_id,
            "message": "Your file is being uploaded and analyzed in the background. This usually takes 30 seconds to a few minutes.",
            "poll_every_seconds": 5
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start upload job: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to start upload processing")


@app.get("/upload-file-status/{job_id}")
def get_upload_file_status(job_id: str, user_email: str = Query(...)):
    result = UPLOAD_JOB_RESULTS.get(job_id)
    if result:
        return result
    return {
        "job_id": job_id,
        "status": "running",
        "message": "Upload is still being processed...",
        "last_checked": datetime.utcnow().isoformat()
    }

@app.post("/build_ml_model_dep", tags=["Deprecated APIs"], deprecated=True)
async def build_ml_model_dep(
    # Required OneLake path for training data
    file_path: str = Form(..., description="Relative path inside lakehouse, e.g. Files/Datasets/.../train.csv"),
    # Optional test file path (also from OneLake)
    test_file_path: Optional[str] = Form(None, description="Optional relative test file path in same lakehouse"),
    
    # Original parameters (unchanged)
    upload_file_path: bool = Form(False),
    user_email: str = Form(...),
    task: Optional[str] = Form(None),
    target: Optional[str] = Form(None),
    models: Optional[str] = Form(None),
    metric: Optional[str] = Form(None),
    horizon: Optional[int] = Form(12),
    transformation_config: Optional[str] = Form(None),
    preprocessing_mode: Optional[str] = Form("simple"),
    use_cleaning: Optional[bool] = Form(True),
    use_feature_selection: Optional[bool] = Form(False),
    use_optuna: Optional[bool] = Form(True),
    optuna_trials: Optional[int] = Form(2),
    time_budget: Optional[int] = Form(300),
    test_size: Optional[float] = Form(0.2)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
            
        user_id = user["user_id"]
        agent_id = user.get("agents", [{}])[0].get("agent_id")
        
       
        original_filename = file_path.split("/")[-1] or "train_dataset.csv"

        # Fetch training file (and optional test file) from whichever
        # source they live in - dispatch logic lives in app/source_fetch.py.
        test_blob = None
        try:
            train_content, test_content, resolved_source_type = fetch_train_and_optional_test_file(
                file_path=file_path,
                test_file_path=test_file_path,
                databricks_handler=databricks_handler_global,
            )
        except SourceFileNotFoundError as e:
            raise HTTPException(404, str(e))
        except SourceCredentialsMissingError as e:
            raise HTTPException(503, str(e))

        content = train_content.getvalue()

        # Process training file 
        class FetchedFile:
            def __init__(self, content: BytesIO, filename: str):
                self.file = content
                self.filename = filename
                self.content_type = "application/octet-stream"

        train_fetched = FetchedFile(train_content, original_filename)
        processed_file, processed_filename, was_converted = await to_csv_if_needed(train_fetched)

        if was_converted:
            original_filename = processed_filename

        # Upload processed training file to your blob
        train_blob = f"{user_id}/build_model/{original_filename}"
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        train_blob_client = container_client.get_blob_client(train_blob)

        train_content.seek(0)
        train_blob_client.upload_blob(train_content.read(), overwrite=True)
        logger.info(f"Processed training file uploaded: {train_blob}")

        # Upload test file if fetched
        if test_content:
            test_original_name = test_file_path.split("/")[-1] or "test_dataset.csv"
            test_blob = f"{user_id}/build_model/{test_original_name}"
            test_blob_client = container_client.get_blob_client(test_blob)
            test_content.seek(0)
            test_blob_client.upload_blob(test_content.read(), overwrite=True)
            logger.info(f"Test file uploaded to blob: {test_blob}")
       
        # Dataset intelligence
        
        dataset_id = None
        analysis_metadata = None
        if upload_file_path:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    train_content.seek(0)
                    tmp.write(train_content.read())
                    tmp_path = tmp.name
                
                try:
                    intelligence_result = dataset_service.process_new_upload(
                        file_path=tmp_path,
                        filename=original_filename,
                        user_id=user_id,
                        user_email=user_email,
                        agent_id=agent_id,
                        blob_path=train_blob,
                        min_similarity_threshold=60.0,
                        source_type=resolved_source_type,
                        source_path=file_path
                    )
                    dataset_id = intelligence_result.get("dataset_id")
                    analysis_metadata = intelligence_result.get("analysis_metadata", {})
                    logger.info(f"Dataset intelligence processed: {dataset_id}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Dataset intelligence failed (non-blocking): {e}")

            feature_suggestions = None
            try:
                feature_suggestions = handler.get_all_task_feature_suggestions(
                    blob_path=train_blob,
                    user_id=user_id,
                    max_preview_rows=200   # or 300, depending on your preference
                )
                logger.info(f"Generated feature suggestions for all tasks - {len(feature_suggestions.get('columns', {}).get('all', []))} columns analyzed")
            except Exception as e:
                logger.warning(f"Feature suggestions generation failed (non-blocking): {str(e)}")
                feature_suggestions = {"status": "failed", "error": str(e)}
            
            return {
                "status": "file_saved",
                "message": "File uploaded successfully! Ready for model building later.",
                "blob_path": train_blob,
                "filename": original_filename,
                "ready_for_training": True,
                "dataset_id": dataset_id,
                "analysis_metadata": {
                    "dataset_structure": analysis_metadata,
                    "intelligence_processed": dataset_id is not None
                },
              "features" : feature_suggestions
            }
        
      
        if not task or not task.strip():
            raise HTTPException(status_code=400, detail="Task is required for training")
        if not target or not target.strip():
            raise HTTPException(status_code=400, detail="Target is required for training")
        
        task = task.lower().strip()
        if task not in TASKS:
            raise HTTPException(status_code=400, detail=f"Invalid task. Must be one of: {TASKS}")

        transformation_config_dict = None

        if transformation_config:
            try:
                transformation_config_dict = json.loads(transformation_config)
                logger.info(f"User provided transformation_config: {transformation_config_dict}")

                transformation_config_dict.setdefault("generated_code", None)
                transformation_config_dict.setdefault("code_metadata", None)
                transformation_config_dict.setdefault("needs_transformation", True)

                # Load dataset for analysis
                temp_df = pd.read_csv(io.BytesIO(train_content.getvalue()))  # use train_content

                # Check what user provided
                has_year_column = bool(transformation_config_dict.get('year_column', '').strip())
                has_month_columns = bool(transformation_config_dict.get('month_columns'))
                
                # Auto-detect year column if missing
                if not has_year_column:
                    logger.info("year_column missing - auto-detecting from dataset...")
                    # Try to find year/date columns
                    year_col = None
                    # Check for columns with 'year' in name
                    year_candidates = [col for col in temp_df.columns if 'year' in col.lower()]
                    if year_candidates:
                        year_col = year_candidates[0]
                        logger.info(f"Found year column by name: {year_col}")
                    
                    # Check for datetime columns
                    if not year_col:
                        datetime_cols = temp_df.select_dtypes(include=['datetime64']).columns.tolist()
                        if datetime_cols:
                            year_col = datetime_cols[0]
                            logger.info(f"Found datetime column: {year_col}")
                        else:
                            # Check for columns that can be converted to datetime
                            for col in temp_df.columns:
                                try:
                                    pd.to_datetime(temp_df[col].dropna().head(5))
                                    year_col = col
                                    logger.info(f"Found parseable date column: {year_col}")
                                    break
                                except:
                                    continue
                    
                    if not year_col:
                        raise HTTPException(
                            status_code=400,
                            detail="Cannot find year/date column in dataset. Please add a column with year values or dates."
                        )
                    
                    transformation_config_dict['year_column'] = year_col
                    logger.info(f"Auto-detected year_column: {year_col}")
                
                # Auto-detect month columns if missing
                if not has_month_columns:
                    logger.info("month_columns missing - detecting from measures...")
                    
                    measures = transformation_config_dict.get('measures', [])
                    if not measures:
                        raise HTTPException(400, "measures are required in transformation_config")
                    
                    # The measures are the month columns in wide format
                    transformation_config_dict['month_columns'] = measures
                    logger.info(f"Month columns set to measures: {len(measures)} columns")
                
                # Validate configuration
                ts_transformer = TimeseriesTransformer()
                
                # Temporarily add needs_transformation for validation
                transformation_config_dict['needs_transformation'] = True
                
                is_valid, error_msg = ts_transformer.validate_transformation_config(
                    temp_df, 
                    transformation_config_dict
                )
                
                if not is_valid:
                    raise HTTPException(400, f"Invalid transformation config: {error_msg}")
                
                logger.info(f"Transformation config validated: {len(transformation_config_dict['measures'])} measures, "
                        f"{len(transformation_config_dict.get('group_by', []))} group_by columns, "
                        f"year_column='{transformation_config_dict['year_column']}'")
                
                # Generate transformation code using LLM
                try:                    
                    llm_transformer = get_llm_transformer()
                    
                    # Create temp file for LLM analysis
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_input:
                        tmp_input.write(content)
                        tmp_input_path = tmp_input.name
                    
                    try:
                        # Get dataset analysis
                        analysis = ts_transformer.analyze_dataset_structure(temp_df)
                        
                        # Generate transformation code with auto detected fields
                        generated_code, code_metadata = llm_transformer.generate_transformation_code(
                            file_path=tmp_input_path,
                            analysis=analysis,
                            transformation_config=transformation_config_dict
                        )
                        
                        # Store generated code
                        transformation_config_dict['generated_code'] = generated_code
                        transformation_config_dict['code_metadata'] = code_metadata
                        
                        logger.info(f"LLM generated transformation code ({len(generated_code)} chars)")
                        # logger.info(f"Code preview: {generated_code}")
                        
                    finally:
                        if os.path.exists(tmp_input_path):
                            os.remove(tmp_input_path)
                
                except Exception as llm_error:
                    logger.error(f"LLM transformation code generation failed: {llm_error}")
                    logger.warning("Falling back to rule-based transformation")
                    
                    transformation_config_dict['generated_code'] = ""
                    transformation_config_dict['needs_transformation'] = True
                
            except json.JSONDecodeError:
                raise HTTPException(400, "Invalid transformation_config JSON")
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Transformation config error: {traceback.format_exc()}")
                raise HTTPException(400, f"Transformation config error: {str(e)}")
            
        if not metric:
            metric = DEFAULT_METRICS[task]
        elif metric not in METRICS[task]:
            raise HTTPException(status_code=400, detail=f"Invalid metric for {task}. Must be one of: {METRICS[task]}")

        target_param = target
        horizon = horizon 
        
        if task == 'multistep_forecasting':
            if transformation_config_dict and transformation_config_dict.get('needs_transformation'):
                # For wide-format datasets, use horizon from transformation_config
                config_horizon = transformation_config_dict.get('horizon')
                if config_horizon and isinstance(config_horizon, int):
                    horizon = config_horizon
                    logger.info(f"Using horizon from transformation_config: {horizon}")
                else:
                  target_param = "target"
                  logger.info("Wide-format: target will be 'target' after transformation")

            # Validate horizon
            if horizon is None or horizon < 2 or horizon > 50:
                raise HTTPException(
                    status_code=400,
                    detail="Horizon must be between 2 and 50 steps for multistep forecasting (received: {horizon})"
                )
            
            targets_list = [t.strip() for t in target.split(',')]
            target_param = ", ".join(targets_list)
            
            logger.info(f"Multistep forecasting: {len(targets_list)} target(s), horizon={horizon}")
            
            available_models = ["xgboost", "lightgbm", "catboost"]
            models_to_train = None
            
            if models:
                models_list = [m.strip() for m in models.split(",")]
                invalid_models = [m for m in models_list if m not in available_models]
                valid_models = [m for m in models_list if m in available_models]
                
                if invalid_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid models for multistep forecasting: {invalid_models}. Valid models: {available_models}"
                    )
                
                models_to_train = valid_models if valid_models else None
        
        else:
            target_param = target
            horizon = None
            models_to_train = None
            
            if models:
                models_list = [m.strip() for m in models.split(",")]
                invalid_models = [m for m in models_list if m not in MODELS[task]]
                if invalid_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid models: {invalid_models}. Valid models for {task}: {MODELS[task]}"
                    )
                models_to_train = models_list
        
        if preprocessing_mode not in ["simple", "advanced"]:
            preprocessing_mode = "simple"
        
        if optuna_trials < 2 or optuna_trials > 50:
            optuna_trials = 5

        if test_size <= 0 or test_size >= 1:
            test_size = 0.2

        if (
            task == "multistep_forecasting"
            and transformation_config_dict
            and transformation_config_dict.get("needs_transformation")
        ):
            target_param = "target"
            logger.info(
                "Wide-format multistep forecasting detected. "
                "Setting target to 'target' after transformation decision."
            )
    

        payload = {
            "blob_file": train_blob,
            "task": task,
            "target": target_param,
            "metric": metric,
            "models": models_to_train,
            "test_blob": test_blob,
            "test_size": test_size,
            "time_budget": time_budget,
            "horizon": horizon,
            "preprocessing_config": {
                "mode": preprocessing_mode,
                "use_cleaning": use_cleaning
            },
            "optuna_config": {
                "use_optuna": use_optuna,
                "optuna_trials": optuna_trials
            },
            "source_type": resolved_source_type
        }
        
        if task == "multistep_forecasting":
            payload["horizon"] = horizon
            logger.info(f"Added horizon={horizon} to payload")
        
        if transformation_config:
            payload["transformation_config"] = transformation_config_dict
            logger.info("Added transformation_config to payload")

        logger.info(f"Triggering AutoML with payload: {payload}")

        response = requests.post(
            FUN1_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            # host.json functionTimeout is 00:10:00. A 300s client
            # timeout abandoned runs FUN1 was still executing.
            timeout=FUN1_TIMEOUT
        )
        
        if response.status_code != 200:
            error_detail = response.text[:500]
            logger.error(f"AutoML failed: {response.status_code} - {error_detail}")
            raise HTTPException(
                status_code=502,
                detail=f"Model training failed: {error_detail}"
            )
        
        result = response.json()
        
        run_id = result.get("run_id")
        best_model = result.get("best_model")
        train_metrics = result.get("train_metrics", {})
        test_metrics = result.get("test_metrics", {})
        all_models = result.get("all_models", {})
        data_quality = result.get("data_quality", {})
        clean_all_models = ModelRegistryHandler.normalize_all_models(all_models, task=task)

        if data_quality.get("warnings"):
            logger.info(f"[DATA QUALITY] {data_quality['warnings']}")

        if task == "multistep_forecasting":
            primary_metric_key = f"avg_{metric}" if not metric.startswith("avg_") else metric
            primary_score = test_metrics.get(primary_metric_key) or train_metrics.get(primary_metric_key)
        else:
            primary_score = test_metrics.get(metric) or train_metrics.get(metric)
        
        if primary_score is None:
            primary_score = "N/A"
        
        try:
            model_id = model_registry.register_model(
                user_id=user_id,
                user_email=user_email,
                run_id=run_id,
                task=task,
                target=target_param,
                model_name=best_model,
                metric=metric,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                all_models=clean_all_models,
                dataset_id=dataset_id,
                blob_path=result.get("run_path"),
                databricks_path=result.get("databricks_run_path"),
                snowflake_path=result.get("snowflake_run_path"),
                onelake_path=result.get("onelake_run_path"),
                original_filename=original_filename,
                original_blob_path=train_blob,
                horizon=horizon if task == "multistep_forecasting" else None,
                targets_list=target_param.split(", ") if task == "multistep_forecasting" else None,
                source_type=resolved_source_type,
                veriton_file_path=file_path
            )

            # Record task completion in dataset history
            if dataset_id:
                try:
                    dataset_service.record_task_completion(
                        dataset_id=dataset_id,
                        user_id=user_id,
                        task_type=task,
                        query=f"Built {task} model using {original_filename}",
                        thread_id=None,
                        status="completed",
                        model_id=model_id,
                        model_name=best_model,
                        results_path=result.get("run_path")
                    )
                    logger.info(f"Recorded model build task in dataset history")
                except Exception as e:
                    logger.warning(f"Failed to record task in dataset history: {e}")

            logger.info(f"Registered model: {model_id}")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
            model_id = None
        
        top_features = (
            result.get("feature_importance", {}).get("top_features") or
            result.get("top_features") or
            []
        )
        if isinstance(top_features, dict):
            top_features = list(top_features.keys())[:5]

        text_summary = handler.generate_build_model_text_summary(
            user_email=user_email,  
            task=task,
            best_model=best_model,
            primary_metric=metric,
            primary_score=primary_score,
            all_models=clean_all_models,
            dataset_name=original_filename,
            top_features=top_features,
            result_details=result
        )
        response_payload = {
            "status": "success",
            "message": "Model built successfully!",
            "dataset": original_filename,
            "dataset_id": dataset_id,
            "model_id": model_id,
            "task_type": task.replace("_", " ").title(),
            "best_model": best_model,
            "primary_metric": metric,
            "primary_score": primary_score,
            "all_models": clean_all_models,
            "text_summary": text_summary
        }

        if task == "multistep_forecasting":
            response_payload["horizon"] = horizon
            response_payload["targets"] = target_param.split(", ")
            logger.info(f"Response includes horizon={horizon}, targets={response_payload['targets']}")

        return response_payload
    
    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Model training timed out after 5 minutes"
        )
    except Exception as e:
        logger.error(f"Build model failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build model: {str(e)}"
        )

@app.post("/test_model_v")
async def test_model_v(
    file_path: str = Form(..., description="Relative path inside lakehouse, e.g. Files/Datasets/.../test.csv"),

    model_id: str = Form(...),
    user_email: str = Form(...),
    return_predictions: bool = Form(True),
    return_probabilities: bool = Form(False),
    save_predictions: bool = Form(True)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")
     
        user_id = user["user_id"]
        
        # Get model info from registry
        model_info = model_registry.get_model_by_id(model_id, user_id)
          
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )
       
        run_id = model_info["run_id"]
        task = model_info["task"]
        target = model_info["target"]
        model_name = model_info["model_name"]
        training_test_metrics = model_info.get("test_metrics", {})

        task_model_name = f"{task}_{model_name}"
        logger.info(f"Testing model: {model_name} (run_id: {run_id})")
        
        # Fetch test file from whichever source it lives in (Databricks
        # Volume or OneLake) - dispatch logic lives in app/source_fetch.py.
        original_filename = file_path.split("/")[-1] or "test_dataset.csv"

        try:
            test_content, resolved_source_type = fetch_source_file(
                file_path=file_path,
                databricks_handler=databricks_handler_global,
            )
        except SourceFileNotFoundError as e:
            raise HTTPException(404, str(e))
        except SourceCredentialsMissingError as e:
            raise HTTPException(503, str(e))

        class FetchedFile:
            def __init__(self, content: BytesIO, filename: str):
                self.file = content
                self.filename = filename
                self.content_type = "application/octet-stream"
        
        test_fetched = FetchedFile(test_content, original_filename)
        
        # Convert if needed (Parquet → CSV, etc.) — same as original
        processed_test_file, processed_test_filename, was_converted = await to_csv_if_needed(test_fetched)
        if was_converted:
            original_filename = processed_test_filename
        
  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_folder_name = f"{task}_{model_name}_test"
        test_blob = f"{user_id}/test_model/{test_folder_name}/{original_filename}"
        
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        test_blob_client = container_client.get_blob_client(test_blob)
        
        test_content.seek(0)  # reset for upload
        test_blob_client.upload_blob(test_content.read(), overwrite=True)
        logger.info(f"Uploaded test file: {test_blob}")
        
        
        # CHECK IF TEST FILE HAS TARGET COLUMN
        test_file_df = None
        has_ground_truth = False
        
        try:
            test_blob_client_read = container_client.get_blob_client(test_blob)
            data = test_blob_client_read.download_blob().readall()
           
            ext = os.path.splitext(test_blob)[1].lower()
            if ext == '.csv':
                test_file_df = pd.read_csv(io.BytesIO(data))
            elif ext in {'.xlsx', '.xls'}:
                test_file_df = pd.read_excel(io.BytesIO(data))
           
            if test_file_df is not None:
                if task not in ['clustering', 'anomaly_detection']:
                    has_ground_truth = target in test_file_df.columns
                    logger.info(f"Ground truth {'FOUND' if has_ground_truth else 'NOT FOUND'} in test file")
                else:
                    has_ground_truth = False
                    logger.info(f"Unsupervised task ({task}) - no ground truth check")
        
        except Exception as e:
            logger.warning(f"Could not read test file for ground truth check: {e}")
            has_ground_truth = False
        
        # === HANDLE MULTISTEP FORECASTING ===
        if task == "multistep_forecasting":
            logger.info("Multistep forecasting model - checking for history buffer")
            
            # Try to load history buffer
            history_buffer_path = f"{user_id}/runs/{run_id}/history_buffer.csv"
            history_blob_client = container_client.get_blob_client(history_buffer_path)
            
            has_history_buffer = False
            if history_blob_client.exists():
                logger.info(f"Found history buffer: {history_buffer_path}")
                has_history_buffer = True
            else:
                logger.warning(f"No history buffer found at {history_buffer_path}")
        
        test_result_id = None
        
        # SCENARIO 1: HAS GROUND TRUTH
        if has_ground_truth:
            logger.info("SCENARIO 1: Ground truth available - Computing fresh test metrics")

            # Separate features and target
            X_test = test_file_df.drop(columns=[target])
            y_test = test_file_df[target]
            
            # HANDLE FORECASTING AND MULTI-HORIZON SEPARATELY
            if task == 'multistep_forecasting':
                logger.info("Multistep forecasting - checking for history buffer")
                
                # Try to load history buffer
                history_buffer_path = f"{user_id}/runs/{run_id}/history_buffer.csv"
                history_blob_client = container_client.get_blob_client(history_buffer_path)
                
                if history_blob_client.exists():
                    logger.info(f"Found history buffer: {history_buffer_path}")
                    
                    # Download history buffer
                    history_data = history_blob_client.download_blob().readall()
                    history_df = pd.read_csv(io.BytesIO(history_data))
                    
                    logger.info(f"History buffer: {len(history_df)} rows")
                    
                    # Combine history + test data for feature engineering
                    df_combined = pd.concat([history_df, test_file_df], ignore_index=True)
                    logger.info(f"Combined: {len(history_df)} history + {len(test_file_df)} test = {len(df_combined)} total")
                    
                    # Use combined data for inference (function will handle feature engineering)
                    inference_blob = test_blob  # Original test file
                else:
                    logger.warning("No history buffer found - using test file directly")
                    inference_blob = test_blob

            elif task in ['forecasting']:
                inference_blob = test_blob
                logger.info(f"Forecasting task - using file: {inference_blob}")

            else:
                # For non-forecasting: create features-only file
                features_only_blob = f"{user_id}/test_model/{test_folder_name}/features_only_{original_filename}"
                features_blob_client = container_client.get_blob_client(features_only_blob)
                
                # Save as CSV
                features_buffer = io.BytesIO()
                X_test.to_csv(features_buffer, index=False)
                features_buffer.seek(0)
                features_blob_client.upload_blob(features_buffer, overwrite=True)
            
                logger.info(f"Saved features-only file: {features_only_blob}")
                inference_blob = features_only_blob
            
            payload = {
                "run_id": run_id,
                "data_blob": inference_blob,
                "return_proba": return_probabilities
            }
            logger.info(f"Triggering inference with payload: {payload}")
            
            response = requests.post(
                FUN2_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=FUN2_TIMEOUT
            )
            
            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"Inference failed: {response.status_code} - {error_detail}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Model testing failed: {error_detail}"
                )
            
            result = response.json()
            predictions = result["predictions"]["predictions"]

            # COMPUTE FRESH TEST METRICS
            fresh_test_metrics = {}
            try:
                if task == 'classification':
                    fresh_test_metrics['accuracy'] = float(accuracy_score(y_test, predictions))
                    fresh_test_metrics['f1'] = float(f1_score(y_test, predictions, average='weighted', zero_division=0))
                    fresh_test_metrics['precision'] = float(precision_score(y_test, predictions, average='weighted', zero_division=0))
                    fresh_test_metrics['recall'] = float(recall_score(y_test, predictions, average='weighted', zero_division=0))

                    if return_probabilities and "probabilities" in result:
                        try:
                            proba = np.array(result["probabilities"])
                            if proba.shape[1] == 2:
                                fresh_test_metrics['roc_auc'] = float(roc_auc_score(y_test, proba[:, 1]))
                            else:
                                fresh_test_metrics['roc_auc'] = float(roc_auc_score(y_test, proba, multi_class='ovr', average='weighted'))
                        except Exception as e:
                            logger.warning(f"Could not compute ROC-AUC: {e}")
               
                elif task == 'regression':
                    fresh_test_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
                    fresh_test_metrics['mae'] = float(mean_absolute_error(y_test, predictions))
                    fresh_test_metrics['r2'] = float(r2_score(y_test, predictions))
                    fresh_test_metrics['mse'] = float(mean_squared_error(y_test, predictions))
               
                elif task == 'forecasting':
                    fresh_test_metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, predictions)))
                    fresh_test_metrics['mae'] = float(mean_absolute_error(y_test, predictions))
                    fresh_test_metrics['r2'] = float(r2_score(y_test, predictions))

                elif task == 'multistep_forecasting':
                    # For multistep: predictions are 2D array
                    predictions_array = np.array(predictions)
                    y_test_array = y_test.values if isinstance(y_test, pd.DataFrame) else y_test
                    
                    if len(predictions_array.shape) == 1:
                        predictions_array = predictions_array.reshape(-1, 1)
                    
                    if len(y_test_array.shape) == 1:
                        y_test_array = y_test_array.reshape(-1, 1)
                    
                    # Compute average metrics across horizons
                    n_horizons = predictions_array.shape[1]
                    rmse_values = []
                    mae_values = []
                    r2_values = []
                    
                    for h in range(n_horizons):
                        y_h = y_test_array[:, h]
                        p_h = predictions_array[:, h]
                        
                        rmse_values.append(np.sqrt(mean_squared_error(y_h, p_h)))
                        mae_values.append(mean_absolute_error(y_h, p_h))
                        r2_values.append(r2_score(y_h, p_h))
                    
                    fresh_test_metrics['avg_rmse'] = float(np.mean(rmse_values))
                    fresh_test_metrics['avg_mae'] = float(np.mean(mae_values))
                    fresh_test_metrics['avg_r2'] = float(np.mean(r2_values))
                    
                    logger.info(f"Multistep metrics: RMSE={fresh_test_metrics['avg_rmse']:.4f}, "
                               f"MAE={fresh_test_metrics['avg_mae']:.4f}, R²={fresh_test_metrics['avg_r2']:.4f}")
               
                elif task == 'clustering':
                    if len(np.unique(predictions)) > 1:
                        fresh_test_metrics['silhouette_score'] = float(silhouette_score(X_test, predictions))
                        fresh_test_metrics['davies_bouldin_score'] = float(davies_bouldin_score(X_test, predictions))
                        fresh_test_metrics['calinski_harabasz'] = float(calinski_harabasz_score(X_test, predictions))
                    fresh_test_metrics['n_clusters'] = int(len(np.unique(predictions)))
               
                elif task == 'anomaly_detection':
                    pred_binary = (np.array(predictions) == -1).astype(int)
                    true_binary = (y_test == -1).astype(int) if y_test.dtype != bool else y_test.astype(int)
                    fresh_test_metrics['accuracy'] = float(accuracy_score(true_binary, pred_binary))
                    fresh_test_metrics['precision'] = float(precision_score(true_binary, pred_binary, zero_division=0))
                    fresh_test_metrics['recall'] = float(recall_score(true_binary, pred_binary, zero_division=0))
                    fresh_test_metrics['f1'] = float(f1_score(true_binary, pred_binary, zero_division=0))
                    fresh_test_metrics['n_anomalies'] = int(np.sum(pred_binary))
               
                logger.info(f"Computed fresh test metrics: {fresh_test_metrics}")
           
            except Exception as e:
                logger.error(f"Failed to compute metrics: {e}")
                fresh_test_metrics = {"error": "Could not compute metrics", "details": str(e)}
           
            # SAVE PREDICTIONS WITH GROUND TRUTH
            predictions_file_path = None
            predictions_filename = None
            predictions_databricks_path = None
            predictions_snowflake_path = None
            predictions_onelake_path = None
            predictions_mirror_path = None
            if save_predictions:
                try:
                    predictions_df = test_file_df.copy()
                    predictions_df[f'{target}_predicted'] = predictions
                   
                    if return_probabilities and "probabilities" in result and task == 'classification':
                        proba = np.array(result["probabilities"])
                        unique_classes = sorted(predictions_df[target].unique())
                        for idx, cls in enumerate(unique_classes):
                            if idx < proba.shape[1]:
                                predictions_df[f'{target}_prob_class_{cls}'] = proba[:, idx]
                   
                    original_filename = os.path.splitext(original_filename)[0]
                    predictions_filename = f"{original_filename}_with_predictions_{timestamp}.csv"
                    predictions_blob_path = f"{user_id}/predictions/{task_model_name}/{predictions_filename}"
                    # predictions_blob_path = f"{user_id}/predictions/{model_name}/{predictions_filename}"
                   
                    predictions_blob_client = container_client.get_blob_client(predictions_blob_path)
                    csv_buffer = io.BytesIO()
                    predictions_df.to_csv(csv_buffer, index=False)
                    csv_buffer.seek(0)
                    predictions_blob_client.upload_blob(csv_buffer, overwrite=True)
                    predictions_file_path = predictions_blob_path

                    # If the test file came from Databricks, Snowflake, or
                    # OneLake, also mirror the predictions into that same
                    # source. Blob remains the authoritative copy either way.
                    predictions_databricks_path = None
                    predictions_snowflake_path = None
                    predictions_onelake_path = None
                    if resolved_source_type == "databricks":
                        csv_buffer.seek(0)
                        predictions_databricks_path = write_back_to_databricks(
                            databricks_handler=databricks_handler_global,
                            source_file_path=file_path,
                            relative_suffix=predictions_blob_path,
                            content=csv_buffer.read(),
                        )
                    elif resolved_source_type == "snowflake":
                        csv_buffer.seek(0)
                        predictions_snowflake_path = write_back_to_snowflake(
                            snowflake_handler=get_snowflake_handler(),
                            source_file_path=file_path,
                            relative_suffix=predictions_blob_path,
                            content=csv_buffer.read(),
                        )
                    elif resolved_source_type == "onelake":
                        csv_buffer.seek(0)
                        predictions_onelake_path = write_back_to_onelake(
                            source_file_path=file_path,
                            relative_suffix=predictions_blob_path,
                            content=csv_buffer.read(),
                        )
                    predictions_mirror_path = predictions_databricks_path or predictions_snowflake_path or predictions_onelake_path
                    logger.info(f"Saved predictions to: {predictions_blob_path}")
               
                except Exception as e:
                    logger.error(f"Failed to save predictions: {e}", exc_info=True)
                    predictions_file_path = None
                    predictions_filename = None
                    predictions_databricks_path = None
                    predictions_snowflake_path = None
                    predictions_onelake_path = None
                    predictions_mirror_path = None
                    
            # Save test result to DB
            try:
                test_result_id = test_metrics_handler.save_test_result(
                    model_id=model_id,
                    user_id=user_id,
                    user_email=user_email,
                    test_file_name=original_filename,
                    test_blob_path=test_blob,
                    has_ground_truth=True,
                    test_metrics=fresh_test_metrics,
                    predictions_blob_path=predictions_file_path,
                    rows_tested=len(predictions),
                    notes=f"Test with ground truth - {task} on {target}",
                    model_name=model_name,
                    task=task,
                    target=target,
                    run_id=run_id,
                    source_type=resolved_source_type,
                    source_path=file_path,
                    predictions_source_path=predictions_mirror_path,
                )
                logger.info(f"Saved test result: {test_result_id}")
            except Exception as e:
                logger.warning(f"Failed to save test result: {e}")
            
            try:
                model_registry.increment_usage(model_id, user_id)
            except Exception as e:
                logger.warning(f"Failed to increment usage: {e}")
            
            response_data = {
                "status": "success",
                "test_result_id": test_result_id,
                "message": "Model tested successfully with ground truth!",
                "scenario": "ground_truth_available",
                "model_info": {
                    "model_id": model_id,
                    "model_name": model_name,
                    "task": task,
                    "target": target,
                    "run_id": run_id
                },
                "test_file": original_filename,
                "rows_tested": len(predictions),
                "has_ground_truth": True,
                "test_metrics": fresh_test_metrics,
                "training_test_metrics": training_test_metrics,
                # "metrics_comparison": {
                #     "note": "Fresh metrics computed from this test file vs. metrics from training",
                #     "fresh": fresh_test_metrics,
                #     "training": training_test_metrics
                # },
                "timestamp": result["timestamp"]
            }
            
            if predictions_file_path:
                response_data["predictions_file"] = {
                    "saved": True,
                    "blob_path": predictions_file_path,
                    "filename": predictions_filename
                }
            else:
                response_data["predictions_file"] = {"saved": False}
           
            if return_predictions:
                response_data["predictions"] = {"predicted": predictions, "actual": y_test.tolist()}
           
            if return_probabilities and "probabilities" in result:
                response_data["probabilities"] = result["probabilities"]
           
            if "top_features" in result:
                response_data["top_features"] = result["top_features"]

        # NO GROUND TRUTH
        else:
            logger.info("SCENARIO 2: No ground truth - Using training test metrics")
           
            payload = {
                "run_id": run_id,
                "data_blob": test_blob,
                "return_proba": return_probabilities
            }
           
            response = requests.post(
                FUN2_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=FUN2_TIMEOUT
            )
           
            if response.status_code != 200:
                error_detail = response.text[:500]
                logger.error(f"Inference failed: {response.status_code} - {error_detail}")
                raise HTTPException(status_code=502, detail=f"Model testing failed: {error_detail}")
           
            result = response.json()
            predictions = result["predictions"]["predictions"]
           
            predictions_file_path = None
            predictions_filename = None
            predictions_databricks_path = None
            predictions_snowflake_path = None
            predictions_onelake_path = None
            predictions_mirror_path = None
            
            if save_predictions:
                try:
                    predictions_df = test_file_df.copy() if test_file_df is not None else None
                    if predictions_df is not None:
                        if task == 'clustering':
                            predictions_df['cluster_assigned'] = predictions
                        elif task == 'anomaly_detection':
                            predictions_df['is_anomaly'] = (np.array(predictions) == -1).astype(int)
                            predictions_df['anomaly_label'] = predictions
                        else:
                            predictions_df[f'{target}_predicted'] = predictions
                       
                        if return_probabilities and "probabilities" in result and task == 'classification':
                            proba = np.array(result["probabilities"])
                            n_classes = proba.shape[1]
                            for idx in range(n_classes):
                                predictions_df[f'{target}_prob_class_{idx}'] = proba[:, idx]
                       
                        original_filename = os.path.splitext(original_filename)[0]
                        predictions_filename = f"{original_filename}_with_predictions_{timestamp}.csv"
                        predictions_blob_path = f"{user_id}/predictions/{task_model_name}/{predictions_filename}"
                        # predictions_blob_path = f"{user_id}/predictions/{model_name}/{predictions_filename}"
                       
                        predictions_blob_client = container_client.get_blob_client(predictions_blob_path)
                        csv_buffer = io.BytesIO()
                        predictions_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)
                        
                        predictions_blob_client.upload_blob(csv_buffer, overwrite=True)
                        predictions_file_path = predictions_blob_path
                        logger.info(f"Saved predictions to: {predictions_blob_path}")

                        # If the test file came from Databricks, Snowflake,
                        # or OneLake, also mirror the predictions into that
                        # same source. Blob remains the authoritative
                        # copy either way.
                        predictions_databricks_path = None
                        predictions_snowflake_path = None
                        predictions_onelake_path = None
                        if resolved_source_type == "databricks":
                            csv_buffer.seek(0)
                            predictions_databricks_path = write_back_to_databricks(
                                databricks_handler=databricks_handler_global,
                                source_file_path=file_path,
                                relative_suffix=predictions_blob_path,
                                content=csv_buffer.read(),
                            )
                        elif resolved_source_type == "snowflake":
                            csv_buffer.seek(0)
                            predictions_snowflake_path = write_back_to_snowflake(
                                snowflake_handler=get_snowflake_handler(),
                                source_file_path=file_path,
                                relative_suffix=predictions_blob_path,
                                content=csv_buffer.read(),
                            )
                        elif resolved_source_type == "onelake":
                            csv_buffer.seek(0)
                            predictions_onelake_path = write_back_to_onelake(
                                source_file_path=file_path,
                                relative_suffix=predictions_blob_path,
                                content=csv_buffer.read(),
                            )
                        predictions_mirror_path = predictions_databricks_path or predictions_snowflake_path or predictions_onelake_path
                
                except Exception as e:
                    logger.error(f"Failed to save predictions: {e}")
                    # predictions_file_path = None
                    predictions_databricks_path = None
                    predictions_snowflake_path = None
                    predictions_onelake_path = None
                    predictions_mirror_path = None
            
            try:
                model_registry.increment_usage(model_id, user_id)
            except Exception as e:
                logger.warning(f"Failed to increment usage: {e}")
           
            try:
                test_result_id = test_metrics_handler.save_test_result(
                    model_id=model_id,
                    user_id=user_id,
                    user_email=user_email,
                    test_file_name=original_filename,
                    test_blob_path=test_blob,
                    has_ground_truth=False,
                    test_metrics=training_test_metrics,
                    predictions_blob_path=predictions_file_path,
                    rows_tested=len(predictions),
                    notes="Inference only - no ground truth available",
                    model_name=model_name,
                    task=task,
                    target=target,
                    run_id=run_id,
                    source_type=resolved_source_type,
                    source_path=file_path,
                    predictions_source_path=predictions_mirror_path,
                )
                logger.info(f"Saved test result (no ground truth): {test_result_id}")
            except Exception as e:
                logger.error(f"Failed to save test result: {e}")
                test_result_id = None


            response_data = {
                "status": "success",
                "test_result_id": test_result_id,
                "message": "Model predictions generated successfully!",
                "scenario": "no_ground_truth",
                "model_info": {
                    "model_id": model_id,
                    "model_name": model_name,
                    "task": task,
                    "target": target,
                    "run_id": run_id
                },
                "test_file": original_filename,
                "rows_tested": len(predictions),
                "has_ground_truth": False,
                "test_metrics": training_test_metrics,
                "metrics_note": "These are test metrics from the training phase (no ground truth in current test file)",
                "timestamp": result["timestamp"]
            }
            
            if predictions_file_path:
                response_data["predictions_file"] = {
                    "saved": True,
                    "blob_path": predictions_file_path,
                    "filename": predictions_filename
                }
            else:
                response_data["predictions_file"] = {"saved": False}
           
            if return_predictions:
                response_data["predictions"] = result["predictions"]
           
            if return_probabilities and "probabilities" in result:
                response_data["probabilities"] = result["probabilities"]
           
            if "top_features" in result:
                response_data["top_features"] = result["top_features"]
        
        return response_data
   
    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Model testing timed out after 3 minutes")
    except Exception as e:
        logger.error(f"Test model failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to test model: {str(e)}")



def _explain_fun1_error(response=None, exc=None):
    detail = str(exc) if exc is not None else ""

    if response is not None:
        detail = f"FUN1 returned {response.status_code}: {response.text[:1500]}"
        try:
            body = response.json()
            reason = body.get("error") or body.get("details")
            if reason:
                reason = str(reason).replace("Pipeline execution failed:", "").strip()
                return reason, detail
        except Exception:
            pass

    low = detail.lower()
    if "rate_limit" in low or "rate limit" in low or "429" in low:
        return ("The AI service is temporarily rate-limited. Please wait a "
                "minute and try again. If this persists, raise the token-per-"
                "minute quota on the gpt-4.1-mini deployment in Azure."), detail
    if "timeout" in low or "timed out" in low:
        return ("Training took longer than the time limit. Try a smaller "
                "dataset, fewer models, or disable hyperparameter tuning."), detail
    if "connection" in low:
        return ("Could not reach the training service. Please try again in a "
                "moment."), detail

    return ("Model building failed. Please review your data and configuration "
            "and try again."), detail


def full_background_training_process(
    job_id: str,                   
    user_email: str,
    user_id: str,
    file_path: str,
    test_file_path: Optional[str],
    upload_file_path: bool,
    task: Optional[str],
    target: Optional[str],
    models: Optional[str],
    metric: Optional[str],
    horizon: Optional[int],
    transformation_config: Optional[str],
    preprocessing_mode: str,
    use_cleaning: bool,
    use_feature_selection: bool,
    use_optuna: bool,
    optuna_trials: int,
    time_budget: int,
    test_size: float
):
    try:
        logger.info(f"[BACKGROUND] Job started – job_id={job_id}")

        user = auth_handler.get_user(user_email)
        if not user:
            logger.error(f"[BACKGROUND] Unauthorized user – job_id={job_id}")
            return

        agent_id = None
        agents = user.get("agents", [])
        if agents and isinstance(agents[0], dict):
            agent_id = agents[0].get("agent_id")

        original_filename = file_path.split("/")[-1] or "train_dataset.csv"

        # Fetch training file from whichever source it lives in - dispatch
        # logic lives in app/source_fetch.py.
        test_blob = None
        try:
            train_content, resolved_source_type = fetch_source_file(
                file_path=file_path,
                databricks_handler=databricks_handler_global,
            )
        except (SourceFileNotFoundError, SourceCredentialsMissingError) as e:
            logger.error(f"[BACKGROUND] Training file not fetched: {e} – job_id={job_id}")
            return

        # Optional test file - non-fatal if missing, matching original behavior
        test_content = None
        if test_file_path:
            try:
                test_content, _ = fetch_source_file(
                    file_path=test_file_path,
                    databricks_handler=databricks_handler_global,
                    source_type=resolved_source_type,
                )
            except (SourceFileNotFoundError, SourceCredentialsMissingError) as e:
                logger.warning(f"[BACKGROUND] Optional test file not fetched: {e} – job_id={job_id}")

        content = train_content.getvalue()

        # Process training file
        class FetchedFile:
            def __init__(self, content: BytesIO, filename: str):
                self.file = content
                self.filename = filename
                self.content_type = "application/octet-stream"

        train_fetched = FetchedFile(train_content, original_filename)
        processed_file, processed_filename, was_converted = asyncio.run(to_csv_if_needed(train_fetched))
                                                        

        if was_converted:
            original_filename = processed_filename

        # Upload to blob – reset stream before upload
        train_blob = f"{user_id}/build_model/{original_filename}"
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        train_blob_client = container_client.get_blob_client(train_blob)

        train_content.seek(0)
        train_blob_client.upload_blob(train_content.read(), overwrite=True)
        logger.info(f"[BACKGROUND] Uploaded training file: {train_blob} – job_id={job_id}")

        if test_content:
            test_original_name = test_file_path.split("/")[-1] or "test_dataset.csv"
            test_blob = f"{user_id}/build_model/{test_original_name}"
            test_blob_client = container_client.get_blob_client(test_blob)

            test_content.seek(0)
            test_blob_client.upload_blob(test_content.read(), overwrite=True)
            logger.info(f"[BACKGROUND] Uploaded test file: {test_blob} – job_id={job_id}")

        # Dataset intelligence
        dataset_id = None
        if upload_file_path:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    train_content.seek(0)
                    tmp.write(train_content.read())
                    tmp_path = tmp.name

                try:
                    intelligence_result = dataset_service.process_new_upload(
                        file_path=tmp_path,
                        filename=original_filename,
                        user_id=user_id,
                        user_email=user_email,
                        agent_id=agent_id,
                        blob_path=train_blob,
                        min_similarity_threshold=60.0,
                        source_type=resolved_source_type,
                        source_path=file_path
                    )
                    dataset_id = intelligence_result.get("dataset_id")
                    logger.info(f"[BACKGROUND] Dataset intelligence processed: {dataset_id} – job_id={job_id}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"[BACKGROUND] Dataset intelligence failed – job_id={job_id}: {e}")

        # Validation & transformation config
        if not task or not task.strip():
            return
        task = task.lower().strip()
        if task not in TASKS:
            return

        transformation_config_dict = None
        if transformation_config:
            try:
                transformation_config_dict = json.loads(transformation_config)
                transformation_config_dict.setdefault("generated_code", None)
                transformation_config_dict.setdefault("code_metadata", None)
                transformation_config_dict.setdefault("needs_transformation", True)

                temp_df = pd.read_csv(io.BytesIO(content))

                has_year_column = bool(transformation_config_dict.get('year_column', '').strip())
                has_month_columns = bool(transformation_config_dict.get('month_columns'))

                if not has_year_column:
                    year_col = None
                    year_candidates = [col for col in temp_df.columns if 'year' in col.lower()]
                    if year_candidates:
                        year_col = year_candidates[0]

                    if not year_col:
                        datetime_cols = temp_df.select_dtypes(include=['datetime64']).columns.tolist()
                        if datetime_cols:
                            year_col = datetime_cols[0]
                        else:
                            for col in temp_df.columns:
                                try:
                                    pd.to_datetime(temp_df[col].dropna().head(5))
                                    year_col = col
                                    break
                                except:
                                    continue

                    if not year_col:
                        return

                    transformation_config_dict['year_column'] = year_col

                if not has_month_columns:
                    measures = transformation_config_dict.get('measures', [])
                    if not measures:
                        return
                    transformation_config_dict['month_columns'] = measures

                ts_transformer = TimeseriesTransformer()
                transformation_config_dict['needs_transformation'] = True

                is_valid, error_msg = ts_transformer.validate_transformation_config(
                    temp_df, transformation_config_dict
                )
                if not is_valid:
                    return

                try:
                    llm_transformer = get_llm_transformer()
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as tmp_input:
                        tmp_input.write(content)
                        tmp_input_path = tmp_input.name

                    try:
                        analysis = ts_transformer.analyze_dataset_structure(temp_df)
                        generated_code, code_metadata = llm_transformer.generate_transformation_code(
                            file_path=tmp_input_path,
                            analysis=analysis,
                            transformation_config=transformation_config_dict
                        )
                        transformation_config_dict['generated_code'] = generated_code
                        transformation_config_dict['code_metadata'] = code_metadata
                    finally:
                        if os.path.exists(tmp_input_path):
                            os.remove(tmp_input_path)

                except Exception as llm_error:
                    logger.warning(f"[BACKGROUND] LLM transformation failed – job_id={job_id}: {llm_error}")
                    transformation_config_dict['generated_code'] = ""
                    transformation_config_dict['needs_transformation'] = True

            except Exception as e:
                logger.warning(f"[BACKGROUND] Transformation config failed – job_id={job_id}: {e}")
                return

        if not metric:
            metric = DEFAULT_METRICS[task]
        elif metric not in METRICS[task]:
            return

        target_param = target
        horizon = horizon

        if task == 'multistep_forecasting':
            if transformation_config_dict and transformation_config_dict.get('needs_transformation'):
                config_horizon = transformation_config_dict.get('horizon')
                if config_horizon and isinstance(config_horizon, int):
                    horizon = config_horizon
                else:
                    target_param = "target"

            if horizon is None or horizon < 2 or horizon > 50:
                return

            targets_list = [t.strip() for t in target.split(',')]
            target_param = ", ".join(targets_list)

            available_models = ["xgboost", "lightgbm", "catboost"]
            models_to_train = None
            if models:
                models_list = [m.strip() for m in models.split(",")]
                invalid_models = [m for m in models_list if m not in available_models]
                valid_models = [m for m in models_list if m in available_models]
                if invalid_models:
                    return
                models_to_train = valid_models if valid_models else None

        else:
            target_param = target
            horizon = None
            models_to_train = None
            if models:
                models_list = [m.strip() for m in models.split(",")]
                invalid_models = [m for m in models_list if m not in MODELS[task]]
                if invalid_models:
                    return
                models_to_train = models_list

        if preprocessing_mode not in ["simple", "advanced"]:
            preprocessing_mode = "simple"

        if optuna_trials < 2 or optuna_trials > 50:
            optuna_trials = 3

        if test_size <= 0 or test_size >= 1:
            test_size = 0.2

        if task == "multistep_forecasting" and transformation_config_dict and transformation_config_dict.get("needs_transformation"):
            target_param = "target"

        payload = {
            "blob_file": train_blob,
            "task": task,
            "target": target_param,
            "metric": metric,
            "models": models_to_train,
            "test_blob": test_blob,
            "test_size": test_size,
            "time_budget": time_budget,
            "horizon": horizon,
            "preprocessing_config": {
                "mode": preprocessing_mode,
                "use_cleaning": use_cleaning
            },
            "optuna_config": {
                "use_optuna": use_optuna,
                "optuna_trials": optuna_trials
            },
            "source_type": resolved_source_type
        }

        if task == "multistep_forecasting":
            payload["horizon"] = horizon

        if transformation_config:
            payload["transformation_config"] = transformation_config_dict
        
        try:
            logger.info(f"[TRAINING] Calling FUN1_URL | job_id={job_id}")

            response = requests.post(
            FUN1_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=FUN1_TIMEOUT
        )

            if response.status_code != 200:
                user_msg, tech_detail = _explain_fun1_error(response=response)
                raise RuntimeError(user_msg) from Exception(tech_detail)
                
            result = response.json()

        except Exception as automl_error:
            user_msg, tech_detail = _explain_fun1_error(exc=automl_error)
            logger.error(f"[TRAINING FAILED] {user_msg} | job_id={job_id}")
            logger.error(traceback.format_exc())

            model_registry.register_failed_job(
                job_id=job_id,
                user_id=user_id,
                user_email=user_email,
                file_path=file_path,
                veriton_file_path=file_path,
                task=task,
                target=target,
                error_message=user_msg,
                error_detail=tech_detail or str(automl_error),
                source_type=resolved_source_type if 'resolved_source_type' in locals() else None
            )
        
            return
        
        # Get the REAL AutoML run_id
        auto_ml_run_id = result.get("run_id")
        if not auto_ml_run_id:
            logger.error(f"[BACKGROUND] FUN1_URL did not return run_id – job_id={job_id}")
            model_registry.register_failed_job(
                job_id=job_id, user_id=user_id, user_email=user_email, 
                file_path=file_path, veriton_file_path=file_path, task=task, error_message="No run_id returned from FUN1",
                source_type=resolved_source_type if 'resolved_source_type' in locals() else None
            )
            return

        logger.info(f"[BACKGROUND] AutoML run completed – auto_ml_run_id={auto_ml_run_id} – job_id={job_id}")

        best_model = result.get("best_model")
        train_metrics = result.get("train_metrics", {})
        test_metrics = result.get("test_metrics", {})
        all_models = result.get("all_models", {})
        data_quality = result.get("data_quality", {})
        clean_all_models = ModelRegistryHandler.normalize_all_models(all_models, task=task)

        if data_quality.get("warnings"):
            logger.info(f"[DATA QUALITY] {data_quality['warnings']}")

        if task == "multistep_forecasting":
            primary_metric_key = f"avg_{metric}" if not metric.startswith("avg_") else metric
            primary_score = test_metrics.get(primary_metric_key) or train_metrics.get(primary_metric_key)
        else:
            primary_score = test_metrics.get(metric) or train_metrics.get(metric)

        if primary_score is None:
            primary_score = "N/A"


        top_features = (
            result.get("feature_importance", {}).get("top_features") or
            result.get("top_features") or []
        )
        if isinstance(top_features, dict):
            top_features = list(top_features.keys())[:5]

        text_summary = handler.generate_build_model_text_summary(
            user_email=user_email,
            task=task,
            best_model=best_model,
            primary_metric=metric,
            primary_score=primary_score,
            all_models=clean_all_models,
            dataset_name=original_filename,
            top_features=top_features,
            result_details=result
        )

        model_id = None
        try:
            model_id = model_registry.register_model(
                user_id=user_id,
                user_email=user_email,
                run_id=auto_ml_run_id,                 
                job_id=job_id,                         
                task=task,
                target=target_param,
                model_name=best_model,
                metric=metric,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                all_models=clean_all_models,
                dataset_id=dataset_id,
                blob_path=result.get("run_path"),
                databricks_path=result.get("databricks_run_path"),
                snowflake_path=result.get("snowflake_run_path"),
                onelake_path=result.get("onelake_run_path"),
                original_filename=original_filename,
                original_blob_path=train_blob,
                horizon=horizon if task == "multistep_forecasting" else None,
                targets_list=target_param.split(", ") if task == "multistep_forecasting" else None,
                text_summary=text_summary,
                veriton_file_path=file_path,
                source_type=resolved_source_type
            )
            logger.info(f"[BACKGROUND] Model registered – model_id={model_id}, job_id={job_id}, auto_ml_run_id={auto_ml_run_id}")
        except Exception as e:
            logger.warning(f"[BACKGROUND] Failed to register model – job_id={job_id}: {e}")

        if dataset_id:
            try:
                dataset_service.record_task_completion(
                    dataset_id=dataset_id,
                    user_id=user_id,
                    task_type=task,
                    query=f"Built {task} model using {original_filename}",
                    thread_id=None,
                    status="completed",
                    model_id=model_id,
                    model_name=best_model,
                    results_path=result.get("run_path")
                )
            except Exception as e:
                logger.warning(f"[BACKGROUND] Failed to record task – job_id={job_id}: {e}")

        logger.info(f"[BACKGROUND] Job fully completed – job_id={job_id}, auto_ml_run_id={auto_ml_run_id}")

    except Exception as e:
        logger.error(f"[BACKGROUND] Job crashed – job_id={job_id}: {traceback.format_exc()}")
        model_registry.register_failed_job(
            job_id=job_id,
            user_id=user_id,
            user_email=user_email,
            file_path=file_path,
            veriton_file_path=file_path,
            task=task,
            error_message="Unexpected error in background job",
            error_detail=str(e),
            source_type=resolved_source_type if 'resolved_source_type' in locals() else None
        )

@app.post("/build_ml_model_v")
async def build_ml_model_v(
    file_path: str = Form(..., description="Relative path"),
    test_file_path: Optional[str] = Form(None, description="Optional relative test file path"),
    upload_file_path: bool = Form(False),
    user_email: str = Form(...),
    task: Optional[str] = Form(None),
    target: Optional[str] = Form(None),
    models: Optional[str] = Form(None),
    metric: Optional[str] = Form(None),
    horizon: Optional[int] = Form(12),
    transformation_config: Optional[str] = Form(None),
    preprocessing_mode: Optional[str] = Form("simple"),
    use_cleaning: Optional[bool] = Form(True),
    use_feature_selection: Optional[bool] = Form(False),
    use_optuna: Optional[bool] = Form(True),
    optuna_trials: Optional[int] = Form(2),
    time_budget: Optional[int] = Form(300),
    test_size: Optional[float] = Form(0.2)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")

        user_id = user["user_id"]
        agents = user.get("agents", [])
        agent_id = None
        if agents and isinstance(agents[0], dict):
            agent_id = agents[0].get("agent_id")

        if upload_file_path == False:

            # Stable job_id – this is what client will poll with forever
            job_id = str(uuid.uuid4())
            logger.info(f"POST received – starting job – job_id={job_id}")

            threading.Thread(
            target=full_background_training_process,
            kwargs={
                "job_id": job_id,
                "user_email": user_email,
                "user_id": user_id,
                "file_path": file_path,
                "test_file_path": test_file_path,
                "upload_file_path": upload_file_path,
                "task": task,
                "target": target,
                "models": models,
                "metric": metric,
                "horizon": horizon,
                "transformation_config": transformation_config,
                "preprocessing_mode": preprocessing_mode,
                "use_cleaning": use_cleaning,
                "use_feature_selection": use_feature_selection,
                "use_optuna": use_optuna,
                "optuna_trials": optuna_trials,
                "time_budget": time_budget,
                "test_size": test_size
            },
            daemon=True
        ).start()

            return {
                "status": "model has started running",
                "job_id": job_id,
                "message": "Training job started successfully in background. This usually takes 1–5 minutes.",
                # "poll_url": f"/training-status/{job_id}?user_email={user_email}",
                "poll_every_seconds": 15,
                # "note": "Keep polling using this job_id until status = 'success' to get full details (model_id, best_model, primary_score, text_summary, etc.)"
            }

        original_filename = file_path.split("/")[-1] or "train_dataset.csv"

        # Fetch training file (and optional test file) from whichever
        # source they live in - dispatch logic lives in app/source_fetch.py.
        test_blob = None
        try:
            train_content, test_content, resolved_source_type = fetch_train_and_optional_test_file(
                file_path=file_path,
                test_file_path=test_file_path,
                databricks_handler=databricks_handler_global,
            )
        except SourceFileNotFoundError as e:
            raise HTTPException(404, str(e))
        except SourceCredentialsMissingError as e:
            raise HTTPException(503, str(e))

        # Process training file
        class FetchedFile:
            def __init__(self, content: BytesIO, filename: str):
                self.file = content
                self.filename = filename
                self.content_type = "application/octet-stream"

        train_fetched = FetchedFile(train_content, original_filename)
        processed_file, processed_filename, was_converted = await to_csv_if_needed(train_fetched)

        if was_converted:
            original_filename = processed_filename

        # Upload to blob
        train_blob = f"{user_id}/build_model/{original_filename}"
        container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
        train_blob_client = container_client.get_blob_client(train_blob)

        train_content.seek(0)
        train_blob_client.upload_blob(train_content.read(), overwrite=True)
        logger.info(f"Processed training file uploaded: {train_blob}")

        if test_content:
            test_original_name = test_file_path.split("/")[-1] or "test_dataset.csv"
            test_blob = f"{user_id}/build_model/{test_original_name}"
            test_blob_client = container_client.get_blob_client(test_blob)
            test_content.seek(0)
            test_blob_client.upload_blob(test_content.read(), overwrite=True)
            logger.info(f"Test file uploaded: {test_blob}")

        # Dataset intelligence + feature suggestions
        dataset_id = None
        analysis_metadata = None
        feature_suggestions = None

        if upload_file_path:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    train_content.seek(0)
                    tmp.write(train_content.read())
                    tmp_path = tmp.name

                try:
                    intelligence_result = dataset_service.process_new_upload(
                        file_path=tmp_path,
                        filename=original_filename,
                        user_id=user_id,
                        user_email=user_email,
                        agent_id=agent_id,
                        blob_path=train_blob,
                        min_similarity_threshold=60.0,
                        source_type=resolved_source_type,
                        source_path=file_path
                    )
                    dataset_id = intelligence_result.get("dataset_id")
                    analysis_metadata = intelligence_result.get("analysis_metadata", {})
                    logger.info(f"Dataset intelligence processed: {dataset_id}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Dataset intelligence failed (non-blocking): {e}")

            try:
                feature_suggestions = handler.get_all_task_feature_suggestions(
                    blob_path=train_blob,
                    user_id=user_id,
                    max_preview_rows=200
                )
                logger.info(f"Generated feature suggestions – {len(feature_suggestions.get('columns', {}).get('all', []))} columns")
            except Exception as e:
                logger.warning(f"Feature suggestions failed: {str(e)}")
                feature_suggestions = {"status": "failed", "error": str(e)}

            # Return immediately – exact match to code1
            return {
                "status": "file_saved",
                "message": "File uploaded successfully! Ready for model building later.",
                "blob_path": train_blob,
                "filename": original_filename,
                "ready_for_training": True,
                "dataset_id": dataset_id,
                "analysis_metadata": {
                    "dataset_structure": analysis_metadata,
                    "intelligence_processed": dataset_id is not None
                },
                "features": feature_suggestions
            }


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start job: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to start model training: {str(e)}")


@app.get("/training-status/{job_id}")
async def get_training_status(
    job_id: str,
    user_email: str = Query(...)
):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")

        user_id = user["user_id"]

        # Query model using job_id (you need to implement this in model_registry)
        model_info = model_registry.get_model_by_job_id(job_id, user_id)

        if model_info:
            if model_info.get("status") == "failed":
                return {
                    "job_id": job_id,
                    "status": "failed",
                    # error_message is now the actionable reason from
                    # FUN1, not a generic 'unexpected issue'.
                    "message": model_info.get("error_message", "Training failed"),
                    "error": model_info.get("error_detail"),
                    "task": model_info.get("task"),
                    "target": model_info.get("target"),
                    "file_path": model_info.get("veriton_file_path")
                }
            
            # Success case
            response_payload = {
                "status": "success",
                "message": "Model built successfully!",
                "job_id": job_id,
                "model_id": model_info["model_id"],
                "auto_ml_run_id": model_info.get("run_id"),  # return the AutoML ID too
                "dataset": model_info.get("original_filename", "Unknown"),
                "dataset_id": model_info.get("dataset_id"),
                "task_type": model_info["task"].replace("_", " ").title(),
                "best_model": model_info["model_name"],
                "primary_metric": model_info["metric"],
                "primary_score": model_info.get("test_metrics", {}).get(model_info["metric"], "N/A"),
                "all_models": model_info["all_models"],
                "text_summary": model_info.get("text_summary")
            }

            if model_info["task"] == "multistep_forecasting":
                response_payload["horizon"] = model_info.get("horizon")
                response_payload["targets"] = model_info.get("targets", [])

            return response_payload

        else:
            return {
                "job_id": job_id,
                "status": "running",
                "model_id": None,
                "message": "Model training is still in progress...",
                "last_checked": datetime.utcnow().isoformat()
            }

    except Exception as e:
        logger.error(f"Status check failed for job_id={job_id}: {str(e)}")
        return {
            "job_id": job_id,
            "status": "error",
            "message": "Failed to check training status",
            "error_detail": str(e)
        }


@app.post("/process_task_query_v")
def process_task_query_endpoint_v(
    session_id: str = Form(...),
    query: str = Form(...),
    user_email: str = Form(...)
):
    job_id = str(uuid.uuid4())

    threading.Thread(
        target=full_background_process_task_query,
        kwargs={
            "job_id": job_id,
            "session_id": session_id,
            "query": query,
            "user_email": user_email
        },
        daemon=True
    ).start()

    return {
        "status": "started",
        "job_id": job_id,
        "message": "Your query is being processed in the background. This usually takes 1–5 minutes.",
        "poll_every_seconds": 10
    }

TASK_QUERY_RESULTS: dict = {} 
def full_background_process_task_query(
    job_id: str,
    session_id: str,
    query: str,
    user_email: str
):
    try:
        def get_dynamic_suggestion():
            # Delegates to the shared, schema-grounded suggestion generator so a
            # failed/"no data found" answer never gets followed by a suggestion
            # that references a column or field that doesn't actually exist in
            # this dataset (it used to hallucinate names like 'plant_plantname'
            # from an unrelated column such as 'employee_employeename').
            try:
                return handler.get_dynamic_suggestion(
                    thread_id=thread_id,
                    user_email=user_email,
                    agent_info=agent_info,
                    analysis_text=analysis_text or "",
                    user_id=user_id,
                    agent_name=agent_name,
                )
            except Exception:
                return "Ask: what's next?"
       
        user = auth_handler.get_user(user_email)
        if not user or len(user.get("agents", [])) == 0:
            raise HTTPException(status_code=400, detail="No agent")
       
        agent_info = user["agents"][0]
        if session_id not in agent_info.get("threads", []):
            raise HTTPException(status_code=403, detail="Invalid thread")
        user_id = user["user_id"]
        agent_name = agent_info["agent_name"]
        thread_id = session_id
        blob_file = handler.get_latest_blob_file(user_id, agent_name)
        dataset_id = thread_metadata_handler.get_current_dataset_id(
            thread_id=thread_id,
            user_id=user_id
        )
        veriton_file_path = None
        try:
            thread_doc = thread_metadata_handler.container.read_item(
                item=thread_id, 
                partition_key=user_id
            )
            veriton_file_path = thread_doc.get("metadata", {}).get("veriton_file_path")
        except:
            veriton_file_path = None
       
        if dataset_id:
            logger.info(f"Using dataset: {dataset_id} for thread: {thread_id}")
        else:
            logger.warning(f"No dataset found for thread: {thread_id}")
        # STEP 1: Run agent to detect intent
        # Wrap the raw question with reasoning-quality rules so plain-language
        # answers actually drill into the data instead of returning an
        # unrelated stat, a raw unsorted dump, or a bare value with no
        # context. This does not affect ML-task detection - the model still
        # sees the user's real question first and can still emit the is_ml
        # control JSON when appropriate.
        filename = blob_file.split('/')[-1] if blob_file else None
        QUERY_ANALYSIS_RULES = _build_query_analysis_rules(filename)
        thread_id, ml_json, analysis_text = handler.process_query_on_main_agent(
            agent_id=agent_info["agent_id"],
            query=f"{query}{QUERY_ANALYSIS_RULES}",
            raw_query=query,
            thread_id=thread_id,
            user_id=user_id,
            agent_name=agent_name
        )

        # Strip any leaked "NEXT_STEPS: a | b | c" control marker before it
        # ever reaches the user - suggestions for a chat query are generated
        # separately via get_dynamic_suggestion(). Prefer the model's own
        # steps as suggestions when it does supply them.
        extracted_steps = []
        if not (ml_json and ml_json.get("is_ml")) and analysis_text:
            analysis_text, extracted_steps = split_next_steps(analysis_text, None)

        # Defensive net: if the agent still leaks a bare control token instead
        # of a real sentence (seen previously as a literal "CANNOT_ANSWER"
        # response reaching the user), replace it with an honest, grounded
        # message that names the real available columns instead of silently
        # showing the placeholder or a hallucinated guess.
        if analysis_text and analysis_text.strip().strip(".").upper() in (
            "CANNOT_ANSWER", "CANNOT ANSWER", "I CANNOT ANSWER", "N/A", "NONE"
        ):
            try:
                available_cols = handler.get_dataset_columns(blob_file, user_id) if blob_file else []
            except Exception:
                available_cols = []
            if available_cols:
                col_preview = ", ".join(available_cols[:15])
                analysis_text = (
                    "I couldn't find data in this dataset that answers that "
                    f"question directly. The columns available are: {col_preview}. "
                    "Could you rephrase your question using one of these?"
                )
            else:
                analysis_text = (
                    "I couldn't find data in this dataset that answers that "
                    "question directly. Could you rephrase it or tell me which "
                    "column you mean?"
                )

        # Defensive net: if the model excused itself out of actually reading
        # the dataset ("Data access limitation in this session prevents me
        # from loading...") and answered with a hypothetical plan instead of
        # a real computed answer. Force a fresh re-attachment and retry ONCE.
        if analysis_text and looks_like_access_excuse(analysis_text):
            logger.warning(f"Access excuse detected for {filename}; forcing re-attachment and retrying once")
            try:
                if hasattr(handler, 'thread_metadata_handler') and handler.thread_metadata_handler and user_id:
                    try:
                        handler.thread_metadata_handler.add_metadata(thread_id, user_id, "attached_file_ids", [])
                    except Exception:
                        pass
                handler._ensure_file_attached_to_thread(thread_id, user_id, agent_info["agent_id"], filename)
                _, ml_json_retry, analysis_text_retry = handler.process_query_on_main_agent(
                    agent_id=agent_info["agent_id"],
                    query=f"{query}{QUERY_ANALYSIS_RULES}",
                    raw_query=query,
                    thread_id=thread_id,
                    user_id=user_id,
                    agent_name=agent_name
                )
                if analysis_text_retry and not looks_like_access_excuse(analysis_text_retry):
                    analysis_text = analysis_text_retry
                    if ml_json_retry:
                        ml_json = ml_json_retry
                else:
                    analysis_text = (
                        f"I had trouble reading {filename or 'your dataset'} just now "
                        "and couldn't compute a real answer. Please try asking again "
                        "in a moment."
                    )
            except Exception as retry_err:
                logger.warning(f"Retry after access excuse failed: {retry_err}")
                analysis_text = (
                    f"I had trouble reading {filename or 'your dataset'} just now "
                    "and couldn't compute a real answer. Please try asking again "
                    "in a moment."
                )

        # Defensive net: if the agent returned a leaked task description
        # (e.g. "Compute sum(productioncost) across all orders." instead of
        # the actual total) OR asked the user a question back instead of
        # answering (e.g. "What price per unit should be used for revenue?"),
        # try to answer from the dataset directly via the rule-based engine
        # before falling back to a retry message.
        if not (ml_json and ml_json.get("is_ml")) and analysis_text and (
            looks_like_leaked_ml_intent(analysis_text)
            or looks_like_agent_asking_question(analysis_text)
        ):
            logger.warning(f"Bad agent response detected (leaked task or question-back); trying rule-based answer.")
            try:
                rule_answer = handler._answer_non_ml_query(
                    query, agent_info["agent_id"], thread_id, user_id, agent_name
                )
                if rule_answer and not looks_like_agent_asking_question(rule_answer):
                    analysis_text = rule_answer
                else:
                    analysis_text = (
                        "I wasn't able to compute a real answer for that just now. "
                        "Please try asking again in a moment."
                    )
            except Exception:
                analysis_text = (
                    "I wasn't able to compute a real answer for that just now. "
                    "Please try asking again in a moment."
                )

        # STEP 2: Handle ML Task
        if ml_json and ml_json.get("is_ml"):
           
            if not blob_file:
                error_msg = "ML task requires a dataset. Please upload a file first."
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", error_msg)
                final_result = {
                    "session_id": thread_id,
                    "task_type": ml_json.get("task_type"),
                    "response": error_msg,
                    "analysis": None,
                    "results": None,
                    "dataset_id": dataset_id,
                    "suggestions": ["Upload a CSV or Excel file"]
                }
           
            task_type = ml_json.get("task_type", "").lower().strip()
            target_raw = ml_json.get("target", "").strip()
            metric = ml_json.get("metric")
            models_requested = ml_json.get("models", [])
           
            # MULTISTEP FORECASTING
            if task_type == "multistep_forecasting":
                horizon = ml_json.get("horizon", 12)
               
                if not isinstance(horizon, int) or horizon < 2:
                    clarification_msg = (
                        f"Invalid horizon value: {horizon}. "
                        f"For multi-step forecasting, horizon must be ≥ 2. "
                        f"Example: 'predict sales for next 12 months' (horizon=12)"
                    )
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", clarification_msg)
                    final_result = {
                        "session_id": thread_id,
                        "task_type": task_type,
                        "response": clarification_msg,
                        "needs_clarification": True,
                        "dataset_id": dataset_id,
                        "suggestions": ["Specify horizon ≥ 2"]
                    }
               
                validation = handler.parse_and_validate_targets(target_raw, blob_file, user_id)
               
                if not validation["valid"]:
                    clarification_msg = validation["message"]
                    if validation.get("similar_columns"):
                        similar_str = "', '".join(validation["similar_columns"][:5])
                        clarification_msg += f"\n\nDid you mean: '{similar_str}'?"
                   
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", clarification_msg)
                   
                    final_result = {
                        "status": "success",
                        "session_id": thread_id,
                        "task_type": task_type,
                        "response": clarification_msg,
                        "needs_clarification": True,
                        "available_columns": validation["available_columns"][:20],
                        "dataset_id": dataset_id,
                        "suggestions": ["Specify valid target column(s)"]
                    }

                    TASK_QUERY_RESULTS[job_id] = final_result
                    model_registry.register_query_result(
                        job_id=job_id,
                        user_id=user_id,
                        user_email=user_email,
                        task=task_type,
                        result=final_result,
                        session_id=thread_id,
                    )
                    return

                target = ", ".join(validation["targets"])
                logger.info(f"Validated targets: {target} (horizon={horizon})")
               
                available_models = ["xgboost", "lightgbm", "catboost"]
                models_to_train = []
                invalid_models = []
               
                if models_requested:
                    for model in models_requested:
                        if model in available_models:
                            models_to_train.append(model)
                        else:
                            invalid_models.append(model)
               
                model_warning = ""
                if invalid_models:
                    model_warning = (
                        f"Note: Models {invalid_models} are not available for multi-step forecasting. "
                        f"Building with: {models_to_train or available_models}"
                    )
                    logger.warning(model_warning)
               
                if not models_to_train:
                    models_to_train = None
           
            # HANDLE OTHER SUPERVISED TASKS
            elif task_type not in ['clustering', 'anomaly_detection']:
                validation = handler.validate_target_column(target_raw, blob_file, user_id)
           
                if not validation["valid"]:
                    possible_targets = handler.extract_targets_from_query(
                        query,
                        validation["available_columns"]
                    )
                   
                    clarification_msg = handler.ask_user_to_clarify_target(
                        possible_targets,
                        validation["available_columns"]
                    )
                   
                    if not clarification_msg:
                        clarification_msg = validation["message"]
                        if validation.get("similar_columns"):
                            similar_str = "', '".join(validation["similar_columns"])
                            clarification_msg += f"\n\nDid you mean one of these? '{similar_str}'"
                   
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                    response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", clarification_msg)
                   
                    final_result = {
                        "status": "success",
                        "session_id": thread_id,
                        "task_type": task_type,
                        "response": clarification_msg,
                        "analysis": None,
                        "results": None,
                        "dataset_id": dataset_id,
                        "needs_clarification": True,
                        "available_columns": validation["available_columns"][:20],
                        "suggestions": ["Specify target column clearly"]
                    }

                    TASK_QUERY_RESULTS[job_id] = final_result
                    model_registry.register_query_result(
                        job_id=job_id,
                        user_id=user_id,
                        user_email=user_email,
                        task=task_type,
                        result=final_result,
                        session_id=thread_id,
                    )
                    return

                # Target is valid - matched_column is guaranteed present here.
                target = validation["matched_column"]
                horizon = None
                models_to_train = models_requested or None
                model_warning = ""
                logger.info(f"Validated target: '{target}' for task: {task_type}")
           
            # UNSUPERVISED TASKS
            else:
                target = target_raw if target_raw and target_raw.lower() != 'none' else None
                horizon = None
                models_to_train = models_requested or None
                model_warning = ""
                logger.info(f"Unsupervised task ({task_type}) - target: {target}")
            # TRIGGER AUTOML
            try:
                # Look up how this dataset originally got into Blob (e.g.
                # "databricks", "onelake", "upload") so run artifacts can
                # also be mirrored back to the right source, same as the
                # explicit /build_ml_model* endpoints already do.
                source_type_for_run = None
                if dataset_id:
                    try:
                        dataset_doc = dataset_service.metadata_handler.get_dataset_by_id(dataset_id, user_id)
                        if dataset_doc:
                            source_type_for_run = dataset_doc.get("source_type")
                    except Exception as e:
                        logger.warning(f"Could not look up dataset source_type: {e}")

                # Fallback: a new session has no thread-linked dataset_id
                # yet, but blob_file (looked up per user+agent, not per
                # session) may still be the same physical file uploaded
                # from Databricks/OneLake in an earlier session. Match on
                # blob_path directly so write-back doesn't silently regress
                # just because this is a fresh session.
                if not source_type_for_run and blob_file:
                    try:
                        dataset_doc = dataset_service.metadata_handler.get_dataset_by_blob_path(user_id, blob_file)
                        if dataset_doc:
                            source_type_for_run = dataset_doc.get("source_type")
                    except Exception as e:
                        logger.warning(f"Could not look up dataset source_type by blob_path: {e}")

                results_filename = handler._trigger_automl_internal(
                    blob_file=blob_file,
                    query=query,
                    time_budget=300,
                    user_id=user_id,
                    task_type=task_type,
                    target=target,
                    horizon=horizon,
                    models=models_to_train,
                    source_type=source_type_for_run
                )
                container_client = handler.blob_service.get_container_client(UPLOAD_CONTAINER)
                blob_client = container_client.get_blob_client(results_filename)
                results_data = json.loads(blob_client.download_blob().readall().decode('utf-8'))
                # The model has already been trained and saved at this point.
                # A failure in the (cosmetic) LLM summary must NOT turn a
                # successful run into a failed job - that is what produced
                # "Model building failed" on a run FUN1 completed in 183s.
                try:
                    analysis_text = handler.generate_ml_analysis(
                        agent_id=agent_info["agent_id"],
                        results_data=results_data,
                        thread_id=thread_id
                    )
                except Exception as analysis_error:
                    logger.warning(
                        f"AI analysis unavailable, using fallback summary: "
                        f"{analysis_error}"
                    )
                    analysis_text = AgentHandler._fallback_summary(results_data)

                # Short, direct answer to the user's actual question, shown
                # above the detailed report - the chatbot-style reply.
                try:
                    query_response = handler.generate_query_response(
                        agent_id=agent_info["agent_id"],
                        query=query,
                        results_data=results_data,
                        task_type=task_type,
                        thread_id=thread_id,
                    )
                except Exception as qr_error:
                    logger.warning(f"Query response unavailable: {qr_error}")
                    query_response = AgentHandler._fallback_query_response(
                        query, results_data, task_type
                    )
               
                if model_warning:
                    analysis_text = f"{model_warning}\n\n{analysis_text}"
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", analysis_text)
               
                best_model = results_data.get('best_model')
                train_metrics = results_data.get('train_metrics', {})
                test_metrics = results_data.get('test_metrics', {})
                all_models = results_data.get('all_models', {})
               
                clean_all_models = ModelRegistryHandler.normalize_all_models(all_models, task=task_type)
                if task_type == "multistep_forecasting":
                    primary_metric = f"avg_{metric}" if not metric.startswith("avg_") else metric
                    primary_score = test_metrics.get(primary_metric) or train_metrics.get(primary_metric)
                else:
                    primary_score = test_metrics.get(metric) or train_metrics.get(metric)
               
                if primary_score is None:
                    primary_score = "N/A"
               
                model_id = None
                if results_data:
                    try:
                        run_id = results_data.get('run_id')
                        run_path = results_data.get('run_path', '')
                       
                        model_id = model_registry.register_model(
                            user_id=user_id,
                            user_email=user_email,
                            run_id=run_id,
                            task=task_type,
                            target=target,
                            model_name=best_model,
                            metric=metric,
                            train_metrics=train_metrics,
                            test_metrics=test_metrics,
                            all_models=clean_all_models,
                            dataset_id=dataset_id,
                            blob_path=run_path,
                            horizon=horizon if task_type == "multistep_forecasting" else None,
                            targets_list=target.split(", ") if task_type == "multistep_forecasting" else None,
                            job_id=job_id,
                            veriton_file_path=veriton_file_path or "N/A",
                            source_type=source_type_for_run,
                            databricks_path=results_data.get("databricks_run_path"),
                            snowflake_path=results_data.get("snowflake_run_path"),
                            onelake_path=results_data.get("onelake_run_path")
                        )
                    except Exception as e:
                        logger.error(f"Failed to register model: {e}", exc_info=True)
                if dataset_id:
                    try:
                        dataset_service.record_task_completion(
                            dataset_id=dataset_id,
                            user_id=user_id,
                            task_type=task_type,
                            query=query,
                            thread_id=thread_id,
                            status="completed",
                            model_id=model_id,
                            model_name=best_model,
                            results_path=results_filename
                        )
                        logger.info(f"Recorded AutoML task in dataset history")
                    except Exception as e:
                        logger.warning(f"Failed to record task: {e}")
                final_result = {
                    "status": "success",
                    "message": "AutoML completed successfully!",
                    "session_id": thread_id,
                    "model_id": model_id,
                    "task_type": task_type.replace("_", " ").title(),
                    "target": target,
                    "best_model": best_model,
                    "primary_metric": metric,
                    "primary_score": primary_score,
                    "all_models": clean_all_models,
                    "analysis": analysis_text,
                    "query_response": query_response,
                    "blob_file_used": blob_file,
                    "results_filename": results_filename,
                    "dataset_id": dataset_id,
                    "suggestions": _suggestions_list(get_dynamic_suggestion())
                }
            except Exception as e:
                logger.error(f"AutoML failed: {e}")
                error_msg = f"AutoML failed: {str(e)}"
               
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
                response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", error_msg)
                final_result = {
                    "status": "error",
                    "session_id": thread_id,
                    "task_type": task_type,
                    "target": target,
                    "blob_file_used": blob_file,
                    "results_filename": None,
                    "results": None,
                    "analysis": error_msg,
                    "response": "ML task failed during processing.",
                    "dataset_id": dataset_id,
                    "suggestions": ["Check your data format", "Try a different task"]
                }
                # Use the real reason (FUN1's own message, or a timeout /
                # connection explanation) rather than a generic string.
                _user_msg, _tech_detail = _explain_fun1_error(exc=e)
                model_registry.register_failed_job(
                        job_id=job_id,
                        user_id=user_id,
                        user_email=user_email,
                        file_path=blob_file or "N/A",
                        veriton_file_path=veriton_file_path or "N/A",
                        task=task_type if 'task_type' in locals() else "query",
                        error_message=_user_msg,
                        error_detail=_tech_detail or str(e),
                        source_type=source_type_for_run if 'source_type_for_run' in locals() else None
                    )
           
        # NON-ML: Just return analysis
        else:
            if not analysis_text:
                raise Exception("No response from agent")
            response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "user", query)
            response_handler.save_response(user_email, user_id, agent_info["agent_id"], thread_id, "assistant", analysis_text)
           
            if dataset_id:
                try:
                    dataset_service.record_task_completion(
                        dataset_id=dataset_id,
                        user_id=user_id,
                        task_type="analysis",
                        query=query,
                        thread_id=thread_id,
                        status="completed"
                    )
                    logger.info(f"Recorded analysis in dataset history")
                except Exception as e:
                    logger.warning(f"Failed to record analysis: {e}")
           
            final_result = {
                "status": "success",
                "session_id": thread_id,
                "task_type": "analysis",
                "blob_file_used": blob_file,
                "results": None,
                "response": analysis_text,
                # For a non-ML query the agent's answer IS the response,
                # so expose it under the same field name the ML path uses.
                "query_response": analysis_text,
                "dataset_id": dataset_id,
                # Prefer steps the model itself supplied (via the now-stripped
                # NEXT_STEPS marker) over asking a fresh LLM call for one.
                "suggestions": extracted_steps or _suggestions_list(get_dynamic_suggestion())
            }
            TASK_QUERY_RESULTS[job_id] = final_result

        model_doc = model_registry.get_model_by_job_id(job_id, user_id)
        if model_doc:
            model_doc["result"] = final_result
            model_registry.container.upsert_item(model_doc)
        else:
            model_registry.register_query_result(
                job_id=job_id,
                user_id=user_id,
                user_email=user_email,
                task=final_result.get("task_type", "analysis"),
                result=final_result,
                session_id=thread_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {traceback.format_exc()}")
        logger.error(f"[QUERY] Background job crashed – job_id={job_id}: {traceback.format_exc()}")
        _user_msg, _tech_detail = _explain_fun1_error(exc=e)
        model_registry.register_failed_job(
            job_id=job_id,
            user_id=user_id if 'user_id' in locals() else "unknown",
            user_email=user_email,
            file_path=blob_file if 'blob_file' in locals() else "N/A",
            veriton_file_path=veriton_file_path or "N/A",
            task="query",
            error_message=_user_msg,
            error_detail=_tech_detail or str(e),
            source_type=source_type_for_run if 'source_type_for_run' in locals() else None
        )


@app.get("/process-task-query-status/{job_id}")
def get_process_task_query_status(job_id: str, user_email: str = Query(...)):
    # First get user_id from email
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            return {"status": "error", "message": "User not found"}

        user_id = user["user_id"]

        model_doc = model_registry.get_model_by_job_id(job_id, user_id)
        if model_doc:
            # Case 1: Job failed
            if model_doc.get("status") == "failed":
                return {
                    "job_id": job_id,
                    "status": "failed",
                    "message": model_doc.get("error_message", "Query processing failed"),
                    "error": model_doc.get("error_detail"),
                    "task": model_doc.get("task"),
                    "session_id": model_doc.get("session_id")
                }

            # Case 2: Job completed successfully and has result
            if "result" in model_doc:
                return model_doc["result"]

            # Case 3: Job exists but no result yet (still processing)
            return {
                "job_id": job_id,
                "status": "running",
                "message": "Processing is still in progress...",
                "last_checked": datetime.utcnow().isoformat()
            }

        # Case 4: No registry record. Fall back to the in-memory store, which
        # holds completed non-ML analysis results.
        in_memory = TASK_QUERY_RESULTS.get(job_id)
        if in_memory:
            return in_memory

        # Case 5: Nothing anywhere yet - still processing
        return {
            "job_id": job_id,
            "status": "running",
            "message": "Processing is still in progress...",
            "last_checked": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Status check failed for query job_id={job_id}: {str(e)}", exc_info=True)
        return {
            "job_id": job_id,
            "status": "error",
            "message": "Failed to check processing status",
            "error_detail": str(e)
        }

KPI_REQUIRED_ROLES = {
    "date": "The column containing the record date or timestamp",
    "plant_id": "Plant / site / facility / location identifier",
    "downtime_hours": "Numeric machine or line downtime, in hours",
    "work_order_type": "Categorical column distinguishing e.g. breakdown vs planned/preventive work orders",
    "production_status": "Categorical column with values like Completed / Delayed / On Hold",
    "machine_id": "Machine / equipment / asset identifier",
    "will_breakdown": "Binary/flag column predicting whether the machine will break down (e.g. 0/1, True/False, Yes/No)",
    "actual_cost": "Numeric actual cost of the work order",
    "defective_quantity": "Numeric count of defective/rejected units",
    "planned_quantity": "Numeric planned production quantity",
    "produced_quantity": "Numeric actual produced quantity",
}

_KPI_DEFAULT_VALUES = {
    "breakdown_value": "Breakdown",
    "risk_true_value": "1",
    "status_completed": "Completed",
    "status_delayed": "Delayed",
    "status_on_hold": "On Hold",
}


def _matches(series: pd.Series, value) -> pd.Series:
    """String-normalized equality so '1' vs 1, 'Yes' vs True, etc. still match."""
    if value is None:
        return pd.Series(False, index=series.index)
    return series.astype(str).str.strip() == str(value).strip()


def _exact_kpi_column_matches(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {role: (role if role in df.columns else None) for role in KPI_REQUIRED_ROLES}


def _resolve_kpi_columns(df: pd.DataFrame, agent_id: str) -> Dict[str, Any]:
    """
    Map this dataset's actual columns (and key categorical values) onto the
    semantic roles the KPI logic needs.
    Returns {"columns": {role: actual_col_name_or_None}, "values": {...}}.
    """
    exact = _exact_kpi_column_matches(df)
    if all(exact.values()):
        return {"columns": exact, "values": dict(_KPI_DEFAULT_VALUES)}

    profile_lines = []
    for c in df.columns:
        try:
            sample_vals = df[c].dropna().astype(str).unique()[:5].tolist()
        except Exception:
            sample_vals = []
        profile_lines.append(f"- {c} (dtype={df[c].dtype}): sample values = {sample_vals}")
    profile_text = "\n".join(profile_lines)

    roles_text = "\n".join(f'- "{role}": {desc}' for role, desc in KPI_REQUIRED_ROLES.items())

    prompt = f"""You are mapping a manufacturing dataset's actual columns onto a fixed set of semantic roles.

Dataset columns:
{profile_text}

Roles to map (use the EXACT column name from the list above, or null if there is truly no match):
{roles_text}

Also identify, using the sample values shown above, the literal value inside the mapped
work_order_type column that represents a breakdown-type work order, the literal value inside
the mapped will_breakdown column that means "yes / at risk", and the literal values inside the
mapped production_status column for Completed / Delayed / On Hold. Use the value exactly as it
appears in "sample values" above. Use null for anything that doesn't apply.

Respond with ONLY a single JSON object, no prose, no markdown fences, in this exact shape:
{{
  "columns": {{
    "date": "...", "plant_id": "...", "downtime_hours": "...", "work_order_type": "...",
    "production_status": "...", "machine_id": "...", "will_breakdown": "...",
    "actual_cost": "...", "defective_quantity": "...", "planned_quantity": "...",
    "produced_quantity": "..."
  }},
  "values": {{
    "breakdown_value": "...", "risk_true_value": "...",
    "status_completed": "...", "status_delayed": "...", "status_on_hold": "..."
  }}
}}"""

    mapping = None
    try:
        raw = handler._ask_agent_for_text(agent_id, prompt)
        mapping = extract_json_from_text(raw)
    except Exception as e:
        logger.warning(f"KPI column-mapping LLM call failed: {e}")

    if not mapping or "columns" not in mapping:
        logger.warning("KPI column-mapping returned no usable JSON - falling back to exact-name matches only")
        return {"columns": exact, "values": dict(_KPI_DEFAULT_VALUES)}

    resolved_cols = {}
    for role in KPI_REQUIRED_ROLES:
        candidate = mapping.get("columns", {}).get(role)
        resolved_cols[role] = candidate if candidate in df.columns else exact.get(role)

    llm_values = mapping.get("values", {}) or {}
    values = {
        "breakdown_value": llm_values.get("breakdown_value") or _KPI_DEFAULT_VALUES["breakdown_value"],
        "risk_true_value": (
            llm_values.get("risk_true_value")
            if llm_values.get("risk_true_value") is not None
            else _KPI_DEFAULT_VALUES["risk_true_value"]
        ),
        "status_completed": llm_values.get("status_completed") or _KPI_DEFAULT_VALUES["status_completed"],
        "status_delayed": llm_values.get("status_delayed") or _KPI_DEFAULT_VALUES["status_delayed"],
        "status_on_hold": llm_values.get("status_on_hold") or _KPI_DEFAULT_VALUES["status_on_hold"],
    }
    logger.info(f"KPI column mapping resolved: {resolved_cols}")
    return {"columns": resolved_cols, "values": values}


KPI_JOB_RESULTS: dict = {}

def full_background_kpi_process(job_id: str, file_path: str, user_email: str):
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            KPI_JOB_RESULTS[job_id] = {"status": "failed", "available": False, "reason": "Unauthorized"}
            return

        # ── Fetch file from whichever source it lives in - dispatch logic
        # lives in app/source_fetch.py. ──
        try:
            file_content, resolved_source_type = fetch_source_file(
                file_path=file_path,
                databricks_handler=databricks_handler_global,
            )
        except SourceFileNotFoundError as e:
            KPI_JOB_RESULTS[job_id] = {"status": "failed", "available": False, "reason": str(e)}
            return
        except SourceCredentialsMissingError as e:
            KPI_JOB_RESULTS[job_id] = {"status": "failed", "available": False, "reason": str(e)}
            return

        content = file_content.read()

        filename = file_path.split("/")[-1]

        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            KPI_JOB_RESULTS[job_id] = {
                "status": "failed", "available": False,
                "reason": "Unsupported file format. Use CSV or Excel."
            }
            return

        agent_id = user["agents"][0]["agent_id"]
        resolved = _resolve_kpi_columns(df, agent_id)
        col = resolved["columns"]
        vals = resolved["values"]

        if not col.get("date"):
            KPI_JOB_RESULTS[job_id] = {
                "status": "failed",
                "available": False,
                "reason": "Could not find a date/time column in this dataset."
            }
            return

        # ── Prep ───────────────────────────────────────────────────────────
        date_col = col["date"]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

        if df.empty:
            KPI_JOB_RESULTS[job_id] = {
                "status": "failed", "available": False,
                "reason": "No valid dated rows found in this dataset."
            }
            return

        latest_date  = df[date_col].max()
        current_month_start  = latest_date.replace(day=1)
        previous_month_start = (current_month_start - pd.DateOffset(months=1))

        current  = df[df[date_col] >= current_month_start]
        previous = df[(df[date_col] >= previous_month_start) & (df[date_col] < current_month_start)]

        def pct_change(curr, prev):
            if not prev:
                return None
            return round(((curr - prev) / prev) * 100, 1)

        def trend_label(pct, lower_is_better=True):
            if pct is None:
                return "No previous data"
            if lower_is_better:
                if pct < 0:
                    return f"↓ Improved {abs(pct)}% vs last month"
                elif pct > 0:
                    return f"↑ Worsened {abs(pct)}% vs last month"
                return "→ No change vs last month"
            else:
                if pct > 0:
                    return f"↑ Improved {abs(pct)}% vs last month"
                elif pct < 0:
                    return f"↓ Dropped {abs(pct)}% vs last month"
                return "→ No change vs last month"

        # ── 1. Plant Health Status ─────────────────────────────────────────
        plant_col = col.get("plant_id")
        downtime_col = col.get("downtime_hours")
        wo_type_col = col.get("work_order_type")
        will_breakdown_col = col.get("will_breakdown")

        plant_health = []
        if plant_col:
            plant_downtime = df.groupby(plant_col)[downtime_col].sum() if downtime_col else pd.Series(dtype=float)
            plant_breakdowns = (
                df[_matches(df[wo_type_col], vals["breakdown_value"])].groupby(plant_col).size()
                if wo_type_col else pd.Series(dtype=int)
            )
            plant_risk_flags = (
                df[_matches(df[will_breakdown_col], vals["risk_true_value"])].groupby(plant_col).size()
                if will_breakdown_col else pd.Series(dtype=int)
            )

            def plant_status(plant):
                downtime = plant_downtime.get(plant, 0)
                risk     = plant_risk_flags.get(plant, 0)
                if downtime > 300 or risk > 25:
                    return "critical"
                elif downtime > 150 or risk > 15:
                    return "warning"
                return "normal"

            status_label = {"critical": "Critical", "warning": "Warning", "normal": "Normal"}

            for plant in sorted(df[plant_col].unique()):
                s    = plant_status(plant)
                dt   = round(float(plant_downtime.get(plant, 0)), 1)
                bd   = int(plant_breakdowns.get(plant, 0))
                risk = int(plant_risk_flags.get(plant, 0))

                plant_prompt = (
                    f"You are summarizing plant health for a non-technical pharma plant manager.\n\n"
                    f"Plant: {plant}\n"
                    f"Status: {s.upper()}\n"
                    f"Total downtime: {dt} hours\n"
                    f"Actual breakdowns this period: {bd}\n"
                    f"Machines flagged as breakdown risk: {risk}\n\n"
                    f"Write exactly ONE sentence (max 30 words) telling the manager what this means "
                    f"and what action to take. Be specific to these exact numbers. "
                    f"No technical jargon. No bullet points. Plain English only."
                )
                insight = handler._ask_agent_for_text(agent_id, plant_prompt)
                if not insight:
                    insight = f"{plant} has {dt} hours of downtime with {risk} machines at risk."

                plant_health.append({
                    "plant_id": plant,
                    "status": s,
                    "status_label": f"{plant} — {status_label[s]}",
                    "insight": insight
                })

        # ── 2. Downtime Trend (current vs previous month) ─────────────────
        status_col = col.get("production_status")

        curr_downtime = round(float(current[downtime_col].sum()), 1) if downtime_col else None
        prev_downtime = round(float(previous[downtime_col].sum()), 1) if downtime_col else None
        downtime_pct  = pct_change(curr_downtime, prev_downtime) if downtime_col else None

        curr_breakdowns = int(_matches(current[wo_type_col], vals["breakdown_value"]).sum()) if wo_type_col else None
        prev_breakdowns = int(_matches(previous[wo_type_col], vals["breakdown_value"]).sum()) if wo_type_col else None
        breakdown_pct   = pct_change(curr_breakdowns, prev_breakdowns) if wo_type_col else None

        curr_delayed = int(_matches(current[status_col], vals["status_delayed"]).sum()) if status_col else None
        prev_delayed = int(_matches(previous[status_col], vals["status_delayed"]).sum()) if status_col else None
        delayed_pct  = pct_change(curr_delayed, prev_delayed) if status_col else None

        # Build trend insight via LLM, using only whichever metrics this dataset has
        trend_facts = [f"Period: {current_month_start.strftime('%B %Y')} vs {previous_month_start.strftime('%B %Y')}"]
        if downtime_col:
            trend_facts.append(
                f"Downtime: {prev_downtime} hrs last month → {curr_downtime} hrs this month "
                f"({'improved' if (downtime_pct or 0) < 0 else 'worsened'} by {abs(downtime_pct or 0)}%)"
            )
        if wo_type_col:
            trend_facts.append(f"Breakdowns: {prev_breakdowns} last month → {curr_breakdowns} this month")
        if status_col:
            trend_facts.append(f"Delayed orders: {prev_delayed} last month → {curr_delayed} this month")

        if len(trend_facts) > 1:
            trend_prompt = (
                "You are summarizing operational trends for a non-technical pharma plant manager.\n\n"
                + "\n".join(trend_facts) +
                "\n\nWrite exactly TWO sentences (max 40 words total) summarising what changed and "
                "whether the plant is moving in the right direction. "
                "Be specific to these numbers. Plain English only. No bullet points."
            )
            trend_insight = handler._ask_agent_for_text(agent_id, trend_prompt)
            if not trend_insight:
                trend_insight = "; ".join(trend_facts[1:]) or "Not enough data to summarize this period."
        else:
            trend_insight = "This dataset doesn't have enough matching fields to summarize a month-over-month trend."

        trend = {
            "period": f"{current_month_start.strftime('%B %Y')} vs {previous_month_start.strftime('%B %Y')}",
            "insight": trend_insight,
            "downtime": {
                "current":    curr_downtime,
                "previous":   prev_downtime,
                "trend":      trend_label(downtime_pct, lower_is_better=True),
                "pct_change": downtime_pct
            },
            "breakdowns": {
                "current":    curr_breakdowns,
                "previous":   prev_breakdowns,
                "trend":      trend_label(breakdown_pct, lower_is_better=True),
                "pct_change": breakdown_pct
            },
            "delayed_orders": {
                "current":    curr_delayed,
                "previous":   prev_delayed,
                "trend":      trend_label(delayed_pct, lower_is_better=True),
                "pct_change": delayed_pct
            }
        }

        # ── 3. Top Machine Risks ───────────────────────────────────────────
        machine_col = col.get("machine_id")
        top_machines = []
        if machine_col:
            machine_downtime  = df.groupby(machine_col)[downtime_col].sum() if downtime_col else pd.Series(dtype=float)
            machine_risk      = (
                df[_matches(df[will_breakdown_col], vals["risk_true_value"])].groupby(machine_col).size()
                if will_breakdown_col else pd.Series(dtype=int)
            )
            machine_breakdown = (
                df[_matches(df[wo_type_col], vals["breakdown_value"])].groupby(machine_col).size()
                if wo_type_col else pd.Series(dtype=int)
            )

            all_machines = set(df[machine_col].unique()) | set(machine_downtime.index) | set(machine_risk.index)
            machine_scores = []
            for m in all_machines:
                dt   = float(machine_downtime.get(m, 0))
                risk = int(machine_risk.get(m, 0))
                bd   = int(machine_breakdown.get(m, 0))
                # weighted risk score: downtime + risk flags weighted higher
                score = (dt * 0.4) + (risk * 5) + (bd * 2)
                machine_scores.append({
                    "machine_id": m,
                    "downtime_hours": round(dt, 1),
                    "predicted_breakdown_risk": risk,
                    "actual_breakdowns": bd,
                    "risk_score": round(score, 1)
                })

            machine_scores.sort(key=lambda x: x["risk_score"], reverse=True)
            top_machines = machine_scores[:3]

            risk_levels = ["High Risk", "Elevated Risk", "Medium Risk"]
            for i, m in enumerate(top_machines):
                m["risk_label"] = risk_levels[i] if i < len(risk_levels) else "Monitor"

                machine_prompt = (
                    f"You are advising a non-technical pharma plant manager about a machine.\n\n"
                    f"Machine: {m['machine_id']}\n"
                    f"Risk level: {m['risk_label']}\n"
                    f"Total downtime: {m['downtime_hours']} hours\n"
                    f"Predicted future breakdowns: {m['predicted_breakdown_risk']}\n"
                    f"Actual breakdowns recorded: {m['actual_breakdowns']}\n\n"
                    f"Write exactly ONE sentence (max 25 words) telling the manager what action "
                    f"to take for this specific machine right now. "
                    f"Be direct and specific. Plain English only. No jargon."
                )
                machine_insight = handler._ask_agent_for_text(agent_id, machine_prompt)
                if not machine_insight:
                    machine_insight = (
                        f"{m['machine_id']} has {m['downtime_hours']} hrs downtime "
                        f"and {m['predicted_breakdown_risk']} predicted breakdowns — review urgently."
                    )
                m["insight"] = machine_insight

        # ── 4. Quick Summary ───────────────────────────────────────────────
        cost_col = col.get("actual_cost")
        defect_col = col.get("defective_quantity")
        produced_col = col.get("produced_quantity")

        total_orders     = len(df)
        completed        = int(_matches(df[status_col], vals["status_completed"]).sum()) if status_col else None
        delayed          = int(_matches(df[status_col], vals["status_delayed"]).sum()) if status_col else None
        on_hold          = int(_matches(df[status_col], vals["status_on_hold"]).sum()) if status_col else None
        total_downtime   = round(float(df[downtime_col].sum()), 1) if downtime_col else None
        total_cost       = round(float(df[cost_col].sum()), 2) if cost_col else None
        avg_defect_rate  = (
            round(float((df[defect_col].sum() / df[produced_col].sum()) * 100), 2)
            if defect_col and produced_col and df[produced_col].sum() > 0 else None
        )

        date_range = f"{df[date_col].min().strftime('%d %b %Y')} – {df[date_col].max().strftime('%d %b %Y')}"

        at_risk_count  = int(_matches(df[will_breakdown_col], vals["risk_true_value"]).sum()) if will_breakdown_col else None
        completion_pct = round((completed / total_orders) * 100, 1) if (completed is not None and total_orders > 0) else None

        summary_facts = [f"Period: {date_range}", f"Total work orders: {total_orders}"]
        if completed is not None:
            summary_facts.append(f"Completed: {completed} ({completion_pct}%)")
        if delayed is not None:
            summary_facts.append(f"Delayed: {delayed}")
        if on_hold is not None:
            summary_facts.append(f"On Hold: {on_hold}")
        if total_downtime is not None:
            summary_facts.append(f"Total downtime: {total_downtime} hours")
        if avg_defect_rate is not None:
            summary_facts.append(f"Defect rate: {avg_defect_rate}%")
        if at_risk_count is not None:
            summary_facts.append(f"Machines at breakdown risk: {at_risk_count}")
        if plant_col:
            summary_facts.append(f"Plants monitored: {int(df[plant_col].nunique())}")

        summary_prompt = (
            "You are giving an overall operations summary to a non-technical pharma plant manager.\n\n"
            + "\n".join(summary_facts) +
            "\n\nWrite exactly TWO sentences (max 45 words total) giving an honest overall picture "
            "of how operations are performing and what the manager should focus on. "
            "Be specific to these numbers. Plain English only. No bullet points. No jargon."
        )
        summary_insight = handler._ask_agent_for_text(agent_id, summary_prompt)
        if not summary_insight:
            summary_insight = "; ".join(summary_facts[1:]) or "Not enough data available to summarize."

        summary = {
            "date_range": date_range,
            "insight": summary_insight,
            "total_work_orders": total_orders,
            "plants_monitored": int(df[plant_col].nunique()) if plant_col else None,
            "machines_monitored": int(df[machine_col].nunique()) if machine_col else None,
            "completed_orders": completed,
            "delayed_orders": delayed,
            "on_hold_orders": on_hold,
            "total_downtime_hours": total_downtime,
            "total_cost": total_cost,
            "defect_rate_pct": avg_defect_rate,
            "at_risk_machines": at_risk_count
        }

        KPI_JOB_RESULTS[job_id] = {
            "status": "success",
            "available": True,
            "plant_health": plant_health,
            "trend": trend,
            "top_machine_risks": top_machines,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"KPI job failed - job_id={job_id}: {e}", exc_info=True)
        KPI_JOB_RESULTS[job_id] = {"status": "failed", "available": False, "reason": str(e)}


@app.post("/dataset_kpis")
async def get_dataset_kpis(
    file_path: str = Form(..., description="Relative path inside lakehouse, e.g. Files/Datasets/.../file.csv"),
    user_email: str = Form(...)
):
   
    try:
        user = auth_handler.get_user(user_email)
        if not user:
            raise HTTPException(status_code=401, detail="Unauthorized")

        job_id = str(uuid.uuid4())
        logger.info(f"dataset_kpis job started - job_id={job_id}")

        threading.Thread(
            target=full_background_kpi_process,
            kwargs={
                "job_id": job_id,
                "file_path": file_path,
                "user_email": user_email
            },
            daemon=True
        ).start()

        return {
            "status": "started",
            "job_id": job_id,
            "message": "KPI generation started in the background. This usually takes 10-60 seconds.",
            "poll_every_seconds": 5
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start dataset_kpis job: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to start KPI processing")


@app.get("/dataset-kpis-status/{job_id}")
def get_dataset_kpis_status(job_id: str, user_email: str = Query(...)):
    result = KPI_JOB_RESULTS.get(job_id)
    if result:
        return result
    return {
        "job_id": job_id,
        "status": "running",
        "available": False,
        "message": "KPI generation is still in progress...",
        "last_checked": datetime.utcnow().isoformat()
    }