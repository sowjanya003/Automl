"""
Model Registry Handler - Stores all trained models metadata
"""
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import uuid
import re

from .config import COSMOS_CONNECTION_STRING, DATABASE_NAME, MODEL_REGISTRY
logger = logging.getLogger(__name__)


class ModelRegistryHandler:
    """Handle model registry in Cosmos DB"""
    
    def __init__(self):
        try:
            self.client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
            self.database = self.client.create_database_if_not_exists(id=DATABASE_NAME)
            
            # Create ModelRegistry container
            self.container = self.database.create_container_if_not_exists(
                id=MODEL_REGISTRY,
                partition_key=PartitionKey(path="/email"),
                # offer_throughput=400
            )
            logger.info("Model Registry initialized successfully")
            # logger.info(f"Partition key path: /email")
        except Exception as e:
            logger.error(f"Failed to initialize Model Registry: {e}")
            raise
    
    def register_model(
        self,
        user_id: str,
        user_email: str,
        run_id: str,
        task: str,
        target: str,
        model_name: str,
        metric: str,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        all_models: Dict = None, 
        dataset_id: str = None,
        blob_path: str = None,
        original_filename: str = None, 
        original_blob_path: str = None,
        text_summary: str = None,
        horizon: int = None,
        targets_list: list = None,
        job_id: str = None,
        veriton_file_path: str = None,
        source_type: str = None,
        databricks_path: str = None,
        snowflake_path: str = None,
        onelake_path: str = None
    ) -> str:
        try:
            model_id = str(uuid.uuid4())
            
            model_doc = {
                "id": model_id,
                "model_id": model_id,
                "user_id": user_id,
                "user_email": user_email,
                "run_id": run_id,
                "task": task,
                "target": target,
                "model_name": model_name,
                "metric": metric,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "all_models": all_models or {},
                "dataset_id": dataset_id,
                "blob_path": blob_path,
                "original_filename": original_filename or "Unknown Dataset",     
                "original_blob_path": original_blob_path,     
                "text_summary": text_summary,  
                "created_at": datetime.now().isoformat(),
                "trained_at": datetime.now().isoformat(),
                "is_active": True,
                "usage_count": 0,
                "veriton_file_path": veriton_file_path or "N/A",
                "source_type": source_type or "upload",
                "databricks_path": databricks_path,
                "snowflake_path": snowflake_path,
                "onelake_path": onelake_path
            }
            if job_id:
                model_doc["job_id"] = job_id
            

            if task == "multistep_forecasting":
                if horizon is not None:
                    model_doc["horizon"] = horizon
                if targets_list:
                    model_doc["targets"] = targets_list
                logger.info(f"Registered multistep model: horizon={horizon}, targets={targets_list}")
            
            self.container.create_item(body=model_doc)
            logger.info(f"Registered model: {model_id} ({model_name})")
            logger.info(f"Source: {source_type or 'upload'} | Source path: {veriton_file_path}")
            if databricks_path:
                logger.info(f"Databricks mirror path: {databricks_path}")
            if snowflake_path:
                logger.info(f"Snowflake mirror path: {snowflake_path}")
            if onelake_path:
                logger.info(f"OneLake mirror path: {onelake_path}")
            
            return model_id
        
        except Exception as e:
            logger.error(f"Failed to register model: {e}", exc_info=True)
            logger.error(f"Model doc that failed: {model_doc}")
            raise
    
    def get_user_models(
        self,
        user_id: str,
        task: str = None,
        limit: int = 50,
        include_failed: bool = False   
    ) -> List[Dict[str, Any]]:
        try:
            if include_failed:
                query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.status = 'failed'"
            else:
                query = "SELECT * FROM c WHERE c.user_id = @user_id AND c.is_active = true"

            parameters = [{"name": "@user_id", "value": user_id}]
        
            if task:
                query += " AND c.task = @task"
                parameters.append({"name": "@task", "value": task})
        
            query += " ORDER BY c.created_at DESC"
        
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=limit
            ))
        
            logger.info(f"Found {len(items)} models for user {user_id} (failed={include_failed})")
            return items
    
        except Exception as e:
            logger.error(f"Failed to get user models: {e}")
            return []
    
    def get_model_by_id(
        self,
        model_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific model by ID"""
        try:
            item = self.container.read_item(item=model_id, partition_key=user_id)
            return item
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Model not found: {model_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get model: {e}")
            return None
    
    def increment_usage(
        self,
        model_id: str,
        user_id: str
    ) -> bool:
        """Increment usage count for a model"""
        try:
            model = self.get_model_by_id(model_id, user_id)
            
            if not model:
                return False
            
            model["usage_count"] = model.get("usage_count", 0) + 1
            model["last_used"] = datetime.now().isoformat()
            
            self.container.upsert_item(body=model)
            return True
        
        except Exception as e:
            logger.error(f"Failed to increment usage: {e}")
            return False
    
    def get_best_models_per_task(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get best model per task for a user"""
        try:
            models = self.get_user_models(user_id, limit=1000)
            
            task_models = {}
            for model in models:
                task = model['task']
                if task not in task_models:
                    task_models[task] = []
                task_models[task].append(model)
            
            # Find best model per task
            best_models = []
            for task, models_list in task_models.items():
                best_model = max(
                    models_list,
                    key=lambda m: self._get_primary_metric_value(m)
                )
                best_models.append(best_model)
            
            return best_models
        
        except Exception as e:
            logger.error(f"Failed to get best models: {e}")
            return []
    
    def _get_primary_metric_value(self, model: Dict) -> float:
        """Extract primary metric value for comparison"""
        metric = model.get('metric', '')
        test_metrics = model.get('test_metrics', {})
        
        # Handle negative metrics (errors)
        if metric in ['rmse', 'mae', 'mape']:
            return -test_metrics.get(metric, 999999)
        
        # Handle positive metrics
        return test_metrics.get(metric, 0)
    
    def delete_model(
        self,
        model_id: str,
        user_id: str
    ):
        """Soft delete a model"""
        try:
            model = self.get_model_by_id(model_id, user_id)
            
            if model:
                model["is_active"] = False
                model["deleted_at"] = datetime.now().isoformat()
                self.container.upsert_item(body=model)
                logger.info(f"Deleted model: {model_id}")
        
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")

    def normalize_all_models(raw_all_models: dict, task: str = None) -> dict:
        normalized = {}

        train_data = raw_all_models.get("train", {})
        test_data = raw_all_models.get("test", {})

        all_model_names = set(train_data.keys()) | set(test_data.keys())

        for model_name in all_model_names:
            train_entry = train_data.get(model_name, {})
            test_entry = test_data.get(model_name, {})

            # Train metrics:
            if isinstance(train_entry.get("train_metrics"), dict):
                train_metrics = train_entry["train_metrics"]
            else:
                train_metrics = {
                    k: v for k, v in train_entry.items()
                    if isinstance(v, (int, float))
                }

            # Test metrics:
            if isinstance(test_entry, dict):
                test_metrics = {
                    k: v for k, v in test_entry.items()
                    if isinstance(v, (int, float))
                }
            else:
                test_metrics = {}
            
            if task == "multistep_forecasting":
                def is_avg_metric(key: str) -> bool:
                    return key.startswith("avg_")
                
                def is_per_horizon_metric(key: str) -> bool:
                    return bool(re.match(r'^h\d+_', key))
                
                def is_per_output_metric(key: str) -> bool:
                    return bool(re.match(r'^output_\d+_', key))
            
                train_metrics = {
                    k: v for k, v in train_metrics.items()
                    if is_avg_metric(k) and not is_per_horizon_metric(k) and not is_per_output_metric(k)
                }
                test_metrics = {
                    k: v for k, v in test_metrics.items()
                    if is_avg_metric(k) and not is_per_horizon_metric(k) and not is_per_output_metric(k)
                }

            normalized[model_name] = {
                "train": train_metrics,
                "test": test_metrics
            }

        return normalized


    def get_model_by_job_id(
        self,
        job_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a model document by its stable job_id and user_id.
        main method used for polling.
        """
        try:
            query = """
                SELECT * FROM c 
                WHERE c.job_id = @job_id 
                AND c.user_id = @user_id 
            """
            parameters = [
                {"name": "@job_id", "value": job_id},
                {"name": "@user_id", "value": user_id}
            ]

            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))

            if items:
                return items[0]  # Return first match
            else:
                logger.info(f"No active model found for job_id={job_id} and user_id={user_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to query model by job_id {job_id}: {str(e)}")
            return None

    def register_failed_job(
        self,
        job_id: str,
        user_id: str,
        user_email: str,
        file_path: str,
        veriton_file_path: Optional[str] = None,  
        task: Optional[str] = None,
        target: Optional[str] = None,
        error_message: str = "AutoML training failed",
        error_detail: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> Optional[str]:
        try:
            failed_doc = {
                "id": f"failed_{job_id}",
                "model_id": f"failed_{job_id}",
                "job_id": job_id,
                "user_id": user_id,
                "user_email": user_email,
                "status": "failed",
                "original_filename": file_path.split("/")[-1] if file_path else "unknown.csv",
                "veriton_file_path":veriton_file_path or "N/A",
                "task": task,
                "target": target,
                "error_message": error_message,
                "error_detail": error_detail[:2000] if error_detail else None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "is_active": False,
                "source_type": source_type
            }

            self.container.create_item(body=failed_doc)
            logger.info(f"Registered FAILED job | job_id={job_id} | task={task}")
            return failed_doc["model_id"]

        except Exception as e:
            logger.error(f"Failed to register failed job {job_id}: {e}", exc_info=True)
            return None

    def register_query_result(
        self,
        job_id: str,
        user_id: str,
        user_email: str,
        result: Dict[str, Any],
        task: str = "analysis",
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist a COMPLETED non-ML query (e.g. a dataset analysis).

        These queries train no model, so there is no model document to attach
        the result to. Previously the caller fell through to
        register_failed_job(), which marked a successful analysis as failed
        ("Query completed but no model record found"). This stores it as a
        completed record that /process-task-query-status can return directly.
        """
        try:
            record_id = f"query_{job_id}"
            query_doc = {
                "id": record_id,
                "model_id": record_id,
                "job_id": job_id,
                "user_id": user_id,
                "user_email": user_email,
                "status": "completed",
                "task": task,
                "session_id": session_id,
                "result": result,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "is_active": False,
            }
            self.container.create_item(body=query_doc)
            logger.info(f"Registered COMPLETED query | job_id={job_id} | task={task}")
            return record_id

        except Exception as e:
            logger.error(f"Failed to register query result {job_id}: {e}", exc_info=True)
            return None