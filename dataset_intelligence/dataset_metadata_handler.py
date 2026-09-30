"""Dataset Metadata Handler - Manages storage and retrieval of dataset fingerprints in Cosmos DB"""

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from typing import Dict, List, Any, Optional
from datetime import datetime, date
import math
import numpy as np
import pandas as pd
import logging
import uuid

from app.config import COSMOS_CONNECTION_STRING, DATABASE_NAME, DATASET_METADATA
logger = logging.getLogger(__name__)


def sanitize_for_cosmos(obj: Any) -> Any:
    """
    Make a value safe to write to Cosmos DB.

    Cosmos rejects a document with "(BadRequest) The request payload is
    invalid" if the serialized JSON contains the bare literals NaN, Infinity
    or -Infinity. Python's json.dumps emits those by default, so a single
    all-null numeric column in a fingerprint is enough to fail the whole
    insert.

    This converts:
      - NaN / +Inf / -Inf        -> None
      - numpy scalars            -> Python int / float / bool
      - numpy arrays, sets, tuples -> list
      - pandas NaT / NA          -> None
      - non-string dict keys     -> str
      - anything else unknown    -> str(obj)
    """
    # None and plain bools pass straight through
    if obj is None or isinstance(obj, bool):
        return obj

    # str first, but normalise numpy's str_ subclass to a builtin str
    if isinstance(obj, str):
        return str(obj)

    # numpy scalar types
    if isinstance(obj, np.generic):
        obj = obj.item()

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    if isinstance(obj, int):
        return obj

    if isinstance(obj, dict):
        return {str(k): sanitize_for_cosmos(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return [sanitize_for_cosmos(v) for v in obj]

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # pandas NaT / pd.NA and friends
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    return str(obj)


class DatasetMetadataHandler:
    """Handle dataset metadata storage in Cosmos DB"""
    
    def __init__(self):
        try:
            self.client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
            self.database = self.client.create_database_if_not_exists(id=DATABASE_NAME)
            
            # Create DatasetMetadata container
            self.container = self.database.create_container_if_not_exists(
                id=DATASET_METADATA,
                partition_key=PartitionKey(path="/user_email"),
                # offer_throughput=200
            )
            logger.info("Dataset Metadata Handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Dataset Metadata Handler: {e}")
            raise
    
    def save_dataset_metadata(
        self,
        user_id: str,
        user_email: str,
        agent_id: str,
        filename: str,
        blob_path: str,
        fingerprint: Dict[str, Any],
        semantic_analysis: Optional[Dict] = None,
        source_type: Optional[str] = None,
        source_path: Optional[str] = None
    ) -> str:
        try:
            dataset_id = str(uuid.uuid4())
            
            metadata_doc = {
                "id": dataset_id,
                "dataset_id": dataset_id,
                "user_id": user_id,
                "user_email": user_email,
                "agent_id": agent_id,
                "filename": filename,
                "blob_path": blob_path,
                "fingerprint": fingerprint,
                "upload_timestamp": datetime.now().isoformat(),
                "tasks_performed": [],
                "last_accessed": datetime.now().isoformat(),
                "is_active": True,
                "semantic_analysis": semantic_analysis or {},
                "source_type": source_type or "upload",
                "source_path": source_path
            }
            
            # Strip NaN / Infinity / numpy types before writing, otherwise
            # Cosmos rejects the whole document with a BadRequest.
            metadata_doc = sanitize_for_cosmos(metadata_doc)

            self.container.create_item(body=metadata_doc)
            logger.info(f"Saved dataset metadata: {dataset_id} for user {user_id}")
            
            return dataset_id
        
        except Exception as e:
            logger.error(f"Failed to save dataset metadata: {e}")
            raise
    
    def get_user_datasets(
        self, 
        user_id: str, 
        include_inactive: bool = False
    ) -> List[Dict[str, Any]]:
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            if not include_inactive:
                query += " AND c.is_active = true"
            
            query += " ORDER BY c.upload_timestamp DESC"
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            logger.info(f"Found {len(items)} datasets for user {user_id}")
            return items
        
        except Exception as e:
            logger.error(f"Failed to get user datasets: {e}")
            return []
    
    def get_dataset_by_id(self, dataset_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific dataset metadata by ID"""
        try:
            item = self.container.read_item(item=dataset_id, partition_key=user_id)
            return item
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Dataset not found: {dataset_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get dataset: {e}")
            return None

    def get_dataset_by_blob_path(self, user_id: str, blob_path: str) -> Optional[Dict[str, Any]]:
        """
        Look up a dataset by its Blob path rather than dataset_id.

        Needed because dataset_id is only known within the SESSION/THREAD
        that originally uploaded the file (via ThreadMetadata's
        current_dataset_id). A later session can still reuse the same
        underlying Blob file (get_latest_blob_file() is scoped by
        user_id+agent_name, not by thread) without ever having its own
        dataset_id - in that case this is the only way to recover which
        source (e.g. "databricks") the file originally came from.

        Returns the most recently uploaded matching, active record if
        there are duplicates (e.g. same filename re-uploaded).
        """
        try:
            query = (
                "SELECT * FROM c WHERE c.user_id = @user_id "
                "AND c.blob_path = @blob_path AND c.is_active = true "
                "ORDER BY c.upload_timestamp DESC"
            )
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@blob_path", "value": blob_path},
            ]
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items[0] if items else None
        except Exception as e:
            logger.error(f"Failed to get dataset by blob_path '{blob_path}': {e}")
            return None
    
    def add_task_to_dataset(
        self,
        dataset_id: str,
        user_id: str,
        task_info: Dict[str, Any]
    ) -> bool:
        try:
            dataset = self.get_dataset_by_id(dataset_id, user_id)
            
            if not dataset:
                logger.warning(f"Dataset {dataset_id} not found")
                return False
            
            # Add timestamp to task info
            task_info["timestamp"] = datetime.now().isoformat()
            
            # Add task to tasks_performed list
            if "tasks_performed" not in dataset:
                dataset["tasks_performed"] = []
            
            dataset["tasks_performed"].append(task_info)
            dataset["last_accessed"] = datetime.now().isoformat()
            
            # Update in Cosmos DB
            self.container.upsert_item(body=sanitize_for_cosmos(dataset))
            
            logger.info(f"Added task to dataset {dataset_id}: {task_info['task_type']}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to add task to dataset: {e}")
            return False
    
    def update_dataset_access_time(self, dataset_id: str, user_id: str):
        """Update last accessed time for a dataset"""
        try:
            dataset = self.get_dataset_by_id(dataset_id, user_id)
            
            if dataset:
                dataset["last_accessed"] = datetime.now().isoformat()
                self.container.upsert_item(body=sanitize_for_cosmos(dataset))
                logger.info(f"Updated access time for dataset {dataset_id}")
        
        except Exception as e:
            logger.warning(f"Failed to update access time: {e}")
    
    def deactivate_dataset(self, dataset_id: str, user_id: str):
        """Mark a dataset as inactive (soft delete)"""
        try:
            dataset = self.get_dataset_by_id(dataset_id, user_id)
            
            if dataset:
                dataset["is_active"] = False
                dataset["deactivated_at"] = datetime.now().isoformat()
                self.container.upsert_item(body=sanitize_for_cosmos(dataset))
                logger.info(f"Deactivated dataset {dataset_id}")
        
        except Exception as e:
            logger.error(f"Failed to deactivate dataset: {e}")
    
    def delete_dataset(self, dataset_id: str, user_id: str):
        """Permanently delete a dataset metadata"""
        try:
            self.container.delete_item(item=dataset_id, partition_key=user_id)
            logger.info(f"Deleted dataset metadata: {dataset_id}")
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Dataset already deleted: {dataset_id}")
        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
    
    def delete_all_user_datasets(self, user_id: str):
        """Delete all dataset metadata for a user"""
        try:
            datasets = self.get_user_datasets(user_id, include_inactive=True)
            
            for dataset in datasets:
                self.delete_dataset(dataset["dataset_id"], user_id)
            
            logger.info(f"Deleted {len(datasets)} datasets for user {user_id}")
            return len(datasets)
        
        except Exception as e:
            logger.error(f"Failed to delete user datasets: {e}")
            return 0
    
    def get_dataset_tasks(self, dataset_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get all tasks performed on a specific dataset"""
        try:
            dataset = self.get_dataset_by_id(dataset_id, user_id)
            
            if dataset and "tasks_performed" in dataset:
                return dataset["tasks_performed"]
            
            return []
        
        except Exception as e:
            logger.error(f"Failed to get dataset tasks: {e}")
            return []
    
    def search_datasets_by_filename(
        self, 
        user_id: str, 
        filename_pattern: str
    ) -> List[Dict[str, Any]]:
        """Search datasets by filename pattern"""
        try:
            query = """
                SELECT * FROM c 
                WHERE c.user_id = @user_id 
                AND CONTAINS(LOWER(c.filename), LOWER(@pattern))
                AND c.is_active = true
                ORDER BY c.upload_timestamp DESC
            """
            
            parameters = [
                {"name": "@user_id", "value": user_id},
                {"name": "@pattern", "value": filename_pattern}
            ]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return items
        
        except Exception as e:
            logger.error(f"Failed to search datasets: {e}")
            return []
    
    def get_datasets_with_task_type(
        self, 
        user_id: str, 
        task_type: str
    ) -> List[Dict[str, Any]]:
        """Get all datasets where a specific task type was performed"""
        try:
            datasets = self.get_user_datasets(user_id)
            
            matching_datasets = []
            for dataset in datasets:
                tasks = dataset.get("tasks_performed", [])
                
                if any(task.get("task_type") == task_type for task in tasks):
                    matching_datasets.append(dataset)
            
            return matching_datasets
        
        except Exception as e:
            logger.error(f"Failed to get datasets with task type: {e}")
            return []
    
    def get_statistics(self, user_id: str) -> Dict[str, Any]:
        try:
            datasets = self.get_user_datasets(user_id, include_inactive=True)
            
            total_tasks = sum(len(d.get("tasks_performed", [])) for d in datasets)
            
            task_type_counts = {}
            for dataset in datasets:
                for task in dataset.get("tasks_performed", []):
                    task_type = task.get("task_type", "unknown")
                    task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            
            return {
                "total_datasets": len(datasets),
                "active_datasets": len([d for d in datasets if d.get("is_active", True)]),
                "total_tasks_performed": total_tasks,
                "task_type_breakdown": task_type_counts,
                "oldest_dataset": min((d["upload_timestamp"] for d in datasets), default=None),
                "newest_dataset": max((d["upload_timestamp"] for d in datasets), default=None)
            }
        
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}