
"""
Thread Metadata Handler
Stores and retrieves metadata associated with threads/sessions in Cosmos DB
Used to track dataset_id, file uploads, and other thread-specific data
"""

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from app.config import COSMOS_CONNECTION_STRING, DATABASE_NAME, THREAD_METADATA
logger = logging.getLogger(__name__)

class ThreadMetadataHandler:
    
    def __init__(self):
        try:
            self.client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
            self.database = self.client.create_database_if_not_exists(id=DATABASE_NAME)
            
            # Create ThreadMetadata container
            self.container = self.database.create_container_if_not_exists(
                id=THREAD_METADATA,
                partition_key=PartitionKey(path="/user_email"),
                # offer_throughput=200
            )
            logger.info("Thread Metadata Handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Thread Metadata Handler: {e}")
            raise
    
    def create_or_update_thread_metadata(
        self,
        thread_id: str,
        user_id: str,
        user_email: str,
        agent_id: str,
        dataset_id: Optional[str] = None,
        blob_path: Optional[str] = None,
        filename: Optional[str] = None,
        veriton_file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            try:
                thread_doc = self.container.read_item(
                    item=thread_id,
                    partition_key=user_id
                )
                logger.info(f"Updating existing thread metadata: {thread_id}")
            except exceptions.CosmosResourceNotFoundError:
                thread_doc = {
                    "id": thread_id,
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "user_email": user_email,
                    "agent_id": agent_id,
                    "created_at": datetime.now().isoformat(),
                    "datasets": [],
                    "files": [],
                    "metadata": {}
                }
                logger.info(f"Creating new thread metadata: {thread_id}")
            
            if dataset_id:
                if "datasets" not in thread_doc:
                    thread_doc["datasets"] = []
                
                if dataset_id not in thread_doc["datasets"]:
                    thread_doc["datasets"].append(dataset_id)
                
                thread_doc["current_dataset_id"] = dataset_id

            if veriton_file_path:
                if "metadata" not in thread_doc:
                    thread_doc["metadata"] = {}
                thread_doc["metadata"]["veriton_file_path"] = veriton_file_path
                thread_doc["metadata"]["original_source"] = "onelake"
            
            if blob_path and filename:
                if "files" not in thread_doc:
                    thread_doc["files"] = []
                
                file_info = {
                    "filename": filename,
                    "blob_path": blob_path,
                    "dataset_id": dataset_id,
                    "veriton_file_path": veriton_file_path,
                    "uploaded_at": datetime.now().isoformat()
                }
                thread_doc["files"].append(file_info)
            
            thread_doc["last_updated"] = datetime.now().isoformat()
            
            # Upsert to Cosmos DB
            updated_doc = self.container.upsert_item(body=thread_doc)
            logger.info(f"Saved thread metadata: {thread_id}")
            
            return updated_doc
        
        except Exception as e:
            logger.error(f"Failed to save thread metadata: {e}")
            raise
    
    def get_thread_metadata(
        self,
        thread_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            thread_doc = self.container.read_item(
                item=thread_id,
                partition_key=user_id
            )
            return thread_doc
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Thread metadata not found: {thread_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get thread metadata: {e}")
            return None
    
    def get_current_dataset_id(
        self,
        thread_id: str,
        user_id: str
    ) -> Optional[str]:
        try:
            thread_doc = self.get_thread_metadata(thread_id, user_id)
            
            if thread_doc:
                return thread_doc.get("current_dataset_id")
            
            return None
        except Exception as e:
            logger.error(f"Failed to get current dataset ID: {e}")
            return None
    
    def get_thread_datasets(
        self,
        thread_id: str,
        user_id: str
    ) -> List[str]:
        try:
            thread_doc = self.get_thread_metadata(thread_id, user_id)
            
            if thread_doc:
                return thread_doc.get("datasets", [])
            
            return []
        except Exception as e:
            logger.error(f"Failed to get thread datasets: {e}")
            return []
    
    def get_thread_files(
        self,
        thread_id: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        try:
            thread_doc = self.get_thread_metadata(thread_id, user_id)
            
            if thread_doc:
                return thread_doc.get("files", [])
            
            return []
        except Exception as e:
            logger.error(f"Failed to get thread files: {e}")
            return []
    
    def get_latest_file(
        self,
        thread_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            files = self.get_thread_files(thread_id, user_id)
            
            if files:
                # Return the last file (most recent)
                return files[-1]
            
            return None
        except Exception as e:
            logger.error(f"Failed to get latest file: {e}")
            return None
    
    def add_metadata(
        self,
        thread_id: str,
        user_id: str,
        key: str,
        value: Any
    ) -> bool:
        try:
            thread_doc = self.get_thread_metadata(thread_id, user_id)
            
            if not thread_doc:
                logger.warning(f"Thread not found: {thread_id}")
                return False
            
            if "metadata" not in thread_doc:
                thread_doc["metadata"] = {}
            
            thread_doc["metadata"][key] = value
            thread_doc["last_updated"] = datetime.now().isoformat()
            
            self.container.upsert_item(body=thread_doc)
            logger.info(f"Added metadata to thread {thread_id}: {key}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to add metadata: {e}")
            return False
    
    def get_metadata(
        self,
        thread_id: str,
        user_id: str,
        key: str
    ) -> Any:
        try:
            thread_doc = self.get_thread_metadata(thread_id, user_id)
            
            if thread_doc and "metadata" in thread_doc:
                return thread_doc["metadata"].get(key)
            
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None
    
    def get_user_threads(
        self,
        user_id: str,
        agent_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id"
            parameters = [{"name": "@user_id", "value": user_id}]
            
            if agent_id:
                query += " AND c.agent_id = @agent_id"
                parameters.append({"name": "@agent_id", "value": agent_id})
            
            query += " ORDER BY c.created_at DESC"
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            return items
        except Exception as e:
            logger.error(f"Failed to get user threads: {e}")
            return []
    
    def delete_thread_metadata(
        self,
        thread_id: str,
        user_id: str
    ):
        try:
            self.container.delete_item(item=thread_id, partition_key=user_id)
            logger.info(f"Deleted thread metadata: {thread_id}")
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Thread metadata already deleted: {thread_id}")
        except Exception as e:
            logger.error(f"Failed to delete thread metadata: {e}")
    
    def delete_user_threads(
        self,
        user_id: str
    ) -> int:
        try:
            threads = self.get_user_threads(user_id)
            
            for thread in threads:
                self.delete_thread_metadata(thread["thread_id"], user_id)
            
            logger.info(f"Deleted {len(threads)} thread metadata records for user {user_id}")
            return len(threads)
        except Exception as e:
            logger.error(f"Failed to delete user threads: {e}")
            return 0
    
    def cleanup_old_threads(
        self,
        user_id: str,
        days_old: int = 30
    ) -> int:
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            threads = self.get_user_threads(user_id)
            
            deleted_count = 0
            for thread in threads:
                created_at = datetime.fromisoformat(thread["created_at"])
                if created_at < cutoff_date:
                    self.delete_thread_metadata(thread["thread_id"], user_id)
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old threads for user {user_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old threads: {e}")
            return 0
    
    def get_thread_summary(
        self,
        thread_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        try:
            thread_doc = self.get_thread_metadata(thread_id, user_id)
            
            if not thread_doc:
                return {
                    "thread_id": thread_id,
                    "exists": False
                }
            
            return {
                "thread_id": thread_id,
                "exists": True,
                "created_at": thread_doc.get("created_at"),
                "last_updated": thread_doc.get("last_updated"),
                "current_dataset_id": thread_doc.get("current_dataset_id"),
                "total_datasets": len(thread_doc.get("datasets", [])),
                "total_files": len(thread_doc.get("files", [])),
                "files": thread_doc.get("files", [])
            }
        except Exception as e:
            logger.error(f"Failed to get thread summary: {e}")
            return {
                "thread_id": thread_id,
                "exists": False,
                "error": str(e)
            }
