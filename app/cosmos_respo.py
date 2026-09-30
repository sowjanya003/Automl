"""Cosmos DB handlers for authentication and chat history"""
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from passlib.context import CryptContext
from datetime import datetime
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from datetime import datetime
import logging
from .config import COSMOS_CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, CONTAINER_CHAT_HISTORY

logger = logging.getLogger(__name__)

# COSMOS DB RESPONSE HANDLER
class CosmosDBResponseHandler:
    def __init__(self):
        try:
            self.client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
            self.database = self.client.create_database_if_not_exists(id=DATABASE_NAME)
            self.container = self.database.create_container_if_not_exists(
                id=CONTAINER_CHAT_HISTORY,
                partition_key=PartitionKey(path="/email"),
                # offer_throughput=200
            )
            logger.info("Chat History initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB Response container: {e}")
            raise

    def _get_or_create_user_document(self, email: str, user_id: str, agent_id: str):
        try:
            return self.container.read_item(item=email, partition_key=email)
        except exceptions.CosmosResourceNotFoundError:
            logger.info(f"Creating new chat history document for {email}")
            user_doc = {
                "id": email,
                "email": email,
                "user_id": user_id,
                "agent_id": agent_id,
                "threads": {},
                "created_at": datetime.now().isoformat(),
                "metadata": {}
            }
            return self.container.create_item(body=user_doc)

    def save_response(
        self, 
        email: str, 
        user_id: str, 
        agent_id: str, 
        thread_id: str, 
        role: str, 
        content: str,
        model_id: str = None
    ):
        try:
            user_doc = self._get_or_create_user_document(email, user_id, agent_id)
            
            if "threads" not in user_doc:
                user_doc["threads"] = {}
            
            if thread_id not in user_doc["threads"]:
                user_doc["threads"][thread_id] = {
                    "agent_id": agent_id,
                    "messages": [],
                    "created_at": datetime.now().isoformat()
                }
            
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat()
            }

            if model_id:
                message["model_id"] = model_id
            
            user_doc["threads"][thread_id]["messages"].append(message)
            user_doc["last_updated"] = datetime.now().isoformat()

            self.container.upsert_item(body=user_doc)
            logger.info(f"Saved {role} message to thread {thread_id} for {email}")
        except Exception as e:
            logger.error(f"Failed to save response: {e}")
            raise

    def get_thread_history(self, email: str, thread_id: str):
        try:
            user_doc = self.container.read_item(item=email, partition_key=email)
            if "threads" not in user_doc or thread_id not in user_doc["threads"]:
                return []
            return user_doc["threads"][thread_id]["messages"]
        except exceptions.CosmosResourceNotFoundError:
            return []

    def get_all_threads(self, email: str):
        try:
            user_doc = self.container.read_item(item=email, partition_key=email)
            if "threads" not in user_doc:
                return {}
            
            thread_summary = {}
            for thread_id, thread_data in user_doc["threads"].items():
                messages = thread_data.get("messages", [])
                thread_summary[thread_id] = {
                    "agent_id": thread_data.get("agent_id"),
                    "created_at": thread_data.get("created_at"),
                    "message_count": len(messages),
                    "last_message": messages[-1] if messages else None
                }
            return thread_summary
        except exceptions.CosmosResourceNotFoundError:
            return {}

    def delete_thread(self, email: str, thread_id: str):
        try:
            user_doc = self.container.read_item(item=email, partition_key=email)
            
            if "threads" in user_doc and thread_id in user_doc["threads"]:
                del user_doc["threads"][thread_id]
                user_doc["last_updated"] = datetime.now().isoformat()
                self.container.upsert_item(body=user_doc)
                logger.info(f"Deleted thread {thread_id} for {email}")
                return True
            return False
        except exceptions.CosmosResourceNotFoundError:
            return False

    def delete_all_threads(self, email: str):
        try:
            user_doc = self.container.read_item(item=email, partition_key=email)
            thread_count = len(user_doc.get("threads", {}))
            user_doc["threads"] = {}
            user_doc["last_updated"] = datetime.now().isoformat()
            self.container.upsert_item(body=user_doc)
            return thread_count
        except exceptions.CosmosResourceNotFoundError:
            return 0

    def delete_user_history(self, email: str):
        try:
            self.container.delete_item(item=email, partition_key=email)
            logger.info(f"Deleted chat history for: {email}")
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Chat history already deleted: {email}")