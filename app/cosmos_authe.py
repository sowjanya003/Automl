"""Cosmos DB handlers for authentication and chat history"""
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from passlib.context import CryptContext
from datetime import datetime
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from passlib.context import CryptContext
from datetime import datetime
import logging, uuid

from .config import COSMOS_CONNECTION_STRING, DATABASE_NAME, CONTAINER_NAME, CONTAINER_CHAT_HISTORY

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# COSMOS DB AUTHENTICATION HANDLER
class CosmosDBAuthHandler:
    def __init__(self):
        try:
            self.client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
            if not COSMOS_CONNECTION_STRING:
                raise ValueError("Missing COSMOS_CONNECTION_STRING environment variable")

            self.database = self.client.create_database_if_not_exists(id=DATABASE_NAME)
            self.container = self.database.create_container_if_not_exists(
                id=CONTAINER_NAME,
                partition_key=PartitionKey(path="/email"),
                # offer_throughput=200
            )
            logger.info("AutomlDB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cosmos DB: {e}")
            raise

    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def create_user(self, email: str, password: str, full_name: str = None, user_id: str = None):
        try:
            if self.user_exists(email):
                raise ValueError(f"User with email '{email}' already exists")

            if user_id:
                existing = self.get_user_by_id(user_id)
                if existing:
                    raise ValueError(
                        f"user_id '{user_id}' is already assigned to email "
                        f"'{existing.get('email')}'"
                    )
            else:
                user_id = str(uuid.uuid4())

            user_doc = {
                "id": email,
                "user_id": user_id,
                "email": email,
                "password": self.hash_password(password),
                "full_name": full_name,
                "created_at": datetime.now().isoformat(),
                "last_login": None,
                "is_active": True,
                "agents": [],
                "metadata": {}
            }
            created_user = self.container.create_item(body=user_doc)
            logger.info(f"User created: {email} with user_id: {user_id}")
            created_user.pop("password", None)
            return created_user
        except ValueError as ve:
            raise ve
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise

    def user_exists(self, email: str) -> bool:
        try:
            self.container.read_item(item=email, partition_key=email)
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking user existence: {e}")
            raise

    def authenticate_user(self, email: str, password: str):
        try:
            user = self.container.read_item(item=email, partition_key=email)
            if not user.get("is_active", False):
                logger.warning(f"Inactive user attempted login: {email}")
                return None
            if not self.verify_password(password, user["password"]):
                logger.warning(f"Invalid password for user: {email}")
                return None
            user["last_login"] = datetime.now().isoformat()
            self.container.upsert_item(body=user)
            logger.info(f"User authenticated: {email}")
            user.pop("password", None)
            return user
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"User not found: {email}")
            return None
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise

    def get_user(self, email: str):
        try:
            user = self.container.read_item(item=email, partition_key=email)
            user.pop("password", None)
            return user
        except exceptions.CosmosResourceNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error fetching user: {e}")
            raise

    def get_user_by_id(self, user_id: str):
        try:
            query = "SELECT * FROM c WHERE c.user_id = @user_id"
            params = [{"name": "@user_id", "value": user_id}]
            results = list(self.container.query_items(
                query=query,
                parameters=params,
                enable_cross_partition_query=True
            ))
            if not results:
                return None
            user = results[0]
            user.pop("password", None)
            return user
        except Exception as e:
            logger.error(f"Error fetching user by user_id: {e}")
            raise

    def add_agent_to_user(self, email: str, agent_id: str, agent_name: str, thread_id: str):
        try:
            user = self.container.read_item(item=email, partition_key=email)
            if "agents" not in user:
                user["agents"] = []
            if len(user["agents"]) >= 1:
                raise ValueError(f"User {email} already has an agent.")
            user["agents"].append({
                "agent_id": agent_id,
                "agent_name": agent_name,
                "threads": [thread_id],
                "created_at": datetime.now().isoformat()
            })
            self.container.upsert_item(body=user)
            logger.info(f"Agent {agent_id} and Thread {thread_id} added to user {email}")
        except Exception as e:
            logger.error(f"Error adding agent to user: {e}")
            raise

    def add_thread_to_user_agent(self, email: str, thread_id: str):
        try:
            user = self.container.read_item(item=email, partition_key=email)
            if "agents" not in user or len(user["agents"]) == 0:
                raise ValueError("User has no agent")
            if thread_id not in user["agents"][0]["threads"]:
                user["agents"][0]["threads"].append(thread_id)
            self.container.upsert_item(body=user)
            logger.info(f"Thread {thread_id} added to user {email}'s agent")
        except Exception as e:
            logger.error(f"Error adding thread to user agent: {e}")
            raise

    def remove_thread_from_user(self, email: str, thread_id: str):
        try:
            user = self.container.read_item(item=email, partition_key=email)
            if "agents" in user and len(user["agents"]) > 0:
                user["agents"][0]["threads"] = [
                    t for t in user["agents"][0]["threads"] if t != thread_id
                ]
                self.container.upsert_item(body=user)
                logger.info(f"Thread {thread_id} removed from user {email}")
        except Exception as e:
            logger.error(f"Error removing thread: {e}")
            raise