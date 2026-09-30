"""
Test Results Handler - Stores test metrics and predictions for deployed models
"""
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import uuid

from app.config import COSMOS_CONNECTION_STRING, DATABASE_NAME, TEST_METRICS

logger = logging.getLogger(__name__)

class TestMetricsHandler:    
    def __init__(self):
        try:
            self.client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
            self.database = self.client.create_database_if_not_exists(id=DATABASE_NAME)
            
            self.container = self.database.create_container_if_not_exists(
                id=TEST_METRICS,
                partition_key=PartitionKey(path="/user_email"),
                # offer_throughput=200
            )
            logger.info("Test Results Handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Test Results Handler: {e}")
            raise
    
    def save_test_result(
        self,
        model_id: str,
        user_id: str,
        user_email: str,
        test_file_name: str,
        test_blob_path: str,
        has_ground_truth: bool,
        test_metrics: Dict[str, float],
        predictions_blob_path: Optional[str] = None,
        rows_tested: int = 0,
        notes: Optional[str] = None,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        target: Optional[str] = None,
        run_id: Optional[str] = None,
        source_type: Optional[str] = None,
        source_path: Optional[str] = None,
        predictions_source_path: Optional[str] = None
    ) -> str:
        
        try:
            test_result_id = str(uuid.uuid4())
            
            test_doc = {
                "id": test_result_id,
                "test_result_id": test_result_id,
                "model_id": model_id,
                "user_id": user_id,
                "user_email": user_email, 
                "test_file_name": test_file_name,
                "test_blob_path": test_blob_path,
                "has_ground_truth": has_ground_truth,
                "test_metrics": test_metrics,
                "predictions_blob_path": predictions_blob_path,
                "rows_tested": rows_tested,
                "notes": notes,
                "tested_at": datetime.now().isoformat(),
                "is_active": True,
                "model_name": model_name,
                "task": task,
                "target": target,
                "run_id": run_id,
                "source_type": source_type or "upload",
                "source_path": source_path,
                "predictions_source_path": predictions_source_path
            }
            
            self.container.create_item(body=test_doc)
            logger.info(f"Saved test result: {test_result_id} for model {model_id}")
            
            return test_result_id
        
        except Exception as e:
            logger.error(f"Failed to save test result: {e}")
            raise
    
    def get_test_result(
        self,
        test_result_id: str,
        user_email: str
    ) -> Optional[Dict[str, Any]]:
        """Get specific test result by ID"""
        try:
            item = self.container.read_item(
                item=test_result_id,
                partition_key=user_email
            )
            return item
        except exceptions.CosmosResourceNotFoundError:
            logger.warning(f"Test result not found: {test_result_id}")
            return None
        except Exception as e:
            logger.error(f"Failed to get test result: {e}")
            return None
    
    def get_model_test_history(
        self,
        model_id: str,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        try:
            query = """
                SELECT * FROM c 
                WHERE c.model_id = @model_id 
                AND c.user_id = @user_id
                AND c.is_active = true
                ORDER BY c.tested_at DESC
            """
            
            parameters = [
                {"name": "@model_id", "value": model_id},
                {"name": "@user_id", "value": user_id}
            ]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=limit
            ))
            
            logger.info(f"Found {len(items)} test results for model {model_id}")
            return items
        
        except Exception as e:
            logger.error(f"Failed to get model test history: {e}")
            return []
    
    def get_user_test_results(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all test results for a user"""
        try:
            query = """
                SELECT * FROM c 
                WHERE c.user_id = @user_id
                AND c.is_active = true
                ORDER BY c.tested_at DESC
            """
            
            parameters = [{"name": "@user_id", "value": user_id}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True,
                max_item_count=limit
            ))
            
            return items
        
        except Exception as e:
            logger.error(f"Failed to get user test results: {e}")
            return []
    
    def get_latest_test_for_model(
        self,
        model_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent test result for a model"""
        try:
            results = self.get_model_test_history(model_id, user_id, limit=1)
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Failed to get latest test: {e}")
            return None
    
    def update_test_result(
        self,
        test_result_id: str,
        user_email: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an existing test result"""
        try:
            test_result = self.get_test_result(test_result_id, user_email)
            
            if not test_result:
                return False
            
            for key, value in updates.items():
                if key not in ["id", "test_result_id", "user_id", "user_email"]:
                    test_result[key] = value
            
            test_result["last_updated"] = datetime.now().isoformat()
            
            self.container.upsert_item(body=test_result)
            logger.info(f"Updated test result: {test_result_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to update test result: {e}")
            return False
    
    def delete_test_result(
        self,
        test_result_id: str,
        user_email: str
    ):
        """Soft delete a test result"""
        try:
            test_result = self.get_test_result(test_result_id, user_email)
            
            if test_result:
                test_result["is_active"] = False
                test_result["deleted_at"] = datetime.now().isoformat()
                self.container.upsert_item(body=test_result)
                logger.info(f"Deleted test result: {test_result_id}")
        
        except Exception as e:
            logger.error(f"Failed to delete test result: {e}")
    
    def get_test_statistics(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get statistics about user's test results"""
        try:
            test_results = self.get_user_test_results(user_id, limit=1000)
            
            total_tests = len(test_results)
            with_ground_truth = sum(1 for t in test_results if t.get("has_ground_truth"))
            total_rows_tested = sum(t.get("rows_tested", 0) for t in test_results)
            
            # Count tests per model
            model_test_counts = {}
            for result in test_results:
                model_id = result.get("model_id")
                if model_id:
                    model_test_counts[model_id] = model_test_counts.get(model_id, 0) + 1
            
            return {
                "total_tests": total_tests,
                "tests_with_ground_truth": with_ground_truth,
                "tests_without_ground_truth": total_tests - with_ground_truth,
                "total_rows_tested": total_rows_tested,
                "models_tested": len(model_test_counts),
                "most_tested_model": max(model_test_counts.items(), key=lambda x: x[1])[0] if model_test_counts else None
            }
        
        except Exception as e:
            logger.error(f"Failed to get test statistics: {e}")
            return {}