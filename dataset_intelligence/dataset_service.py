"""
Dataset Service - High-level service combining fingerprinting, similarity detection, and metadata management
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
from .dataset_fingerprint import create_fingerprint_from_file
# (
#     DatasetFingerprint, 
#     create_fingerprint_from_file, 
#     create_fingerprint_from_dataframe
# )
from .similarity_detector import SimilarityDetector, find_similar_datasets
from .dataset_metadata_handler import DatasetMetadataHandler
from app.timeseries_transformer import TimeseriesTransformer
logger = logging.getLogger(__name__)

class DatasetService:
    
    def __init__(self):
        self.metadata_handler = DatasetMetadataHandler()
        self.similarity_detector = SimilarityDetector()
        logger.info("Dataset Service initialized")
    
    def process_new_upload(
        self,
        file_path: str,
        filename: str,
        user_id: str,
        user_email: str,
        agent_id: str,
        blob_path: str,
        min_similarity_threshold: float = 60.0,
        source_type: str = None,
        source_path: str = None
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Generating fingerprint for {filename}")
            fingerprint = create_fingerprint_from_file(file_path, filename)
            
            existing_datasets = self.metadata_handler.get_user_datasets(user_id)
            
            similar_datasets = []
            if existing_datasets:
                logger.info(f"Checking against {len(existing_datasets)} existing datasets")
                
                # Prepare existing fingerprints for comparison
                existing_fingerprints = [
                    {
                        "dataset_id": ds["dataset_id"],
                        "filename": ds["filename"],
                        "upload_date": ds["upload_timestamp"],
                        "tasks_performed": ds.get("tasks_performed", []),
                        "fingerprint": ds["fingerprint"]
                    }
                    for ds in existing_datasets
                ]
                
                similar_datasets = find_similar_datasets(
                    fingerprint,
                    existing_fingerprints,
                    min_similarity_threshold
                )

            semantic_analysis = {}
            try:
                # Python-based structural analysis
                df_temp = pd.read_csv(file_path)
                ts_transformer = TimeseriesTransformer()
                python_analysis = ts_transformer.analyze_dataset_structure(df_temp)
                
                # LLM semantic analysis using dedicated Azure Foundry agent
                llm_analysis = None
                try:
                    from app.analysis_agent_handler import get_analysis_agent
                    analysis_agent = get_analysis_agent()
                    llm_analysis = analysis_agent.analyze_structure(file_path)
                    logger.info(f"Dedicated agent analysis complete")
                except Exception as llm_error:
                    logger.warning(f"LLM analysis failed, using Python fallback: {llm_error}")
                
                # Merge analyses
                if llm_analysis:
                    semantic_analysis = ts_transformer.merge_analyses(
                        python_analysis=python_analysis,
                        llm_analysis=llm_analysis,
                        df=df_temp
                    )
                    logger.info("Successfully merged LLM and Python analyses")
                else:
                    semantic_analysis = python_analysis
                    logger.info("Using Python-only analysis (LLM unavailable)")
                
            except Exception as e:
                logger.error(f"Dataset structure analysis failed: {e}")
                semantic_analysis = {}

            dataset_id = self.metadata_handler.save_dataset_metadata(
                user_id=user_id,
                user_email=user_email,
                agent_id=agent_id,
                filename=filename,
                blob_path=blob_path,
                fingerprint=fingerprint,
                semantic_analysis=semantic_analysis,
                source_type=source_type,
                source_path=source_path
            )
            
            suggestions = self._generate_suggestions(similar_datasets)
            
            result = {
                "dataset_id": dataset_id,
                "fingerprint": fingerprint,
                "similar_datasets": similar_datasets,
                "has_matches": len(similar_datasets) > 0,
                "suggestions": suggestions,
                "message": self._create_user_message(similar_datasets),
                "semantic_analysis": semantic_analysis,
                "analysis_metadata": semantic_analysis 
            }
            
            logger.info(f"Processed upload: {filename} - Found {len(similar_datasets)} matches")
            return result
        
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            raise
    
    def _generate_suggestions(self, similar_datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        suggestions = []
        
        for match in similar_datasets[:3]:  # Top 3 matches
            dataset_suggestions = []
            
            # Get tasks performed on this dataset
            tasks = match.get("tasks_performed", [])
            
            if tasks:
                # Group tasks by type
                task_types = {}
                for task in tasks:
                    task_type = task.get("task_type", "unknown")
                    if task_type not in task_types:
                        task_types[task_type] = []
                    task_types[task_type].append(task)
                
                # Create suggestions for each task type
                for task_type, task_list in task_types.items():
                    latest_task = max(task_list, key=lambda t: t.get("timestamp", ""))
                    
                    dataset_suggestions.append({
                        "action": "repeat_task",
                        "task_type": task_type,
                        "original_query": latest_task.get("query", ""),
                        "results_path": latest_task.get("results_path"),
                        "suggestion_text": f"Previously, you ran '{task_type}' on this dataset. Would you like to run it again?"
                    })
            
            # Add comparison suggestion
            if match["similarity_score"] >= 80:
                dataset_suggestions.append({
                    "action": "compare_datasets",
                    "suggestion_text": f"This is very similar to '{match['filename']}'. Would you like to compare them?"
                })
            
            suggestions.append({
                "dataset_id": match["dataset_id"],
                "filename": match["filename"],
                "similarity_score": match["similarity_score"],
                "match_type": match["match_type"],
                "actions": dataset_suggestions
            })
        
        return suggestions
    
    def _create_user_message(self, similar_datasets: List[Dict[str, Any]]) -> str:
        if not similar_datasets:
            return "This is a new dataset. No similar datasets found in your history."
        
        top_match = similar_datasets[0]
        score = top_match["similarity_score"]
        match_type = top_match["match_type"]
        details = top_match.get("details", {})
        
        if match_type == "evolved_dataset":
            return (
                f"This appears to be an updated version of '{top_match['filename']}'. "
                f"{details.get('message', '')} "
                f"Would you like to run the same analysis you did previously?"
            )
        elif match_type == "exact_match" or score >= 90:
            tasks_count = len(top_match.get("tasks_performed", []))
            if tasks_count > 0:
                return (
                    f"You've uploaded a very similar dataset before ('{top_match['filename']}'). "
                    f"You performed {tasks_count} task(s) on it. "
                    f"Would you like to review those results or repeat any analysis?"
                )
            else:
                return (
                    f"This dataset is very similar to '{top_match['filename']}' "
                    f"that you uploaded previously."
                )
        elif match_type == "similar_evolved":
            return (
                f"This dataset is similar to '{top_match['filename']}'. "
                f"{details.get('message', '')} "
                f"Check out the suggestions for related analyses."
            )
        else:
            return (
                f"Found {len(similar_datasets)} similar dataset(s) in your history. "
                f"The most similar is '{top_match['filename']}' (score: {score:.1f}%)."
            )
    
    def record_task_completion(
        self,
        dataset_id: str,
        user_id: str,
        task_type: str,
        query: str,
        thread_id: str,
        status: str = "completed",
        model_id: Optional[str] = None,  
        model_name: Optional[str] = None,
        results_path: Optional[str] = None
    ) -> bool:
        task_info = {
            "task_type": task_type,
            "query": query,
            "thread_id": thread_id,
            "status": status,
            "model_id": model_id,
            "model_name": model_name,
            "results_path": results_path
        }
        
        return self.metadata_handler.add_task_to_dataset(dataset_id, user_id, task_info)
    
    def get_dataset_history(
        self,
        dataset_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        return self.metadata_handler.get_dataset_by_id(dataset_id, user_id)
    
    def find_datasets_by_task(
        self,
        user_id: str,
        task_type: str
    ) -> List[Dict[str, Any]]:
        return self.metadata_handler.get_datasets_with_task_type(user_id, task_type)
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        return self.metadata_handler.get_statistics(user_id)
    
    def search_similar_to_file(
        self,
        file_path: str,
        filename: str,
        user_id: str,
        min_similarity_threshold: float = 60.0
    ) -> List[Dict[str, Any]]:
        try:
            fingerprint = create_fingerprint_from_file(file_path, filename)
            existing_datasets = self.metadata_handler.get_user_datasets(user_id)
            
            if not existing_datasets:
                return []
            
            existing_fingerprints = [
                {
                    "dataset_id": ds["dataset_id"],
                    "filename": ds["filename"],
                    "upload_date": ds["upload_timestamp"],
                    "tasks_performed": ds.get("tasks_performed", []),
                    "fingerprint": ds["fingerprint"]
                }
                for ds in existing_datasets
            ]
            
            similar = find_similar_datasets(
                fingerprint,
                existing_fingerprints,
                min_similarity_threshold
            )
            return similar
        
        except Exception as e:
            logger.error(f"Error searching for similar datasets: {e}")
            return []
    
    def cleanup_user_data(self, user_id: str):
        return self.metadata_handler.delete_all_user_datasets(user_id)
    
    def compare_two_datasets(
        self,
        dataset_id_1: str,
        dataset_id_2: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            dataset1 = self.metadata_handler.get_dataset_by_id(dataset_id_1, user_id)
            dataset2 = self.metadata_handler.get_dataset_by_id(dataset_id_2, user_id)
            
            if not dataset1 or not dataset2:
                logger.warning("One or both datasets not found")
                return None
            
            comparison = self.similarity_detector.compare_datasets(
                dataset1["fingerprint"],
                dataset2["fingerprint"]
            )
            
            return {
                "dataset_1": {
                    "id": dataset_id_1,
                    "filename": dataset1["filename"],
                    "upload_date": dataset1["upload_timestamp"]
                },
                "dataset_2": {
                    "id": dataset_id_2,
                    "filename": dataset2["filename"],
                    "upload_date": dataset2["upload_timestamp"]
                },
                "comparison": comparison
            }
        
        except Exception as e:
            logger.error(f"Error comparing datasets: {e}")
            return None