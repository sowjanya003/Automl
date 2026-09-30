"""
Dataset Similarity Detection Module - Compares dataset fingerprints to detect similar or evolved datasets
"""

from typing import Dict, List, Any, Tuple
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class SimilarityDetector:
    """Detect similarity between datasets using their fingerprints"""
    
    # Similarity thresholds
    EXACT_MATCH_THRESHOLD = 95
    EVOLVED_MATCH_THRESHOLD = 80
    SIMILAR_MATCH_THRESHOLD = 60
    
    def __init__(self):
        pass
    
    def compare_datasets(
        self, 
        new_fingerprint: Dict[str, Any], 
        existing_fingerprint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two dataset fingerprints and return similarity score and details
        
        Args:
            new_fingerprint: Fingerprint of newly uploaded dataset
            existing_fingerprint: Fingerprint of existing dataset
            
        Returns:
            Dictionary with similarity score and match details
        """
        try:
            # Calculate component scores
            schema_score = self._calculate_schema_similarity(
                new_fingerprint["schema_signature"],
                existing_fingerprint["schema_signature"]
            )
            
            statistical_score = self._calculate_statistical_similarity(
                new_fingerprint["statistical_signature"],
                existing_fingerprint["statistical_signature"]
            )
            
            structural_score = self._calculate_structural_similarity(
                new_fingerprint["structural_signature"],
                existing_fingerprint["structural_signature"]
            )
            
            content_score = self._calculate_content_similarity(
                new_fingerprint["content_hash"],
                existing_fingerprint["content_hash"]
            )
            
            # Weighted overall score
            overall_score = (
                schema_score * 0.40 +
                statistical_score * 0.30 +
                structural_score * 0.20 +
                content_score * 0.10
            )
            
            # Determine match type
            match_type = self._determine_match_type(
                overall_score, 
                schema_score,
                structural_score,
                new_fingerprint["structural_signature"],
                existing_fingerprint["structural_signature"]
            )
            
            return {
                "overall_score": round(overall_score, 2),
                "match_type": match_type,
                "component_scores": {
                    "schema": round(schema_score, 2),
                    "statistical": round(statistical_score, 2),
                    "structural": round(structural_score, 2),
                    "content": round(content_score, 2)
                },
                "details": self._generate_match_details(
                    new_fingerprint,
                    existing_fingerprint,
                    match_type
                ),
                "is_match": overall_score >= self.SIMILAR_MATCH_THRESHOLD
            }
        except Exception as e:
            logger.error(f"Error comparing datasets: {e}")
            return {
                "overall_score": 0,
                "match_type": "no_match",
                "is_match": False,
                "error": str(e)
            }
    
    def _calculate_schema_similarity(
        self, 
        new_schema: Dict[str, Any], 
        existing_schema: Dict[str, Any]
    ) -> float:
        """Calculate schema similarity (0-100)"""
        score = 0.0
        
        # Check schema hash for exact match
        if new_schema.get("schema_hash") == existing_schema.get("schema_hash"):
            return 100.0
        
        new_cols = set(new_schema.get("columns", []))
        existing_cols = set(existing_schema.get("columns", []))
        
        # Exact column name match (40 points)
        if new_cols and existing_cols:
            column_match_ratio = len(new_cols & existing_cols) / len(new_cols | existing_cols)
            score += column_match_ratio * 40
        
        # Data type match (30 points)
        new_dtypes = new_schema.get("dtypes", [])
        existing_dtypes = existing_schema.get("dtypes", [])
        
        if new_dtypes and existing_dtypes:
            # Compare dtypes for matching columns
            matching_dtype_count = 0
            for i, col in enumerate(new_schema.get("columns", [])):
                if col in existing_schema.get("columns", []):
                    existing_idx = existing_schema["columns"].index(col)
                    if new_dtypes[i] == existing_dtypes[existing_idx]:
                        matching_dtype_count += 1
            
            if len(new_cols & existing_cols) > 0:
                dtype_match_ratio = matching_dtype_count / len(new_cols & existing_cols)
                score += dtype_match_ratio * 30
        
        # Column count similarity (15 points)
        new_col_count = new_schema.get("num_columns", 0)
        existing_col_count = existing_schema.get("num_columns", 0)
        
        if new_col_count and existing_col_count:
            col_count_ratio = min(new_col_count, existing_col_count) / max(new_col_count, existing_col_count)
            score += col_count_ratio * 15
        
        # Fuzzy column name matching (15 points)
        fuzzy_score = self._fuzzy_column_match(
            new_schema.get("columns", []),
            existing_schema.get("columns", [])
        )
        score += fuzzy_score * 15
        
        return min(score, 100.0)
    
    def _fuzzy_column_match(self, new_cols: List[str], existing_cols: List[str]) -> float:
        """Calculate fuzzy similarity for column names"""
        if not new_cols or not existing_cols:
            return 0.0
        
        total_similarity = 0
        for new_col in new_cols:
            max_similarity = 0
            for existing_col in existing_cols:
                similarity = SequenceMatcher(None, new_col.lower(), existing_col.lower()).ratio()
                max_similarity = max(max_similarity, similarity)
            total_similarity += max_similarity
        
        return total_similarity / len(new_cols)
    
    def _calculate_statistical_similarity(
        self, 
        new_stats: Dict[str, Any], 
        existing_stats: Dict[str, Any]
    ) -> float:
        """Calculate statistical similarity (0-100)"""
        if not new_stats or not existing_stats:
            return 0.0
        
        common_columns = set(new_stats.keys()) & set(existing_stats.keys())
        
        if not common_columns:
            return 0.0
        
        total_similarity = 0
        comparable_columns = 0
        
        for col in common_columns:
            new_col_stats = new_stats[col]
            existing_col_stats = existing_stats[col]
            
            # For numerical columns
            # Stats for an all-empty column are stored as None (they used to be
            # NaN, which Cosmos rejects). Skip those instead of raising
            # "unsupported operand type(s) for -: 'float' and 'NoneType'".
            def _num(d, key, default=0.0):
                v = d.get(key, default)
                return default if v is None else float(v)

            if new_col_stats.get("mean") is not None and existing_col_stats.get("mean") is not None:
                try:
                    new_mean = _num(new_col_stats, "mean")
                    old_mean = _num(existing_col_stats, "mean")
                    mean_diff = abs(new_mean - old_mean)
                    mean_avg = (abs(new_mean) + abs(old_mean)) / 2
                    
                    if mean_avg > 0:
                        mean_similarity = max(0, 1 - (mean_diff / mean_avg))
                    else:
                        mean_similarity = 1.0
                    
                    std_diff = abs(_num(new_col_stats, "std", 0.0) - _num(existing_col_stats, "std", 0.0))
                    std_avg = (abs(_num(new_col_stats, "std", 1.0)) + abs(_num(existing_col_stats, "std", 1.0))) / 2
                    
                    if std_avg > 0:
                        std_similarity = max(0, 1 - (std_diff / std_avg))
                    else:
                        std_similarity = 1.0
                    
                    col_similarity = (mean_similarity + std_similarity) / 2
                    total_similarity += col_similarity
                    comparable_columns += 1
                except Exception as e:
                    logger.warning(f"Error comparing stats for {col}: {e}")
                    continue
            
            # For categorical columns
            elif "unique_count" in new_col_stats and "unique_count" in existing_col_stats:
                try:
                    unique_ratio = min(
                        new_col_stats["unique_count"],
                        existing_col_stats["unique_count"]
                    ) / max(
                        new_col_stats["unique_count"],
                        existing_col_stats["unique_count"]
                    )
                    total_similarity += unique_ratio
                    comparable_columns += 1
                except Exception as e:
                    logger.warning(f"Error comparing unique counts for {col}: {e}")
                    continue
        
        if comparable_columns == 0:
            return 0.0
        
        return (total_similarity / comparable_columns) * 100
    
    def _calculate_structural_similarity(
        self, 
        new_structure: Dict[str, Any], 
        existing_structure: Dict[str, Any]
    ) -> float:
        """Calculate structural similarity (0-100)"""
        score = 0.0
        
        # Column count match (40 points)
        new_cols = new_structure.get("num_columns", 0)
        existing_cols = existing_structure.get("num_columns", 0)
        
        if new_cols == existing_cols:
            score += 40
        elif new_cols and existing_cols:
            col_ratio = min(new_cols, existing_cols) / max(new_cols, existing_cols)
            score += col_ratio * 30
        
        # Row count similarity (30 points)
        new_rows = new_structure.get("num_rows", 0)
        existing_rows = existing_structure.get("num_rows", 0)
        
        if new_rows >= existing_rows:
            if existing_rows > 0:
                row_growth_factor = new_rows / existing_rows
                if row_growth_factor <= 2:
                    score += 30
                elif row_growth_factor <= 5:  
                    score += 20
                else:  
                    score += 10
        else:
            # Fewer rows (might be filtered subset)
            if new_rows > 0 and existing_rows > 0:
                row_ratio = new_rows / existing_rows
                score += row_ratio * 20
        
        # Missing value pattern similarity (30 points)
        new_missing = new_structure.get("missing_value_pct", 0)
        existing_missing = existing_structure.get("missing_value_pct", 0)
        
        missing_diff = abs(new_missing - existing_missing)
        if missing_diff <= 5:  # Within 5% difference
            score += 30
        elif missing_diff <= 15:  # Within 15% difference
            score += 20
        elif missing_diff <= 30:  # Within 30% difference
            score += 10
        
        return min(score, 100.0)
    
    def _calculate_content_similarity(self, new_hash: str, existing_hash: str) -> float:
        """Calculate content similarity based on hash (0-100)"""
        if new_hash == existing_hash:
            return 100.0
        return 0.0
    
    def _determine_match_type(
        self,
        overall_score: float,
        schema_score: float,
        structural_score: float,
        new_structure: Dict[str, Any],
        existing_structure: Dict[str, Any]
    ) -> str:
        """Determine the type of match based on scores and patterns"""
        
        if overall_score >= self.EXACT_MATCH_THRESHOLD:
            # Check if it's data evolution (more rows)
            new_rows = new_structure.get("num_rows", 0)
            existing_rows = existing_structure.get("num_rows", 0)
            
            if new_rows > existing_rows and schema_score >= 90:
                return "evolved_dataset"
            else:
                return "exact_match"
        
        elif overall_score >= self.EVOLVED_MATCH_THRESHOLD:
            return "similar_evolved"
        
        elif overall_score >= self.SIMILAR_MATCH_THRESHOLD:
            return "similar_dataset"
        
        else:
            return "no_match"
    
    def _generate_match_details(
        self,
        new_fingerprint: Dict[str, Any],
        existing_fingerprint: Dict[str, Any],
        match_type: str
    ) -> Dict[str, Any]:
        """Generate detailed match information"""
        
        new_structure = new_fingerprint["structural_signature"]
        existing_structure = existing_fingerprint["structural_signature"]
        
        details = {
            "row_difference": new_structure.get("num_rows", 0) - existing_structure.get("num_rows", 0),
            "new_rows": new_structure.get("num_rows", 0),
            "existing_rows": existing_structure.get("num_rows", 0),
            "column_difference": new_structure.get("num_columns", 0) - existing_structure.get("num_columns", 0)
        }
        
        # Add match type specific details
        if match_type == "evolved_dataset":
            details["message"] = f"This appears to be an updated version of a previous dataset with {details['row_difference']} additional rows."
        elif match_type == "exact_match":
            details["message"] = "This dataset is very similar to a previously uploaded file."
        elif match_type == "similar_evolved":
            details["message"] = "This dataset has similar structure and features to a previous upload."
        elif match_type == "similar_dataset":
            details["message"] = "This dataset shares some characteristics with a previously uploaded file."
        else:
            details["message"] = "No similar datasets found."
        
        return details


def find_similar_datasets(
    new_fingerprint: Dict[str, Any],
    existing_fingerprints: List[Dict[str, Any]],
    min_threshold: float = 60.0
) -> List[Dict[str, Any]]:
    """Find similar datasets from existing fingerprints"""
    detector = SimilarityDetector()
    matches = []
    
    for existing in existing_fingerprints:
        comparison = detector.compare_datasets(new_fingerprint, existing["fingerprint"])
        
        if comparison["overall_score"] >= min_threshold:
            matches.append({
                "dataset_id": existing.get("dataset_id"),
                "filename": existing.get("filename"),
                "upload_date": existing.get("upload_date"),
                "similarity_score": comparison["overall_score"],
                "match_type": comparison["match_type"],
                "component_scores": comparison["component_scores"],
                "details": comparison["details"],
                "tasks_performed": existing.get("tasks_performed", [])
            })
    
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    return matches