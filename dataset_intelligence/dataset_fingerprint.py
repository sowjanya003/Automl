"""
Dataset Fingerprinting Module - Generates unique signatures for datasets to enable similarity detection
"""

import hashlib
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatasetFingerprint:
    """Generate and manage dataset fingerprints for similarity detection"""
    
    def __init__(self, dataframe: pd.DataFrame, filename: str):
        self.df = dataframe
        self.filename = filename
        self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> Dict[str, Any]:
        """Generate complete fingerprint for the dataset"""
        try:
            return {
                "schema_signature": self._generate_schema_signature(),
                "statistical_signature": self._generate_statistical_signature(),
                "structural_signature": self._generate_structural_signature(),
                "content_hash": self._generate_content_hash(),
                "metadata": {
                    "filename": self.filename,
                    "generated_at": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error generating fingerprint: {e}")
            raise
    
    def _generate_schema_signature(self) -> Dict[str, Any]:
        """Generate schema-based signature"""
        columns_info = []
        
        for col in self.df.columns:
            col_info = {
                "name": col,
                "dtype": str(self.df[col].dtype),
                "nullable": bool(self.df[col].isnull().any())
            }
            columns_info.append(col_info)
        
        # Generate schema hash for quick comparison
        schema_string = "_".join([f"{c['name']}:{c['dtype']}" for c in columns_info])
        schema_hash = hashlib.md5(schema_string.encode()).hexdigest()
        
        return {
            "columns": [col["name"] for col in columns_info],
            "dtypes": [col["dtype"] for col in columns_info],
            "column_details": columns_info,
            "schema_hash": schema_hash,
            "num_columns": len(self.df.columns)
        }
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """
        Convert to float, mapping NaN and +/-Infinity to None.

        Cosmos DB rejects documents containing the bare JSON literals NaN or
        Infinity with "The request payload is invalid" - they are not valid
        JSON per RFC 8259, and Python's json.dumps emits them by default.
        A fully-empty column (e.g. DateKey at 99.7% null) makes mean/std/
        min/max/median all NaN, which is what broke /upload_file_V.
        """
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    def _generate_statistical_signature(self) -> Dict[str, Any]:
        """Generate statistical signature for numerical columns"""
        stats = {}
        
        # Numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            try:
                sf = self._safe_float
                stats[col] = {
                    "mean": sf(self.df[col].mean()),
                    "std": sf(self.df[col].std()),
                    "min": sf(self.df[col].min()),
                    "max": sf(self.df[col].max()),
                    "median": sf(self.df[col].median()),
                    "missing_pct": sf(self.df[col].isnull().sum() / len(self.df) * 100)
                }
            except Exception as e:
                logger.warning(f"Could not compute stats for {col}: {e}")
                continue
        
        # Categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:10]:  # Limit to first 10 categorical columns
            try:
                stats[col] = {
                    "unique_count": int(self.df[col].nunique()),
                    "top_value": str(self.df[col].mode()[0]) if len(self.df[col].mode()) > 0 else None,
                    "missing_pct": float(self.df[col].isnull().sum() / len(self.df) * 100)
                }
            except Exception as e:
                logger.warning(f"Could not compute stats for {col}: {e}")
                continue
        
        # Datetime columns
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            try:
                stats[col] = {
                    "min_date": str(self.df[col].min()),
                    "max_date": str(self.df[col].max()),
                    "date_range_days": int((self.df[col].max() - self.df[col].min()).days),
                    "missing_pct": float(self.df[col].isnull().sum() / len(self.df) * 100)
                }
            except Exception as e:
                logger.warning(f"Could not compute stats for {col}: {e}")
                continue
        
        return stats
    
    def _generate_structural_signature(self) -> Dict[str, Any]:
        """Generate structural signature"""
        return {
            "num_rows": int(len(self.df)),
            "num_columns": int(len(self.df.columns)),
            "memory_usage_mb": float(self.df.memory_usage(deep=True).sum() / 1024 / 1024),
            "total_missing_values": int(self.df.isnull().sum().sum()),
            "missing_value_pct": float(self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100),
            "duplicate_rows": int(self.df.duplicated().sum())
        }
    
    def _generate_content_hash(self, sample_rows: int = 100) -> str:
        """Generate hash based on sample of data"""
        try:
            # Take first N rows (or all if fewer)
            sample_size = min(sample_rows, len(self.df))
            sample_df = self.df.head(sample_size)
            
            # Convert to string representation and hash
            content_string = sample_df.to_csv(index=False)
            content_hash = hashlib.sha256(content_string.encode()).hexdigest()
            
            return content_hash
        except Exception as e:
            logger.warning(f"Could not generate content hash: {e}")
            return "unknown"
    
    def get_fingerprint(self) -> Dict[str, Any]:
        """Return the complete fingerprint"""
        return self.fingerprint


def create_fingerprint_from_file(file_path: str, filename: str) -> Dict[str, Any]:
    try:
        # Read file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Create fingerprint
        fingerprinter = DatasetFingerprint(df, filename)
        return fingerprinter.get_fingerprint()
    
    except Exception as e:
        logger.error(f"Error creating fingerprint from file: {e}")
        raise


def create_fingerprint_from_dataframe(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    try:
        fingerprinter = DatasetFingerprint(df, filename)
        return fingerprinter.get_fingerprint()
    except Exception as e:
        logger.error(f"Error creating fingerprint from DataFrame: {e}")
        raise