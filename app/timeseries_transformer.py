import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class TimeseriesTransformer:
    """Transform wide-format datasets to long-format for multistep forecasting"""
    MONTH_MAP = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    
    def __init__(self):
        pass
    
    def detect_wide_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            month_cols = self._detect_month_columns(df)
            
            return {
                "is_wide": len(month_cols) >= 6,
                "month_columns": month_cols,
                "detected_months": len(month_cols),
                "confidence": "high" if len(month_cols) == 12 else "medium" if len(month_cols) >= 6 else "low"
            }
        except Exception as e:
            logger.error(f"Error detecting wide format: {e}")
            return {
                "is_wide": False,
                "month_columns": [],
                "detected_months": 0,
                "confidence": "none"
            }
    
    def _detect_month_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that represent months"""
        month_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Check if column name matches month pattern
            for month_name in self.MONTH_MAP.keys():
                if col_lower == month_name or col_lower.startswith(month_name):
                    month_cols.append(col)
                    break
        
        return month_cols
    
    def detect_time_definition(
        self, 
        df: pd.DataFrame, 
        wide_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect time-related columns (year, date, etc.)"""
        try:
            year_col = self._detect_year_column(df)
            datetime_cols = self._detect_datetime_columns(df)
            
            if wide_info["is_wide"]:
                return {
                    "year_column": year_col,
                    "month_columns": wide_info["month_columns"],
                    "datetime_column": None,
                    "frequency": "monthly"
                }
            
            # Long format
            return {
                "year_column": year_col,
                "month_columns": [],
                "datetime_column": datetime_cols[0] if datetime_cols else None,
                "frequency": "unknown"
            }
        
        except Exception as e:
            logger.error(f"Error detecting time definition: {e}")
            return {
                "year_column": None,
                "month_columns": [],
                "datetime_column": None,
                "frequency": "unknown"
            }
    
    def _detect_year_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            col_lower = str(col).lower()
            
            # Check by name
            if col_lower in ["year", "yr"] or col_lower.endswith("_year"):
                return col
            
            # Check by values (years between 1990-2100)
            if df[col].dtype in ['int64', 'float64']:
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if 1990 <= min_val and max_val <= 2100:
                        return col
                except:
                    continue
        
        return None
    
    def _detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        return [
            col for col in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
    
    def detect_grouping_columns(
        self,
        df: pd.DataFrame,
        wide_info: Dict[str, Any]
    ) -> List[str]:
        grouping_cols = []
        excluded_cols = set(wide_info["month_columns"])
        
        for col in df.columns:
            if col in excluded_cols:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                nunique = df[col].nunique()
                if 2 <= nunique <= min(50, len(df) * 0.5):
                    grouping_cols.append(col)
        
        return grouping_cols
    
    def detect_measure_columns(
        self, 
        df: pd.DataFrame,
        wide_info: Dict[str, Any]
    ) -> List[str]:
        if wide_info["is_wide"]:
            return wide_info["month_columns"]
        
        # In long format, detect numeric columns
        measures = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].nunique() / len(df) > 0.95:
                continue
            
            measures.append(col)
        return measures
    
    def analyze_dataset_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        try:
            wide_info = self.detect_wide_format(df)
            time_def = self.detect_time_definition(df, wide_info)
            grouping_cols = self.detect_grouping_columns(df, wide_info)
            measure_cols = self.detect_measure_columns(df, wide_info)
            
            analysis = {
                "data_shape": {
                    "rows": len(df),
                    "columns": len(df.columns)
                },
                "format": "wide" if wide_info["is_wide"] else "long",
                "is_wide_format": wide_info["is_wide"],
                "dimensions": grouping_cols,
                "measures": measure_cols,
                "time_definition": time_def,
                "needs_transformation": wide_info["is_wide"],
                "supports": {
                    "single_series": True,
                    "multiple_series": len(grouping_cols) > 0,
                    "multistep_forecasting": True
                }
            }
            
            logger.info(
                f"Dataset analysis: format={analysis['format']}, "
                f"dimensions={len(grouping_cols)}, measures={len(measure_cols)}"
            )
            
            return analysis
        
        except Exception as e:
            logger.error(f"Dataset analysis failed: {e}")
            raise
    
    def transform_wide_to_long(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Transform wide-format dataset to long-format"""
        try:
            measures = config.get('measures', [])
            group_by = config.get('group_by', [])
            year_col = config.get('year_column')
            
            if not measures:
                raise ValueError("No measures specified for transformation")
            
            # Validate columns exist
            missing_measures = [m for m in measures if m not in df.columns]
            if missing_measures:
                raise ValueError(f"Measures not found in dataset: {missing_measures}")
            
            if year_col and year_col not in df.columns:
                raise ValueError(f"Year column not found: {year_col}")
            
            # Prepare ID columns (dimensions + year)
            id_vars = group_by.copy() if group_by else []
            if year_col:
                id_vars.append(year_col)
            
            # Melt the DataFrame
            df_long = pd.melt(
                df,
                id_vars=id_vars if id_vars else None,
                value_vars=measures,
                var_name='month',
                value_name='target'
            )
            
            # Convert month names to numbers
            df_long['month'] = df_long['month'].apply(
                lambda x: self._month_to_number(x)
            )
            
            # Create datetime if year column exists
            if year_col:
                df_long['date'] = pd.to_datetime(
                    df_long[year_col].astype(str) + '-' + 
                    df_long['month'].astype(str).str.zfill(2) + '-01'
                )
                # Sort by date
                df_long = df_long.sort_values(['date'] + group_by if group_by else ['date'])
            else:
                # Sort by month
                df_long = df_long.sort_values(['month'] + group_by if group_by else ['month'])
            
            df_long = df_long.dropna(subset=['target'])
            df_long = df_long.reset_index(drop=True)
            
            logger.info(
                f"Transformed wide to long: {len(df)} rows → {len(df_long)} rows"
            )
            
            return df_long
        
        except Exception as e:
            logger.error(f"Wide-to-long transformation failed: {e}")
            raise

    def _month_to_number(self, month_name: str) -> int:
        month_lower = str(month_name).lower().strip()
        
        # Anything ending with underscore + M + number (Revenue_M1, Sales_M12)
        match = re.search(r'[_\s]?m(\d+)$', month_lower)
        if match:
            month_num = int(match.group(1))
            if 1 <= month_num <= 12:
                logger.debug(f"Extracted month {month_num} from underscore pattern: {month_name}")
                return month_num
        
        # Anything ending directly with a number (Sales1, Revenue2, Month12)
        match = re.search(r'(\d+)$', month_lower)
        if match:
            month_num = int(match.group(1))
            if 1 <= month_num <= 12:
                logger.debug(f"Extracted month {month_num} from suffix pattern: {month_name}")
                return month_num
        
        # Month names (Jan, February, etc.)
        for key, value in self.MONTH_MAP.items():
            if month_lower == key or month_lower.startswith(key):
                logger.debug(f"Matched month name: {month_name} → {value}")
                return value
        
        # Pure numeric (already a number 1-12)
        try:
            month_num = int(month_name)
            if 1 <= month_num <= 12:
                return month_num
        except:
            pass
        raise ValueError(f"Cannot map month: {month_name}. Expected format: 'Revenue_M1', 'M12', 'Jan', or 1-12")
    
    def create_transformation_config(
        self,
        analysis: Dict[str, Any],
        user_selection: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            'needs_transformation': analysis['needs_transformation'],
            'measures': user_selection.get('measures', analysis.get('measures', [])),
            'group_by': user_selection.get('group_by', []),
            'year_column': analysis['time_definition'].get('year_column'),
            'month_columns': analysis['time_definition'].get('month_columns', []),
            'horizon': user_selection.get('horizon', 12)
        }
    
    def validate_transformation_config(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate transformation configuration"""
        try:
            # Check required keys
            required_keys = ['measures', 'group_by', 'year_column', 'horizon']
            missing_keys = [k for k in required_keys if k not in config]
            
            if missing_keys:
                return False, f"Missing required config keys: {missing_keys}"
            
            # Check measures exist
            measures = config['measures']
            if not measures:
                return False, "No measures specified"
            
            missing_measures = [m for m in measures if m not in df.columns]
            if missing_measures:
                return False, f"Measures not found: {missing_measures}"
            
            # Check group_by columns exist
            group_by = config.get('group_by', [])
            if group_by:
                missing_groups = [g for g in group_by if g not in df.columns]
                if missing_groups:
                    return False, f"Group columns not found: {missing_groups}"
            
            # Check year column
            year_col = config.get('year_column')
            if year_col and year_col not in df.columns:
                return False, f"Year column not found: {year_col}"
            
            # Check horizon
            horizon = config.get('horizon')
            if not isinstance(horizon, int) or horizon < 2:
                return False, f"Invalid horizon: {horizon}. Must be integer ≥ 2"
            
            return True, "Configuration is valid"
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def merge_analyses(
        self, 
        python_analysis: Dict[str, Any], 
        llm_analysis: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Merge LLM semantic analysis with Python heuristics"""
        if not llm_analysis:
            logger.info("No LLM analysis provided, using Python-only")
            return python_analysis
        
        # Validate LLM output structure
        required_keys = ["dataset_format", "time_definition", "dimensions", "measures", "needs_transformation"]
        missing_keys = [k for k in required_keys if k not in llm_analysis]
        
        if missing_keys:
            logger.warning(f"LLM analysis missing keys {missing_keys}, using Python fallback")
            return python_analysis
        
        # Get dataset columns for validation
        df_columns = set(df.columns)
        
        # Start with structure from LLM
        merged = {
            "data_shape": python_analysis.get("data_shape", {"rows": len(df), "columns": len(df.columns)}),
            "format": llm_analysis.get("dataset_format", "unknown"),
            "is_wide_format": llm_analysis.get("dataset_format") == "wide",
            "time_definition": llm_analysis.get("time_definition", {}),
            "dimensions": [],
            "measures": [],
            "needs_transformation": llm_analysis.get("needs_transformation", False),
            "supports": python_analysis.get("supports", {
                "single_series": True,
                "multiple_series": False,
                "multistep_forecasting": True
            })
        }
        
        # Validate and filter dimensions (LLM may hallucinate)
        llm_dims = llm_analysis.get("dimensions", [])
        valid_dims = [d for d in llm_dims if d in df_columns]
        invalid_dims = [d for d in llm_dims if d not in df_columns]
        
        if invalid_dims:
            logger.warning(f"LLM suggested invalid dimensions: {invalid_dims}")
        
        # Use Python dims as fallback
        if not valid_dims and python_analysis.get("dimensions"):
            logger.info("No valid LLM dimensions, using Python analysis")
            valid_dims = python_analysis["dimensions"]
        
        merged["dimensions"] = valid_dims
        llm_measures = llm_analysis.get("measures", [])
        valid_measures = [m for m in llm_measures if m in df_columns]
        invalid_measures = [m for m in llm_measures if m not in df_columns]
        
        if invalid_measures:
            logger.warning(f"LLM suggested invalid measures: {invalid_measures}")
        
        # Use Python measures as fallback
        if not valid_measures and python_analysis.get("measures"):
            logger.info("No valid LLM measures, using Python analysis")
            valid_measures = python_analysis["measures"]
        
        merged["measures"] = valid_measures
        time_def = merged["time_definition"]
        
        if time_def.get("year_column") and time_def["year_column"] not in df_columns:
            logger.warning(f"Invalid year column from LLM: {time_def['year_column']}")
            time_def["year_column"] = python_analysis["time_definition"].get("year_column")
        
        if time_def.get("datetime_column") and time_def["datetime_column"] not in df_columns:
            logger.warning(f"Invalid datetime column from LLM: {time_def['datetime_column']}")
            time_def["datetime_column"] = python_analysis["time_definition"].get("datetime_column")
        
        if time_def.get("month_columns"):
            valid_months = [m for m in time_def["month_columns"] if m in df_columns]
            time_def["month_columns"] = valid_months
        
        if python_analysis.get("is_wide_format") != merged["is_wide_format"]:
            logger.info(f"Format conflict - Python: {python_analysis['is_wide_format']}, LLM: {merged['is_wide_format']}")
            logger.info("Trusting Python's structural detection")
            merged["is_wide_format"] = python_analysis["is_wide_format"]
            merged["format"] = "wide" if python_analysis["is_wide_format"] else "long"
        
        merged["supports"]["multiple_series"] = len(merged["dimensions"]) > 0
        
        logger.info(
            f"Merged analysis complete: format={merged['format']}, "
            f"dims={len(merged['dimensions'])}, measures={len(merged['measures'])}, "
            f"needs_transform={merged['needs_transformation']}"
        )
        return merged
    
    def canonicalize_time(
    self, 
    df: pd.DataFrame, 
    time_def: Dict[str, Any], 
    dimensions: List[str], 
    measure: str
) -> pd.DataFrame:
        if time_def["type"] == "year_month":
            df_long = df.melt(
                id_vars=dimensions + [time_def["year_column"]],
                value_vars=time_def["month_columns"],
                var_name="month",
                value_name="y"
            )
            df_long["month_num"] = df_long["month"].apply(self._parse_month_number)
            df_long["ds"] = pd.to_datetime(
                df_long[time_def["year_column"]].astype(str) + "-" +
                df_long["month_num"].astype(str) + "-01"
            )
            df_long = df_long.drop(columns=["month", "month_num"])
        
        elif time_def["type"] == "datetime":
            df_long = df.copy()
            df_long["ds"] = pd.to_datetime(df_long[time_def["datetime_column"]])
            df_long["y"] = df_long[measure]
        
        elif time_def["type"] == "period":
            raise NotImplementedError("Period transformation not yet implemented")  # Add as needed
        
        else:
            raise ValueError("Unsupported time format")
        
        # Sort and fill missing
        df_long = df_long.sort_values("ds")
        return df_long
