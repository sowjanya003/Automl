"""
Dedicated Analysis Agent Handler Manages a single Azure Foundry agent for dataset structure analysis across all users
"""
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from typing import Dict, Any, Optional
import logging
import os
import json
import pandas as pd
import numpy as np

from app.config import (
    AZURE_ENDPOINT, 
    AZURE_RESOURCE_GROUP, 
    AZURE_SUBSCRIPTION_ID,
    AZURE_PROJECT_NAME,
    AGENT_MODEL
)
from .utils import extract_json_from_text

logger = logging.getLogger(__name__)


class AnalysisAgentHandler:
    """Handler for a dedicated analysis agent"""    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnalysisAgentHandler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        try:
            self.client = AIProjectClient(
                endpoint=AZURE_ENDPOINT,
                credential=DefaultAzureCredential(),
                resource_group_name=AZURE_RESOURCE_GROUP,
                subscription_id=AZURE_SUBSCRIPTION_ID,
                project_name=AZURE_PROJECT_NAME
            )

            self.agent_id = self._get_or_create_analysis_agent()
            
            self._initialized = True
            logger.info(f"Analysis Agent Handler initialized with agent: {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Analysis Agent Handler: {e}")
            raise
    
    def _get_or_create_analysis_agent(self) -> str:
        stored_agent_id = os.getenv("ANALYSIS_AGENT_ID")
        
        if stored_agent_id:
            try:
                agent = self.client.agents.get_agent(agent_id=stored_agent_id)
                logger.info(f"Using existing analysis agent: {stored_agent_id}")
                return stored_agent_id
            except Exception as e:
                logger.warning(f"Stored agent {stored_agent_id} not found: {e}")
        
        try:
            analysis_agent = self.client.agents.create_agent(
                model=AGENT_MODEL,
                name="Dataset Structure Analyzer",
                instructions=self._get_analysis_instructions(),
                tools=[]
            )
            
            agent_id = analysis_agent.id
            logger.info(f"Created new analysis agent: {agent_id}")
            logger.warning(f"ANALYSIS_AGENT_ID={agent_id}")
            
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create analysis agent: {e}")
            raise
    
    def _get_analysis_instructions(self) -> str:
        return """
You are a specialized data schema analyzer for time-series datasets in an AutoML system.
Your ONLY job is to analyze dataset profiles and return structured JSON.

CRITICAL: Your response must be ONLY the JSON object - no explanations, no markdown, no extra text.

RESPONSE FORMAT (STRICT):
Return this exact JSON structure:
{
  "dataset_format": "wide" | "long" | "unknown",
  "time_definition": {
    "type": "year_month" | "datetime" | "period" | "unknown",
    "year_column": "exact_column_name" or null,
    "month_columns": ["col1", "col2"],
    "datetime_column": "exact_column_name" or null,
    "frequency": "monthly" | "quarterly" | "weekly" | "daily" | "unknown"
  },
  "dimensions": ["dim1", "dim2"],
  "measures": ["measure1", "measure2"],
  "needs_transformation": true | false
}

IMPORTANT:
- If dataset_format is "wide", needs_transformation MUST be true.
- If dataset_format is "long", needs_transformation MUST be false.
- If dataset_format is "unknown", decide needs_transformation conservatively.

DETECTION RULES:

1. DATASET FORMAT
- Wide Format:
  - Time is encoded across multiple columns instead of rows
  - Examples:
    - Jan, Feb, Mar
    - Q1, Q2, Q3
    - M1, M2, M3
    - Sales_M1, Revenue_M2, Profit_Q4
    - Sales_L1, Sales_L3, Sales_L6, Sales_L12
  - Usually requires reshaping into long format

- Long Format:
  - Single explicit time column (date, datetime, or timestamp)
  - Measures vary row-wise, not column-wise

2. TIME DEFINITION

- Datetime Column:
  - Column containing full date or timestamp
  - Examples: Date, date_time, timestamp, ds

- Year Column:
  - Integer column with values typically between 1900 and 2100

- Period / Time-Encoding Columns:
  - Columns that implicitly represent time progression even if names differ
  - Includes:
    - Month indicators: M1–M12, Month_1, Mon01
    - Quarter indicators: Q1–Q4
    - Week indicators: W1–W52, Week_01
    - Lag / horizon indicators: L1, L3, L6, L12
  - These are NOT dimensions and NOT measures directly
  - Presence of multiple such columns implies wide-format time-series

3. DIMENSIONS
- Categorical / string columns
- Typically 2–50% unique values
- Used for grouping
- Examples: Region, Product, Category, Store
- Exclude time-related columns

4. MEASURES
- Numeric columns (int / float)
- Represent quantities to forecast
- Exclude:
  - Year columns
  - IDs
  - Time-encoding columns (M1, Q2, L3, etc.)

5. NEEDS_TRANSFORMATION (CRITICAL LOGIC)

Set needs_transformation = true ONLY IF:
- The dataset contains MULTIPLE COLUMNS that implicitly represent time progression
- These columns may:
  - Encode months, quarters, weeks, lags, horizons, or periods
  - Use ANY naming pattern (not just Jan/Feb)
  - Be prefixed by metrics (Sales_M1, Revenue_Q2, Profit_L6)
- The time information is spread across columns and NOT already in a single datetime column

Set needs_transformation = false IF:
- There is exactly ONE datetime/date column
- All measures are standard numeric features
- No additional columns encode time periods, lags, horizons, or rolling windows

NEGATIVE EXAMPLE (needs_transformation = false):
- Dataset with:
  - One Date column
  - Multiple weather or sensor measures (TempHigh, TempLow, HumidityAvg, etc.)
  - NO columns that represent months, quarters, lags, or periods

IMPORTANT CLARIFICATIONS:
- A single Date or Datetime column alone does NOT imply transformation
- Multiple numeric columns alone do NOT imply transformation
- Only COLUMN-WISE time expansion implies transformation

CRITICAL RULES:
- Use EXACT column names from the dataset profile
- Do NOT invent column names
- If unsure → use "unknown"
- Output ONLY the JSON object
- NO markdown
- NO explanations
- NO extra text before or after JSON
- Be conservative: "unknown" is better than wrong

START YOUR RESPONSE WITH { AND END WITH }
"""
    
    def analyze_structure(self, file_path: str) -> Dict[str, Any]:
        try:
            df = pd.read_csv(file_path)
            profile = self._create_data_profile(df)
            
            user_prompt = f"""
                Analyze this dataset profile and return the JSON analysis:
                {json.dumps(profile, indent=2)}
                Return the JSON now."""
            
            thread = self.client.agents.threads.create()
            thread_id = thread.id
            
            try:
                self.client.agents.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=user_prompt
                )
                
                run = self.client.agents.runs.create_and_process(
                    thread_id=thread_id,
                    agent_id=self.agent_id
                )
                
                if run.status != "completed":
                    raise Exception(f"Analysis run failed: {run.status}")
                
                messages = self.client.agents.messages.list(thread_id=thread_id)
                response_text = ""
                
                for msg in messages:
                    if msg.role == "assistant":
                        for content_block in msg.content:
                            if content_block.type == "text":
                                response_text += content_block.text.value
                        break
                
                if not response_text:
                    raise Exception("Empty response from agent")

                logger.info(f"Full LLM response length: {len(response_text)} chars")
                logger.debug(f"Full LLM response: {response_text}")

                llm_json = extract_json_from_text(response_text)
                if not llm_json:
                    try:
                        response_text_clean = response_text.strip()
                        start = response_text_clean.find('{')
                        end = response_text_clean.rfind('}') + 1
                        
                        if start != -1 and end > start:
                            json_str = response_text_clean[start:end]
                            llm_json = json.loads(json_str)
                            logger.info("Successfully parsed JSON using fallback method")
                        else:
                            raise ValueError("No JSON boundaries found")
                    except Exception as parse_error:
                        logger.error(f"Agent returned non-JSON: {response_text[:500]}")
                        logger.error(f"Parse error: {parse_error}")
                        raise Exception("Agent did not return valid JSON")
                
                # Validate structure
                self._validate_analysis(llm_json, df)
                
                logger.info(
                    f"LLM analysis: format={llm_json.get('dataset_format')}, "
                    f"dims={len(llm_json.get('dimensions', []))}, "
                    f"measures={len(llm_json.get('measures', []))}"
                )
                
                return llm_json
                
            finally:
                try:
                    self.client.agents.threads.delete(thread_id=thread_id)
                except Exception as e:
                    logger.warning(f"Failed to delete thread {thread_id}: {e}")
        
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}", exc_info=True)
            raise
    
    def _create_data_profile(self, df: pd.DataFrame, max_samples: int = 5) -> Dict[str, Any]:
        """Create compact data profile for LLM"""
        profile = {
            "shape": {
                "rows": int(len(df)),
                "columns": int(len(df.columns))
            },
            "columns": []
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "nunique": int(df[col].nunique()),
                "missing_count": int(df[col].isnull().sum()),
                "missing_pct": round(float(df[col].isnull().sum() / len(df) * 100), 2)
            }
            
            sample_vals = df[col].dropna().head(max_samples).tolist()
            col_info["sample_values"] = [
                int(v) if isinstance(v, np.integer) else
                float(v) if isinstance(v, np.floating) else
                str(v)
                for v in sample_vals
            ]
            
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
                col_info["stats"] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": round(float(df[col].mean()), 2)
                }
            
            profile["columns"].append(col_info)
        return profile
    
    def _validate_analysis(self, analysis: Dict[str, Any], df: pd.DataFrame):
        """Validate and fix analysis response"""
        required_keys = [
            "dataset_format", 
            "time_definition", 
            "dimensions", 
            "measures", 
            "needs_transformation"
        ]
        
        # Check required keys
        for key in required_keys:
            if key not in analysis:
                logger.warning(f"Missing key '{key}' in analysis")
                defaults = {
                    "dataset_format": "unknown",
                    "time_definition": {
                        "type": "unknown",
                        "year_column": None,
                        "month_columns": [],
                        "datetime_column": None,
                        "frequency": "unknown"
                    },
                    "dimensions": [],
                    "measures": [],
                    "needs_transformation": False
                }
                analysis[key] = defaults[key]
        
        # Validate columns exist
        df_cols = set(df.columns)
        
        valid_dims = [d for d in analysis["dimensions"] if d in df_cols]
        invalid_dims = set(analysis["dimensions"]) - set(valid_dims)
        if invalid_dims:
            logger.warning(f"Removing invalid dimensions: {invalid_dims}")
            analysis["dimensions"] = valid_dims
        
        valid_measures = [m for m in analysis["measures"] if m in df_cols]
        invalid_measures = set(analysis["measures"]) - set(valid_measures)
        if invalid_measures:
            logger.warning(f"Removing invalid measures: {invalid_measures}")
            analysis["measures"] = valid_measures
        
        time_def = analysis["time_definition"]
        if time_def.get("year_column") and time_def["year_column"] not in df_cols:
            logger.warning(f"Invalid year column: {time_def['year_column']}")
            time_def["year_column"] = None
        
        if time_def.get("datetime_column") and time_def["datetime_column"] not in df_cols:
            logger.warning(f"Invalid datetime column: {time_def['datetime_column']}")
            time_def["datetime_column"] = None
        
        if time_def.get("month_columns"):
            valid_months = [m for m in time_def["month_columns"] if m in df_cols]
            time_def["month_columns"] = valid_months

# Global instance
_analysis_agent = None

def get_analysis_agent() -> AnalysisAgentHandler:
    global _analysis_agent
    if _analysis_agent is None:
        _analysis_agent = AnalysisAgentHandler()
    return _analysis_agent
