"""LLM-Based Dataset Transformer - Uses Azure AI Agent to generate transformation code dynamically"""
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import CodeInterpreterTool, MessageAttachment, CodeInterpreterToolDefinition
import pandas as pd
import logging
import os, re
import json
from typing import Dict, Any, Optional, Tuple

from app.config import (
    AZURE_ENDPOINT, AZURE_RESOURCE_GROUP, AZURE_SUBSCRIPTION_ID,
    AZURE_PROJECT_NAME, AGENT_MODEL
)

logger = logging.getLogger(__name__)

class LLMTransformerHandler:
    """Handler for LLM-based dataset transformation using Azure AI Agent"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMTransformerHandler, cls).__new__(cls)
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
            
            self.transformer_agent_id = self._get_or_create_transformer_agent()
            self._initialized = True
            logger.info(f"LLM Transformer Handler initialized with agent: {self.transformer_agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM Transformer Handler: {e}")
            raise
    
    def _get_or_create_transformer_agent(self) -> str:
        """Get or create dedicated transformation agent"""
        stored_agent_id = os.getenv("TRANSFORMER_AGENT_ID")
        
        if stored_agent_id:
            try:
                agent = self.client.agents.get_agent(agent_id=stored_agent_id)
                logger.info(f"Using existing transformer agent: {stored_agent_id}")
                return stored_agent_id
            except Exception as e:
                logger.warning(f"Stored agent {stored_agent_id} not found: {e}")
        
        try:
            code_interpreter = CodeInterpreterTool()
            transformer_agent = self.client.agents.create_agent(
                model=AGENT_MODEL,
                name="Dataset Transformation Specialist",
                instructions=self._get_transformation_instructions(),
                tools=code_interpreter.definitions
            )
            
            agent_id = transformer_agent.id
            logger.info(f"Created new transformer agent: {agent_id}")
            logger.warning(f"Store this in .env: TRANSFORMER_AGENT_ID={agent_id}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create transformer agent: {e}")
            raise

    def _get_transformation_instructions(self) -> str:
        """System instructions for the transformation agent"""
        return """
You are an expert data transformation specialist. Your job is to transform wide-format time-series datasets into long-format (tidy) data suitable for forecasting models.
CRITICAL EXECUTION CONTRACT (MUST FOLLOW)

- INPUT_FILE_PATH and OUTPUT_FILE_PATH are PREDEFINED READ-ONLY VARIABLES.
- DO NOT assign values to INPUT_FILE_PATH or OUTPUT_FILE_PATH.
- DO NOT redefine them.
- DO NOT put quotes around INPUT_FILE_PATH or OUTPUT_FILE_PATH.
- DO NOT hardcode file paths.
- ALWAYS use them exactly like this:
    df = pd.read_csv(INPUT_FILE_PATH)
    df.to_csv(OUTPUT_FILE_PATH, index=False)

Any violation of these rules will cause execution failure.

GENERAL RULES

1. ALWAYS assume the dataset is already available at INPUT_FILE_PATH
2. NEVER assume column names or structure
3. Generate ONLY executable Python code (no markdown, no explanations outside comments)
4. Handle ALL edge cases (missing years, custom month patterns, multi-level columns, etc.)
5. Do NOT print paths or debug file locations
6. The code must run via exec() without modification

TRANSFORMATION GOAL

Transform a wide-format time-series dataset into a long-format (tidy) dataset suitable for forecasting.

OUTPUT MUST CONTAIN EXACTLY THESE COLUMNS:
- date   → datetime (YYYY-MM-DD)
- target → float
- all original dimension columns preserved (Region, Product, Category, etc.)

STEP 1: LOAD DATA (MANDATORY)

```python
import pandas as pd
import numpy as np
import re
from datetime import datetime

df = pd.read_csv(INPUT_FILE_PATH)

STEP 2: AUTO-DETECT YEAR COLUMN (MANDATORY)

You MUST create a column named year.
Detection priority:
Explicit year column (year, Year, YEAR)
Date/datetime columns (Snapshot_Month, Date, Timestamp, Period, etc.)
String dates ("2022-01", "2022/01/15")
Fallback: current year (ONLY if nothing else exists)
Examples (DO NOT COPY VERBATIM — adapt to dataset):

```python
year_col = None
for c in df.columns:
    if c.lower() == "year":
        year_col = c
        break

if year_col:
    df["year"] = df[year_col]
else:
    for c in df.columns:
        try:
            df["year"] = pd.to_datetime(df[c]).dt.year
            break
        except:
            continue

if "year" not in df.columns:
    df["year"] = datetime.now().year
```

STEP 3: IDENTIFY DIMENSIONS & TIME COLUMNS
Dimensions = non-time, non-measure columns (e.g., Product, Region)
Month/period columns may include:
Jan, Feb, M1, M2, Sales_M1, Revenue_M12, Q1, Q2, Week1, etc.
You MUST infer these from the dataset — never hardcode blindly.

STEP 4: MONTH / PERIOD EXTRACTION (CRITICAL — MUST HANDLE DATE STRINGS)

You MUST correctly extract month information even when the dataset ALREADY
contains a time column like "_Month" with values such as:
- "2022-01"
- "2022-02"
- "2022-03"
FAILURE TO HANDLE THIS WILL PRODUCE INVALID OUTPUT.
Define a function to extract a month number using the following PRIORITY ORDER:

1. FULL DATE OR YEAR-MONTH STRINGS (HIGHEST PRIORITY)
   Examples: "2022-01", "2022/02", "2022-03-15"
   → Use pandas datetime parsing and extract .month
2. COLUMN NAMES WITH MONTH INDEX
   Examples:
   - Revenue_M1, Sales_M12, M3
   - Sales1, Revenue2, Month12
   - m1, m02
   → Extract trailing numeric month (1–12)
3. MONTH NAMES
   Examples:
   Jan, January, Feb, March, etc.
4. PURE NUMERIC MONTH VALUES
   Examples:
   1, 2, 12

IMPLEMENTATION (DO NOT CHANGE LOGIC ORDER):
```python
def extract_month_number(value):
    import re
    import pandas as pd

    # Try full date or year-month parsing FIRST
    try:
        dt = pd.to_datetime(value, errors="raise")
        return int(dt.month)
    except:
        pass
    val = str(value).lower().strip()

    # Underscore or suffix month patterns (Revenue_M1, Sales12, m3)
    m = re.search(r'[_\s]?m(\d+)$', val)
    if m:
        month_num = int(m.group(1))
        if 1 <= month_num <= 12:
            return month_num

    m = re.search(r'(\d+)$', val)
    if m:
        month_num = int(m.group(1))
        if 1 <= month_num <= 12:
            return month_num

    # Month name detection
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9,
        "oct": 10, "nov": 11, "dec": 12
    }
    for k, v in months.items():
        if val.startswith(k):
            return int(v)

    # Pure numeric fallback
    try:
        month_num = int(float(val))
        if 1 <= month_num <= 12:
            return month_num
    except:
        pass

    return None
```

STEP 5: MELT TO LONG FORMAT
```python
df_long = df.melt(
    id_vars=dimensions + ["year"],
    value_vars=month_columns,
    var_name="month_col",
    value_name="target"
)
```

STEP 6: CREATE DATE COLUMN
```python
df_long["month_num"] = df_long["month_col"].apply(extract_month_number)
df_long["month_num"] = (df_long["month_num"].astype("Int64").astype(int))

df_long["date"] = pd.to_datetime(
    df_long["year"].astype(int).astype(str) + "-" +
    df_long["month_num"].astype(str).str.zfill(2) + "-01",
    errors="raise"
)
```

STEP 7: CLEAN & SORT
```python
df_long = df_long.dropna(subset=["target"])
df_long = df_long.drop(columns=["month_col", "month_num", "year"])
df_long = df_long.sort_values(["date"] + dimensions)
df_long = df_long.reset_index(drop=True)
```
FINAL STEP: SAVE OUTPUT (MANDATORY)
```python
df_long.to_csv(OUTPUT_FILE_PATH, index=False)
```
FINAL CHECKLIST (MUST PASS):
INPUT_FILE_PATH is used ONLY inside pd.read_csv()
OUTPUT_FILE_PATH is used ONLY inside to_csv()
No assignments to INPUT_FILE_PATH or OUTPUT_FILE_PATH
No quotes around INPUT_FILE_PATH or OUTPUT_FILE_PATH
No hardcoded file paths
Code is directly executable via exec()
"""

    def generate_transformation_code(
        self,
        file_path: str,
        analysis: Dict[str, Any],
        transformation_config: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Use LLM to generate transformation code based on dataset analysis"""
        try:
            context = self._create_transformation_context(analysis, transformation_config)
            
            uploaded_file = self.client.agents.files.upload_and_poll(
                file_path=file_path,
                purpose="assistants"
            )
            
            thread = self.client.agents.threads.create()
            thread_id = thread.id
            
            try:                
                attachment = MessageAttachment(
                    file_id=uploaded_file.id,
                    tools=[CodeInterpreterToolDefinition()]
                )
                
                prompt = f"""
ANALYZE THIS DATASET AND GENERATE TRANSFORMATION CODE.

**DATASET ANALYSIS:**
{json.dumps(context, indent=2)}

**YOUR TASK:**
1. Use Code Interpreter to load and inspect the dataset
2. Verify the analysis (dimensions, measures, time columns)
3. Generate Python code to transform from wide to long format
4. Code must create: date (datetime), target (float), + dimension columns
5. Handle ALL edge cases in month extraction and year handling

**OUTPUT REQUIREMENTS:**
- Return ONLY executable Python code
- No markdown code fences
- Include all necessary imports
- Add comments explaining key steps
- Use 'INPUT_FILE_PATH' and 'OUTPUT_FILE_PATH' as placeholders

Generate the transformation code now.
"""
                
                self.client.agents.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=prompt,
                    attachments=[attachment]
                )
                
                run = self.client.agents.runs.create_and_process(
                    thread_id=thread_id,
                    agent_id=self.transformer_agent_id
                )
                
                if run.status != "completed":
                    raise Exception(f"Code generation failed: {run.status}")
                
                messages = self.client.agents.messages.list(thread_id=thread_id)
                code_response = ""
                
                for msg in messages:
                    if msg.role == "assistant":
                        for content_block in msg.content:
                            if content_block.type == "text":
                                code_response += content_block.text.value
                        break
                
                if not code_response:
                    raise Exception("Empty response from agent")
                
                # Extract Python code from response
                generated_code = self._extract_python_code(code_response)
                
                if not generated_code:
                    raise Exception("No valid Python code found in response")
                
                # Validate generated code
                validation_result = self._validate_generated_code(generated_code)
                
                if not validation_result["valid"]:
                    logger.warning(f"Generated code validation warning: {validation_result['message']}")
                
                metadata = {
                    "generation_timestamp": pd.Timestamp.now().isoformat(),
                    "agent_id": self.transformer_agent_id,
                    "thread_id": thread_id,
                    "validation": validation_result,
                    "code_length": len(generated_code)
                }
                
                logger.info(f"Generated transformation code: {len(generated_code)} chars")
                return generated_code, metadata
                
            finally:
                try:
                    self.client.agents.threads.delete(thread_id=thread_id)
                    self.client.agents.files.delete(file_id=uploaded_file.id)
                except Exception as e:
                    logger.warning(f"Failed to cleanup resources: {e}")
        
        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            raise
    
    def _create_transformation_context(
        self,
        analysis: Dict[str, Any],
        transformation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed context for LLM"""
        return {
            "format": analysis.get("format", "unknown"),
            "dimensions": transformation_config.get("group_by", []),
            "measures": transformation_config.get("measures", []),
            "year_column": transformation_config.get("year_column"),
            "month_columns": transformation_config.get("month_columns", []),
            "time_definition": analysis.get("time_definition", {}),
            "horizon": transformation_config.get("horizon", 12),
            "data_shape": analysis.get("data_shape", {})
        }
    
    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        text = text.strip()
        
        # Try to extract from ```python blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Try to extract from ``` blocks
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, check if entire response is code
        if 'import' in text and ('pd.melt' in text or 'pd.to_datetime' in text):
            return text.strip()
        
        return None
    
    def _validate_generated_code(self, code: str) -> Dict[str, Any]:
        """Validate generated transformation code"""
        required_elements = {
            "imports": ["import pandas as pd"],
            "operations": ["pd.melt", "pd.to_datetime"],
            "output": ["to_csv", "df_long"]
        }
        
        issues = []
        if not any(imp in code for imp in required_elements["imports"]):
            issues.append("Missing pandas import")
        
        # Check operations
        if not any(op in code for op in required_elements["operations"]):
            issues.append("Missing required transformation operations")
        
        # Check output
        if not any(out in code for out in required_elements["output"]):
            issues.append("Missing output/save operation")
        
        # Check for placeholders
        if 'INPUT_FILE_PATH' not in code or 'OUTPUT_FILE_PATH' not in code:
            issues.append("Missing file path placeholders")
        
        return {
            "valid": len(issues) == 0,
            "message": "; ".join(issues) if issues else "Code validation passed",
            "issues": issues
        }
    
    def execute_transformation(
        self,
        code: str,
        input_file_path: str,
        output_file_path: str
    ) -> pd.DataFrame:
        """Execute the generated transformation code safely"""
        try:
            code = code.replace('INPUT_FILE_PATH', f"'{input_file_path}'")
            code = code.replace('OUTPUT_FILE_PATH', f"'{output_file_path}'")
            
            # Create safe execution environment
            local_vars = {
                'pd': pd,
                'np': __import__('numpy'),
                're': __import__('re')
            }
            
            # Execute code
            exec(code, local_vars)
            
            df_transformed = pd.read_csv(output_file_path)
            
            required_cols = ['date', 'target']
            missing_cols = [col for col in required_cols if col not in df_transformed.columns]
            
            if missing_cols:
                raise ValueError(f"Transformation output missing required columns: {missing_cols}")
            
            df_transformed['date'] = pd.to_datetime(df_transformed['date'])
            df_transformed['target'] = pd.to_numeric(df_transformed['target'], errors='coerce')
            initial_rows = len(df_transformed)
            df_transformed = df_transformed.dropna(subset=['target'])
            
            if len(df_transformed) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(df_transformed)} rows with NaN targets")
            logger.info(f"Transformation executed: {len(df_transformed)} rows, {len(df_transformed.columns)} columns")
            
            return df_transformed
        
        except Exception as e:
            logger.error(f"Code execution failed: {e}", exc_info=True)
            raise

# Global instance
_llm_transformer = None

def get_llm_transformer() -> LLMTransformerHandler:
    """Get the global LLM transformer instance (singleton)"""
    global _llm_transformer
    if _llm_transformer is None:
        _llm_transformer = LLMTransformerHandler()
    return _llm_transformer
