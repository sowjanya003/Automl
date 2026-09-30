"""Utility functions"""
import re, os, io
import json, numpy as np
import pandas as pd
import logging
from typing import Optional, Dict
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

def extract_json_from_text(text: str) -> Optional[Dict]:
    """Extract first valid JSON object from text, even if wrapped in markdown."""
    if not text:
        return None
    
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()
    
    stack = []
    start_idx = -1
    
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    # Found complete JSON object
                    json_str = text[start_idx:i+1]
                    try:
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON parse failed: {e}")
                        start_idx = -1
                        continue
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    return None

async def to_csv_if_needed(file: UploadFile) -> tuple[UploadFile, str, bool]:
    """Convert Parquet file to CSV if needed. Returns (file, filename, converted_flag)."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in {'.csv', '.xlsx', '.xls'}:
        return file, file.filename, False
    
    if ext != '.parquet':
        raise HTTPException(400, f"Unsupported: {ext}. Use CSV/XLSX/Parquet.")
    
    await file.seek(0)
    content = await file.read()
    df = pd.read_parquet(io.BytesIO(content))
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    new_name = os.path.splitext(file.filename)[0] + '.csv'
    converted_file = UploadFile(filename=new_name, file=csv_buffer)
    
    logger.info(f"Converted {file.filename} to {new_name}")
    return converted_file, new_name, True

def split_next_steps(raw_text: str, fallback_suggestion=None, max_steps: int = 4):
    """
    Remove the model's internal 'NEXT_STEPS: a | b | c' control line from a
    visible response, returning (clean_text, steps_list).

    Handles the marker whether the model writes it as 'NEXT_STEPS:' or
    'NEXT STEPS:' (space instead of underscore), in any case, and whether the
    '|'-separated payload sits on the same line as the marker or on the very
    next line - all variants seen in practice. This is a defensive net: even
    on endpoints/prompts that never asked for this marker, the agent thread's
    history can carry the habit over from an earlier turn, so this always
    strips it before the text reaches the user.
    """
    text = (raw_text or "").strip()
    if not text:
        return text, ([fallback_suggestion] if fallback_suggestion else [])

    lines = text.split("\n")
    n = len(lines)
    kept_lines = []
    steps = []
    marker_re = re.compile(r'^\s*NEXT[\s_]+STEPS\s*:\s*(.*)$', re.IGNORECASE)

    i = 0
    while i < n:
        line = lines[i]
        m = marker_re.match(line)
        if m:
            payload = m.group(1).strip()
            if not payload:
                # Payload may be on the next non-empty line instead of the
                # same line as the marker.
                j = i + 1
                while j < n and lines[j].strip() == "":
                    j += 1
                if j < n:
                    payload = lines[j].strip()
                    i = j
            steps = [s.strip() for s in payload.split("|") if s.strip()]
        else:
            kept_lines.append(line)
        i += 1

    clean_text = "\n".join(kept_lines).strip()
    if not steps:
        steps = [fallback_suggestion] if fallback_suggestion else []
    return clean_text, steps[:max_steps]


def looks_like_access_excuse(text: str) -> bool:
    """
    True if `text` is the model excusing itself out of actually reading the
    dataset (e.g. 'I can't access the file directly from here', 'Data access
    limitation in this session prevents me from loading...') instead of
    running Code Interpreter and giving a real, computed answer. Used to
    catch a hypothetical 'here's what I would do once I can read the file'
    reply before it reaches the user.
    """
    if not text:
        return False
    t = text.lower()
    markers = (
        "can't access the file",
        "cannot access the file",
        "can't access this file",
        "unable to access the file",
        "unable to load the file",
        "data access limitation",
        "don't have access to the file",
        "do not have access to the file",
        "prevents me from loading",
        "can't load the file directly",
        "cannot load the file directly",
        "i can't access it directly",
        "couldn't read the dataset file",
        "could not read the dataset file",
        "can't read the dataset file",
        "cannot read the dataset file",
        "couldn't read the file",
        "could not read the file",
        "please re-upload",
        "provide an accessible path",
        # Additional patterns seen in production
        "cannot access dataset",
        "can't access dataset",
        "unable to access dataset",
        "cannot access the dataset",
        "can't access the dataset",
        "no file was uploaded",
        "no dataset was uploaded",
        "i don't have access to the dataset",
        "i do not have access to the dataset",
        "at the provided path",
        "access the file at",
        "cannot find the file",
        "file not found",
        "unable to find the file",
    )
    return any(m in t for m in markers)


def looks_like_leaked_task_description(text: str) -> bool:
    """
    True if `text` is the model describing an action it meant to actually
    perform (e.g. 'Forecast 12 months of quantityproduced using
    calendar_year and calendar_month.' or 'Compute sum of productioncost
    grouped by product_productname.') instead of either running it (Code
    Interpreter, or the is_ml control JSON for ML requests) and returning
    the real computed result, or giving a genuine prose answer. This is a
    control-message leak just like a bare CANNOT_ANSWER token or an access
    excuse - a short, imperative, single-sentence description naming a task
    verb plus a column/duration, with no real computed numbers in it, is
    never a genuine answer to show the user - whether the task it
    describes is an ML request or a plain aggregation.

    Two things matter for avoiding false positives on genuine short answers:
      - the verb must be the OPENING word, not merely present anywhere in
        the sentence (a real answer can legitimately say "...we aggregate
        cost by supplier..." in the middle of an explanation).
      - "no real numbers" must recognize plain integers and thousands-
        separated figures (a real total like "3,245,678" or a real count
        like "245"), not just currency/decimal/percent formatting - while
        still not being fooled by a bare duration/horizon count like
        "12 months" or "30 days", which names a task PARAMETER rather than
        a computed RESULT.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) > 200 or "\n" in t:
        return False  # a real analysis is longer/multi-line; this is not
    verbs = (
        "forecast", "predict", "classify", "cluster",
        "detect anomalies", "detect anomaly", "build a model", "train a model",
        "compute", "aggregate", "calculate", "group by",
        "show the trend", "rank the", "compare total",
    )
    tl = t.lower()
    # Only the OPENING action counts - a verb appearing mid-sentence in a
    # real explanatory answer must not trigger this.
    if not any(tl.startswith(v) for v in verbs):
        return False

    # Strip duration/horizon phrases like "12 months", "30 days", "6
    # quarters" before checking for numbers - those are task PARAMETERS the
    # model names, not computed RESULTS, and would otherwise make a leaked
    # task description look like it "has real numbers".
    t_without_durations = re.sub(
        r'\d+\s*(day|days|week|weeks|month|months|quarter|quarters|year|years|step|steps)\b',
        '', t, flags=re.IGNORECASE
    )
    has_real_numbers = bool(re.search(r'\d', t_without_durations))
    return not has_real_numbers


# Backward-compatible alias - the original name was ML-specific; kept so
# any existing import of it keeps working unchanged after broadening the
# check to also cover leaked plain-aggregation task descriptions.
looks_like_leaked_ml_intent = looks_like_leaked_task_description


def looks_like_agent_asking_question(text: str) -> bool:
    """
    True if the agent replied by asking the USER a clarifying question about
    HOW to compute something, instead of computing a real answer.
    E.g. "What price per unit should be used for revenue?" when asked
    "What is the total revenue?" - asking the user for methodology/config,
    not for data. This is never a valid chatbot response to a data question.

    Very narrow to avoid false positives: must be a short, number-free,
    single-sentence question that asks about methodology (price, rate, threshold,
    value, formula) rather than asking a data question about the dataset.
    """
    if not text:
        return False
    t = text.strip()
    if "\n" in t or len(t) > 150:
        return False
    if not t.endswith("?"):
        return False
    import re as _re
    if _re.search(r'\d', t):
        return False
    tl = t.lower()
    # Must contain a methodology/config keyword - the agent is asking HOW,
    # not asking WHAT the data shows.
    methodology_words = (
        "price", "rate", "cost per", "value per", "formula",
        "threshold", "should i use", "should be used", "which column",
        "which field", "which metric", "what unit", "what formula",
        "what method", "how should", "how would",
    )
    return any(w in tl for w in methodology_words)



def to_python_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_types(item) for item in obj)
    else:
        return obj