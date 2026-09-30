# app/forecasting_template.py
import pandas as pd
import io
import re
from azure.storage.blob import BlobServiceClient
from typing import Optional


class SmartForecastAnswer:

    def __init__(self, blob_service: BlobServiceClient):
        self.blob = blob_service

    def make_answer(self, run_path: str, query: str) -> str:
        df = self._load_predictions(run_path)
        if df is None or "prediction" not in df.columns:
            return "No forecast data found."

        total = df["prediction"].sum()
        horizon_days = self._detect_horizon(query, len(df))

        return self._format(total, horizon_days)

    def _load_predictions(self, run_path: str) -> Optional[pd.DataFrame]:
        try:
            client = self.blob.get_blob_client(
                container="uploaded-file",
                blob=f"{run_path}/predictions.csv"
            )
            if not client.exists():
                return None
            data = client.download_blob().readall()
            return pd.read_csv(io.BytesIO(data))
        except:
            return None

    def _detect_horizon(self, query: str, default: int) -> int:
        q = query.lower()

        # 1. Direct: "next 30 days", "next 2 months"
        match = re.search(r"next\s+(\d+)\s*(day|week|month|quarter|year)s?", q)
        if match:
            num, unit = match.groups()
            num = int(num)
            if "week" in unit: return num * 7
            if "month" in unit: return num * 30
            if "quarter" in unit: return num * 90
            if "year" in unit: return num * 365
            return num

        # 2. "Q1", "Q2", "Q3", "Q4"
        if "q1" in q: return 90
        if "q2" in q: return 90
        if "q3" in q: return 90
        if "q4" in q: return 90

        # 3. "this month", "next quarter"
        if "this month" in q: return 30
        if "next quarter" in q: return 90

        # 4. Default: use length of predictions
        return default

    def _format(self, total: float, days: int) -> str:
        total_str = f"${total:,.0f}"

        if days % 365 == 0:
            years = days // 365
            unit = "year" if years == 1 else "years"
            return f"Predicted **{total_str}** for next {years} {unit}"
        if days % 90 == 0:
            quarters = days // 90
            unit = "quarter" if quarters == 1 else "quarters"
            return f"Predicted **{total_str}** for next {quarters} {unit}"
        if days % 30 == 0:
            months = days // 30
            unit = "month" if months == 1 else "months"
            return f"Predicted **{total_str}** for next {months} {unit}"
        if days % 7 == 0:
            weeks = days // 7
            unit = "week" if weeks == 1 else "weeks"
            return f"Predicted **{total_str}** for next {weeks} {unit}"

        unit = "day" if days == 1 else "days"
        return f"Predicted **{total_str}** for next {days} {unit}"