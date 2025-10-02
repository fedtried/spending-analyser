from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple, IO, Optional
import os
import io
import random

import pandas as pd
import base64

from .models import AnalysisResult, Transaction, Insight


def _normalize_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize various CSV schemas to the internal schema.

    Expected output columns:
      - timestamp: datetime
      - amount: float (negative for outflows/spend, positive for inflows/credits)
      - currency: str
      - category: Optional[str]
      - merchant: Optional[str]
      - description: Optional[str]
    """
    df = raw_df.copy()

    # Common bank export schema mapping (from demo-data.csv)
    column_map = {
        "Transaction_Date": "timestamp",
        "Posting_Date": None,  # ignored for now
        "Description": "description",
        "Transaction_Type": None,
        "Merchant_Category": "category",
        "Amount": "amount",
        "Location": None,
        "Balance_After": "balance_after",
    }

    # Only rename columns that exist
    rename_dict = {k: v for k, v in column_map.items() if v is not None and k in df.columns}
    df = df.rename(columns=rename_dict)

    # Timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        # Try any column with 'date' in name
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df["timestamp"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        else:
            df["timestamp"] = pd.NaT

    # Amount: internal convention is negative for spend and positive for income.
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        df["amount"] = pd.NA
    
    # Balance After: convert to numeric
    if "balance_after" in df.columns:
        df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
    else:
        df["balance_after"] = pd.NA

    # Currency: default to £ if missing
    if "currency" not in df.columns:
        df["currency"] = "£"

    # Merchant: if missing, fall back to description
    if "merchant" not in df.columns:
        if "description" in df.columns:
            df["merchant"] = df["description"].astype(str)
        else:
            df["merchant"] = pd.NA

    # Category optional; ensure string type where present
    if "category" in df.columns:
        df["category"] = df["category"].astype(str)

    # Coerce dtypes and drop rows without timestamp or amount
    available_columns = ["timestamp", "amount", "currency", "category", "merchant", "description", "balance_after"]
    df = df[[c for c in available_columns if c in df.columns]]
    df = df.dropna(subset=["timestamp", "amount"])  # require core fields
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_demo_dataframe() -> pd.DataFrame:
    """Load a randomly selected demo CSV and normalize to internal schema."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    demo_folder = os.path.join(repo_root, "demo-data")
    
    # Dynamically find all demo-data-N.csv files in the demo-data folder
    import glob
    demo_pattern = os.path.join(demo_folder, "demo-data-*.csv")
    demo_files = glob.glob(demo_pattern)
    
    if not demo_files:
        raise FileNotFoundError(f"No demo-data-*.csv files found in {demo_folder}")
    
    # Extract just the filenames for random selection
    demo_filenames = [os.path.basename(f) for f in demo_files]
    selected_file = random.choice(demo_filenames)
    csv_path = os.path.join(demo_folder, selected_file)
    
    raw_df = pd.read_csv(csv_path)
    return _normalize_dataframe(raw_df)


def _encode_pdf_to_base64(file_obj: IO[bytes]) -> str:
    """Encode PDF file to base64 string for sending to Gemini."""
    file_obj.seek(0)
    pdf_bytes = file_obj.read()
    return base64.b64encode(pdf_bytes).decode('utf-8')


def load_user_dataframe(file_obj: IO[bytes]) -> pd.DataFrame:
    """Load a user-uploaded PDF file-like into a normalized DataFrame."""
    # Check if it's a PDF file
    file_obj.seek(0)
    header = file_obj.read(4)
    file_obj.seek(0)
    
    if header.startswith(b'%PDF'):
        # It's a PDF file - encode to base64 for Gemini processing
        base64_pdf = _encode_pdf_to_base64(file_obj)
        # For now, return demo data structure since we'll process with Gemini
        # TODO: Replace this with actual Gemini API call
        return load_demo_dataframe()
    else:
        # Fallback to CSV processing for backward compatibility
        raw_df = pd.read_csv(file_obj)
        return _normalize_dataframe(raw_df)


def analyze_dataframe(df: pd.DataFrame) -> AnalysisResult:
    transactions = [
        Transaction(
            id=f"tx_{i}",
            timestamp=row.timestamp,
            amount=float(row.amount),
            currency=row.currency,
            category=row.category,
            merchant=row.merchant,
            description=row.description,
        )
        for i, row in df.iterrows()
    ]

    totals: dict[str, float] = {}
    for t in transactions:
        # Include spending transactions (negative amounts) in category totals as positive magnitudes
        if t.amount < 0:
            key = t.category or "Uncategorized"
            totals[key] = totals.get(key, 0.0) + (-t.amount)

    total_spent = abs(sum(a for a in (t.amount for t in transactions) if a < 0))
    total_income = sum(a for a in (t.amount for t in transactions) if a > 0)
    total_net = total_income - total_spent

    insights = [
        Insight(
            title="Consistent weekly spending",
            summary="Your spending is steady across the week with small peaks on weekends.",
            kind="trend",
            score=0.6,
        )
    ]

    return AnalysisResult(
        transactions=transactions,
        insights=insights,
        totals_by_category=totals,
        total_spent=total_spent,
        total_income=total_income,
        total_net=total_net,
        num_transactions=len(transactions),
    )


# Backwards compatibility alias
def analyze_demo_dataframe(df: pd.DataFrame) -> AnalysisResult:
    return analyze_dataframe(df)
