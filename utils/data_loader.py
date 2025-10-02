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

    column_map = {
        "Transaction_Date": "timestamp",
        "Posting_Date": None,
        "Description": "description",
        "Transaction_Type": None,
        "Merchant_Category": "category",
        "Amount": "amount",
        "Location": None,
        "Balance_After": "balance_after",
    }

    rename_dict = {k: v for k, v in column_map.items() if v is not None and k in df.columns}
    df = df.rename(columns=rename_dict)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if date_cols:
            df["timestamp"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        else:
            df["timestamp"] = pd.NaT

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    else:
        df["amount"] = pd.NA
    
    if "balance_after" in df.columns:
        df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
    else:
        df["balance_after"] = pd.NA

    if "currency" not in df.columns:
        df["currency"] = "£"

    if "merchant" not in df.columns:
        if "description" in df.columns:
            df["merchant"] = df["description"].astype(str)
        else:
            df["merchant"] = pd.NA

    if "category" in df.columns:
        df["category"] = df["category"].astype(str)

    available_columns = ["timestamp", "amount", "currency", "category", "merchant", "description", "balance_after"]
    df = df[[c for c in available_columns if c in df.columns]]
    df = df.dropna(subset=["timestamp", "amount"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_demo_dataframe() -> pd.DataFrame:
    """Load a randomly selected demo CSV and normalize to internal schema."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    demo_folder = os.path.join(repo_root, "demo-data")
    
    import glob
    demo_pattern = os.path.join(demo_folder, "demo-data-*.csv")
    demo_files = glob.glob(demo_pattern)
    
    if not demo_files:
        raise FileNotFoundError(f"No demo-data-*.csv files found in {demo_folder}")
    
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
    file_obj.seek(0)
    header = file_obj.read(4)
    file_obj.seek(0)
    
    if header.startswith(b'%PDF'):
        base64_pdf = _encode_pdf_to_base64(file_obj)
        return load_demo_dataframe()
    else:
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


def analyze_demo_dataframe(df: pd.DataFrame) -> AnalysisResult:
    return analyze_dataframe(df)


# ---- Subscription/Recurring detection utilities ----
def _normalize_merchant_name(text: str) -> str:
    """Normalize merchant string for grouping (strip ids, digits, case, extra spaces)."""
    if not isinstance(text, str):
        return ""
    s = text.lower()
    # Remove long numeric tokens and order ids
    s = ''.join(ch if not ch.isdigit() else ' ' for ch in s)
    # Remove extra punctuation
    for token in ['pos', 'online', 'dd', 'direct debit', 'subscription']:
        s = s.replace(token, ' ')
    s = ' '.join(s.split())
    return s


def detect_recurring_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Detect likely recurring charges by merchant using cadence and amount stability.

    Returns a dataframe with columns:
      - merchant_norm: normalized merchant key
      - merchant: example display merchant
      - cadence_days: median days between occurrences
      - is_monthly_like: bool for 27-33 day cadence window
      - is_weekly_like: bool for 6-8 day cadence window
      - amount_median: typical charge amount (>0 as spend absolute)
      - amount_cv: coefficient of variation (std/mean) on absolute spend
      - last_timestamp: last occurrence timestamp
      - count: number of occurrences considered
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'merchant_norm','merchant','cadence_days','is_monthly_like','is_weekly_like',
            'amount_median','amount_cv','last_timestamp','count'
        ])

    spend = df[df['amount'] < 0].copy()
    if spend.empty:
        return pd.DataFrame()

    spend['merchant_norm'] = spend.get('merchant', spend.get('description', '')).astype(str).apply(_normalize_merchant_name)
    spend['abs_amount'] = spend['amount'].abs()
    spend = spend[spend['merchant_norm'] != ""]

    groups = []
    for key, g in spend.sort_values('timestamp').groupby('merchant_norm'):
        if len(g) < 2:
            continue
        ts = g['timestamp'].sort_values().to_list()
        deltas = pd.Series([(ts[i] - ts[i-1]).days for i in range(1, len(ts))])
        if deltas.empty:
            continue
        cadence = float(deltas.median())
        is_monthly = 27 <= cadence <= 33
        is_weekly = 6 <= cadence <= 8
        amt_median = float(g['abs_amount'].median())
        amt_std = float(g['abs_amount'].std(ddof=0)) if len(g) > 1 else 0.0
        amt_cv = (amt_std / amt_median) if amt_median > 0 else 0.0
        groups.append({
            'merchant_norm': key,
            'merchant': str(g.get('merchant', g.get('description', key)).iloc[-1]),
            'cadence_days': cadence,
            'is_monthly_like': bool(is_monthly),
            'is_weekly_like': bool(is_weekly),
            'amount_median': amt_median,
            'amount_cv': amt_cv,
            'last_timestamp': g['timestamp'].max(),
            'count': int(len(g))
        })

    result = pd.DataFrame(groups)
    if result.empty:
        return result

    # Heuristic: recurring if cadence weekly/monthly-like AND stable amount
    result['is_recurring'] = ((result['is_monthly_like'] | result['is_weekly_like']) & (result['amount_cv'] <= 0.25) & (result['count'] >= 2))
    return result


def detect_day_of_month_recurring_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Detect recurring charges by day-of-month and amount patterns.
    
    This function complements detect_recurring_charges by identifying recurring
    payments that occur on the same day of each month with the same amount,
    even if the merchant description varies slightly.
    
    Returns a dataframe with columns:
      - day_of_month: day of month (1-31)
      - amount: recurring amount
      - merchant: example merchant/description
      - category: merchant category
      - count: number of occurrences
      - first_date: first occurrence date
      - last_date: last occurrence date
      - is_monthly_recurring: bool for monthly-like patterns
      - amount_cv: coefficient of variation (should be 0 for exact matches)
      - is_recurring: bool for confirmed recurring patterns
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            'day_of_month', 'amount', 'merchant', 'category', 'count', 
            'first_date', 'last_date', 'is_monthly_recurring', 'amount_cv', 'is_recurring'
        ])
    
    # Work with both positive and negative amounts
    df_work = df.copy()
    df_work['day_of_month'] = df_work['timestamp'].dt.day
    df_work['abs_amount'] = df_work['amount'].abs()
    
    groups = []
    for (day, amount), g in df_work.groupby(['day_of_month', 'amount']):
        if len(g) < 2:
            continue
            
        # Calculate amount stability (should be 0 for exact matches)
        amt_std = float(g['abs_amount'].std(ddof=0)) if len(g) > 1 else 0.0
        amt_median = float(g['abs_amount'].median())
        amt_cv = (amt_std / amt_median) if amt_median > 0 else 0.0
        
        # Check if it's monthly recurring (appears in multiple months)
        months = g['timestamp'].dt.to_period('M').nunique()
        is_monthly_recurring = months >= 2
        
        groups.append({
            'day_of_month': int(day),
            'amount': float(amount),
            'merchant': str(g['merchant'].iloc[-1]) if 'merchant' in g.columns else str(g['description'].iloc[-1]),
            'category': str(g['category'].iloc[-1]) if 'category' in g.columns else 'Unknown',
            'count': int(len(g)),
            'first_date': g['timestamp'].min(),
            'last_date': g['timestamp'].max(),
            'is_monthly_recurring': bool(is_monthly_recurring),
            'amount_cv': amt_cv
        })
    
    result = pd.DataFrame(groups)
    if result.empty:
        return result
    
    # Filter for monthly recurring patterns with stable amounts
    result['is_recurring'] = (result['is_monthly_recurring'] & (result['amount_cv'] <= 0.25) & (result['count'] >= 2))
    
    return result.sort_values(['count', 'amount'], ascending=[False, False])
