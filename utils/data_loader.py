from __future__ import annotations

from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd

from .models import AnalysisResult, Transaction, Insight


def load_demo_dataframe() -> pd.DataFrame:
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    data = {
        "timestamp": [datetime.combine(d, datetime.min.time()) for d in days],
        "amount": [-12.5, -8.9, -24.0, -3.2, -15.0, -7.7, -18.3],
        "currency": ["£"] * 7,
        "category": [
            "Food & Drink",
            "Transport",
            "Entertainment",
            "Food & Drink",
            "Groceries",
            "Transport",
            "Social",
        ],
        "merchant": [
            "Pret",
            "Uber",
            "Spotify",
            "Costa",
            "Tesco",
            "TfL",
            "Pub",
        ],
        "description": [
            "Coffee and snack",
            "Ride to meetup",
            "Monthly subscription",
            "Latte",
            "Weekly groceries",
            "Tube fare",
            "Friends night out",
        ],
    }
    return pd.DataFrame(data)


def analyze_demo_dataframe(df: pd.DataFrame) -> AnalysisResult:
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
        totals[t.category or "Uncategorized"] = totals.get(t.category or "Uncategorized", 0.0) + t.amount

    total_spent = sum(a for a in (t.amount for t in transactions) if a < 0)

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
        total_spent=abs(total_spent),
        num_transactions=len(transactions),
    )
