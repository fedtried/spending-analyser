from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime
    amount: float = Field(..., description="Signed amount; negative for spend")
    currency: str = Field("£", description="Currency symbol or code")
    category: Optional[str] = Field(None, description="High-level category")
    merchant: Optional[str] = Field(None, description="Merchant or payee")
    description: Optional[str] = None


class Insight(BaseModel):
    title: str
    summary: str
    kind: str = Field("general", description="Type of insight, e.g., trend, anomaly")
    score: Optional[float] = Field(None, ge=0, le=1, description="Confidence or impact score")


class AnalysisResult(BaseModel):
    transactions: List[Transaction]
    insights: List[Insight] = []
    totals_by_category: dict[str, float] = {}
    total_spent: float = 0.0
    total_income: float = 0.0
    total_net: float = 0.0
    num_transactions: int = 0
