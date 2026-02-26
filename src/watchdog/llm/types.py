from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouterDecision(BaseModel):
    relevant: bool
    market_slugs: list[str] = Field(default_factory=list)
    impact_direction: Literal["up", "down", "uncertain"] = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""


class ExecutorDecision(BaseModel):
    trade: bool
    side: Literal["YES", "NO", "NONE"] = "NONE"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limit_price: float | None = Field(default=None, ge=0.0, le=1.0)
    reason_might_be_wrong: str
    rationale: str
