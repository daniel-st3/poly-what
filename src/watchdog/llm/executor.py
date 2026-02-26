from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from watchdog.core.config import Settings
from watchdog.llm.types import ExecutorDecision

LOGGER = logging.getLogger(__name__)


class BaseExecutorAgent(ABC):
    @abstractmethod
    async def decide(self, context: dict[str, Any]) -> ExecutorDecision:
        raise NotImplementedError


class MockExecutorAgent(BaseExecutorAgent):
    async def decide(self, context: dict[str, Any]) -> ExecutorDecision:
        divergence = float(context.get("divergence", 0.0))
        model_probability = float(context.get("model_probability", 0.5))
        market_probability = float(context.get("market_probability", 0.5))

        if divergence < float(context.get("min_divergence", 0.15)):
            return ExecutorDecision(
                trade=False,
                side="NONE",
                confidence=0.20,
                limit_price=None,
                reason_might_be_wrong="Signal not strong enough for execution",
                rationale="Divergence below threshold",
            )

        side = "YES" if model_probability > market_probability else "NO"
        limit_price = model_probability if side == "YES" else (1.0 - model_probability)

        return ExecutorDecision(
            trade=True,
            side=side,
            confidence=min(0.65 + divergence, 0.95),
            limit_price=max(0.01, min(0.99, limit_price)),
            reason_might_be_wrong="Calibration may be stale during regime shift",
            rationale="Mock decision based on calibrated divergence",
        )


class AnthropicExecutorAgent(BaseExecutorAgent):
    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for executor_provider=anthropic")
        self.api_key = settings.anthropic_api_key
        self.model = settings.executor_model

    async def decide(self, context: dict[str, Any]) -> ExecutorDecision:
        schema_instruction = (
            "Return ONLY valid JSON with keys: trade (bool), side ('YES'|'NO'|'NONE'), "
            "confidence (0..1), limit_price (0..1 or null), reason_might_be_wrong (string), rationale (string)."
        )
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a disciplined prediction market execution agent. "
                    + schema_instruction
                    + " Context: "
                    + json.dumps(context)
                ),
            }
        ]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 400,
            "temperature": 0,
            "messages": messages,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        text_out = _extract_text_from_anthropic_response(data)
        try:
            return ExecutorDecision.model_validate_json(text_out)
        except Exception as exc:
            LOGGER.warning("Executor produced non-JSON output: %s", exc)
            return ExecutorDecision(
                trade=False,
                side="NONE",
                confidence=0.0,
                limit_price=None,
                reason_might_be_wrong="Executor output parse failure",
                rationale="Invalid executor response JSON",
            )


def _extract_text_from_anthropic_response(data: dict[str, Any]) -> str:
    content = data.get("content", [])
    for item in content:
        if item.get("type") == "text":
            return str(item.get("text", "")).strip()
    return "{}"


def build_executor(settings: Settings) -> BaseExecutorAgent:
    if settings.executor_provider == "mock":
        return MockExecutorAgent()
    if settings.executor_provider == "anthropic":
        return AnthropicExecutorAgent(settings)
    raise ValueError(f"Unsupported executor provider: {settings.executor_provider}")
