from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import httpx

from watchdog.core.config import Settings
from watchdog.llm.types import RouterDecision
from watchdog.news.models import NewsItem

LOGGER = logging.getLogger(__name__)


class BaseRouterAgent(ABC):
    @abstractmethod
    async def route(self, news_item: NewsItem, tracked_market_slugs: list[str]) -> RouterDecision:
        raise NotImplementedError


class MockRouterAgent(BaseRouterAgent):
    KEYWORDS: ClassVar[dict[str, str]] = {
        "crypto": "crypto",
        "election": "politics",
        "fed": "macro",
        "inflation": "macro",
        "war": "geopolitics",
        "etf": "crypto",
        "bitcoin": "crypto",
    }

    async def route(self, news_item: NewsItem, tracked_market_slugs: list[str]) -> RouterDecision:
        text = f"{news_item.headline} {news_item.raw_text or ''}".lower()
        matched = [k for k in self.KEYWORDS if k in text]
        if not matched:
            return RouterDecision(relevant=False, market_slugs=[], impact_direction="uncertain", confidence=0.12)

        linked = tracked_market_slugs[:3]
        return RouterDecision(
            relevant=True,
            market_slugs=linked,
            impact_direction="uncertain",
            confidence=min(0.4 + 0.1 * len(matched), 0.9),
            rationale=f"Matched keywords: {', '.join(matched[:4])}",
        )


class OpenAIRouterAgent(BaseRouterAgent):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for router_provider=openai")
        self.api_key = settings.openai_api_key
        self.model = settings.router_model

    async def route(self, news_item: NewsItem, tracked_market_slugs: list[str]) -> RouterDecision:
        schema_instruction = (
            "Return ONLY valid JSON with keys: relevant (bool), market_slugs (array of strings), "
            "impact_direction ('up'|'down'|'uncertain'), confidence (0..1), rationale (string)."
        )
        user_prompt = {
            "headline": news_item.headline,
            "source": news_item.source,
            "text": news_item.raw_text,
            "tracked_market_slugs": tracked_market_slugs,
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a prediction market relevance router. " + schema_instruction,
                },
                {
                    "role": "user",
                    "content": json.dumps(user_prompt),
                },
            ],
            "temperature": 0,
            "response_format": { "type": "json_object" }
        }

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        text_out = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
        try:
            return RouterDecision.model_validate_json(text_out)
        except Exception as exc:
            LOGGER.warning("Router produced non-JSON output: %s", exc)
            return RouterDecision(
                relevant=False,
                market_slugs=[],
                impact_direction="uncertain",
                confidence=0.0,
                rationale="Invalid router response JSON",
            )


def build_router(settings: Settings) -> BaseRouterAgent:
    if settings.router_provider == "mock":
        return MockRouterAgent()
    if settings.router_provider == "openai":
        return OpenAIRouterAgent(settings)
    raise ValueError(f"Unsupported router provider: {settings.router_provider}")
