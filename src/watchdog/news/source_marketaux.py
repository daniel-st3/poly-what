from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from watchdog.core.config import Settings
from watchdog.news.models import NewsItem

LOGGER = logging.getLogger(__name__)

MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"


async def fetch_marketaux(settings: Settings, max_items: int = 25) -> list[NewsItem]:
    """Fetch financial news with entities + sentiment from Marketaux API.

    Free tier: 100 requests/day.
    """
    if not settings.marketaux_enabled:
        return []

    if not settings.marketaux_api_key:
        LOGGER.warning("Marketaux enabled but MARKETAUX_API_KEY not set; skipping")
        return []

    params = {
        "filter_entities": "true",
        "language": "en",
        "api_token": settings.marketaux_api_key,
        "limit": str(min(max_items, 50)),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(MARKETAUX_URL, params=params)
        response.raise_for_status()
        data = response.json()

    articles = data.get("data", [])
    out: list[NewsItem] = []
    now = datetime.now(UTC)

    for article in articles:
        title = str(article.get("title") or "").strip()
        if not title:
            continue

        # Extract sentiment from Marketaux response
        sentiment_info = ""
        if article.get("entities"):
            entities = article["entities"]
            sentiments = [
                f"{e.get('symbol', '?')}:{e.get('sentiment_score', 0)}"
                for e in entities[:5]
                if isinstance(e, dict)
            ]
            sentiment_info = f" | entities: {', '.join(sentiments)}"

        raw_text = (article.get("description") or article.get("snippet") or "") + sentiment_info

        out.append(
            NewsItem(
                headline=title,
                source="marketaux",
                url=article.get("url"),
                raw_text=raw_text,
                domain_tags="financial_news",
                received_at=now,
            )
        )

    LOGGER.info("Marketaux fetched %d articles", len(out))
    return out
