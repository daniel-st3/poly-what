from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import httpx

from watchdog.core.config import Settings
from watchdog.news.models import NewsItem

LOGGER = logging.getLogger(__name__)

GDELT_GKG_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


async def _query_gkg(keyword: str, max_records: int = 10) -> list[dict]:
    """Query GDELT GKG for a single keyword, returning article dicts."""
    params = {
        "query": keyword,
        "mode": "artlist",
        "maxrecords": str(max_records),
        "format": "json",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(GDELT_GKG_URL, params=params)
        response.raise_for_status()
        data = response.json()

    return data.get("articles", [])


async def fetch_gdelt_gkg(
    settings: Settings,
    keywords: list[str] | None = None,
    max_records_per_keyword: int = 10,
) -> list[NewsItem]:
    """Fetch GDELT GKG tone-scored articles for given keywords.

    Uses the free GKG API to get article-level tone scores (-10 to +10)
    which map directly to sentiment signals.

    Keywords are typically extracted from active Polymarket market questions.
    """
    if not settings.gdelt_gkg_enabled:
        return []

    if not keywords:
        # Default keywords relevant to prediction markets
        keywords = ["election", "crypto", "bitcoin", "inflation", "fed", "war", "trade"]

    # Fetch all keywords concurrently
    tasks = [_query_gkg(kw, max_records=max_records_per_keyword) for kw in keywords[:10]]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    now = datetime.now(UTC)
    seen_urls: set[str] = set()
    out: list[NewsItem] = []

    for keyword, result in zip(keywords, results, strict=False):
        if isinstance(result, Exception):
            LOGGER.warning("GDELT GKG query failed for '%s': %s", keyword, result)
            continue

        for article in result:
            if not isinstance(article, dict):
                continue

            title = str(article.get("title") or "").strip()
            url = str(article.get("url") or "").strip()
            if not title:
                continue

            # Dedupe by URL
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)

            # Extract tone score from GDELT (-10 to +10)
            tone = _safe_float(article.get("tone", 0.0))
            seen_date = str(article.get("seendate") or "")
            domain = str(article.get("domain") or "")

            raw_text = (
                f"keyword={keyword} | tone={tone:.2f} | "
                f"seen={seen_date} | domain={domain}"
            )

            out.append(
                NewsItem(
                    headline=title,
                    source="gdelt_gkg",
                    url=url or None,
                    raw_text=raw_text,
                    domain_tags="gdelt_gkg",
                    received_at=now,
                )
            )

    LOGGER.info("GDELT GKG fetched %d articles across %d keywords", len(out), len(keywords))
    return out


def _safe_float(value: object) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
