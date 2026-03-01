from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx
import praw

from watchdog.core.config import Settings
from watchdog.news.models import NewsItem
from watchdog.news.source_gdelt_gkg import fetch_gdelt_gkg
from watchdog.news.source_marketaux import fetch_marketaux
from watchdog.news.source_polymarket_volume import fetch_polymarket_volume_spikes

LOGGER = logging.getLogger(__name__)

RSS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/worldNews",
    "ap": "https://feeds.apnews.com/apf-topnews",
    "nyt": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
}

REDDIT_SUBREDDITS = ["worldnews", "politics", "cryptocurrency", "sports", "finance"]


async def fetch_gdelt(settings: Settings, max_records: int = 25) -> list[NewsItem]:
    if not settings.gdelt_enabled:
        return []

    params = {
        "query": "(election OR crypto OR fed OR inflation OR war)",
        "mode": "ArtList",
        "format": "json",
        "sort": "DateDesc",
        "maxrecords": str(max_records),
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()

    articles = data.get("articles", [])
    out: list[NewsItem] = []

    for article in articles:
        title = str(article.get("title") or "").strip()
        if not title:
            continue
        out.append(
            NewsItem(
                headline=title,
                source="gdelt",
                url=article.get("url"),
                raw_text=article.get("seendate"),
                domain_tags="global_news",
                received_at=datetime.now(UTC),
            )
        )
    return out


def _parse_rss_feed(name: str, url: str, limit_per_feed: int) -> list[NewsItem]:
    parsed = feedparser.parse(url)
    now = datetime.now(UTC)
    items: list[NewsItem] = []
    for entry in parsed.entries[:limit_per_feed]:
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        summary = str(entry.get("summary") or "")
        items.append(
            NewsItem(
                headline=title,
                source=f"rss:{name}",
                url=entry.get("link"),
                raw_text=summary,
                domain_tags="rss",
                received_at=now,
            )
        )
    return items


async def fetch_rss(settings: Settings, limit_per_feed: int = 15) -> list[NewsItem]:
    if not settings.rss_enabled:
        return []

    tasks = [
        asyncio.to_thread(_parse_rss_feed, name=name, url=url, limit_per_feed=limit_per_feed)
        for name, url in RSS_FEEDS.items()
    ]
    grouped = await asyncio.gather(*tasks, return_exceptions=True)

    out: list[NewsItem] = []
    for result in grouped:
        if isinstance(result, Exception):
            LOGGER.warning("RSS fetch failed: %s", result)
            continue
        out.extend(result)
    return out


def _fetch_reddit_sync(settings: Settings, limit_per_subreddit: int = 10) -> list[NewsItem]:
    if not settings.reddit_enabled:
        return []

    if not settings.reddit_client_id or not settings.reddit_client_secret:
        LOGGER.warning("Reddit enabled but credentials missing; skipping reddit ingestion")
        return []

    client = praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
    )

    out: list[NewsItem] = []
    now = datetime.now(UTC)
    for subreddit_name in REDDIT_SUBREDDITS:
        subreddit = client.subreddit(subreddit_name)
        for post in subreddit.hot(limit=limit_per_subreddit):
            out.append(
                NewsItem(
                    headline=post.title,
                    source=f"reddit:{subreddit_name}",
                    url=getattr(post, "url", None),
                    raw_text=getattr(post, "selftext", None),
                    domain_tags="reddit",
                    received_at=now,
                )
            )
    return out


async def fetch_reddit(settings: Settings, limit_per_subreddit: int = 10) -> list[NewsItem]:
    return await asyncio.to_thread(_fetch_reddit_sync, settings, limit_per_subreddit)


async def collect_news_once(settings: Settings) -> list[NewsItem]:
    tasks = [
        fetch_gdelt(settings),
        fetch_rss(settings),
        fetch_reddit(settings),
        # New high-signal sources
        fetch_marketaux(settings),
        fetch_polymarket_volume_spikes(settings),
        fetch_gdelt_gkg(settings),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: list[NewsItem] = []
    for result in results:
        if isinstance(result, Exception):
            LOGGER.warning("News source failed: %s", result)
            continue
        merged.extend(result)

    # naive dedupe by (headline, source)
    deduped: dict[tuple[str, str], NewsItem] = {}
    for item in merged:
        deduped[(item.headline, item.source)] = item

    return list(deduped.values())



async def brave_fallback(settings: Settings, query: str) -> list[dict[str, Any]]:
    if not settings.brave_api_key:
        return []

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.brave_api_key,
    }
    params = {"q": query, "count": 10}

    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.get("https://api.search.brave.com/res/v1/web/search", params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    return data.get("web", {}).get("results", [])
