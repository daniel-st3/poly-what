from __future__ import annotations

import asyncio
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from watchdog.core.config import Settings
from watchdog.db.models import NewsEvent
from watchdog.news.models import NewsItem
from watchdog.news.sources import collect_news_once

analyzer = SentimentIntensityAnalyzer()


def persist_news(session: Session, items: Iterable[NewsItem]) -> int:
    inserted = 0
    for item in items:
        existing = session.execute(
            select(NewsEvent.id).where(NewsEvent.headline == item.headline, NewsEvent.source == item.source)
        ).first()
        if existing:
            continue

        sentiment = analyzer.polarity_scores(item.headline + " " + (item.raw_text or ""))["compound"]
        session.add(
            NewsEvent(
                headline=item.headline,
                source=item.source,
                url=item.url,
                raw_text=item.raw_text,
                domain_tags=item.domain_tags,
                sentiment_score=sentiment,
                processed=False,
            )
        )
        inserted += 1

    session.commit()
    return inserted


async def ingest_news_once(settings: Settings, session_factory: sessionmaker[Session]) -> int:
    items = await collect_news_once(settings)
    with session_factory() as session:
        return persist_news(session, items)


def ingest_news_once_sync(settings: Settings, session_factory: sessionmaker[Session]) -> int:
    return asyncio.run(ingest_news_once(settings, session_factory))
