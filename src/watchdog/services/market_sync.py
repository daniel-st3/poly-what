from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from dateutil import parser as dtparser
from sqlalchemy import select
from sqlalchemy.orm import Session

from watchdog.db.models import Market
from watchdog.market_data.polymarket_cli import PolymarketCli


def _safe_parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = dtparser.parse(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _extract_markets_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("markets", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def sync_markets_once(session: Session, cli: PolymarketCli, limit: int = 200) -> int:
    response = cli.list_markets(limit=limit)
    market_rows = _extract_markets_payload(response.payload)
    upserts = 0

    for row in market_rows:
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue

        existing = session.execute(select(Market).where(Market.slug == slug)).scalar_one_or_none()
        if existing is None:
            market = Market(
                slug=slug,
                question=str(row.get("question") or row.get("title") or slug),
                domain=str(row.get("domain") or row.get("category") or "unknown").lower(),
                yes_token_id=str(row.get("yesTokenId") or row.get("yes_token_id") or "") or None,
                no_token_id=str(row.get("noTokenId") or row.get("no_token_id") or "") or None,
                condition_id=str(row.get("conditionId") or row.get("condition_id") or "") or None,
                resolution_time=_safe_parse_datetime(row.get("endDate") or row.get("resolution_time")),
                status=str(row.get("status") or "active").lower(),
                resolution_outcome=str(row.get("outcome") or "") or None,
            )
            session.add(market)
        else:
            existing.question = str(row.get("question") or row.get("title") or existing.question)
            existing.domain = str(row.get("domain") or row.get("category") or existing.domain).lower()
            existing.yes_token_id = str(row.get("yesTokenId") or row.get("yes_token_id") or existing.yes_token_id or "") or None
            existing.no_token_id = str(row.get("noTokenId") or row.get("no_token_id") or existing.no_token_id or "") or None
            existing.condition_id = str(row.get("conditionId") or row.get("condition_id") or existing.condition_id or "") or None
            existing.resolution_time = _safe_parse_datetime(
                row.get("endDate") or row.get("resolution_time") or existing.resolution_time
            )
            existing.status = str(row.get("status") or existing.status).lower()
            existing.resolution_outcome = str(row.get("outcome") or existing.resolution_outcome or "") or None
        upserts += 1

    session.commit()
    return upserts
