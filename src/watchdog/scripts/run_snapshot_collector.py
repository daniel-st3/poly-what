from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from sqlalchemy import select

from watchdog.core.config import get_settings
from watchdog.core.logging import configure_logging
from watchdog.db.init import init_db
from watchdog.db.models import Market, MarketSnapshot
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.polymarket_cli import PolymarketCli
from watchdog.services.market_sync import sync_markets_once

LOGGER = logging.getLogger(__name__)


def _extract_orderbook_metrics(payload: Any) -> dict[str, float | None]:
    data = payload
    if isinstance(payload, dict):
        for key in ("orderbook", "data"):
            if isinstance(payload.get(key), dict):
                data = payload[key]
                break

    bids = data.get("bids", []) if isinstance(data, dict) else []
    asks = data.get("asks", []) if isinstance(data, dict) else []

    def _normalize(levels: Any) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        if not isinstance(levels, list):
            return out
        for level in levels:
            if isinstance(level, dict):
                price = float(level.get("price") or level.get("px") or 0.0)
                size = float(level.get("size") or level.get("qty") or level.get("amount") or 0.0)
            elif isinstance(level, list) and len(level) >= 2:
                price = float(level[0])
                size = float(level[1])
            else:
                continue
            if price > 0 and size >= 0:
                out.append((price, size))
        return out

    bid_lvls = _normalize(bids)
    ask_lvls = _normalize(asks)

    bid = max((p for p, _ in bid_lvls), default=None)
    ask = min((p for p, _ in ask_lvls), default=None)
    mid = (bid + ask) / 2 if bid is not None and ask is not None else bid or ask

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": (ask - bid) if bid is not None and ask is not None else None,
        "bid_volume": sum(size for _, size in bid_lvls),
        "ask_volume": sum(size for _, size in ask_lvls),
        "total_volume": sum(size for _, size in bid_lvls) + sum(size for _, size in ask_lvls),
    }


def _collect_cycle(session_factory, cli: PolymarketCli) -> tuple[int, int]:
    with session_factory() as session:
        synced_markets = sync_markets_once(session, cli, limit=500)
        active_markets = session.execute(
            select(Market).where(Market.status.in_(["active", "open"])).order_by(Market.id.asc())
        ).scalars().all()

        snap_count = 0
        for market in active_markets:
            response = cli.orderbook(market.slug)
            metrics = _extract_orderbook_metrics(response.payload)
            session.add(
                MarketSnapshot(
                    market_id=market.id,
                    bid=metrics.get("bid"),
                    ask=metrics.get("ask"),
                    mid=metrics.get("mid"),
                    spread=metrics.get("spread"),
                    bid_volume=metrics.get("bid_volume"),
                    ask_volume=metrics.get("ask_volume"),
                    total_volume=metrics.get("total_volume"),
                    cli_latency_ms=response.latency_ms,
                    raw_json=str(response.payload),
                )
            )
            snap_count += 1

        session.commit()
        return synced_markets, snap_count


async def main() -> None:
    settings = get_settings()
    configure_logging(settings)

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    cli = PolymarketCli(settings)
    try:
        cli.startup_check()
    except Exception as exc:
        LOGGER.error("Snapshot collector startup failed: %s", exc)
        return

    LOGGER.info("Starting snapshot collector interval=%ss", settings.snapshot_interval_seconds)

    while True:
        cycle_start = time.perf_counter()
        retries = 0

        while True:
            try:
                synced_count, snap_count = await asyncio.to_thread(_collect_cycle, session_factory, cli)
                break
            except Exception as exc:  # pragma: no cover - runtime loop safeguard
                retries += 1
                if retries > settings.snapshot_retry_max:
                    LOGGER.exception("Snapshot cycle failed after retries: %s", exc)
                    synced_count, snap_count = 0, 0
                    break
                delay = settings.snapshot_retry_base_seconds * (2 ** (retries - 1))
                LOGGER.warning(
                    "Snapshot cycle error (attempt %s/%s). Backing off %ss: %s",
                    retries,
                    settings.snapshot_retry_max,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        elapsed = time.perf_counter() - cycle_start
        LOGGER.info(
            "Snapshot cycle complete synced_markets=%s snapshots=%s duration=%.2fs",
            synced_count,
            snap_count,
            elapsed,
        )

        sleep_for = max(0.0, settings.snapshot_interval_seconds - elapsed)
        await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Snapshot collector stopped by user.")
