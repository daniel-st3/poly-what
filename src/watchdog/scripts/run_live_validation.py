from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from sqlalchemy import select

from watchdog.core.config import get_settings
from watchdog.core.exceptions import LiveTradingDisabledError
from watchdog.core.logging import configure_logging
from watchdog.db.init import init_db
from watchdog.db.models import Market, NewsEvent, Signal, Telemetry, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.llm.executor import build_executor
from watchdog.llm.router import build_router
from watchdog.market_data.manifold_client import ManifoldClient, ManifoldMarket
from watchdog.market_data.polymarket_cli import PolymarketCli
from watchdog.news.ingest import persist_news
from watchdog.news.models import NewsItem
from watchdog.news.sources import collect_news_once
from watchdog.services.market_sync import sync_markets_once
from watchdog.services.pipeline import _extract_orderbook_metrics, _hours_to_resolution
from watchdog.signals.calibration import CalibrationSurfaceService

LOGGER = logging.getLogger(__name__)


def _ensure_live_enabled() -> None:
    settings = get_settings()
    if not settings.enable_live_trading:
        raise LiveTradingDisabledError(
            "run_live_validation requires ENABLE_LIVE_TRADING=true. "
            "This script does not run in live mode unless explicitly enabled."
        )


def _now_utc() -> datetime:
    return datetime.now(UTC)


async def _fetch_market_prob(
    platform: Literal["polymarket", "manifold"],
    slug: str,
    cli: PolymarketCli | None,
    manifold: ManifoldClient | None,
) -> float:
    if platform == "polymarket":
        assert cli is not None
        response = await asyncio.to_thread(cli.orderbook, slug)
        metrics = _extract_orderbook_metrics(response.payload)
        return float(metrics.get("mid") or 0.5)

    assert manifold is not None
    market = await manifold.get_market_by_slug(slug)
    return float(market.probability)


async def _update_future_price(
    delay_seconds: int,
    telemetry_id: int,
    column: Literal["market_price_1m", "market_price_5m"],
    platform: Literal["polymarket", "manifold"],
    slug: str,
    session_factory,
    cli: PolymarketCli | None,
    manifold: ManifoldClient | None,
) -> None:
    await asyncio.sleep(delay_seconds)
    try:
        price = await _fetch_market_prob(platform, slug, cli, manifold)
        with session_factory() as session:
            row = session.get(Telemetry, telemetry_id)
            if row is None:
                return
            setattr(row, column, price)
            session.commit()
    except Exception as exc:  # pragma: no cover - runtime loop safety
        LOGGER.warning("Deferred %s update failed for telemetry_id=%s: %s", column, telemetry_id, exc)


def _market_from_manifold_row(session, row: ManifoldMarket) -> Market:
    domain = str(row.raw.get("groupSlug") or "unknown").lower()
    existing = session.execute(select(Market).where(Market.slug == row.slug)).scalar_one_or_none()
    if existing is not None:
        existing.question = row.question
        existing.domain = domain
        existing.resolution_time = row.close_time
        existing.status = "resolved" if row.is_resolved else "active"
        existing.resolution_outcome = str(row.outcome) if row.outcome is not None else None
        return existing

    market = Market(
        slug=row.slug,
        question=row.question,
        domain=domain,
        resolution_time=row.close_time,
        status="resolved" if row.is_resolved else "active",
        resolution_outcome=str(row.outcome) if row.outcome is not None else None,
    )
    session.add(market)
    session.flush()
    return market


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run $50 live validation experiment")
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--bankroll", type=float, default=50.0)
    parser.add_argument("--platform", type=str, choices=["polymarket", "manifold"], default="polymarket")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()

    _ensure_live_enabled()
    if args.bankroll > 50:
        raise ValueError("Bankroll cap exceeded: max bankroll is $50 for live validation")

    settings = get_settings()
    configure_logging(settings)

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    platform: Literal["polymarket", "manifold"] = args.platform
    cli = PolymarketCli(settings) if platform == "polymarket" else None
    manifold = (
        ManifoldClient(settings.manifold_api_base_url, settings.manifold_api_key)
        if platform == "manifold"
        else None
    )

    if cli is not None:
        cli.startup_check()

    router = build_router(settings)
    executor = build_executor(settings)
    calibration = CalibrationSurfaceService()

    LOGGER.warning(
        "LIVE VALIDATION MODE: experiment_id=%s platform=%s bankroll=%.2f",
        args.experiment_id,
        platform,
        args.bankroll,
    )

    deferred_tasks: set[asyncio.Task] = set()

    while True:
        with session_factory() as session:
            calibration.load_from_db(session)

        # Ingest news if queue is empty.
        with session_factory() as session:
            pending = session.execute(
                select(NewsEvent)
                .where(NewsEvent.processed.is_(False))
                .order_by(NewsEvent.received_at.asc())
                .limit(20)
            ).scalars().all()

            if not pending:
                new_items = await collect_news_once(settings)
                if new_items:
                    persist_news(session, new_items)
                pending = session.execute(
                    select(NewsEvent)
                    .where(NewsEvent.processed.is_(False))
                    .order_by(NewsEvent.received_at.asc())
                    .limit(20)
                ).scalars().all()

        if platform == "polymarket":
            with session_factory() as session:
                sync_markets_once(session, cli, limit=200)
                tracked = session.execute(
                    select(Market).where(Market.status.in_(["active", "open"]))
                ).scalars().all()
                tracked_by_slug = {m.slug: m for m in tracked}
        else:
            active = await manifold.get_active_markets(limit=200)
            with session_factory() as session:
                tracked_by_slug = {}
                for item in active:
                    row = _market_from_manifold_row(session, item)
                    tracked_by_slug[row.slug] = row
                session.commit()

        tracked_slugs = list(tracked_by_slug.keys())

        for news_row in pending:
            with session_factory() as session:
                open_positions = session.execute(
                    select(Trade.id)
                    .join(Signal, Trade.signal_id == Signal.id)
                    .where(Trade.closed_at.is_(None), Signal.experiment_id == args.experiment_id)
                ).all()
            if len(open_positions) >= settings.live_validation_max_positions:
                LOGGER.info("Max open positions reached (%s). Skipping new entries.", len(open_positions))
                break

            ts_news_received = news_row.received_at
            if ts_news_received.tzinfo is None:
                ts_news_received = ts_news_received.replace(tzinfo=UTC)

            news_item = NewsItem(
                headline=news_row.headline,
                source=news_row.source,
                url=news_row.url,
                raw_text=news_row.raw_text,
                domain_tags=news_row.domain_tags,
                received_at=ts_news_received,
            )

            ts_router_start = _now_utc()
            route = await router.route(news_item, tracked_slugs)
            ts_router_completed = _now_utc()

            if not route.relevant or not route.market_slugs:
                with session_factory() as session:
                    session.add(
                        Telemetry(
                            pipeline_id=uuid.uuid4().hex[:12],
                            news_event_id=news_row.id,
                            ts_news_received=ts_news_received,
                            ts_router_completed=ts_router_completed,
                            router_latency_ms=int((ts_router_completed - ts_router_start).total_seconds() * 1000),
                            total_latency_ms=int((ts_router_completed - ts_news_received).total_seconds() * 1000),
                        )
                    )
                    row = session.get(NewsEvent, news_row.id)
                    if row:
                        row.processed = True
                    session.commit()
                continue

            for slug in route.market_slugs[:3]:
                market = tracked_by_slug.get(slug)
                if market is None:
                    continue

                market_probability = float(np.clip(await _fetch_market_prob(platform, slug, cli, manifold), 0.01, 0.99))

                ts_calibration_start = _now_utc()
                calibration_result = calibration.calibrate(
                    market_probability=market_probability,
                    hours_to_resolution=_hours_to_resolution(market.resolution_time),
                    domain=market.domain,
                    sentiment_score=float(news_row.sentiment_score or 0.0),
                )
                ts_calibration_completed = _now_utc()

                divergence = abs(calibration_result.model_probability - market_probability)
                if divergence <= settings.min_divergence_live:
                    continue

                ts_executor_start = _now_utc()
                decision = await executor.decide(
                    {
                        "market_slug": slug,
                        "market_question": market.question,
                        "market_probability": market_probability,
                        "model_probability": calibration_result.model_probability,
                        "divergence": divergence,
                        "min_divergence": settings.min_divergence_live,
                    }
                )
                ts_executor_completed = _now_utc()

                if not decision.trade or decision.side not in {"YES", "NO"}:
                    continue

                quote_price = decision.limit_price or market_probability
                quote_price = float(np.clip(quote_price, 0.01, 0.99))
                position_usdc = min(settings.live_validation_position_size_usdc, args.bankroll)
                order_size = position_usdc / quote_price

                ts_order_submitted: datetime | None = None
                order_id = f"sim-{uuid.uuid4().hex[:10]}"
                is_paper = platform != "polymarket"

                if platform == "polymarket":
                    ts_order_submitted = _now_utc()
                    response = await asyncio.to_thread(
                        cli.place_limit_order,
                        slug,
                        decision.side,
                        quote_price,
                        order_size,
                    )
                    payload = response.payload if isinstance(response.payload, dict) else {}
                    order_id = str(payload.get("order_id") or payload.get("id") or order_id)
                else:
                    ts_order_submitted = _now_utc()

                pipeline_id = uuid.uuid4().hex[:12]
                with session_factory() as session:
                    signal = Signal(
                        market_id=market.id,
                        news_event_id=news_row.id,
                        model_probability=calibration_result.model_probability,
                        market_probability=market_probability,
                        divergence=divergence,
                        signal_type="live_validation",
                        router_confidence=route.confidence,
                        calibration_adjustments=calibration_result.adjustment,
                        executor_confidence=decision.confidence,
                        experiment_id=args.experiment_id,
                        should_trade=True,
                        rationale=decision.rationale,
                    )
                    session.add(signal)
                    session.flush()

                    trade = Trade(
                        market_id=market.id,
                        signal_id=signal.id,
                        side=decision.side,
                        size=order_size,
                        entry_price=quote_price,
                        slippage=0.0,
                        kelly_fraction=min(settings.max_position_per_market, position_usdc / max(args.bankroll, 1e-9)),
                        confidence_score=decision.confidence,
                        order_id=order_id,
                        is_paper=is_paper,
                        opened_at=ts_order_submitted,
                    )
                    session.add(trade)

                    telemetry = Telemetry(
                        pipeline_id=pipeline_id,
                        market_id=market.id,
                        news_event_id=news_row.id,
                        ts_news_received=ts_news_received,
                        ts_router_completed=ts_router_completed,
                        ts_calibration_completed=ts_calibration_completed,
                        ts_executor_completed=ts_executor_completed,
                        ts_order_submitted=ts_order_submitted,
                        market_price_at_signal=market_probability,
                        router_latency_ms=int((ts_router_completed - ts_router_start).total_seconds() * 1000),
                        calibration_latency_ms=int((ts_calibration_completed - ts_calibration_start).total_seconds() * 1000),
                        executor_latency_ms=int((ts_executor_completed - ts_executor_start).total_seconds() * 1000),
                        order_submission_latency_ms=int((_now_utc() - ts_order_submitted).total_seconds() * 1000),
                        total_latency_ms=int((_now_utc() - ts_news_received).total_seconds() * 1000),
                    )
                    session.add(telemetry)

                    row = session.get(NewsEvent, news_row.id)
                    if row:
                        row.processed = True

                    session.commit()
                    telemetry_id = telemetry.id

                task_1m = asyncio.create_task(
                    _update_future_price(
                        delay_seconds=60,
                        telemetry_id=telemetry_id,
                        column="market_price_1m",
                        platform=platform,
                        slug=slug,
                        session_factory=session_factory,
                        cli=cli,
                        manifold=manifold,
                    )
                )
                task_5m = asyncio.create_task(
                    _update_future_price(
                        delay_seconds=300,
                        telemetry_id=telemetry_id,
                        column="market_price_5m",
                        platform=platform,
                        slug=slug,
                        session_factory=session_factory,
                        cli=cli,
                        manifold=manifold,
                    )
                )

                deferred_tasks.update({task_1m, task_5m})
                task_1m.add_done_callback(deferred_tasks.discard)
                task_5m.add_done_callback(deferred_tasks.discard)

            with session_factory() as session:
                row = session.get(NewsEvent, news_row.id)
                if row:
                    row.processed = True
                    session.commit()

        await asyncio.sleep(max(args.poll_seconds, 3))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except LiveTradingDisabledError as exc:
        print(f"Live validation halted: {exc}")
        raise SystemExit(1) from None
