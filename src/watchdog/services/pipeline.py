from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from watchdog.core.config import Settings
from watchdog.db.models import Market, MarketSnapshot, NewsEvent, Signal, Telemetry, Trade
from watchdog.llm.executor import BaseExecutorAgent
from watchdog.llm.router import BaseRouterAgent
from watchdog.market_data.polymarket_cli import PolymarketCli
from watchdog.news.models import NewsItem
from watchdog.risk.kelly import EmpiricalKellySizer
from watchdog.risk.vpin import TradeFlow, VPINCalculator, should_halt_maker
from watchdog.services.market_sync import sync_markets_once
from watchdog.signals.calibration import CalibrationSurfaceService
from watchdog.signals.telegram_bot import TelegramAlerter

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineStats:
    processed_news: int = 0
    generated_signals: int = 0
    executed_trades: int = 0


def _extract_orderbook_metrics(payload: Any) -> dict[str, float | None]:
    data = payload
    if isinstance(payload, dict):
        for key in ("orderbook", "data"):
            if isinstance(payload.get(key), dict):
                data = payload[key]
                break

    bids = data.get("bids", []) if isinstance(data, dict) else []
    asks = data.get("asks", []) if isinstance(data, dict) else []

    def _normalize_levels(levels: Any) -> list[tuple[float, float]]:
        out: list[tuple[float, float]] = []
        if not isinstance(levels, list):
            return out
        for item in levels:
            if isinstance(item, dict):
                px = float(item.get("price") or item.get("px") or 0.0)
                sz = float(item.get("size") or item.get("qty") or item.get("amount") or 0.0)
            elif isinstance(item, list) and len(item) >= 2:
                px = float(item[0])
                sz = float(item[1])
            else:
                continue
            if px > 0 and sz >= 0:
                out.append((px, sz))
        return out

    bid_lvls = _normalize_levels(bids)
    ask_lvls = _normalize_levels(asks)

    bid = max((px for px, _ in bid_lvls), default=None)
    ask = min((px for px, _ in ask_lvls), default=None)
    mid = (bid + ask) / 2 if bid is not None and ask is not None else bid or ask
    spread = (ask - bid) if bid is not None and ask is not None else None
    bid_volume = sum(sz for _, sz in bid_lvls)
    ask_volume = sum(sz for _, sz in ask_lvls)
    total_volume = bid_volume + ask_volume

    return {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": spread,
        "bid_volume": bid_volume,
        "ask_volume": ask_volume,
        "total_volume": total_volume,
    }


def _hours_to_resolution(resolution_time: datetime | None) -> float:
    if resolution_time is None:
        return 24.0
    now = datetime.now(UTC)
    if resolution_time.tzinfo is None:
        resolution_time = resolution_time.replace(tzinfo=UTC)
    return max((resolution_time - now).total_seconds() / 3600.0, 0.1)


def _news_item_from_row(row: NewsEvent) -> NewsItem:
    received_at = row.received_at
    if received_at.tzinfo is None:
        received_at = received_at.replace(tzinfo=UTC)
    return NewsItem(
        headline=row.headline,
        source=row.source,
        url=row.url,
        raw_text=row.raw_text,
        domain_tags=row.domain_tags,
        received_at=received_at,
    )


def _load_historical_arrays(session: Session, market_id: int) -> tuple[np.ndarray, np.ndarray]:
    signal_rows = session.execute(
        select(Signal.divergence).where(Signal.market_id == market_id, Signal.should_trade.is_(True)).limit(500)
    ).all()
    trade_rows = session.execute(
        select(Trade.pnl, Trade.entry_price)
        .where(Trade.market_id == market_id, Trade.pnl.is_not(None), Trade.entry_price > 0)
        .limit(500)
    ).all()

    edges = np.array([abs(float(row[0])) for row in signal_rows], dtype=np.float64)
    returns = np.array([float(pnl) / float(entry) for pnl, entry in trade_rows], dtype=np.float64)
    return edges, returns


class PipelineRunner:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        cli: PolymarketCli,
        router: BaseRouterAgent,
        executor: BaseExecutorAgent,
        calibration: CalibrationSurfaceService,
        sizer: EmpiricalKellySizer,
        vpin_calc: VPINCalculator,
        alerter: TelegramAlerter | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.cli = cli
        self.router = router
        self.executor = executor
        self.calibration = calibration
        self.sizer = sizer
        self.vpin_calc = vpin_calc
        self.alerter = alerter
        self._background_tasks: set[asyncio.Task] = set()

    async def _update_telemetry_price_after_delay(
        self,
        telemetry_id: int,
        token_id: str,
        delay_seconds: int,
        field_name: str,
        experiment_id: str,
    ) -> None:
        await asyncio.sleep(delay_seconds)
        try:
            response = await asyncio.to_thread(self.cli.orderbook, token_id)
            metrics = _extract_orderbook_metrics(response.payload)
            price = float(metrics.get("mid") or 0.5)

            with self.session_factory() as session:
                row = session.get(Telemetry, telemetry_id)
                if row is None:
                    return
                setattr(row, field_name, price)

                if (
                    row.market_price_at_signal is not None
                    and row.market_price_5m is not None
                    and row.market_price_at_signal < 0.90
                    and row.market_price_5m > 0.90
                ):
                    LOGGER.warning(
                        "LATENCY ALERT: Market moved significantly after signal. "
                        "Possible edge erosion. Review experiment_id: %s",
                        experiment_id,
                    )
                    if self.alerter is not None:
                        delta = row.market_price_5m - row.market_price_at_signal
                        self.alerter.send_latency_alert(
                            experiment_id=experiment_id,
                            signal_ms=int(row.total_latency_ms or 0),
                            price_moved=float(delta),
                        )
                session.commit()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            LOGGER.warning(
                "Failed deferred telemetry update for telemetry_id=%s field=%s: %s",
                telemetry_id,
                field_name,
                exc,
            )

    async def _process_volume_spikes(
        self, session: Session, stats: PipelineStats, experiment_id: str
    ) -> None:
        """Process volume spike news events with a fast path (skip router, lower threshold)."""
        VOLUME_SPIKE_MIN_DIVERGENCE = 0.10

        spike_news = (
            session.execute(
                select(NewsEvent)
                .where(
                    NewsEvent.processed.is_(False),
                    NewsEvent.source == "polymarket_volume_spike",
                )
                .order_by(NewsEvent.received_at.asc())
                .limit(20)
            )
            .scalars()
            .all()
        )

        if not spike_news:
            return

        LOGGER.info("Processing %d volume spike events (fast path)", len(spike_news))

        for news_row in spike_news:
            stats.processed_news += 1
            pipeline_id = uuid.uuid4().hex[:12]
            t0 = time.perf_counter()
            ts_news_received = news_row.received_at

            # Extract slug from raw_text (format: "VOLUME SPIKE: {slug} | ...")
            raw = news_row.raw_text or ""
            slug_part = raw.split("|")[0].replace("VOLUME SPIKE:", "").strip() if raw else ""

            # Try to find a matching market
            market = None
            if slug_part:
                market = session.execute(
                    select(Market).where(Market.slug == slug_part)
                ).scalar_one_or_none()

            if market is None:
                # Try matching by headline keyword in existing markets
                markets = session.execute(
                    select(Market).where(Market.status.in_(["open", "active"])).limit(50)
                ).scalars().all()
                if markets:
                    market = markets[0]  # Use first available as fallback

            if market is None or not market.yes_token_id:
                news_row.processed = True
                session.commit()
                continue

            # Fetch orderbook directly (skip router LLM)
            orderbook_resp = self.cli.orderbook(market.yes_token_id)
            metrics = _extract_orderbook_metrics(orderbook_resp.payload)
            market_prob = float(metrics.get("mid") or 0.5)

            snapshot = MarketSnapshot(
                market_id=market.id,
                bid=metrics.get("bid"),
                ask=metrics.get("ask"),
                mid=metrics.get("mid"),
                spread=metrics.get("spread"),
                bid_volume=metrics.get("bid_volume"),
                ask_volume=metrics.get("ask_volume"),
                total_volume=metrics.get("total_volume"),
                cli_latency_ms=orderbook_resp.latency_ms,
                raw_json=str(orderbook_resp.payload),
            )
            session.add(snapshot)

            calib = self.calibration.calibrate(
                market_probability=market_prob,
                hours_to_resolution=_hours_to_resolution(market.resolution_time),
                domain=market.domain,
                sentiment_score=float(news_row.sentiment_score or 0.0),
            )

            divergence = abs(calib.model_probability - market_prob)

            # Extract volume info from raw_text for logging
            vol_str = "unknown"
            for part in raw.split("|"):
                if "24h_vol" in part:
                    vol_str = part.strip()
                    break

            LOGGER.warning(
                "VOLUME SPIKE DETECTED: %s 24h_vol=%s",
                market.slug,
                vol_str,
            )

            signal = Signal(
                market_id=market.id,
                news_event_id=news_row.id,
                model_probability=calib.model_probability,
                market_probability=market_prob,
                divergence=divergence,
                signal_type="volume_spike",
                router_confidence=1.0,  # No router needed for volume spikes
                calibration_adjustments=calib.adjustment,
                vpin_score=None,
                experiment_id=experiment_id,
                should_trade=False,
                rationale=f"Volume spike fast-path: {raw[:200]}",
            )

            # Use lower divergence threshold for volume spikes (0.10 vs default 0.15)
            if divergence >= VOLUME_SPIKE_MIN_DIVERGENCE:
                # Go straight to executor (skip router)
                executor_ctx = {
                    "market_slug": market.slug,
                    "market_probability": market_prob,
                    "model_probability": calib.model_probability,
                    "divergence": divergence,
                    "router_confidence": 1.0,
                    "vpin": 0.0,
                    "orderbook": metrics,
                    "market_question": market.question,
                    "resolution_time": market.resolution_time.isoformat() if market.resolution_time else None,
                    "min_divergence": VOLUME_SPIKE_MIN_DIVERGENCE,
                    "signal_type": "volume_spike",
                }
                executor_decision = await self.executor.decide(executor_ctx)
                signal.executor_confidence = executor_decision.confidence
                signal.rationale = (signal.rationale or "") + f" | {executor_decision.rationale}"

                if executor_decision.trade and executor_decision.side in {"YES", "NO"}:
                    signal.should_trade = True
                    stats.executed_trades += 1

            session.add(signal)
            stats.generated_signals += 1
            news_row.processed = True
            session.commit()

    async def run_once(self, max_news: int = 10, experiment_id: str = "exp50") -> PipelineStats:
        stats = PipelineStats()

        with self.session_factory() as session:
            synced = sync_markets_once(session, self.cli, limit=200)
            LOGGER.info("Synced %d markets", synced)
            loaded = self.calibration.load_from_db(session)
            LOGGER.info("Loaded %d calibration cells", loaded)

            # --- Volume spike fast path (skip router) ---
            await self._process_volume_spikes(session, stats, experiment_id)


            pending_news = session.execute(
                select(NewsEvent).where(NewsEvent.processed.is_(False)).order_by(NewsEvent.received_at.asc()).limit(max_news)
            ).scalars().all()

            tracked_slugs = [
                row[0]
                for row in session.execute(
                    select(Market.slug).where(Market.status.in_(["open", "active"]))
                ).all()
            ]

            for news_row in pending_news:
                stats.processed_news += 1
                pipeline_id = uuid.uuid4().hex[:12]
                t0 = time.perf_counter()
                ts_news_received = news_row.received_at

                news_item = _news_item_from_row(news_row)

                router_start = time.perf_counter()
                route_decision = await self.router.route(news_item, tracked_slugs)
                router_latency_ms = int((time.perf_counter() - router_start) * 1000)
                ts_router = datetime.now(UTC)

                if not route_decision.relevant or not route_decision.market_slugs:
                    session.add(
                        Telemetry(
                            pipeline_id=pipeline_id,
                            news_event_id=news_row.id,
                            ts_news_received=ts_news_received,
                            ts_router_completed=ts_router,
                            router_latency_ms=router_latency_ms,
                            total_latency_ms=int((time.perf_counter() - t0) * 1000),
                        )
                    )
                    news_row.processed = True
                    session.commit()
                    continue

                for slug in route_decision.market_slugs[:3]:
                    market = session.execute(select(Market).where(Market.slug == slug)).scalar_one_or_none()
                    if market is None:
                        continue

                    if not market.yes_token_id:
                        continue
                    orderbook_resp = self.cli.orderbook(market.yes_token_id)
                    metrics = _extract_orderbook_metrics(orderbook_resp.payload)
                    market_prob = float(metrics.get("mid") or 0.5)

                    snapshot = MarketSnapshot(
                        market_id=market.id,
                        bid=metrics.get("bid"),
                        ask=metrics.get("ask"),
                        mid=metrics.get("mid"),
                        spread=metrics.get("spread"),
                        bid_volume=metrics.get("bid_volume"),
                        ask_volume=metrics.get("ask_volume"),
                        total_volume=metrics.get("total_volume"),
                        cli_latency_ms=orderbook_resp.latency_ms,
                        raw_json=str(orderbook_resp.payload),
                    )
                    session.add(snapshot)

                    calibration_start = time.perf_counter()
                    calib = self.calibration.calibrate(
                        market_probability=market_prob,
                        hours_to_resolution=_hours_to_resolution(market.resolution_time),
                        domain=market.domain,
                        sentiment_score=float(news_row.sentiment_score or 0.0),
                    )
                    calibration_latency_ms = int((time.perf_counter() - calibration_start) * 1000)
                    ts_calibration = datetime.now(UTC)

                    divergence = abs(calib.model_probability - market_prob)
                    bid_vol = float(metrics.get("bid_volume") or 0.0)
                    ask_vol = float(metrics.get("ask_volume") or 0.0)
                    vpin = self.vpin_calc.compute(
                        [TradeFlow(side="SELL", volume=bid_vol), TradeFlow(side="BUY", volume=ask_vol)]
                    )

                    maker_halt = should_halt_maker(
                        vpin=vpin,
                        vpin_kill_threshold=self.settings.vpin_kill_threshold,
                        resolution_time=market.resolution_time,
                        near_resolution_hours=self.settings.near_resolution_hours,
                        near_resolution_fraction=self.settings.near_resolution_fraction,
                    )

                    signal = Signal(
                        market_id=market.id,
                        news_event_id=news_row.id,
                        model_probability=calib.model_probability,
                        market_probability=market_prob,
                        divergence=divergence,
                        signal_type="news_calibrated",
                        router_confidence=route_decision.confidence,
                        calibration_adjustments=calib.adjustment,
                        vpin_score=vpin,
                        experiment_id=experiment_id,
                        should_trade=False,
                        rationale=route_decision.rationale,
                    )

                    should_consider_trade = (
                        divergence >= self.settings.min_divergence
                        and not maker_halt
                        and float(metrics.get("total_volume") or 0.0) >= 1.0
                    )

                    executor_decision = None
                    executor_latency_ms = None
                    ts_executor = None
                    ts_order = None
                    order_submission_latency_ms = None

                    if should_consider_trade:
                        executor_ctx = {
                            "market_slug": slug,
                            "market_probability": market_prob,
                            "model_probability": calib.model_probability,
                            "divergence": divergence,
                            "router_confidence": route_decision.confidence,
                            "vpin": vpin,
                            "orderbook": metrics,
                            "market_question": market.question,
                            "resolution_time": market.resolution_time.isoformat() if market.resolution_time else None,
                            "min_divergence": self.settings.min_divergence,
                        }
                        executor_start = time.perf_counter()
                        executor_decision = await self.executor.decide(executor_ctx)
                        executor_latency_ms = int((time.perf_counter() - executor_start) * 1000)
                        ts_executor = datetime.now(UTC)

                        signal.executor_confidence = executor_decision.confidence
                        signal.rationale = (signal.rationale or "") + f" | {executor_decision.rationale}"

                        if executor_decision.trade and executor_decision.side in {"YES", "NO"}:
                            hist_edges, hist_returns = _load_historical_arrays(session, market.id)
                            sizing = self.sizer.size(
                                p_model=calib.model_probability,
                                p_market=market_prob,
                                side=executor_decision.side,
                                historical_edge_estimates=hist_edges,
                                historical_trade_returns=hist_returns,
                            )
                            fraction = min(sizing.empirical_fraction, self.settings.max_position_per_market)
                            size_units = max(0.0, fraction)

                            if size_units > 0:
                                signal.should_trade = True
                                order_id = f"paper-{pipeline_id}"
                                is_paper = not self.settings.enable_live_trading

                                if self.settings.enable_live_trading:
                                    order_start = time.perf_counter()
                                    limit_price = executor_decision.limit_price or market_prob
                                    order_resp = self.cli.place_limit_order(
                                        market_slug=slug,
                                        side=executor_decision.side,
                                        price=limit_price,
                                        size=size_units,
                                    )
                                    order_submission_latency_ms = int((time.perf_counter() - order_start) * 1000)
                                    payload = order_resp.payload
                                    if isinstance(payload, dict):
                                        order_id = str(payload.get("order_id") or payload.get("id") or order_id)
                                    ts_order = datetime.now(UTC)
                                else:
                                    ts_order = datetime.now(UTC)
                                    order_submission_latency_ms = 0

                                trade = Trade(
                                    market_id=market.id,
                                    signal_id=None,
                                    side=executor_decision.side,
                                    size=size_units,
                                    entry_price=executor_decision.limit_price or market_prob,
                                    kelly_fraction=fraction,
                                    confidence_score=executor_decision.confidence,
                                    order_id=order_id,
                                    is_paper=is_paper,
                                    opened_at=datetime.now(UTC),
                                )
                                session.add(trade)
                                stats.executed_trades += 1
                                if self.alerter is not None:
                                    signal.is_paper = is_paper
                                    signal.kelly_fraction = fraction
                                    self.alerter.send_signal_alert(signal=signal, market=market, snapshot=metrics)

                    session.add(signal)
                    session.flush()

                    telemetry = Telemetry(
                        pipeline_id=pipeline_id,
                        market_id=market.id,
                        news_event_id=news_row.id,
                        ts_news_received=ts_news_received,
                        ts_router_completed=ts_router,
                        ts_calibration_completed=ts_calibration,
                        ts_executor_completed=ts_executor,
                        ts_order_submitted=ts_order,
                        market_price_at_signal=market_prob,
                        market_price_1m=None,
                        market_price_5m=None,
                        router_latency_ms=router_latency_ms,
                        calibration_latency_ms=calibration_latency_ms,
                        executor_latency_ms=executor_latency_ms,
                        order_submission_latency_ms=order_submission_latency_ms,
                        total_latency_ms=int((time.perf_counter() - t0) * 1000),
                    )
                    session.add(telemetry)
                    session.flush()

                    task_1m = asyncio.create_task(
                        self._update_telemetry_price_after_delay(
                            telemetry_id=telemetry.id,
                            token_id=market.yes_token_id,
                            delay_seconds=60,
                            field_name="market_price_1m",
                            experiment_id=experiment_id,
                        )
                    )
                    task_5m = asyncio.create_task(
                        self._update_telemetry_price_after_delay(
                            telemetry_id=telemetry.id,
                            token_id=market.yes_token_id,
                            delay_seconds=300,
                            field_name="market_price_5m",
                            experiment_id=experiment_id,
                        )
                    )
                    self._background_tasks.update({task_1m, task_5m})
                    task_1m.add_done_callback(self._background_tasks.discard)
                    task_5m.add_done_callback(self._background_tasks.discard)
                    stats.generated_signals += 1

                news_row.processed = True
                session.commit()

        return stats
