from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from watchdog.core.config import Settings
from watchdog.db.init import init_db
from watchdog.db.models import CalibrationSurface, NewsEvent, Telemetry, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.llm.types import ExecutorDecision, RouterDecision
from watchdog.market_data.polymarket_cli import CliResponse
from watchdog.risk.vpin import VPINCalculator
from watchdog.services.pipeline import PipelineRunner
from watchdog.signals.calibration import CalibrationSurfaceService


class FakeRouter:
    async def route(self, news_item, tracked_market_slugs):
        return RouterDecision(relevant=True, market_slugs=["mkt-1"], impact_direction="up", confidence=0.9, rationale="mock")


class FakeExecutor:
    async def decide(self, context):
        return ExecutorDecision(
            trade=True,
            side="YES",
            confidence=0.95,
            limit_price=0.6,
            reason_might_be_wrong="none",
            rationale="mock",
        )


class FakeSizer:
    class _Result:
        empirical_fraction = 0.1

    def size(self, *args, **kwargs):
        return self._Result()


class FakeCli:
    def __init__(self, high_vpin: bool) -> None:
        self.high_vpin = high_vpin

    def list_markets(self, limit=50):
        return CliResponse(
            payload=[
                {
                    "slug": "mkt-1",
                    "question": "Will X happen?",
                    "domain": "politics",
                    "status": "active",
                    "endDate": datetime(2026, 11, 1, tzinfo=UTC).isoformat(),
                }
            ],
            latency_ms=3,
        )

    def orderbook(self, market_slug: str):
        if self.high_vpin:
            bids = [{"price": 0.49, "size": 0}]
            asks = [{"price": 0.51, "size": 10000}]
        else:
            bids = [{"price": 0.49, "size": 2500}]
            asks = [{"price": 0.51, "size": 2500}]
        return CliResponse(payload={"orderbook": {"bids": bids, "asks": asks}}, latency_ms=5)

    def place_limit_order(self, market_slug: str, side: str, price: float, size: float):
        return CliResponse(payload={"order_id": "paper-1"}, latency_ms=2)


class FastPipelineRunner(PipelineRunner):
    async def _update_telemetry_price_after_delay(self, telemetry_id, market_slug, delay_seconds, field_name, experiment_id):
        with self.session_factory() as session:
            row = session.get(Telemetry, telemetry_id)
            if row is not None:
                setattr(row, field_name, row.market_price_at_signal)
                session.commit()


@pytest.mark.asyncio
async def test_pipeline_should_trade_false_when_vpin_high() -> None:
    settings = Settings(database_url="sqlite:///:memory:", enable_live_trading=False)
    engine = build_engine(settings)
    init_db(engine)
    sf = build_session_factory(engine)

    with sf() as session:
        session.add(
            CalibrationSurface(
                price_bucket=50,
                time_bucket_hours=2160,
                domain="politics",
                dataset_source="test",
                sample_size=100,
                empirical_outcome_rate=0.8,
                model_adjustment=0.30,
            )
        )
        session.add(
            NewsEvent(
                headline="Election surprise",
                source="unit",
                url=None,
                raw_text="news",
                domain_tags="politics",
                sentiment_score=0.1,
                processed=False,
            )
        )
        session.commit()

    runner = FastPipelineRunner(
        settings=settings,
        session_factory=sf,
        cli=FakeCli(high_vpin=True),
        router=FakeRouter(),
        executor=FakeExecutor(),
        calibration=CalibrationSurfaceService(),
        sizer=FakeSizer(),
        vpin_calc=VPINCalculator(),
    )

    await runner.run_once(max_news=1)

    with sf() as session:
        trades = session.execute(select(Trade)).scalars().all()
        assert len(trades) == 0


@pytest.mark.asyncio
async def test_pipeline_populates_timestamps_and_forces_paper_mode() -> None:
    settings = Settings(database_url="sqlite:///:memory:", enable_live_trading=False)
    engine = build_engine(settings)
    init_db(engine)
    sf = build_session_factory(engine)

    with sf() as session:
        session.add(
            CalibrationSurface(
                price_bucket=50,
                time_bucket_hours=2160,
                domain="politics",
                dataset_source="test",
                sample_size=100,
                empirical_outcome_rate=0.8,
                model_adjustment=0.30,
            )
        )
        session.add(
            NewsEvent(
                headline="Election surprise",
                source="unit",
                url=None,
                raw_text="news",
                domain_tags="politics",
                sentiment_score=0.1,
                processed=False,
            )
        )
        session.commit()

    runner = FastPipelineRunner(
        settings=settings,
        session_factory=sf,
        cli=FakeCli(high_vpin=False),
        router=FakeRouter(),
        executor=FakeExecutor(),
        calibration=CalibrationSurfaceService(),
        sizer=FakeSizer(),
        vpin_calc=VPINCalculator(),
    )

    await runner.run_once(max_news=1)
    await asyncio.sleep(0)

    with sf() as session:
        telemetry = session.execute(select(Telemetry)).scalars().first()
        assert telemetry is not None
        assert telemetry.ts_news_received is not None
        assert telemetry.ts_router_completed is not None
        assert telemetry.ts_calibration_completed is not None
        assert telemetry.ts_executor_completed is not None
        assert telemetry.ts_order_submitted is not None

        trade = session.execute(select(Trade)).scalars().first()
        assert trade is not None
        assert trade.is_paper is True
