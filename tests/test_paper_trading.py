from __future__ import annotations

import pytest
from sqlalchemy import select

from watchdog.core.config import get_settings
from watchdog.db.models import Signal, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.manifold_client import ManifoldClient
from watchdog.scripts.run_paper_trading import run_paper_trading_loop
from watchdog.signals.calibration import CalibrationResult, CalibrationSurfaceService


@pytest.mark.asyncio
async def test_manifold_paper_vpin_proxy_allows_trade(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db_path = tmp_path / "paper_vpin_ok.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MANIFOLD_API_KEY", "test-key")
    monkeypatch.setenv("PAPER_LOOP_SECONDS", "5")
    get_settings.cache_clear()

    monkeypatch.setattr(
        ManifoldClient,
        "get_markets",
        lambda self, limit=100: [
            {
                "id": "mkt-1",
                "slug": "market-1",
                "question": "Will test event happen?",
                "probability": 0.50,
                "isResolved": False,
                "closeTime": "2030-01-01T00:00:00Z",
            }
        ],
    )
    monkeypatch.setattr(
        ManifoldClient,
        "get_orderbook",
        lambda self, market_id: {
            "bid": 0.48,
            "ask": 0.52,
            "mid": 0.50,
            "spread": 0.04,
            "probability": 0.50,
            "bid_volume": 500.0,
            "ask_volume": 500.0,
        },
    )
    monkeypatch.setattr(ManifoldClient, "get_market", lambda self, market_id: {"id": market_id, "probability": 0.50})

    bet_calls: list[tuple[str, str, float]] = []

    def _place_bet(self, market_id: str, outcome: str, amount: float):
        bet_calls.append((market_id, outcome, amount))
        return {"betId": "bet-1"}

    monkeypatch.setattr(ManifoldClient, "place_bet", _place_bet)

    def _calibrate(self, market_probability: float, hours_to_resolution: float, domain: str, sentiment_score: float = 0.0):
        return CalibrationResult(
            model_probability=0.80,
            adjustment=0.30,
            price_bucket=80,
            time_bucket_hours=24,
            domain=domain,
        )

    monkeypatch.setattr(CalibrationSurfaceService, "calibrate", _calibrate)

    await run_paper_trading_loop(
        virtual_bankroll=500.0,
        platform="manifold",
        max_markets=1,
        iterations=1,
    )

    settings = get_settings()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        signals = session.execute(select(Signal)).scalars().all()
        trades = session.execute(select(Trade)).scalars().all()

    assert len(signals) == 1
    assert signals[0].vpin_score == pytest.approx(0.0)
    assert signals[0].should_trade is True
    assert signals[0].rationale == "divergence_triggered"
    assert len(trades) == 1
    assert len(bet_calls) == 1

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_manifold_paper_vpin_guardrail_skips_trade(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    db_path = tmp_path / "paper_vpin_halt.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("MANIFOLD_API_KEY", "test-key")
    monkeypatch.setenv("PAPER_LOOP_SECONDS", "5")
    monkeypatch.setenv("VPIN_KILL_THRESHOLD", "0.70")
    get_settings.cache_clear()

    monkeypatch.setattr(
        ManifoldClient,
        "get_markets",
        lambda self, limit=100: [
            {
                "id": "mkt-2",
                "slug": "market-2",
                "question": "Will toxic flow test trigger?",
                "probability": 0.50,
                "isResolved": False,
                "closeTime": "2030-01-01T00:00:00Z",
            }
        ],
    )
    monkeypatch.setattr(
        ManifoldClient,
        "get_orderbook",
        lambda self, market_id: {
            "bid": 0.48,
            "ask": 0.52,
            "mid": 0.50,
            "spread": 0.04,
            "probability": 0.50,
            "bid_volume": 0.0,
            "ask_volume": 10000.0,
        },
    )
    monkeypatch.setattr(ManifoldClient, "get_market", lambda self, market_id: {"id": market_id, "probability": 0.50})

    bet_calls: list[tuple[str, str, float]] = []

    def _place_bet(self, market_id: str, outcome: str, amount: float):
        bet_calls.append((market_id, outcome, amount))
        return {"betId": "bet-2"}

    monkeypatch.setattr(ManifoldClient, "place_bet", _place_bet)

    def _calibrate(self, market_probability: float, hours_to_resolution: float, domain: str, sentiment_score: float = 0.0):
        return CalibrationResult(
            model_probability=0.85,
            adjustment=0.35,
            price_bucket=85,
            time_bucket_hours=24,
            domain=domain,
        )

    monkeypatch.setattr(CalibrationSurfaceService, "calibrate", _calibrate)

    await run_paper_trading_loop(
        virtual_bankroll=500.0,
        platform="manifold",
        max_markets=1,
        iterations=1,
    )

    settings = get_settings()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        signals = session.execute(select(Signal)).scalars().all()
        trades = session.execute(select(Trade)).scalars().all()

    assert len(signals) == 1
    assert signals[0].should_trade is False
    assert signals[0].rationale == "vpin_kill_switch"
    assert signals[0].vpin_score is not None and signals[0].vpin_score >= settings.vpin_kill_threshold
    assert len(trades) == 0
    assert len(bet_calls) == 0

    get_settings.cache_clear()
