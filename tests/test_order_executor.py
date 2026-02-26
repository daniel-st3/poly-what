from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from watchdog.core.config import Settings
from watchdog.db.init import init_db
from watchdog.db.models import Market, Signal, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.execution.order_executor import ExecutionSnapshot, OrderExecutor
from watchdog.market_data.polymarket_cli import CliResponse


class FakeCli:
    def create_limit_order(self, token_id: str, side: str, price: float, size: float, post_only: bool = True):
        return CliResponse(payload={"order_id": "live-1"}, latency_ms=1)

    def cancel_all(self):
        return CliResponse(payload={"ok": True}, latency_ms=1)


def _bootstrap_db(settings: Settings):
    engine = build_engine(settings)
    init_db(engine)
    sf = build_session_factory(engine)

    with sf() as session:
        market = Market(
            slug="mkt-1",
            question="Will X happen?",
            domain="politics",
            resolution_time=datetime(2025, 11, 1, tzinfo=UTC),
            status="active",
        )
        session.add(market)
        session.flush()

        signal = Signal(
            market_id=market.id,
            model_probability=0.7,
            market_probability=0.5,
            divergence=0.2,
            signal_type="test",
            executor_confidence=0.95,
            should_trade=True,
        )
        session.add(signal)
        session.commit()
        return sf, market.id, signal.id


@pytest.mark.parametrize(
    "snapshot,kelly,setup_open_positions,expected_reason",
    [
        (ExecutionSnapshot(1, "mkt-1", ask=0.51, mid=0.5, volume=100, liquidity=1000), {"position_fraction": 0.1, "bankroll": 1000}, 0, "volume_below_minimum"),
        (ExecutionSnapshot(1, "mkt-1", ask=0.51, mid=0.5, volume=2000, liquidity=100), {"position_fraction": 0.1, "bankroll": 1000}, 0, "liquidity_below_minimum"),
        (ExecutionSnapshot(1, "mkt-1", ask=0.51, mid=0.5, volume=2000, liquidity=1000), {"position_fraction": 0.0, "bankroll": 1000}, 0, "non_positive_position_fraction"),
    ],
)
def test_prechecks_abort(snapshot, kelly, setup_open_positions, expected_reason) -> None:
    settings = Settings(database_url="sqlite:///:memory:")
    sf, _market_id, signal_id = _bootstrap_db(settings)
    executor = OrderExecutor(settings, sf, FakeCli())

    with sf() as session:
        signal = session.get(Signal, signal_id)
        if expected_reason == "executor_confidence_below_threshold":
            signal.executor_confidence = 0.2
            session.commit()
        result = executor.execute(signal=signal, snapshot=snapshot, kelly_result=kelly, is_paper=True)

    assert result["executed"] is False
    assert result["reason"] == expected_reason


def test_precheck_confidence_and_limits() -> None:
    settings = Settings(database_url="sqlite:///:memory:", max_positions_simultaneous=1)
    sf, market_id, signal_id = _bootstrap_db(settings)
    executor = OrderExecutor(settings, sf, FakeCli())

    with sf() as session:
        signal = session.get(Signal, signal_id)
        signal.executor_confidence = 0.2
        session.commit()

        result = executor.execute(
            signal=signal,
            snapshot=ExecutionSnapshot(market_id, "mkt-1", ask=0.51, mid=0.5, volume=2000, liquidity=2000),
            kelly_result={"position_fraction": 0.1, "bankroll": 1000},
            is_paper=True,
        )
        assert result["reason"] == "executor_confidence_below_threshold"

        # Add an open trade to hit max position precheck.
        session.add(
            Trade(
                market_id=market_id,
                side="YES",
                size=10,
                entry_price=0.5,
                kelly_fraction=0.1,
                is_paper=True,
                status="open",
            )
        )
        signal.executor_confidence = 0.95
        session.commit()

        result = executor.execute(
            signal=signal,
            snapshot=ExecutionSnapshot(market_id, "mkt-1", ask=0.51, mid=0.5, volume=2000, liquidity=2000),
            kelly_result={"position_fraction": 0.1, "bankroll": 1000},
            is_paper=True,
        )
        assert result["reason"] == "max_open_positions_reached"


def test_paper_fill_price_is_ask_and_cancel_all() -> None:
    settings = Settings(database_url="sqlite:///:memory:", max_positions_simultaneous=5)
    sf, market_id, signal_id = _bootstrap_db(settings)
    executor = OrderExecutor(settings, sf, FakeCli())

    with sf() as session:
        signal = session.get(Signal, signal_id)
        result = executor.execute(
            signal=signal,
            snapshot=ExecutionSnapshot(market_id, "mkt-1", ask=0.53, mid=0.5, volume=2000, liquidity=2000),
            kelly_result={"position_fraction": 0.1, "bankroll": 1000},
            is_paper=True,
        )
        assert result["executed"] is True

    with sf() as session:
        trade = session.execute(select(Trade).where(Trade.id == result["trade_id"])).scalar_one()
        assert trade.entry_price == 0.53
        assert trade.status == "open"

    cancel_result = executor.cancel_all(reason="unit_test")
    assert cancel_result["cancel_response"]["ok"] is True

    with sf() as session:
        open_rows = session.execute(select(Trade).where(Trade.status == "open")).scalars().all()
        canceled_rows = session.execute(select(Trade).where(Trade.status == "canceled")).scalars().all()
        assert len(open_rows) == 0
        assert len(canceled_rows) >= 1
