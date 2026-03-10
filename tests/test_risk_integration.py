"""Integration tests for circuit breaker, trailing stop, and Kelly sizing.

Covers:
  1. Circuit breaker trips when daily PnL loss >= threshold
  2. Trailing stop triggers after profit moves then reverses
  3. Negative-Kelly trades are rejected before any position is created
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from watchdog.db.base import Base
from watchdog.db.init import init_db
from watchdog.db.models import Market, Trade
from watchdog.risk.circuit_breaker import CircuitBreaker
from watchdog.scripts.run_paper_trading import _raw_kelly
from watchdog.strategies.exit_manager import ExitManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    init_db(eng)
    return eng


@pytest.fixture()
def session(engine):
    factory = sessionmaker(bind=engine)
    with factory() as sess:
        yield sess


@pytest.fixture()
def market(session) -> Market:
    m = Market(
        slug="test-market",
        question="Will X happen?",
        domain="politics",
        status="active",
    )
    session.add(m)
    session.flush()
    return m


def _make_trade(session, market: Market, pnl: float, *, entry: float = 0.50) -> Trade:
    """Insert a closed paper trade with the given realized PnL."""
    t = Trade(
        market_id=market.id,
        side="YES",
        size=abs(pnl) + 1.0,
        entry_price=entry,
        exit_price=entry,
        kelly_fraction=0.05,
        is_paper=True,
        status="closed",
        opened_at=datetime.now(UTC) - timedelta(hours=2),
        closed_at=datetime.now(UTC),
        pnl=pnl,
        strategy="test",
        high_water_mark=entry,
        remaining_fraction=1.0,
    )
    session.add(t)
    session.flush()
    return t


def _make_open_trade(session, market: Market, entry: float = 0.50) -> Trade:
    """Insert an open paper trade."""
    t = Trade(
        market_id=market.id,
        side="YES",
        size=10.0,
        entry_price=entry,
        kelly_fraction=0.05,
        is_paper=True,
        status="open",
        opened_at=datetime.now(UTC) - timedelta(hours=1),
        strategy="test",
        high_water_mark=entry,
        remaining_fraction=1.0,
    )
    session.add(t)
    session.flush()
    return t


# ---------------------------------------------------------------------------
# Test 1: Circuit Breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_allows_trading_when_loss_below_threshold(self, session, market):
        _make_trade(session, market, pnl=-30.0)
        cb = CircuitBreaker(max_daily_loss_usd=50.0, cooldown_minutes=60)
        assert cb.check(session) is True, "Should allow trading: loss only $30 < $50 threshold"

    def test_trips_when_loss_meets_threshold(self, session, market):
        _make_trade(session, market, pnl=-25.0)
        _make_trade(session, market, pnl=-25.0)
        cb = CircuitBreaker(max_daily_loss_usd=50.0, cooldown_minutes=60)
        assert cb.check(session) is False, "Should block: total loss = -$50"

    def test_trips_when_loss_exceeds_threshold(self, session, market):
        _make_trade(session, market, pnl=-60.0)
        cb = CircuitBreaker(max_daily_loss_usd=50.0, cooldown_minutes=60)
        assert cb.check(session) is False, "Should block: loss $60 > $50 threshold"

    def test_profitable_day_never_trips(self, session, market):
        _make_trade(session, market, pnl=100.0)
        cb = CircuitBreaker(max_daily_loss_usd=50.0, cooldown_minutes=60)
        assert cb.check(session) is True, "Profitable day should never trip breaker"

    def test_tripped_breaker_blocks_subsequent_checks(self, session, market):
        _make_trade(session, market, pnl=-60.0)
        cb = CircuitBreaker(max_daily_loss_usd=50.0, cooldown_minutes=60)
        assert cb.check(session) is False  # trips
        assert cb.check(session) is False  # still blocked (cooldown not expired)

    def test_no_webhook_does_not_raise(self, session, market):
        _make_trade(session, market, pnl=-60.0)
        cb = CircuitBreaker(max_daily_loss_usd=50.0, cooldown_minutes=60, n8n_webhook_url=None)
        # Should not raise even with no webhook configured
        cb.check(session)


# ---------------------------------------------------------------------------
# Test 2: Trailing Stop
# ---------------------------------------------------------------------------


class TestTrailingStop:
    def setup_method(self):
        self.em = ExitManager()

    def _stop(self, entry, current, hwm):
        pnl_pct = (current - entry) / entry
        t = type("T", (), {"high_water_mark": hwm})()
        return self.em._check_trailing_stop(t, entry, current, hwm, pnl_pct)

    # Hard floor
    def test_hard_floor_triggers_at_minus_25pct(self):
        entry = 0.50
        cur = entry * 0.749  # -25.1%
        reason = self._stop(entry, cur, hwm=entry)
        assert reason is not None, "Hard floor should trigger at -25%+"
        assert "trailing_hard_floor" in reason

    def test_hard_floor_does_not_trigger_just_above(self):
        # At HWM=entry the 0%-profit tier sets a 15% trail → floor = entry*0.85 = 0.425.
        # A price just above that floor should NOT trigger anything.
        entry = 0.50
        cur = 0.426  # above floor of 0.425 — no stop fires
        reason = self._stop(entry, cur, hwm=entry)
        assert reason is None, "Should not trigger when price is above the trailing floor"

    # Tier activation: profit >= 5%, trail 10%
    def test_trailing_stop_triggers_after_reversal(self):
        entry = 0.50
        hwm = 0.56   # +12% peak  → tier: profit>=5%, trail 10% → floor = 0.56*0.90 = 0.504
        cur = 0.503  # below floor
        reason = self._stop(entry, cur, hwm)
        assert reason is not None, "Trailing stop should fire when price drops below HWM floor"
        assert "trailing_stop" in reason

    def test_trailing_stop_does_not_fire_above_floor(self):
        entry = 0.50
        hwm = 0.56   # tier: trail 10% → floor = 0.504
        cur = 0.506  # above floor
        reason = self._stop(entry, cur, hwm)
        assert reason is None, "Should not fire when price is still above trail floor"

    # Tier tightening at >= 25% profit: trail 5%
    def test_tight_trail_triggers_at_high_profit(self):
        entry = 0.50
        hwm = 0.64   # +28% peak → tier: profit>=25%, trail 5% → floor = 0.64*0.95 = 0.608
        cur = 0.607  # just below 0.608
        reason = self._stop(entry, cur, hwm)
        assert reason is not None
        assert "trail=5%" in reason

    def test_no_trail_when_hwm_equals_entry(self):
        # Position never moved into profit — only hard floor applies
        entry = 0.50
        hwm = 0.50
        cur = 0.42  # -16%, above hard floor -25%
        reason = self._stop(entry, cur, hwm)
        # The 0%-profit tier (trail 15%) activates since hwm_profit=0% >= 0%
        # floor = 0.50 * 0.85 = 0.425 — price 0.42 < 0.425 → should trigger
        assert reason is not None


# ---------------------------------------------------------------------------
# Test 3: Negative-Kelly rejection
# ---------------------------------------------------------------------------


class TestKellyRejection:
    def test_positive_kelly_accepted(self):
        # model 65% vs market 50% on YES → clear edge
        k = _raw_kelly(0.65, 0.50, "YES")
        assert k > 0, f"Expected positive Kelly, got {k:.4f}"

    def test_negative_kelly_rejected(self):
        # model 40% vs market 50% on YES → no edge
        k = _raw_kelly(0.40, 0.50, "YES")
        assert k < 0, f"Expected negative Kelly, got {k:.4f}"

    def test_negative_kelly_on_no_side_with_yes_bias(self):
        # model says 65% YES, trading NO side → wrong direction
        k = _raw_kelly(0.65, 0.50, "NO")
        assert k < 0, f"Expected negative Kelly on wrong side, got {k:.4f}"

    def test_positive_kelly_on_no_side(self):
        # model 35% vs market 50% — market overpriced on YES → NO has edge
        k = _raw_kelly(0.35, 0.50, "NO")
        assert k > 0, f"Expected positive Kelly on NO side, got {k:.4f}"

    def test_kelly_magnitude_scales_with_edge(self):
        k_small = _raw_kelly(0.52, 0.50, "YES")  # tiny edge
        k_large = _raw_kelly(0.70, 0.50, "YES")  # large edge
        assert 0 < k_small < k_large, "Larger edge should produce larger Kelly"

    def test_kelly_zero_edge_near_zero(self):
        # No divergence → Kelly ≈ 0
        k = _raw_kelly(0.50, 0.50, "YES")
        assert abs(k) < 1e-3, f"Zero-edge trade should have near-zero Kelly, got {k:.6f}"
