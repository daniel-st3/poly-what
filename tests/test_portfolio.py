"""Tests for portfolio state computation truthfulness.

Covers:
- Breakeven classification (pnl == 0 must not be counted as a loss)
- Win/loss/breakeven counts are accurate
- get_current_portfolios() returns live counts, not fake zeros
- btc_scalp trades are excluded from the normal-bot portfolio scope
- NULL-strategy trades are included in the normal-bot portfolio scope
- Open PnL estimation: partial coverage and no-coverage cases
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from watchdog.backtest.portfolio import (
    PortfolioState,
    get_current_portfolios,
    update_portfolios,
)
from watchdog.core.config import get_settings
from watchdog.db.init import init_db
from watchdog.db.models import Market, MarketSnapshot, PortfolioSnapshot, Trade
from watchdog.db.session import build_engine, build_session_factory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_db(tmp_path):
    """Create a temp file-backed SQLite DB, return (session_factory, session)."""
    db_path = tmp_path / "test.db"
    import os
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    get_settings.cache_clear()
    settings = get_settings()
    engine = build_engine(settings)
    init_db(engine)
    return build_session_factory(engine)


_market_counter = 0


def _make_market(session) -> Market:
    """Create and flush a minimal Market row, returning it."""
    global _market_counter
    _market_counter += 1
    mkt = Market(
        slug=f"test-market-{_market_counter}",
        question="Will this test pass?",
        domain="test",
        status="open",
    )
    session.add(mkt)
    session.flush()
    return mkt


def _make_trade(
    session,
    market_id: int,
    pnl: float | None,
    status: str = "closed",
    strategy: str | None = None,
    side: str = "YES",
    entry_price: float = 0.50,
    size: float = 10.0,
) -> Trade:
    """Create and flush a minimal paper Trade row."""
    now = datetime.now(UTC)
    trade = Trade(
        market_id=market_id,
        side=side,
        size=size,
        entry_price=entry_price,
        kelly_fraction=0.05,
        is_paper=True,
        status=status,
        pnl=pnl if status == "closed" else None,
        strategy=strategy,
        opened_at=now,
        closed_at=now if status == "closed" else None,
    )
    session.add(trade)
    session.flush()
    return trade


def _seed_snapshot(session, cap: float) -> None:
    """Write a minimal PortfolioSnapshot so get_current_portfolios has data."""
    snap = PortfolioSnapshot(
        timestamp=datetime.utcnow(),
        starting_capital=cap,
        current_balance=cap,
        peak_balance=cap,
        run_pnl=0.0,
        total_return_pct=0.0,
        drawdown_pct=0.0,
        trades_today=0,
    )
    session.add(snap)
    session.flush()


# ---------------------------------------------------------------------------
# Tests: PortfolioState has all required fields
# ---------------------------------------------------------------------------

def test_portfolio_state_has_all_new_fields() -> None:
    """PortfolioState NamedTuple exposes all 8 new fields."""
    required = {
        "breakevens", "open_count", "realized_pnl",
        "deployed_capital", "free_capital",
        "open_pnl_estimate", "open_pnl_priced_count", "equity",
    }
    assert required.issubset(set(PortfolioState._fields))


# ---------------------------------------------------------------------------
# Tests: Breakeven / W/L/BE classification
# ---------------------------------------------------------------------------

def test_breakeven_not_counted_as_loss(tmp_path) -> None:
    """A trade with pnl=0.0 must be a breakeven, not a loss."""
    sf = _setup_db(tmp_path)
    with sf() as session:
        mkt = _make_market(session)
        _make_trade(session, mkt.id, pnl=0.0)
        session.commit()
        states = update_portfolios(session)

    p = states[100.0]
    assert p.losses == 0
    assert p.wins == 0
    assert p.breakevens == 1
    assert p.closed_count == 1


def test_win_loss_breakeven_counts(tmp_path) -> None:
    """3W + 2L + 1BE produces exactly those counts."""
    sf = _setup_db(tmp_path)
    with sf() as session:
        mkt = _make_market(session)
        for pnl in [1.0, 2.0, 3.0]:          # 3 wins
            _make_trade(session, mkt.id, pnl=pnl)
        for pnl in [-1.0, -2.0]:             # 2 losses
            _make_trade(session, mkt.id, pnl=pnl)
        _make_trade(session, mkt.id, pnl=0.0) # 1 breakeven
        session.commit()
        states = update_portfolios(session)

    p = states[100.0]
    assert p.wins == 3
    assert p.losses == 2
    assert p.breakevens == 1
    assert p.closed_count == 6
    assert p.wins + p.losses + p.breakevens == p.closed_count


# ---------------------------------------------------------------------------
# Tests: get_current_portfolios — no fake zeros
# ---------------------------------------------------------------------------

def test_no_fake_zeros_in_get_current_portfolios(tmp_path) -> None:
    """After seeding, get_current_portfolios must return real counts, not zeros."""
    sf = _setup_db(tmp_path)
    with sf() as session:
        mkt = _make_market(session)
        _make_trade(session, mkt.id, pnl=1.0)
        _make_trade(session, mkt.id, pnl=-0.5)
        _make_trade(session, mkt.id, pnl=0.0)
        # Write a snapshot so get_current_portfolios has something to read
        update_portfolios(session)

    with sf() as session:
        portfolios = get_current_portfolios(session)

    for cap in [100.0, 500.0]:
        p = portfolios[cap]
        assert p is not None
        assert p.closed_count == 3, f"expected 3 but got {p.closed_count} for ${cap}"
        assert p.wins == 1
        assert p.losses == 1
        assert p.breakevens == 1
        assert p.wins + p.losses + p.breakevens == p.closed_count


# ---------------------------------------------------------------------------
# Tests: Strategy scope — NULL included, btc_scalp excluded
# ---------------------------------------------------------------------------

def test_null_strategy_included_in_portfolio(tmp_path) -> None:
    """Trades with strategy=None (calibration) must be counted in the portfolio."""
    sf = _setup_db(tmp_path)
    with sf() as session:
        mkt = _make_market(session)
        _make_trade(session, mkt.id, pnl=1.0, strategy=None)
        session.commit()
        states = update_portfolios(session)

    p = states[100.0]
    assert p.closed_count == 1
    assert p.wins == 1


def test_btc_scalp_excluded_from_portfolio(tmp_path) -> None:
    """btc_scalp trades must NOT appear in the normal-bot portfolio counts."""
    sf = _setup_db(tmp_path)
    with sf() as session:
        mkt = _make_market(session)
        _make_trade(session, mkt.id, pnl=5.0, strategy=None)           # normal
        _make_trade(session, mkt.id, pnl=10.0, strategy="btc_scalp")   # excluded
        session.commit()
        states = update_portfolios(session)

    p = states[100.0]
    assert p.closed_count == 1, "only the normal trade should be counted"
    assert p.wins == 1
    assert p.realized_pnl == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Tests: Open PnL estimation
# ---------------------------------------------------------------------------

def test_open_pnl_no_coverage_returns_none(tmp_path) -> None:
    """If no MarketSnapshot rows exist, open_pnl_estimate must be None."""
    sf = _setup_db(tmp_path)
    with sf() as session:
        mkt = _make_market(session)
        _make_trade(session, mkt.id, pnl=None, status="open")
        _make_trade(session, mkt.id, pnl=None, status="open")
        session.commit()
        states = update_portfolios(session)

    p = states[100.0]
    assert p.open_count == 2
    assert p.open_pnl_estimate is None
    assert p.open_pnl_priced_count == 0


def test_open_pnl_partial_coverage(tmp_path) -> None:
    """Partial snapshot coverage: priced_count reflects only priced trades; estimate is a float."""
    sf = _setup_db(tmp_path)
    with sf() as session:
        mkt1 = _make_market(session)
        mkt2 = _make_market(session)
        mkt3 = _make_market(session)

        now = datetime.now(UTC)
        # Open trades: entry_price=0.40, current mid=0.60 → YES gain = size*(0.60-0.40)=2.0
        _make_trade(session, mkt1.id, pnl=None, status="open", entry_price=0.40, size=10.0, side="YES")
        _make_trade(session, mkt2.id, pnl=None, status="open", entry_price=0.40, size=10.0, side="YES")
        # mkt3 has no snapshot
        _make_trade(session, mkt3.id, pnl=None, status="open", entry_price=0.40, size=10.0, side="YES")

        # Snapshots only for mkt1 and mkt2
        for mkt in [mkt1, mkt2]:
            snap = MarketSnapshot(
                market_id=mkt.id,
                captured_at=now,
                mid=0.60,
                cli_latency_ms=0,
            )
            session.add(snap)
        session.commit()
        states = update_portfolios(session)

    p = states[100.0]
    assert p.open_count == 3
    assert p.open_pnl_priced_count == 2
    assert p.open_pnl_estimate is not None
    assert p.open_pnl_estimate == pytest.approx(4.0)  # 2 trades * 10 * (0.60-0.40)
