"""Tests for BTC scalp paper stats helper, bankroll formatting, and open risk cap guard."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from watchdog.core.config import get_settings
from watchdog.db.init import init_db
from watchdog.db.models import Market, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.strategies.btc_scalp import BtcScalpStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_market_counter = 0


def _setup_db(tmp_path):
    db_path = tmp_path / "test.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    get_settings.cache_clear()
    settings = get_settings()
    engine = build_engine(settings)
    init_db(engine)
    return build_session_factory(engine)


def _make_market(session) -> Market:
    global _market_counter
    _market_counter += 1
    mkt = Market(
        slug=f"btc-test-market-{_market_counter}",
        question="Will BTC be above $80,000?",
        domain="crypto",
        status="resolved",
    )
    session.add(mkt)
    session.flush()
    return mkt


def _make_trade(
    session,
    market_id: int,
    pnl: float | None,
    closed_at: datetime | None = None,
    status: str = "closed",
    size: float = 2.0,
    entry_price: float = 0.42,
    exit_price: float | None = None,
) -> Trade:
    t = Trade(
        market_id=market_id,
        side="Up",
        size=size,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        kelly_fraction=0.0,
        is_paper=True,
        status=status,
        opened_at=datetime.now(UTC),
        closed_at=closed_at,
        strategy="btc_scalp",
    )
    session.add(t)
    session.flush()
    return t


def _make_strategy(session_factory=None, db_url: str | None = None) -> BtcScalpStrategy:
    """Build a BtcScalpStrategy instance without running __init__ (no DB/WS setup)."""
    strat = object.__new__(BtcScalpStrategy)
    strat._tg_token = None
    strat._tg_chat = None
    strat._open_risk_cap_notified = False
    strat._notified_msgs: list[str] = []
    strat._notify = lambda msg: strat._notified_msgs.append(msg)  # type: ignore[method-assign]
    return strat


# ---------------------------------------------------------------------------
# _btc_paper_stats() tests
# ---------------------------------------------------------------------------


def test_btc_paper_stats_counts(tmp_path):
    session_factory = _setup_db(tmp_path)
    today = datetime.now(UTC)

    with session_factory() as session:
        mkt = _make_market(session)
        # WIN today
        _make_trade(session, mkt.id, pnl=1.16, closed_at=today)
        # LOSS today
        _make_trade(session, mkt.id, pnl=-0.84, closed_at=today)
        # BREAKEVEN today
        _make_trade(session, mkt.id, pnl=0.0, closed_at=today)
        # pnl=None today — counts in closed but not W/L/BE
        _make_trade(session, mkt.id, pnl=None, closed_at=today)
        session.commit()

    strat = _make_strategy()
    stats = strat._btc_paper_stats()

    assert stats["alltime_closed"] == 4
    assert stats["today_closed"] == 4
    assert stats["alltime_wins"] == 1
    assert stats["alltime_losses"] == 1
    assert stats["alltime_be"] == 1
    assert stats["today_wins"] == 1
    assert stats["today_losses"] == 1
    assert stats["today_be"] == 1
    assert abs(stats["alltime_pnl"] - (1.16 - 0.84)) < 1e-9
    assert abs(stats["today_pnl"] - (1.16 - 0.84)) < 1e-9
    assert stats["open_count"] == 0
    assert stats["open_stake"] == 0.0


def test_btc_paper_stats_today_filter(tmp_path):
    session_factory = _setup_db(tmp_path)
    today = datetime.now(UTC)
    yesterday = today - timedelta(days=1)

    with session_factory() as session:
        mkt = _make_market(session)
        _make_trade(session, mkt.id, pnl=1.16, closed_at=today)
        _make_trade(session, mkt.id, pnl=-0.84, closed_at=yesterday)
        session.commit()

    strat = _make_strategy()
    stats = strat._btc_paper_stats()

    assert stats["alltime_closed"] == 2
    assert stats["today_closed"] == 1
    assert stats["alltime_wins"] == 1
    assert stats["alltime_losses"] == 1
    assert stats["today_wins"] == 1
    assert stats["today_losses"] == 0
    assert abs(stats["alltime_pnl"] - (1.16 - 0.84)) < 1e-9
    assert abs(stats["today_pnl"] - 1.16) < 1e-9


def test_btc_paper_stats_open_trades(tmp_path):
    session_factory = _setup_db(tmp_path)

    with session_factory() as session:
        mkt = _make_market(session)
        _make_trade(session, mkt.id, pnl=None, closed_at=None, status="open", size=2.0)
        _make_trade(session, mkt.id, pnl=None, closed_at=None, status="open", size=2.0)
        session.commit()

    strat = _make_strategy()
    stats = strat._btc_paper_stats()

    assert stats["open_count"] == 2
    assert stats["open_stake"] == 4.0
    assert stats["alltime_closed"] == 0


def test_btc_paper_stats_excludes_other_strategies(tmp_path):
    """Trades from other strategies must not appear in btc_scalp stats."""
    session_factory = _setup_db(tmp_path)
    today = datetime.now(UTC)

    with session_factory() as session:
        mkt = _make_market(session)
        # btc_scalp trade
        _make_trade(session, mkt.id, pnl=1.0, closed_at=today)
        # other strategy trade — should be excluded
        other = Trade(
            market_id=mkt.id,
            side="YES",
            size=2.0,
            entry_price=0.5,
            pnl=5.0,
            kelly_fraction=0.0,
            is_paper=True,
            status="closed",
            opened_at=today,
            closed_at=today,
            strategy="ofi_signal",
        )
        session.add(other)
        session.commit()

    strat = _make_strategy()
    stats = strat._btc_paper_stats()

    assert stats["alltime_closed"] == 1
    assert abs(stats["alltime_pnl"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _bankroll_block() tests
# ---------------------------------------------------------------------------


def test_bankroll_block_format():
    strat = _make_strategy()
    stats = {
        "alltime_pnl": 1.16,
        "today_closed": 3,
        "today_wins": 2,
        "today_losses": 1,
        "today_be": 0,
        "today_pnl": 0.48,
        "open_count": 1,
        "open_stake": 2.0,
    }
    block = strat._bankroll_block(stats)
    assert "$50" in block
    assert "$100" in block
    assert "hypothetical" in block
    assert "%" in block
    # $50 bankroll: 50.00 + 1.16 = 51.16, return = +2.3%
    assert "51.16" in block
    assert "101.16" in block
    assert "W2/L1/B0" in block


def test_bankroll_block_negative_pnl():
    strat = _make_strategy()
    stats = {
        "alltime_pnl": -4.0,
        "today_closed": 2,
        "today_wins": 0,
        "today_losses": 2,
        "today_be": 0,
        "today_pnl": -4.0,
        "open_count": 0,
        "open_stake": 0.0,
    }
    block = strat._bankroll_block(stats)
    assert "46.00" in block
    assert "96.00" in block
    assert "-8.0%" in block
    assert "-4.0%" in block


# ---------------------------------------------------------------------------
# _check_open_risk_cap() tests
# ---------------------------------------------------------------------------


def _mock_settings_with_cap(cap: float | None):
    s = MagicMock()
    s.btc_scalp_max_open_risk_usd = cap
    return s


def _build_mock_session(open_trades: list[Trade]) -> MagicMock:
    """Build a minimal mock session whose .query().filter().all() returns open_trades."""
    mock_session = MagicMock()
    mock_q = MagicMock()
    mock_session.query.return_value = mock_q
    mock_q.filter.return_value = mock_q
    mock_q.all.return_value = open_trades
    return mock_session


def _make_open_trade_stub(size: float = 2.0) -> SimpleNamespace:
    return SimpleNamespace(size=size, strategy="btc_scalp", is_paper=True, status="open")


def test_check_open_risk_cap_no_cap():
    strat = _make_strategy()
    mock_session = _build_mock_session([])
    with patch("watchdog.strategies.btc_scalp.get_settings", return_value=_mock_settings_with_cap(None)):
        result = strat._check_open_risk_cap(mock_session)
    assert result is False
    assert strat._notified_msgs == []


def test_check_open_risk_cap_under_cap():
    strat = _make_strategy()
    open_trades = [_make_open_trade_stub(2.0)]  # 2.0 open
    mock_session = _build_mock_session(open_trades)
    # cap = 10, open=2.0, new=2.0 → 4.0 <= 10 → no block
    with patch("watchdog.strategies.btc_scalp.get_settings", return_value=_mock_settings_with_cap(10.0)):
        result = strat._check_open_risk_cap(mock_session)
    assert result is False
    assert strat._notified_msgs == []


def test_check_open_risk_cap_exactly_at_cap():
    strat = _make_strategy()
    open_trades = [_make_open_trade_stub(2.0), _make_open_trade_stub(2.0)]  # 4.0 open
    mock_session = _build_mock_session(open_trades)
    # cap = 4.0, open=4.0, new=2.0 → 6.0 > 4.0 → block
    with patch("watchdog.strategies.btc_scalp.get_settings", return_value=_mock_settings_with_cap(4.0)):
        result = strat._check_open_risk_cap(mock_session)
    assert result is True
    assert len(strat._notified_msgs) == 1
    assert "cap" in strat._notified_msgs[0].lower()


def test_check_open_risk_cap_notifies_only_once():
    strat = _make_strategy()
    open_trades = [_make_open_trade_stub(4.0)]  # 4.0 open, cap=4.0 → 6.0 > 4.0
    mock_session = _build_mock_session(open_trades)
    with patch("watchdog.strategies.btc_scalp.get_settings", return_value=_mock_settings_with_cap(4.0)):
        strat._check_open_risk_cap(mock_session)
        strat._check_open_risk_cap(mock_session)
        strat._check_open_risk_cap(mock_session)
    # Should notify exactly once even though called 3 times
    assert len(strat._notified_msgs) == 1


def test_check_open_risk_cap_resets_after_trade_open():
    """After _open_risk_cap_notified is reset to False, the next cap hit notifies again."""
    strat = _make_strategy()
    open_trades = [_make_open_trade_stub(4.0)]
    mock_session = _build_mock_session(open_trades)
    with patch("watchdog.strategies.btc_scalp.get_settings", return_value=_mock_settings_with_cap(4.0)):
        strat._check_open_risk_cap(mock_session)
        assert len(strat._notified_msgs) == 1

        # Simulate a trade being opened (resets flag)
        strat._open_risk_cap_notified = False

        strat._check_open_risk_cap(mock_session)
        assert len(strat._notified_msgs) == 2
