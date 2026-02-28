import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from watchdog.backtest.go_live_gate import check_go_live_gate
from watchdog.db.base import Base
from watchdog.db.models import Market, Trade


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as sess:
        yield sess


def test_gate_fails_zero_trades(session):
    passed, reasons = check_go_live_gate(session)
    assert not passed
    assert any("insufficient_trades" in r for r in reasons)


def test_gate_fails_all_lose(session):
    market = Market(slug="m1", question="Q1", domain="test", status="open")
    session.add(market)
    session.flush()

    for _ in range(50):
        t = Trade(
            market_id=market.id,
            signal_id=None,
            side="YES",
            size=10.0,
            entry_price=0.5,
            pnl=-1.0,
            kelly_fraction=0.1,
            is_paper=True,
            status="closed"
        )
        session.add(t)
    session.commit()

    passed, reasons = check_go_live_gate(session)
    assert not passed
    assert any("win_rate_below_threshold" in r for r in reasons)


def test_gate_passes_exact_thresholds(session):
    market = Market(slug="m2", question="Q2", domain="test", status="open")
    session.add(market)
    session.flush()

    for i in range(50):
        if i < 28:
            t = Trade(
                market_id=market.id,
                side="YES",
                size=10.0,
                entry_price=0.8,
                pnl=2.0,
                kelly_fraction=0.1,
                is_paper=True,
                status="closed"
            )
        else:
            t = Trade(
                market_id=market.id,
                side="YES",
                size=10.0,
                entry_price=0.6,
                pnl=-0.5,
                kelly_fraction=0.1,
                is_paper=True,
                status="closed"
            )
        session.add(t)
    session.commit()

    passed, reasons = check_go_live_gate(session)
    assert passed
    assert reasons == []


def test_gate_fails_negative_pnl(session):
    market = Market(slug="m3", question="Q3", domain="test", status="open")
    session.add(market)
    session.flush()

    for i in range(50):
        if i < 30:
            t = Trade(
                market_id=market.id,
                side="YES",
                size=10.0,
                entry_price=0.9,
                pnl=0.1,
                kelly_fraction=0.1,
                is_paper=True,
                status="closed"
            )
        else:
            t = Trade(
                market_id=market.id,
                side="YES",
                size=10.0,
                entry_price=0.9,
                pnl=-10.0,
                kelly_fraction=0.1,
                is_paper=True,
                status="closed"
            )
        session.add(t)
    session.commit()

    passed, reasons = check_go_live_gate(session)
    assert not passed
    assert any("total_pnl_below_threshold" in r for r in reasons)
