from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from watchdog.core.config import Settings
from watchdog.db.init import init_db
from watchdog.db.models import MakerQuote, Market
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.polymarket_cli import CliResponse
from watchdog.risk.vpin import VPINCalculator
from watchdog.scripts.run_market_maker import run_market_maker_cycle
from watchdog.trading.maker_model import AvellanedaStoikovPredictionMM


class FakeCli:
    def __init__(self, high_vpin: bool) -> None:
        self.high_vpin = high_vpin
        self.order_calls: list[tuple[str, str, float, float, bool]] = []

    def list_markets(self, limit: int = 50) -> CliResponse:
        return CliResponse(
            payload=[
                {
                    "slug": "mkt-1",
                    "question": "Will X happen?",
                    "domain": "politics",
                    "status": "active",
                    "endDate": (datetime.now(UTC) + timedelta(days=20)).isoformat(),
                    "yes_token_id": "yes-1",
                    "no_token_id": "no-1",
                }
            ],
            latency_ms=1,
        )

    def orderbook(self, market_slug: str) -> CliResponse:
        if self.high_vpin:
            payload = {
                "orderbook": {
                    "bids": [{"price": 0.50, "size": 0}],
                    "asks": [{"price": 0.52, "size": 10000}],
                }
            }
        else:
            payload = {
                "orderbook": {
                    "bids": [{"price": 0.49, "size": 5000}],
                    "asks": [{"price": 0.51, "size": 5000}],
                }
            }
        return CliResponse(payload=payload, latency_ms=1)

    def create_limit_order(self, token_id: str, side: str, price: float, size: float, post_only: bool = True) -> CliResponse:
        self.order_calls.append((token_id, side, price, size, post_only))
        return CliResponse(payload={"order_id": f"{token_id}-{side}"}, latency_ms=1)


def _setup_session(settings: Settings):
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    session = session_factory()
    session.add(
        Market(
            slug="mkt-1",
            question="Will X happen?",
            domain="politics",
            yes_token_id="yes-1",
            no_token_id="no-1",
            resolution_time=datetime.now(UTC) + timedelta(days=20),
            status="active",
        )
    )
    session.commit()
    return session


def test_vpin_above_threshold_skips_quotes() -> None:
    settings = Settings(database_url="sqlite:///:memory:", vpin_kill_threshold=0.7)
    session = _setup_session(settings)
    cli = FakeCli(high_vpin=True)

    counts = run_market_maker_cycle(
        session=session,
        cli=cli,
        settings=settings,
        maker=AvellanedaStoikovPredictionMM(),
        vpin_calc=VPINCalculator(bucket_volume=100.0, rolling_buckets=10),
        max_markets=5,
        market_slug=None,
        dry_run=False,
        quote_size=1.0,
    )

    assert counts["quoted"] == 0
    assert counts["skipped"] >= 1
    assert len(cli.order_calls) == 0


def test_vpin_below_threshold_places_quotes() -> None:
    settings = Settings(database_url="sqlite:///:memory:", backtest_spread_proxy=0.005)
    session = _setup_session(settings)
    cli = FakeCli(high_vpin=False)

    counts = run_market_maker_cycle(
        session=session,
        cli=cli,
        settings=settings,
        maker=AvellanedaStoikovPredictionMM(),
        vpin_calc=VPINCalculator(bucket_volume=10000.0, rolling_buckets=10),
        max_markets=5,
        market_slug=None,
        dry_run=False,
        quote_size=2.0,
    )

    assert counts["quoted"] == 1
    assert len(cli.order_calls) == 2

    yes_call = cli.order_calls[0]
    no_call = cli.order_calls[1]

    assert yes_call[0] == "yes-1"
    assert yes_call[1] == "YES"
    assert round(yes_call[2], 2) == 0.49
    assert yes_call[4] is True

    assert no_call[0] == "no-1"
    assert no_call[1] == "NO"
    assert round(no_call[2], 2) == 0.49
    assert no_call[4] is True

    rows = session.execute(select(MakerQuote)).scalars().all()
    assert len(rows) >= 1
