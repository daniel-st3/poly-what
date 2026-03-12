from __future__ import annotations

import sys
from pathlib import Path

from watchdog.db.models import Market, Trade

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_exits_only import _infer_platform


def test_infer_platform_prefers_order_id_prefix() -> None:
    market = Market(slug="test-market", question="Test?", domain="other")
    trade = Trade(
        market_id=1,
        side="YES",
        size=10.0,
        entry_price=0.5,
        kelly_fraction=0.1,
        order_id="paper-polymarket-test-market-1",
        is_paper=True,
        status="open",
    )

    assert _infer_platform(trade, market) == "polymarket"


def test_infer_platform_falls_back_to_condition_id() -> None:
    market = Market(
        slug="test-market",
        question="Test?",
        domain="other",
        condition_id="0xcondition",
    )
    trade = Trade(
        market_id=1,
        side="YES",
        size=10.0,
        entry_price=0.5,
        kelly_fraction=0.1,
        order_id="arb-yes-test-market-1",
        is_paper=True,
        status="open",
    )

    assert _infer_platform(trade, market) == "polymarket"
