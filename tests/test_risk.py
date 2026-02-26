from __future__ import annotations

import numpy as np

from watchdog.risk.kelly import EmpiricalKellySizer
from watchdog.risk.vpin import TradeFlow, VPINCalculator
from watchdog.trading.maker_model import AvellanedaStoikovPredictionMM


def test_full_kelly_positive_for_edge() -> None:
    f = EmpiricalKellySizer.full_kelly_fraction(p_model=0.60, p_market=0.50, side="YES")
    assert f > 0


def test_empirical_kelly_scales_with_uncertainty() -> None:
    sizer = EmpiricalKellySizer(kelly_fraction=0.25, max_drawdown_p95=0.35)
    edges = np.array([0.10, 0.11, 0.12, 0.09, 0.10])
    returns = np.array([0.05, -0.03, 0.04, -0.02, 0.06, -0.01])

    result = sizer.size(
        p_model=0.58,
        p_market=0.50,
        side="YES",
        historical_edge_estimates=edges,
        historical_trade_returns=returns,
    )

    assert 0 <= result.empirical_fraction <= 1
    assert result.drawdown_p95 >= 0


def test_vpin_higher_for_imbalanced_flow() -> None:
    vpin = VPINCalculator(bucket_volume=100.0, rolling_buckets=10)

    balanced = [TradeFlow(side="BUY", volume=50), TradeFlow(side="SELL", volume=50)] * 10
    imbalanced = [TradeFlow(side="BUY", volume=90), TradeFlow(side="SELL", volume=10)] * 10

    assert vpin.compute(imbalanced) > vpin.compute(balanced)


def test_maker_quotes_within_bounds() -> None:
    mm = AvellanedaStoikovPredictionMM(inventory_limit=1.0)
    quote = mm.compute_quotes(mid_price=0.55, inventory=0.2, tau_hours=12)
    assert 0.01 <= quote.bid_price <= 0.99
    assert 0.01 <= quote.ask_price <= 0.99
    assert quote.ask_price >= quote.bid_price
