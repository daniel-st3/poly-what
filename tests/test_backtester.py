from __future__ import annotations

import pandas as pd

from watchdog.backtest.backtester import Backtester


def test_backtester_runs_and_logs_no_trade_decisions() -> None:
    markets = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "domain": "politics",
                "outcome": 1,
                "resolution_time": "2025-11-05T00:00:00Z",
                "price_30d_before": 0.40,
                "price_7d_before": 0.45,
                "price_1d_before": 0.55,
                "pre_registered": True,
            },
            {
                "market_id": "m2",
                "domain": "politics",
                "outcome": 0,
                "resolution_time": "2025-11-06T00:00:00Z",
                "price_30d_before": 0.60,
                "price_7d_before": 0.58,
                "price_1d_before": 0.52,
                "pre_registered": True,
            },
        ]
    )

    calibration = pd.DataFrame(
        [
            {"p_bucket": 0.40, "t_bucket": "long", "domain": "politics", "empirical_win_rate": 0.65},
            {"p_bucket": 0.45, "t_bucket": "mid", "domain": "politics", "empirical_win_rate": 0.62},
            {"p_bucket": 0.55, "t_bucket": "short", "domain": "politics", "empirical_win_rate": 0.70},
            {"p_bucket": 0.60, "t_bucket": "long", "domain": "politics", "empirical_win_rate": 0.35},
            {"p_bucket": 0.55, "t_bucket": "mid", "domain": "politics", "empirical_win_rate": 0.38},
            {"p_bucket": 0.50, "t_bucket": "short", "domain": "politics", "empirical_win_rate": 0.40},
        ]
    )

    backtester = Backtester(min_divergence_backtest=0.15)
    result = backtester.run_backtest(markets, calibration, strategy_mode="taker", verbose=False)

    assert result.n_total_markets == 2
    assert result.all_trades.shape[0] == 6  # 2 markets * (30d, 7d, 1d)
    assert result.n_traded >= 0
    assert result.equity_curve.size >= 1
