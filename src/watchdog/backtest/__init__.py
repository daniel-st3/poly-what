"""Backtesting interfaces and quantitative metrics."""

from watchdog.backtest.backtester import Backtester, BacktestResults
from watchdog.backtest.historical_loader import BeckerHistoricalLoader
from watchdog.backtest.metrics import (
    brier_score,
    compute_calibration_error,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_win_rate,
    evaluate_pass_fail,
    monte_carlo_drawdown_distribution,
)

__all__ = [
    "BacktestResults",
    "Backtester",
    "BeckerHistoricalLoader",
    "brier_score",
    "compute_calibration_error",
    "compute_max_drawdown",
    "compute_sharpe_ratio",
    "compute_win_rate",
    "evaluate_pass_fail",
    "monte_carlo_drawdown_distribution",
]
