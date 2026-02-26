from __future__ import annotations

import numpy as np

from watchdog.backtest.metrics import (
    brier_score,
    compute_calibration_error,
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_win_rate,
    evaluate_pass_fail,
    monte_carlo_drawdown_distribution,
)


def test_brier_score() -> None:
    y_true = [1, 0, 1, 1]
    p_pred = [0.9, 0.2, 0.7, 0.6]
    score = brier_score(y_true, p_pred)
    assert 0 <= score <= 1


def test_drawdown_and_sharpe() -> None:
    equity = np.array([100, 110, 90, 95, 120], dtype=np.float64)
    max_dd = compute_max_drawdown(equity)
    assert max_dd > 0

    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.04], dtype=np.float64)
    sharpe = compute_sharpe_ratio(returns, periods_per_year=252, risk_free=0.0)
    assert isinstance(sharpe, float)


def test_win_rate_and_ece() -> None:
    outcomes = [1, 0, 1, 1, 0]
    preds = [0.8, 0.2, 0.7, 0.55, 0.4]
    win = compute_win_rate(outcomes, preds)
    ece = compute_calibration_error(outcomes, preds, n_bins=5)
    assert 0 <= win <= 1
    assert ece >= 0


def test_monte_carlo_drawdown_distribution() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.015], dtype=np.float64)
    median, p95, all_dd = monte_carlo_drawdown_distribution(returns, n_simulations=500, percentile=0.95)
    assert median >= 0
    assert p95 >= median
    assert all_dd.size == 500


def test_evaluate_pass_fail() -> None:
    passed, reasons = evaluate_pass_fail(0.60, 0.18, 80, 0.20)
    assert passed
    assert reasons == []

    failed, reasons = evaluate_pass_fail(0.50, 0.25, 20, 0.30)
    assert not failed
    assert len(reasons) >= 1
