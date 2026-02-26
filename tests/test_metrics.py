from __future__ import annotations

import numpy as np

from watchdog.backtest.metrics import (
    brier_score,
    compute_max_drawdown,
    evaluate_pass_fail,
    monte_carlo_drawdown_distribution,
)


def test_brier_score_known_values() -> None:
    y_true = [1, 0]
    p_pred = [0.9, 0.2]
    assert brier_score(y_true, p_pred) == ((0.9 - 1) ** 2 + (0.2 - 0) ** 2) / 2


def test_compute_max_drawdown_known_curve() -> None:
    equity = np.array([100, 120, 90, 95, 80, 130], dtype=np.float64)
    max_dd = compute_max_drawdown(equity)
    assert round(max_dd, 4) == round((120 - 80) / 120, 4)


def test_mc_drawdown_p95_above_median() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.015], dtype=np.float64)
    median, p95, _ = monte_carlo_drawdown_distribution(returns, n_simulations=1000, percentile=0.95)
    assert p95 >= median


def test_evaluate_pass_fail_edge_cases() -> None:
    passed, reasons = evaluate_pass_fail(0.55, 0.21, 50, 0.24)
    assert passed
    assert reasons == []

    failed, reasons = evaluate_pass_fail(0.55, 0.22, 50, 0.24)
    assert not failed
    assert any("brier" in r for r in reasons)
