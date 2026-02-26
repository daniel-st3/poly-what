from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def brier_score(y_true: Sequence[float], p_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    if y.size == 0 or p.size == 0:
        return float("nan")
    if y.size != p.size:
        raise ValueError("y_true and p_pred must have the same length")
    return float(np.mean((p - y) ** 2))


def compute_max_drawdown(equity_curve: Sequence[float]) -> float:
    equity = np.asarray(equity_curve, dtype=np.float64)
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / np.maximum(running_max, 1e-12)
    return float(np.max(drawdown))


def compute_sharpe_ratio(
    returns: Sequence[float],
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> float:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    excess = arr - (risk_free / periods_per_year)
    stdev = float(np.std(excess, ddof=1))
    if stdev <= 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(excess) / stdev)


def compute_win_rate(outcomes: Sequence[float], predictions: Sequence[float]) -> float:
    out = np.asarray(outcomes, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if out.size == 0:
        return 0.0
    if out.size != pred.size:
        raise ValueError("outcomes and predictions must have the same length")

    pred_dir = (pred >= 0.5).astype(np.int8)
    outcome_dir = (out >= 0.5).astype(np.int8)
    return float(np.mean(pred_dir == outcome_dir))


def compute_calibration_error(
    y_true: Sequence[float],
    p_pred: Sequence[float],
    n_bins: int = 10,
) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_pred, dtype=np.float64)
    if y.size == 0 or p.size == 0:
        return 0.0
    if y.size != p.size:
        raise ValueError("y_true and p_pred must have the same length")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    for idx in range(n_bins):
        mask = bin_ids == idx
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(p[mask]))
        avg_acc = float(np.mean(y[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(avg_conf - avg_acc)
    return float(ece)


def monte_carlo_drawdown_distribution(
    returns: Sequence[float],
    n_simulations: int = 10_000,
    percentile: float = 0.95,
) -> tuple[float, float, np.ndarray]:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        zeros = np.zeros(1, dtype=np.float64)
        return 0.0, 0.0, zeros

    rng = np.random.default_rng(42)
    drawdowns = np.empty(n_simulations, dtype=np.float64)

    for idx in range(n_simulations):
        reordered = rng.permutation(arr)
        equity = np.cumprod(1 + reordered)
        equity = np.insert(equity, 0, 1.0)
        drawdowns[idx] = compute_max_drawdown(equity)

    median_dd = float(np.quantile(drawdowns, 0.50))
    pctl_dd = float(np.quantile(drawdowns, percentile))
    return median_dd, pctl_dd, drawdowns


def evaluate_pass_fail(
    win_rate: float,
    brier: float,
    n_trades: int,
    max_drawdown: float,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if n_trades < 50:
        reasons.append(f"insufficient_trades: {n_trades} < 50")
    if win_rate < 0.55:
        reasons.append(f"win_rate_below_threshold: {win_rate:.4f} < 0.55")
    if brier >= 0.22:
        reasons.append(f"brier_above_threshold: {brier:.4f} >= 0.22")
    if max_drawdown >= 0.25:
        reasons.append(f"max_drawdown_above_threshold: {max_drawdown:.4f} >= 0.25")

    return len(reasons) == 0, reasons
