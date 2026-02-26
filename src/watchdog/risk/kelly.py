from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SizingResult:
    full_kelly: float
    empirical_fraction: float
    cv_edge: float
    drawdown_p95: float


class EmpiricalKellySizer:
    def __init__(self, kelly_fraction: float = 0.25, max_drawdown_p95: float = 0.30) -> None:
        self.kelly_fraction = kelly_fraction
        self.max_drawdown_p95 = max_drawdown_p95

    @staticmethod
    def full_kelly_fraction(p_model: float, p_market: float, side: str) -> float:
        p_model = float(np.clip(p_model, 1e-6, 1 - 1e-6))
        p_market = float(np.clip(p_market, 1e-6, 1 - 1e-6))

        if side.upper() == "YES":
            q = p_model
            b = (1 - p_market) / p_market
        elif side.upper() == "NO":
            q = 1 - p_model
            p_no = 1 - p_market
            b = (1 - p_no) / p_no
        else:
            return 0.0

        numerator = b * q - (1 - q)
        f = numerator / b
        return float(np.clip(f, 0.0, 1.0))

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / np.maximum(running_max, 1e-12)
        return float(np.max(drawdowns))

    def _drawdown_p95(self, historical_returns: np.ndarray, bet_fraction: float, sims: int = 10_000) -> float:
        if historical_returns.size == 0 or bet_fraction <= 0:
            return 0.0

        rng = np.random.default_rng(42)
        drawdowns = np.empty(sims, dtype=np.float64)

        for i in range(sims):
            reordered = rng.permutation(historical_returns)
            equity = np.cumprod(1 + bet_fraction * reordered)
            equity = np.insert(equity, 0, 1.0)
            drawdowns[i] = self._max_drawdown(equity)

        return float(np.quantile(drawdowns, 0.95))

    def size(
        self,
        p_model: float,
        p_market: float,
        side: str,
        historical_edge_estimates: np.ndarray,
        historical_trade_returns: np.ndarray,
    ) -> SizingResult:
        full_kelly = self.full_kelly_fraction(p_model=p_model, p_market=p_market, side=side)

        edges = np.asarray(historical_edge_estimates, dtype=np.float64)
        if edges.size == 0:
            cv_edge = 1.0
        else:
            mean = float(np.mean(edges))
            stdev = float(np.std(edges))
            cv_edge = stdev / max(abs(mean), 1e-8)

        uncertainty_scale = float(np.clip(1 - cv_edge, 0.0, 1.0))
        base_fraction = full_kelly * self.kelly_fraction * uncertainty_scale

        returns = np.asarray(historical_trade_returns, dtype=np.float64)
        drawdown_p95 = self._drawdown_p95(returns, base_fraction)

        if drawdown_p95 <= self.max_drawdown_p95:
            return SizingResult(
                full_kelly=full_kelly,
                empirical_fraction=base_fraction,
                cv_edge=cv_edge,
                drawdown_p95=drawdown_p95,
            )

        low, high = 0.0, base_fraction
        for _ in range(20):
            mid = (low + high) / 2
            dd = self._drawdown_p95(returns, mid, sims=1000)
            if dd > self.max_drawdown_p95:
                high = mid
            else:
                low = mid

        scaled_fraction = low
        final_dd = self._drawdown_p95(returns, scaled_fraction)
        return SizingResult(
            full_kelly=full_kelly,
            empirical_fraction=scaled_fraction,
            cv_edge=cv_edge,
            drawdown_p95=final_dd,
        )
