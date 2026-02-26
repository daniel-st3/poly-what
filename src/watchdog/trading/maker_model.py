from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class QuotePlan:
    reservation_price: float
    optimal_spread: float
    bid_price: float
    ask_price: float
    reward_eligible: bool


class AvellanedaStoikovPredictionMM:
    """
    Bounded-price adaptation of Avellaneda-Stoikov for prediction markets.

    - Price dynamics operate in log-odds space.
    - Quotes are projected back to [0,1].
    - Inventory bounds widen spreads as position approaches limits.
    """

    def __init__(
        self,
        gamma: float = 0.15,
        k: float = 8.0,
        sigma: float = 0.08,
        inventory_limit: float = 1.0,
        min_spread: float = 0.01,
    ) -> None:
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.inventory_limit = max(inventory_limit, 1e-6)
        self.min_spread = min_spread

    @staticmethod
    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    @staticmethod
    def _inv_logit(x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    def compute_quotes(self, mid_price: float, inventory: float, tau_hours: float) -> QuotePlan:
        mid_price = float(np.clip(mid_price, 1e-4, 1 - 1e-4))
        tau = max(tau_hours / 24.0, 1e-6)

        inventory_norm = float(np.clip(inventory / self.inventory_limit, -1.0, 1.0))
        mid_logit = self._logit(mid_price)

        # Inventory-adjusted reservation price in log-odds space.
        reservation_logit = mid_logit - inventory_norm * self.gamma * (self.sigma**2) * tau
        reservation_price = self._inv_logit(reservation_logit)

        # Spread = inventory risk term + informational/liquidity term.
        inventory_risk_term = self.gamma * (self.sigma**2) * tau
        profit_term = (2 / self.gamma) * np.log(1 + self.gamma / self.k)

        inventory_pressure = abs(inventory_norm)
        bounded_inventory_widening = 1 + 1.5 * inventory_pressure

        spread_logit = (inventory_risk_term + profit_term) * bounded_inventory_widening
        spread_prob = max(self._inv_logit(spread_logit) - 0.5, self.min_spread / 2)
        optimal_spread = 2 * spread_prob

        bid_price = float(np.clip(reservation_price - optimal_spread / 2, 0.01, 0.99))
        ask_price = float(np.clip(reservation_price + optimal_spread / 2, 0.01, 0.99))

        reward_eligible = optimal_spread <= 0.10
        return QuotePlan(
            reservation_price=reservation_price,
            optimal_spread=optimal_spread,
            bid_price=bid_price,
            ask_price=ask_price,
            reward_eligible=reward_eligible,
        )
