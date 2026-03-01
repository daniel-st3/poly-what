from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

LOGGER = logging.getLogger(__name__)


class ArbitrageDetector:
    """Detect single-market and cross-platform arbitrage opportunities."""

    def __init__(
        self,
        min_arb_spread: float = 0.03,
        max_arb_position_size: float = 20.0,
    ) -> None:
        self.min_arb_spread = min_arb_spread
        self.max_arb_position_size = max_arb_position_size

    def check_single_market_arb(
        self,
        slug: str,
        yes_price: float,
        no_price: float,
    ) -> dict[str, Any] | None:
        """Check if YES + NO < 1 (minus spread buffer).

        If total_cost < (1 - min_arb_spread), buying both sides guarantees profit.

        Returns:
            Dict with arb details, or None if no arb.
        """
        total_cost = yes_price + no_price
        threshold = 1.0 - self.min_arb_spread

        if total_cost < threshold:
            profit_pct = (1.0 - total_cost) * 100
            LOGGER.warning(
                "ARB DETECTED: %s | YES=%.4f + NO=%.4f = %.4f | profit=%.2f%%",
                slug,
                yes_price,
                no_price,
                total_cost,
                profit_pct,
            )
            return {
                "type": "single_market",
                "slug": slug,
                "profit_pct": profit_pct,
                "yes_price": yes_price,
                "no_price": no_price,
                "total_cost": total_cost,
            }

        return None

    def check_cross_platform_arb(
        self,
        manifold_markets: list[dict[str, Any]],
        polymarket_markets: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Match markets across platforms by fuzzy question similarity.

        Returns list of cross-platform arb opportunities with spread > min_arb_spread.
        """
        arbs: list[dict[str, Any]] = []

        for m_market in manifold_markets:
            m_question = str(m_market.get("question") or "").lower()
            if not m_question:
                continue

            m_prob = float(m_market.get("probability") or 0.5)

            for p_market in polymarket_markets:
                p_question = str(p_market.get("question") or "").lower()
                if not p_question:
                    continue

                similarity = SequenceMatcher(None, m_question, p_question).ratio()
                if similarity < 0.60:
                    continue

                p_prob = float(p_market.get("probability") or 0.5)
                spread = abs(m_prob - p_prob)

                if spread > self.min_arb_spread:
                    arb = {
                        "type": "cross_platform",
                        "manifold_slug": m_market.get("slug"),
                        "manifold_question": m_market.get("question"),
                        "manifold_prob": m_prob,
                        "polymarket_slug": p_market.get("slug"),
                        "polymarket_question": p_market.get("question"),
                        "polymarket_prob": p_prob,
                        "spread_pct": spread * 100,
                        "similarity": similarity,
                    }
                    arbs.append(arb)
                    LOGGER.warning(
                        "CROSS-PLATFORM ARB: %s vs %s | spread=%.2f%% | sim=%.2f",
                        m_market.get("slug"),
                        p_market.get("slug"),
                        spread * 100,
                        similarity,
                    )

        return arbs
