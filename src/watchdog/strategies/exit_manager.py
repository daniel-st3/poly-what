from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from watchdog.db.models import Market, Trade
from watchdog.services.pipeline import _hours_to_resolution

LOGGER = logging.getLogger(__name__)


class ExitManager:
    """Manage take-profit, stop-loss, and time-based exits for open paper trades."""

    def __init__(
        self,
        take_profit_pct: float = 0.30,
        stop_loss_pct: float = 0.15,
        max_hold_days: int = 7,
    ) -> None:
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_hold_days = max_hold_days

    def check_exits(
        self,
        session: Session,
        *,
        current_prices: dict[int, float],
        platform: str,
        manifold: Any | None = None,
        polymarket: Any | None = None,
    ) -> int:
        """Check all open paper trades for exit conditions.

        Args:
            session: Active DB session.
            current_prices: Dict mapping market_id -> current_probability.
            platform: 'manifold' or 'polymarket'.
            manifold: ManifoldClient instance (optional).
            polymarket: PolymarketCli instance (optional).

        Returns:
            Number of trades closed.
        """
        open_trades = session.execute(
            select(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .where(Trade.is_paper.is_(True), Trade.status == "open")
        ).all()

        closed_count = 0

        for trade, market in open_trades:
            current_prob = current_prices.get(market.id)
            if current_prob is None:
                continue

            current_prob = max(0.01, min(0.99, current_prob))
            current_price = current_prob if trade.side == "YES" else (1 - current_prob)
            entry = max(trade.entry_price, 1e-9)
            pnl_pct = (current_price - entry) / entry

            # Determine exit reason (if any)
            exit_reason: str | None = None

            if pnl_pct >= self.take_profit_pct:
                exit_reason = f"take_profit ({pnl_pct:+.1%})"
            elif pnl_pct <= -self.stop_loss_pct:
                exit_reason = f"stop_loss ({pnl_pct:+.1%})"
            elif market.resolution_time is not None:
                # Resolution-proximity exit: lock gains if <2h to close
                hours_left = _hours_to_resolution(market.resolution_time)
                if hours_left < 2 and pnl_pct > 0:
                    exit_reason = f"resolution_imminent ({hours_left:.1f}h left, {pnl_pct:+.1%})"

            # Time-based exit: held too long with profit
            if exit_reason is None and trade.opened_at is not None:
                opened = trade.opened_at
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=UTC)
                age_days = (datetime.now(UTC) - opened).total_seconds() / 86400
                if age_days >= self.max_hold_days and pnl_pct > 0:
                    exit_reason = f"time_exit ({age_days:.1f}d, {pnl_pct:+.1%})"

            if exit_reason is None:
                continue

            # Calculate PnL in USDC
            contracts = trade.size / entry
            pnl_usdc = contracts * (current_price - entry)

            # Execute counter-bet on Manifold
            opposite_side = "NO" if trade.side == "YES" else "YES"
            if platform == "manifold" and manifold is not None:
                try:
                    manifold.place_bet(
                        market.yes_token_id or market.slug,
                        opposite_side,
                        trade.size,
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Exit bet failed for %s (%s). Recording simulated exit.",
                        market.slug,
                        exc,
                    )

            # Update trade record
            trade.status = "closed"
            trade.exit_price = current_price
            trade.pnl = pnl_usdc
            trade.closed_at = datetime.now(UTC)

            closed_count += 1
            LOGGER.warning(
                "EXIT %s | %s %s @ %.4f → %.4f | PnL=$%.2f | reason=%s",
                market.slug,
                trade.side,
                trade.size,
                entry,
                current_price,
                pnl_usdc,
                exit_reason,
            )

        return closed_count
