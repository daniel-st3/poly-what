from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from watchdog.db.models import Market, Trade
from watchdog.services.pipeline import _hours_to_resolution

LOGGER = logging.getLogger(__name__)

# Sentinel values stored in Trade.close_reason while the trade is still OPEN,
# indicating that the first tier has already been partially exited.
_PARTIAL_TP1 = "partial_tp1"
_PARTIAL_SL1 = "partial_sl1"

# Trailing stop tiers: (min_profit_pct, trail_pct)
# When profit >= min_profit_pct, trail by trail_pct below high-water mark.
# Evaluated in reverse order (tightest tier wins).
_TRAILING_TIERS: list[tuple[float, float]] = [
    (0.25, 0.05),   # profit >= 25% → trail 5% below HWM
    (0.15, 0.07),   # profit >= 15% → trail 7% below HWM
    (0.05, 0.10),   # profit >= 5%  → trail 10% below HWM
    (0.00, 0.15),   # profit >= 0%  → trail 15% below HWM (initial soft stop)
]
_HARD_FLOOR_PCT = 0.25  # absolute stop-loss floor: -25% from entry (always active)


class ExitManager:
    """Manage take-profit, stop-loss, and time-based exits for open paper trades.

    Tiered exit strategy
    --------------------
    Take-profit:
      Tier 1 — exit 50 % of position when gain >= take_profit_pct   (default +10 %)
      Tier 2 — exit remaining 50 % when gain >= take_profit_2_pct   (default +20 %)

    Stop-loss:
      Tier 1 — exit 50 % of position when loss >= stop_loss_pct     (default -15 %)
      Tier 2 — exit remaining 50 % when loss >= stop_loss_2_pct     (default -25 %)

    Time-based exit:
      Force-close the full remaining position after max_hold_hours   (default 48 h)
      regardless of P&L direction.

    Resolution-proximity exit (unchanged):
      Full exit when < 2 h to resolution AND position is profitable.
    """

    def __init__(
        self,
        take_profit_pct: float = 0.10,
        take_profit_2_pct: float = 0.20,
        stop_loss_pct: float = 0.15,
        stop_loss_2_pct: float = 0.25,
        max_hold_hours: int = 48,
    ) -> None:
        self.take_profit_pct = take_profit_pct
        self.take_profit_2_pct = take_profit_2_pct
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_2_pct = stop_loss_2_pct
        self.max_hold_hours = max_hold_hours

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_exits(
        self,
        session: Session,
        *,
        current_prices: dict[int, float],
        platform: str,
        manifold: Any | None = None,
        polymarket: Any | None = None,
        polymarket_rest: Any | None = None,
    ) -> int:
        """Check all open paper trades for exit conditions.

        Returns the number of exit events recorded (each partial exit counts
        as one event; a full exit also counts as one event).
        """
        open_trades = session.execute(
            select(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .where(Trade.is_paper.is_(True), Trade.status == "open")
        ).all()

        closed_count = 0
        time_exit_live = 0
        time_exit_fallback = 0

        for trade, market in open_trades:
            # Compute age first — time exit does not require a live price.
            age_hours: float = 0.0
            if trade.opened_at is not None:
                opened = trade.opened_at
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=UTC)
                age_hours = (datetime.now(UTC) - opened).total_seconds() / 3600

            current_prob = current_prices.get(market.id)

            if current_prob is None:
                # No live price in the scan batch — only time exit can fire.
                if age_hours >= self.max_hold_hours:
                    # Try a targeted live fetch so P&L reflects reality.
                    fetched_price = self._fetch_live_price(
                        market, platform, manifold, polymarket_rest
                    )
                    if fetched_price is not None:
                        cur = fetched_price if trade.side == "YES" else (1 - fetched_price)
                        entry = max(trade.entry_price, 1e-9)
                        pnl_pct = (cur - entry) / entry
                        exit_reason = f"time_exit ({age_hours:.0f}h, {pnl_pct:+.1%})"
                        time_exit_live += 1
                    else:
                        # True fallback: no price anywhere — assume 0 % drift.
                        cur = trade.entry_price
                        exit_reason = f"time_exit ({age_hours:.0f}h, +0.0%)"
                        time_exit_fallback += 1
                    closed_count += self._full_exit(
                        session=session,
                        trade=trade,
                        market=market,
                        current_price=cur,
                        exit_reason=exit_reason,
                        platform=platform,
                        manifold=manifold,
                    )
                continue  # TP/SL/resolution checks need a live price — skip them

            # Live price available — compute derived values.
            current_prob = max(0.01, min(0.99, current_prob))
            current_price = current_prob if trade.side == "YES" else (1 - current_prob)
            entry = max(trade.entry_price, 1e-9)
            pnl_pct = (current_price - entry) / entry

            # --- Update high-water mark ---
            hwm = trade.high_water_mark if trade.high_water_mark is not None else entry
            if current_price > hwm:
                trade.high_water_mark = current_price
                hwm = current_price

            # --- Trailing stop check (runs before tiered fixed exits) ---
            trailing_exit = self._check_trailing_stop(trade, entry, current_price, hwm, pnl_pct)
            if trailing_exit is not None:
                closed_count += self._full_exit(
                    session=session,
                    trade=trade,
                    market=market,
                    current_price=current_price,
                    exit_reason=trailing_exit,
                    platform=platform,
                    manifold=manifold,
                )
                continue

            # Has tier-1 already fired?
            at_tier2 = trade.close_reason in (_PARTIAL_TP1, _PARTIAL_SL1)

            exit_reason: str | None = None
            is_partial = False  # True → only exit 50 %, keep rest open

            if at_tier2:
                # ---- Waiting for tier-2 ----
                if trade.close_reason == _PARTIAL_TP1 and pnl_pct >= self.take_profit_2_pct:
                    exit_reason = f"take_profit_2 ({pnl_pct:+.1%})"
                elif trade.close_reason == _PARTIAL_SL1 and pnl_pct <= -self.stop_loss_2_pct:
                    exit_reason = f"stop_loss_2 ({pnl_pct:+.1%})"

                # Time exit overrides everything — full close on aged position
                if exit_reason is None and age_hours >= self.max_hold_hours:
                    exit_reason = f"time_exit ({age_hours:.0f}h, {pnl_pct:+.1%})"

                # Resolution-proximity override (full close even mid-tier)
                if exit_reason is None and market.resolution_time is not None:
                    hours_left = _hours_to_resolution(market.resolution_time)
                    if hours_left < 2 and pnl_pct > 0:
                        exit_reason = f"resolution_imminent ({hours_left:.1f}h left, {pnl_pct:+.1%})"

            else:
                # ---- Fresh position — check tier-1 ----
                if pnl_pct >= self.take_profit_pct:
                    exit_reason = f"take_profit_1 ({pnl_pct:+.1%})"
                    is_partial = True
                elif pnl_pct <= -self.stop_loss_pct:
                    exit_reason = f"stop_loss_1 ({pnl_pct:+.1%})"
                    is_partial = True
                elif market.resolution_time is not None:
                    hours_left = _hours_to_resolution(market.resolution_time)
                    if hours_left < 2 and pnl_pct > 0:
                        exit_reason = f"resolution_imminent ({hours_left:.1f}h left, {pnl_pct:+.1%})"
                        # Full exit — no partial on resolution
                elif age_hours >= self.max_hold_hours:
                    exit_reason = f"time_exit ({age_hours:.0f}h, {pnl_pct:+.1%})"
                    # Full exit on time — don't bother with partial

            if exit_reason is None:
                continue

            if is_partial:
                closed_count += self._partial_exit(
                    session=session,
                    trade=trade,
                    market=market,
                    current_price=current_price,
                    exit_reason=exit_reason,
                    platform=platform,
                    manifold=manifold,
                )
            else:
                closed_count += self._full_exit(
                    session=session,
                    trade=trade,
                    market=market,
                    current_price=current_price,
                    exit_reason=exit_reason,
                    platform=platform,
                    manifold=manifold,
                )

        if time_exit_live or time_exit_fallback:
            LOGGER.warning(
                "time_exit summary: %d closed with live price, %d closed at entry (no price available)",
                time_exit_live,
                time_exit_fallback,
            )

        return closed_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_trailing_stop(
        self,
        trade: Any,
        entry: float,
        current_price: float,
        hwm: float,
        pnl_pct: float,
    ) -> str | None:
        """Return an exit reason string if the trailing stop has been hit, else None.

        Hard floor: always exit at -25% from entry regardless of HWM.
        Tiered trail: once position is profitable, trail below HWM by a
        percentage that tightens as peak profit grows.
        """
        # Hard floor — unconditional
        if pnl_pct <= -_HARD_FLOOR_PCT:
            return f"trailing_hard_floor ({pnl_pct:+.1%})"

        # Trailing tiers (only activate once HWM is above entry)
        hwm_profit_pct = (hwm - entry) / entry
        for min_profit, trail_pct in _TRAILING_TIERS:
            if hwm_profit_pct >= min_profit:
                trail_floor = hwm * (1 - trail_pct)
                if current_price <= trail_floor:
                    return (
                        f"trailing_stop (hwm={hwm:.4f}, trail={trail_pct:.0%}, "
                        f"floor={trail_floor:.4f}, cur={current_price:.4f}, {pnl_pct:+.1%})"
                    )
                break  # only the tightest applicable tier applies

        return None

    def _fetch_live_price(
        self,
        market: Market,
        platform: str,
        manifold: Any | None,
        polymarket_rest: Any | None,
    ) -> float | None:
        """Try to fetch a current YES probability for a market not in the scan batch.

        Returns the YES probability (0–1) on success, None on any failure.
        """
        try:
            if platform == "manifold" and manifold is not None:
                data = manifold.get_market(market.yes_token_id or market.slug)
                prob = data.get("probability")
                if prob is not None:
                    return max(0.01, min(0.99, float(prob)))

            elif platform == "polymarket" and polymarket_rest is not None:
                token_id = market.yes_token_id
                if token_id:
                    book = polymarket_rest.get_orderbook(token_id)
                    prob = book.get("probability")
                    if prob is not None:
                        return max(0.01, min(0.99, float(prob)))
        except Exception as exc:
            LOGGER.debug("Live price fetch failed for %s: %s", market.slug, exc)

        return None

    def _full_exit(
        self,
        *,
        session: Session,
        trade: Trade,
        market: Market,
        current_price: float,
        exit_reason: str,
        platform: str,
        manifold: Any | None,
    ) -> int:
        entry = max(trade.entry_price, 1e-9)
        contracts = trade.size / entry
        pnl_usdc = contracts * (current_price - entry)

        self._place_counter_bet(trade, market, platform, manifold)

        trade.status = "closed"
        trade.exit_price = current_price
        trade.pnl = pnl_usdc
        trade.closed_at = datetime.now(UTC)
        trade.close_reason = exit_reason

        LOGGER.warning(
            "EXIT %s | %s %.2f @ %.4f → %.4f | PnL=$%.2f | reason=%s",
            market.slug, trade.side, trade.size, entry, current_price, pnl_usdc, exit_reason,
        )
        return 1

    def _partial_exit(
        self,
        *,
        session: Session,
        trade: Trade,
        market: Market,
        current_price: float,
        exit_reason: str,
        platform: str,
        manifold: Any | None,
    ) -> int:
        entry = max(trade.entry_price, 1e-9)
        exit_size = trade.size * 0.5
        partial_contracts = exit_size / entry
        partial_pnl = partial_contracts * (current_price - entry)

        self._place_counter_bet(trade, market, platform, manifold, size_override=exit_size)

        # Create a closed record for the exited half
        closed_half = Trade(
            market_id=trade.market_id,
            signal_id=trade.signal_id,
            side=trade.side,
            size=exit_size,
            entry_price=trade.entry_price,
            exit_price=current_price,
            slippage=trade.slippage,
            pnl=partial_pnl,
            kelly_fraction=trade.kelly_fraction,
            confidence_score=trade.confidence_score,
            order_id=(trade.order_id or "") + "-partial1",
            is_paper=trade.is_paper,
            status="closed",
            opened_at=trade.opened_at,
            closed_at=datetime.now(UTC),
            strategy=trade.strategy,
            close_reason=exit_reason,
        )
        session.add(closed_half)

        # Reduce remaining position and mark tier-1 as done
        trade.size = exit_size
        trade.close_reason = _PARTIAL_TP1 if "take_profit" in exit_reason else _PARTIAL_SL1

        LOGGER.warning(
            "PARTIAL EXIT %s | %s %.2f (50%%) @ %.4f → %.4f | PnL=$%.2f | reason=%s | remaining=%.2f",
            market.slug, trade.side, exit_size, entry, current_price, partial_pnl, exit_reason, exit_size,
        )
        return 1

    def _place_counter_bet(
        self,
        trade: Trade,
        market: Market,
        platform: str,
        manifold: Any | None,
        size_override: float | None = None,
    ) -> None:
        opposite_side = "NO" if trade.side == "YES" else "YES"
        size = size_override if size_override is not None else trade.size
        if platform == "manifold" and manifold is not None:
            try:
                manifold.place_bet(
                    market.yes_token_id or market.slug,
                    opposite_side,
                    size,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Exit bet failed for %s (%s). Recording simulated exit.",
                    market.slug,
                    exc,
                )
