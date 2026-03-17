"""Module 3: Resolution Proximity Filter.

Scans all open paper trades and flags them based on proximity to resolution:

- resolution_flag = "near_certain": within 24h AND prob > 0.88 or < 0.12
  Action: paper-close at current price (close_reason="resolution_near_certain")

- resolution_flag = "high_risk_close": within 24h AND 0.45 <= prob <= 0.55
  Action: reduce position 50% (paper partial exit), tighten flag for monitoring

- resolution_flag = "normal": everything else

Does NOT touch exit_manager.py. Runs as a standalone pre-processor.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy.orm import Session

from watchdog.db.models import Market, Trade

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

LOGGER = logging.getLogger(__name__)

RESOLUTION_WINDOW_HOURS = 24
NEAR_CERTAIN_HIGH = 0.88
NEAR_CERTAIN_LOW = 0.12
HIGH_RISK_LOW = 0.45
HIGH_RISK_HIGH = 0.55


class ResolutionProximityFilter:
    """Flag and act on open trades approaching resolution."""

    def run(
        self,
        session: Session,
        current_prices: dict[int, float],  # market_id → current mid-price
        is_paper: bool = True,
    ) -> dict[str, Any]:
        """Run resolution check. Returns summary stats dict."""
        print("\n⏰ RESOLUTION CHECK")

        now = datetime.now(UTC)
        cutoff = now + timedelta(hours=RESOLUTION_WINDOW_HOURS)

        open_trades = (
            session.query(Trade)
            .filter(Trade.status == "open", Trade.is_paper == is_paper)
            .all()
        )

        checked = 0
        near_certain = 0
        high_risk = 0
        normal = 0

        for trade in open_trades:
            market: Market | None = session.get(Market, trade.market_id)
            if market is None:
                continue

            # Skip if no resolution time or not within window
            if market.resolution_time is None:
                trade.resolution_flag = "normal"
                normal += 1
                checked += 1
                continue

            res_time = market.resolution_time
            if res_time.tzinfo is None:
                res_time = res_time.replace(tzinfo=UTC)

            if res_time > cutoff:
                trade.resolution_flag = "normal"
                normal += 1
                checked += 1
                continue

            # Get current probability for this market
            prob = current_prices.get(trade.market_id)
            if prob is None:
                # No price available — log and skip flagging
                LOGGER.warning(
                    "RESOLUTION CHECK: no price for market_id=%d (%s) — skipping",
                    trade.market_id,
                    market.slug,
                )
                checked += 1
                continue

            # Determine effective probability for the trade direction
            effective_prob = prob if trade.side == "YES" else (1.0 - prob)

            if effective_prob > NEAR_CERTAIN_HIGH or effective_prob < NEAR_CERTAIN_LOW:
                trade.resolution_flag = "near_certain"
                near_certain += 1
                if is_paper:
                    self._paper_close(session, trade, prob, now, "resolution_near_certain")
                LOGGER.info(
                    "RESOLUTION CHECK: near_certain exit — %s side=%s prob=%.3f",
                    market.slug,
                    trade.side,
                    prob,
                )

            elif HIGH_RISK_LOW <= prob <= HIGH_RISK_HIGH:
                trade.resolution_flag = "high_risk_close"
                high_risk += 1
                if is_paper and trade.size > 0:
                    self._paper_partial_exit(session, trade, prob, now)
                LOGGER.info(
                    "RESOLUTION CHECK: high_risk_close — %s prob=%.3f, reducing 50%%",
                    market.slug,
                    prob,
                )

            else:
                trade.resolution_flag = "normal"
                normal += 1

            checked += 1

        session.commit()

        print(
            f"⏰ RESOLUTION CHECK done: {checked} trades checked, "
            f"{near_certain} near-certain exit(s), {high_risk} high-risk flagged"
        )
        return {
            "trades_checked": checked,
            "near_certain": near_certain,
            "high_risk": high_risk,
            "normal": normal,
        }

    def _paper_close(
        self,
        session: Session,
        trade: Trade,
        current_price: float,
        now: datetime,
        close_reason: str,
    ) -> None:
        """Fully close a paper trade at current price."""
        exit_price = current_price if trade.side == "YES" else (1.0 - current_price)
        pnl = (exit_price - trade.entry_price) * trade.size
        trade.status = "closed"
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.close_reason = close_reason
        trade.closed_at = now

    def _paper_partial_exit(
        self,
        session: Session,
        trade: Trade,
        current_price: float,
        now: datetime,
    ) -> None:
        """Reduce position 50% — close half and shrink the original trade."""
        exit_price = current_price if trade.side == "YES" else (1.0 - current_price)
        half_size = trade.size / 2.0
        pnl = (exit_price - trade.entry_price) * half_size

        # Create a closed record for the exited half
        closed_half = Trade(
            market_id=trade.market_id,
            signal_id=trade.signal_id,
            side=trade.side,
            size=half_size,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            slippage=trade.slippage,
            pnl=pnl,
            kelly_fraction=trade.kelly_fraction,
            confidence_score=trade.confidence_score,
            is_paper=trade.is_paper,
            status="closed",
            opened_at=trade.opened_at,
            closed_at=now,
            strategy=trade.strategy,
            close_reason="resolution_high_risk_partial",
            resolution_flag="high_risk_close",
        )
        session.add(closed_half)

        # Shrink original trade by 50%
        trade.size = half_size
        if trade.remaining_fraction is not None:
            trade.remaining_fraction = trade.remaining_fraction * 0.5

    def check_resolved_markets(self, session: Session) -> dict[str, Any]:
        """Close open paper trades for markets that have actually resolved (outcome known).

        For each open paper trade:
          - Checks Market.resolution_outcome first (may already be set by sync-markets)
          - Falls back to Gamma API via condition_id if not set
          - YES resolution: YES-side exits at 1.0 (profit), NO-side exits at 0.0 (loss)
          - NO  resolution: NO-side exits at 1.0 (profit), YES-side exits at 0.0 (loss)

        PnL is written to Trade.pnl and picked up automatically by the net_pnl sum in
        send-daily-telegram (it is a closed non-arb trade).
        """
        print("\n⏰ RESOLUTION CLOSE CHECK")

        open_trades = (
            session.query(Trade)
            .filter(Trade.status == "open", Trade.is_paper.is_(True))
            .all()
        )

        if not open_trades:
            print("⏰ RESOLUTION CLOSE CHECK done: no open trades")
            return {"resolved_trades": 0, "resolution_pnl": 0.0}

        market_ids = {t.market_id for t in open_trades}
        now = datetime.now(UTC)
        resolved_count = 0
        resolution_pnl = 0.0

        for market_id in market_ids:
            market: Market | None = session.get(Market, market_id)
            if market is None:
                continue

            # 1. Use resolution_outcome already stored on the Market row
            outcome = (market.resolution_outcome or "").upper()

            # 2. If unknown, ask Gamma API
            if outcome not in ("YES", "NO") and market.condition_id:
                try:
                    resp = httpx.get(
                        f"{GAMMA_API_BASE}/markets",
                        params={"conditionId": market.condition_id},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        items = data if isinstance(data, list) else [data]
                        for item in items:
                            if item.get("resolved"):
                                raw = (item.get("resolutionOutcome") or "").upper()
                                if raw in ("YES", "NO"):
                                    outcome = raw
                                    market.resolution_outcome = outcome
                                    market.status = "resolved"
                                    break
                except Exception as exc:
                    LOGGER.warning(
                        "RESOLUTION CLOSE: Gamma API failed for %s — %s",
                        market.slug,
                        exc,
                    )

            if outcome not in ("YES", "NO"):
                continue

            # Close every open trade for this market with correct PnL
            for trade in open_trades:
                if trade.market_id != market_id:
                    continue
                if trade.side == "YES":
                    exit_price = 1.0 if outcome == "YES" else 0.0
                else:  # side == "NO"
                    exit_price = 1.0 if outcome == "NO" else 0.0

                pnl = (exit_price - trade.entry_price) * trade.size
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.status = "closed"
                trade.closed_at = now
                trade.close_reason = "resolution"
                resolution_pnl += pnl
                resolved_count += 1
                LOGGER.info(
                    "RESOLUTION CLOSE: trade #%d %s side=%s outcome=%s pnl=%.4f",
                    trade.id,
                    market.slug,
                    trade.side,
                    outcome,
                    pnl,
                )

        if resolved_count > 0:
            session.commit()

        sign = "+" if resolution_pnl >= 0 else ""
        print(
            f"⏰ RESOLUTION CLOSE CHECK done: {resolved_count} trade(s) closed, "
            f"PnL {sign}${resolution_pnl:.4f}"
        )
        return {"resolved_trades": resolved_count, "resolution_pnl": resolution_pnl}
