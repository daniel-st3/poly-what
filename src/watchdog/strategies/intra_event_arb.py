"""Module 1: Intra-Event YES Sum Arbitrage Scanner (Polymarket only).

For each active multi-outcome event, sums all YES outcome prices.
- sum < 0.97: "underpriced" — buying all YES costs < $1, one pays $1 → guaranteed profit
- sum > 1.03: "overpriced" — buying all NO guarantees profit

Logs opportunities to arb_opportunities table.
In paper mode, records linked paper trades in trades table (strategy="intra_event_arb").
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from watchdog.db.models import ArbOpportunity, Market, Trade

if TYPE_CHECKING:
    from watchdog.market_data.polymarket_rest import PolymarketRestClient

LOGGER = logging.getLogger(__name__)

MIN_EVENT_VOLUME_USD = 5_000.0
MIN_PROFIT_CENTS = 3.0
UNDERPRICED_THRESHOLD = 0.97
OVERPRICED_THRESHOLD = 1.03


class IntraEventArbScanner:
    """Scan Polymarket multi-outcome events for YES-sum arbitrage opportunities."""

    def scan(
        self,
        session: Session,
        rest_client: "PolymarketRestClient",
        is_paper: bool = True,
    ) -> dict[str, Any]:
        """Run one scan cycle. Returns summary stats dict."""
        print("\n📊 ARB SCAN")

        try:
            events = rest_client.get_events(limit=100)
        except Exception as exc:
            LOGGER.warning("ARB SCAN: failed to fetch events — %s", exc)
            print("📊 ARB SCAN done: fetch failed, skipping")
            return {"events_scanned": 0, "opportunities": 0, "est_daily_profit_cents": 0.0}

        events_scanned = len(events)
        opportunities_found = 0
        total_profit_cents = 0.0

        for event in events:
            result = self._process_event(session, event, is_paper)
            if result:
                opportunities_found += 1
                total_profit_cents += result["profit_cents"]

        session.commit()

        est_daily = total_profit_cents * 3  # runs 3x/day
        print(
            f"📊 ARB SCAN done: {events_scanned} events scanned, "
            f"{opportunities_found} opportunities found, "
            f"est. ${est_daily / 100:.2f}/day"
        )
        return {
            "events_scanned": events_scanned,
            "opportunities": opportunities_found,
            "est_daily_profit_cents": est_daily,
        }

    def _process_event(
        self,
        session: Session,
        event: dict[str, Any],
        is_paper: bool,
    ) -> dict[str, Any] | None:
        event_slug = event.get("event_slug", "")
        markets = event.get("markets", [])

        if len(markets) < 2:
            return None

        # Sum YES outcome prices
        yes_prices: list[float] = []
        total_volume = 0.0
        for m in markets:
            outcome_prices = m.get("outcome_prices") or []
            if isinstance(outcome_prices, str):
                import json
                try:
                    outcome_prices = json.loads(outcome_prices)
                except Exception:
                    outcome_prices = []
            if outcome_prices:
                try:
                    yes_prices.append(float(outcome_prices[0]))
                except (TypeError, ValueError, IndexError):
                    pass
            total_volume += float(m.get("volume") or 0)

        if not yes_prices or len(yes_prices) < 2:
            return None

        # Volume filter
        if total_volume < MIN_EVENT_VOLUME_USD:
            return None

        yes_sum = sum(yes_prices)

        # Determine arb type
        if yes_sum < UNDERPRICED_THRESHOLD:
            arb_type = "underpriced"
            profit_cents = (1.0 - yes_sum) * 100
        elif yes_sum > OVERPRICED_THRESHOLD:
            arb_type = "overpriced"
            profit_cents = (yes_sum - 1.0) * 100
        else:
            return None

        # Min profit guard
        if profit_cents < MIN_PROFIT_CENTS:
            return None

        LOGGER.warning(
            "ARB OPPORTUNITY [%s] event=%s sum_yes=%.4f profit=%.1f¢",
            arb_type,
            event_slug,
            yes_sum,
            profit_cents,
        )

        # Persist to arb_opportunities
        opp = ArbOpportunity(
            event_slug=event_slug,
            type=arb_type,
            sum_of_yes=yes_sum,
            profit_cents=profit_cents,
            timestamp=datetime.now(UTC),
            was_actioned=False,
            strategy="intra_event_arb",
        )
        session.add(opp)

        # Paper trade (deduplication guard)
        if is_paper:
            self._create_paper_trades(session, event_slug, markets, yes_prices, arb_type)

        return {"profit_cents": profit_cents}

    def _create_paper_trades(
        self,
        session: Session,
        event_slug: str,
        markets: list[dict[str, Any]],
        yes_prices: list[float],
        arb_type: str,
    ) -> None:
        """Create paper Trade records for each YES leg — skip if already open."""
        # Deduplication: check for any existing arb trades for this event (open or closed)
        existing = session.execute(
            text("SELECT COUNT(*) FROM trades WHERE order_id LIKE :pattern"),
            {"pattern": f"arb_{event_slug}_%"},
        ).scalar()
        if existing and existing > 0:
            LOGGER.debug("ARB SCAN: skipping paper trades for %s — already recorded", event_slug)
            return

        now = datetime.now(UTC)
        group_id = f"arb_{event_slug}_{int(now.timestamp())}"

        for i, (m, yes_price) in enumerate(zip(markets, yes_prices)):
            # Skip legs with very low prices (< 1%) — these are near-zero-probability
            # markets that cause exit_manager to produce misleading PnL figures.
            if yes_price < 0.01:
                LOGGER.debug("ARB SCAN: skipping sub-1%% leg in %s (price=%.4f)", event_slug, yes_price)
                continue

            slug = m.get("slug") or m.get("condition_id") or f"{event_slug}_leg_{i}"

            # Find or create Market row
            market_row = session.query(Market).filter_by(slug=slug).first()
            if market_row is None:
                market_row = Market(
                    slug=slug,
                    question=m.get("question") or slug,
                    domain="other",
                    yes_token_id=m.get("yes_token_id"),
                    condition_id=m.get("condition_id"),
                    status="active",
                )
                session.add(market_row)
                session.flush()  # get market_row.id

            # Mark as immediately closed (status="closed", pnl=0.0) so that
            # exit_manager does not process arb paper trades as real held positions.
            # Actual arb profit data lives in the arb_opportunities table.
            trade = Trade(
                market_id=market_row.id,
                signal_id=None,
                side="YES",
                size=1.0,
                entry_price=yes_price,
                exit_price=yes_price,
                pnl=0.0,
                kelly_fraction=0.0,
                is_paper=True,
                status="closed",
                opened_at=now,
                closed_at=now,
                strategy="intra_event_arb",
                close_reason="arb_paper_entry",
                order_id=f"{group_id}_leg{i}",
                remaining_fraction=1.0,
            )
            session.add(trade)

        LOGGER.info("ARB SCAN: created paper trades for event %s (%s)", event_slug, arb_type)
