"""Module 2: YES+NO Pair Cost Scanner (Binary markets).

For binary markets, fetches both YES and NO orderbooks and checks:
- best_ask_YES + best_ask_NO < 0.97 → buy+merge opportunity
- best_bid_YES + best_bid_NO > 1.03 → split+sell opportunity

Applies rate-limiting (100ms between fetches) and 429 retry logic.
Logs all findings to arb_opportunities with strategy="pair_cost_arb".
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from sqlalchemy.orm import Session

from watchdog.db.models import ArbOpportunity, Market

if TYPE_CHECKING:
    from watchdog.market_data.polymarket_rest import PolymarketRestClient

LOGGER = logging.getLogger(__name__)

GAS_COST = 0.002
MIN_PROFIT_CENTS = 3.0
BUY_MERGE_THRESHOLD = 0.97
SPLIT_SELL_THRESHOLD = 1.03
FETCH_SLEEP_S = 0.1
RETRY_BACKOFF_S = 30.0


class PairCostScanner:
    """Scan binary Polymarket markets for YES+NO pair cost arbitrage."""

    def scan(
        self,
        session: Session,
        rest_client: PolymarketRestClient,
        is_paper: bool = True,
    ) -> dict[str, Any]:
        """Run one scan cycle. Returns summary stats dict."""
        print("\n🔄 PAIR COST")

        candidates = self._gather_candidates(session, rest_client)
        checked = 0
        opportunities = 0

        for slug, yes_token_id, no_token_id in candidates:
            result = self._check_market(session, rest_client, slug, yes_token_id, no_token_id)
            checked += 1
            if result:
                opportunities += 1

        session.commit()
        print(f"🔄 PAIR COST done: {checked} markets checked, {opportunities} opportunities found")
        return {"markets_checked": checked, "opportunities": opportunities}

    def _gather_candidates(
        self,
        session: Session,
        rest_client: PolymarketRestClient,
    ) -> list[tuple[str, str, str]]:
        """Return list of (slug, yes_token_id, no_token_id) to check."""
        seen: set[str] = set()
        candidates: list[tuple[str, str, str]] = []

        # (a) Markets we currently have open positions in
        open_market_ids = (
            session.query(Market)
            .join(Market.trades)
            .filter(
                Market.yes_token_id.isnot(None),
                Market.no_token_id.isnot(None),
            )
            .all()
        )
        for m in open_market_ids:
            if m.slug not in seen and m.yes_token_id and m.no_token_id:
                seen.add(m.slug)
                candidates.append((m.slug, m.yes_token_id, m.no_token_id))

        # (b) Top 50 liquid active markets from REST API
        try:
            active = rest_client.list_active_markets(limit=50)
        except Exception as exc:
            LOGGER.warning("PAIR COST: failed to fetch active markets — %s", exc)
            active = []

        for m in active:
            slug = m.get("slug", "")
            yes_token = m.get("yes_token_id")
            no_token = m.get("no_token_id")
            if slug and yes_token and no_token and slug not in seen:
                seen.add(slug)
                candidates.append((slug, yes_token, no_token))

        return candidates

    def _fetch_book_with_retry(
        self,
        rest_client: PolymarketRestClient,
        token_id: str,
    ) -> dict[str, Any] | None:
        """Fetch orderbook with 429 retry logic. Returns None on unrecoverable failure."""
        time.sleep(FETCH_SLEEP_S)
        try:
            return rest_client.get_orderbook(token_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                LOGGER.warning("PAIR COST: 429 rate limit — backing off %ss", RETRY_BACKOFF_S)
                time.sleep(RETRY_BACKOFF_S)
                try:
                    return rest_client.get_orderbook(token_id)
                except Exception as retry_exc:
                    LOGGER.warning("PAIR COST: retry failed for %s — %s", token_id, retry_exc)
                    return None
            LOGGER.warning("PAIR COST: HTTP error for %s — %s", token_id, exc)
            return None
        except Exception as exc:
            LOGGER.warning("PAIR COST: failed to fetch book for %s — %s", token_id, exc)
            return None

    def _check_market(
        self,
        session: Session,
        rest_client: PolymarketRestClient,
        slug: str,
        yes_token_id: str,
        no_token_id: str,
    ) -> dict[str, Any] | None:
        yes_book = self._fetch_book_with_retry(rest_client, yes_token_id)
        if yes_book is None:
            return None

        no_book = self._fetch_book_with_retry(rest_client, no_token_id)
        if no_book is None:
            return None

        yes_ask = yes_book["ask"]
        no_ask = no_book["ask"]
        yes_bid = yes_book["bid"]
        no_bid = no_book["bid"]
        yes_spread = yes_book["spread"]
        no_spread = no_book["spread"]

        cost_sum = yes_ask + no_ask
        revenue_sum = yes_bid + no_bid

        opp_type: str | None = None
        profit_cents: float = 0.0

        # Buy+merge check
        net_buy_merge = (1.0 - cost_sum) - yes_spread - no_spread - GAS_COST
        if net_buy_merge > 0 and (1.0 - cost_sum) * 100 >= MIN_PROFIT_CENTS:
            opp_type = "buy_merge"
            profit_cents = (1.0 - cost_sum) * 100

        # Split+sell check (takes priority if both somehow true)
        net_split_sell = (revenue_sum - 1.0) - yes_spread - no_spread - GAS_COST
        if net_split_sell > 0 and (revenue_sum - 1.0) * 100 >= MIN_PROFIT_CENTS:
            opp_type = "split_sell"
            profit_cents = (revenue_sum - 1.0) * 100

        if opp_type is None:
            return None

        LOGGER.warning(
            "PAIR COST ARB [%s] market=%s cost_sum=%.4f rev_sum=%.4f profit=%.1f¢",
            opp_type,
            slug,
            cost_sum,
            revenue_sum,
            profit_cents,
        )

        opp = ArbOpportunity(
            event_slug=slug,
            type=opp_type,
            sum_of_yes=cost_sum if opp_type == "buy_merge" else revenue_sum,
            profit_cents=profit_cents,
            timestamp=datetime.now(UTC),
            was_actioned=False,
            strategy="pair_cost_arb",
        )
        session.add(opp)
        return {"type": opp_type, "profit_cents": profit_cents}
