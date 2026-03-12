"""Exit-only checker — runs every 15 minutes via launchd.

Fetches live prices for all open paper positions and fires ExitManager.check_exits().
Does NOT scan for new opportunities and does NOT place new trades.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime

from sqlalchemy import select

sys.path.insert(0, "src")

from watchdog.core.config import get_settings
from watchdog.core.logging import configure_logging
from watchdog.db.init import init_db
from watchdog.db.models import Market, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.manifold_client import ManifoldClient
from watchdog.strategies.exit_manager import ExitManager

LOGGER = logging.getLogger(__name__)


def _infer_platform(trade: Trade, market: Market) -> str:
    """Infer the source venue for an open position.

    Paper trades created by this repo encode the venue in ``order_id``.
    Fall back to market metadata for older rows and arbitrage records.
    """
    order_id = (trade.order_id or "").lower()
    if order_id.startswith("paper-polymarket-"):
        return "polymarket"
    if order_id.startswith("paper-manifold-"):
        return "manifold"

    if market.condition_id:
        return "polymarket"
    return "manifold"


async def _fetch_prices_manifold(
    open_markets: list[Market],
    manifold: ManifoldClient,
) -> dict[int, float]:
    """Fetch current YES probability for each market from Manifold API.

    BINARY markets expose ``probability`` directly.  MULTIPLE_CHOICE and other
    non-binary market types do not — for those we fall back to 0.5 (neutral,
    no P&L movement) so that time-based exits can still fire.
    """
    prices: dict[int, float] = {}
    for market in open_markets:
        token = market.yes_token_id or market.slug
        if not token:
            LOGGER.warning("No token/slug for market id=%d — skipping price fetch", market.id)
            continue
        try:
            data = await asyncio.to_thread(manifold.get_market, token)
            prob = data.get("probability")
            if prob is not None:
                prices[market.id] = max(0.01, min(0.99, float(prob)))
            else:
                # Non-binary market (MULTIPLE_CHOICE, FREE_RESPONSE, etc.)
                # — no single YES probability exists.  Use 0.5 so exit logic
                # can still evaluate time-based exits without false TP/SL.
                outcome_type = data.get("outcomeType", "unknown")
                LOGGER.warning(
                    "Market %s is %s (no scalar probability) — using 0.5 neutral price",
                    market.slug,
                    outcome_type,
                )
                prices[market.id] = 0.5
        except Exception as exc:
            LOGGER.warning("Price fetch failed for %s: %s", market.slug, exc)
    return prices


async def _fetch_prices_polymarket(
    open_markets: list[Market],
) -> dict[int, float]:
    """Fetch current mid price for each market from Polymarket REST API."""
    from watchdog.market_data.polymarket_rest import PolymarketRestClient

    rest = PolymarketRestClient()
    prices: dict[int, float] = {}
    for market in open_markets:
        token = market.yes_token_id
        if not token:
            continue
        try:
            book = await asyncio.to_thread(rest.get_orderbook, token)
            mid = book.get("mid") or book.get("probability")
            if mid is not None:
                prices[market.id] = max(0.01, min(0.99, float(mid)))
        except Exception as exc:
            LOGGER.debug("Price fetch failed for %s: %s", market.slug, exc)
    return prices


async def run_exit_check(platform: str) -> None:
    settings = get_settings()
    configure_logging(settings)

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    manifold = (
        ManifoldClient(
            base_url=settings.manifold_api_base_url,
            api_key=settings.manifold_api_key,
        )
        if platform == "manifold"
        else None
    )
    polymarket_rest = None
    if platform == "polymarket":
        from watchdog.market_data.polymarket_rest import PolymarketRestClient
        polymarket_rest = PolymarketRestClient()

    exit_manager = ExitManager(
        take_profit_pct=settings.take_profit_pct,
        take_profit_2_pct=settings.take_profit_2_pct,
        stop_loss_pct=settings.stop_loss_pct,
        stop_loss_2_pct=settings.stop_loss_2_pct,
        max_hold_hours=settings.max_hold_hours,
    )

    LOGGER.warning(
        "=== Exit check started | platform=%s | %s ===",
        platform,
        datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    # --- Fetch open positions and their markets ---
    with session_factory() as session:
        rows = session.execute(
            select(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .where(Trade.is_paper.is_(True), Trade.status == "open")
        ).all()

        all_open_markets = list({m.id: m for _, m in rows}.values())
        LOGGER.warning("Open positions: %d across %d markets", len(rows), len(all_open_markets))

        if not all_open_markets:
            LOGGER.warning("No open positions — nothing to check.")
            return

        # --- Filter markets to those belonging to this platform ---
        platform_rows = [(trade, market) for trade, market in rows if _infer_platform(trade, market) == platform]
        open_markets = list({m.id: m for _, m in platform_rows}.values())
        skipped_positions = len(rows) - len(platform_rows)

        if skipped_positions:
            other_platform = "Polymarket" if platform == "manifold" else "Manifold"
            LOGGER.warning(
                "Skipping %d %s position(s) during %s exit check",
                skipped_positions,
                other_platform,
                platform.capitalize(),
            )

        if not open_markets:
            LOGGER.warning("No %s positions to check — exiting early.", platform.capitalize())
            LOGGER.warning(
                "=== Exit check complete | exits_fired=0 | platform=%s ===", platform
            )
            return

        # --- Fetch live prices ---
        if platform == "manifold" and manifold is not None:
            current_prices = await _fetch_prices_manifold(open_markets, manifold)
        else:
            current_prices = await _fetch_prices_polymarket(open_markets)

        priced = len(current_prices)
        LOGGER.warning(
            "Live prices fetched: %d/%d markets",
            priced,
            len(open_markets),
        )

        # --- Run exit checks ---
        exits = exit_manager.check_exits(
            session,
            current_prices=current_prices,
            platform=platform,
            manifold=manifold,
            polymarket_rest=polymarket_rest,
        )

        try:
            session.commit()
        except Exception:
            session.rollback()
            raise

        LOGGER.warning(
            "=== Exit check complete | exits_fired=%d | platform=%s ===",
            exits,
            platform,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check exits on open paper positions")
    parser.add_argument(
        "--platform",
        choices=["manifold", "polymarket"],
        default="manifold",
    )
    args = parser.parse_args()
    asyncio.run(run_exit_check(args.platform))


if __name__ == "__main__":
    main()
