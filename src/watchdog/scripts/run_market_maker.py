from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC

from sqlalchemy import case, func, select

from watchdog.core.config import get_settings
from watchdog.core.exceptions import LiveTradingDisabledError
from watchdog.core.logging import configure_logging
from watchdog.db.init import init_db
from watchdog.db.models import MakerQuote, Market, MarketSnapshot, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.polymarket_cli import PolymarketCli
from watchdog.risk.vpin import TradeFlow, VPINCalculator, should_halt_maker
from watchdog.services.market_sync import sync_markets_once
from watchdog.services.pipeline import _extract_orderbook_metrics, _hours_to_resolution
from watchdog.trading.maker_model import AvellanedaStoikovPredictionMM

LOGGER = logging.getLogger(__name__)


def _open_inventory(session, market_id: int) -> float:
    expr = func.coalesce(
        func.sum(
            case(
                (Trade.side == "YES", Trade.size),
                (Trade.side == "NO", -Trade.size),
                else_=0.0,
            )
        ),
        0.0,
    )
    value = session.execute(
        select(expr).where(Trade.market_id == market_id, Trade.closed_at.is_(None))
    ).scalar_one()
    return float(value)


def _recent_vpin(session, market_id: int, calculator: VPINCalculator) -> float:
    snapshots = session.execute(
        select(MarketSnapshot)
        .where(MarketSnapshot.market_id == market_id)
        .order_by(MarketSnapshot.captured_at.desc())
        .limit(50)
    ).scalars().all()

    flows: list[TradeFlow] = []
    for row in reversed(snapshots):
        flows.append(TradeFlow(side="SELL", volume=float(row.bid_volume or 0.0)))
        flows.append(TradeFlow(side="BUY", volume=float(row.ask_volume or 0.0)))
    return calculator.compute(flows)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Watchdog maker loop")
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--max-markets", type=int, default=30)
    parser.add_argument("--quote-size", type=float, default=1.0)
    parser.add_argument("--market-slug", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-inventory", type=float, default=None)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)

    if not settings.enable_live_trading and not args.dry_run:
        raise LiveTradingDisabledError("run_market_maker requires ENABLE_LIVE_TRADING=true")

    LOGGER.warning(
        "MARKET MAKING MODE: This script posts real limit orders if LIVE_TRADING=True. "
        "Ensure bankroll, VPIN threshold, and inventory limits are configured before proceeding. "
        "Recommended for Week 4+ only after backtest and paper trading validation."
    )
    if args.dry_run:
        LOGGER.warning("Dry-run enabled: quotes will be logged but not posted to the exchange.")

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    cli = PolymarketCli(settings)
    if not args.dry_run:
        cli.startup_check()

    vpin_calc = VPINCalculator()
    maker = AvellanedaStoikovPredictionMM(
        inventory_limit=max(args.max_inventory or settings.max_position_per_market, 1e-6),
    )

    while True:
        with session_factory() as session:
            sync_markets_once(session, cli, limit=max(200, args.max_markets))
            markets = session.execute(
                select(Market)
                .where(Market.status.in_(["active", "open"]))
                .order_by(Market.id.asc())
                .limit(args.max_markets)
            ).scalars().all()
            if args.market_slug:
                markets = [m for m in markets if m.slug == args.market_slug]

            for market in markets:
                response = cli.orderbook(market.slug)
                metrics = _extract_orderbook_metrics(response.payload)
                mid = float(metrics.get("mid") or 0.0)
                if mid <= 0:
                    continue

                session.add(
                    MarketSnapshot(
                        market_id=market.id,
                        bid=metrics.get("bid"),
                        ask=metrics.get("ask"),
                        mid=metrics.get("mid"),
                        spread=metrics.get("spread"),
                        bid_volume=metrics.get("bid_volume"),
                        ask_volume=metrics.get("ask_volume"),
                        total_volume=metrics.get("total_volume"),
                        cli_latency_ms=response.latency_ms,
                        raw_json=str(response.payload),
                    )
                )
                session.flush()

                vpin = _recent_vpin(session, market.id, vpin_calc)
                inventory_before = _open_inventory(session, market.id)

                halt = should_halt_maker(
                    vpin=vpin,
                    vpin_kill_threshold=settings.vpin_kill_threshold,
                    resolution_time=market.resolution_time,
                    near_resolution_hours=settings.near_resolution_hours,
                    near_resolution_fraction=settings.near_resolution_fraction,
                    market_opened_at=market.created_at.replace(tzinfo=UTC)
                    if market.created_at and market.created_at.tzinfo is None
                    else market.created_at,
                )

                if halt:
                    cancel_payload = {"dry_run": True} if args.dry_run else cli.cancel_all().payload
                    LOGGER.info("Maker halted for %s, cancel-all payload=%s", market.slug, cancel_payload)
                    session.add(
                        MakerQuote(
                            market_id=market.id,
                            reservation_price=mid,
                            optimal_spread=0.0,
                            bid_price=mid,
                            ask_price=mid,
                            vpin_score=vpin,
                            reward_eligible=False,
                            inventory_before=inventory_before,
                            inventory_after=inventory_before,
                            canceled=True,
                        )
                    )
                    continue

                quote = maker.compute_quotes(
                    mid_price=mid,
                    inventory=inventory_before,
                    tau_hours=_hours_to_resolution(market.resolution_time),
                )

                ask_no_price = float(max(0.01, min(0.99, 1 - quote.ask_price)))
                if args.dry_run:
                    LOGGER.info(
                        "DRY-RUN quote market=%s bid_yes=%.4f ask_yes=%.4f vpin=%.4f inv=%.4f",
                        market.slug,
                        quote.bid_price,
                        quote.ask_price,
                        vpin,
                        inventory_before,
                    )
                    bid_payload: dict[str, str] = {"order_id": f"dry-bid-{market.id}"}
                    ask_payload: dict[str, str] = {"order_id": f"dry-ask-{market.id}"}
                else:
                    bid_resp = cli.place_limit_order(
                        market_slug=market.slug,
                        side="YES",
                        price=quote.bid_price,
                        size=args.quote_size,
                    )
                    ask_resp = cli.place_limit_order(
                        market_slug=market.slug,
                        side="NO",
                        price=ask_no_price,
                        size=args.quote_size,
                    )
                    bid_payload = bid_resp.payload if isinstance(bid_resp.payload, dict) else {}
                    ask_payload = ask_resp.payload if isinstance(ask_resp.payload, dict) else {}

                session.add(
                    MakerQuote(
                        market_id=market.id,
                        reservation_price=quote.reservation_price,
                        optimal_spread=quote.optimal_spread,
                        bid_price=quote.bid_price,
                        ask_price=quote.ask_price,
                        vpin_score=vpin,
                        reward_eligible=quote.reward_eligible,
                        bid_order_id=str(bid_payload.get("order_id") or bid_payload.get("id") or ""),
                        ask_order_id=str(ask_payload.get("order_id") or ask_payload.get("id") or ""),
                        inventory_before=inventory_before,
                        inventory_after=inventory_before,
                        canceled=False,
                    )
                )

            session.commit()

        await asyncio.sleep(max(args.interval, 2))


if __name__ == "__main__":
    asyncio.run(main())
