from __future__ import annotations

# VERIFIED
import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import case, func, select

from watchdog.core.config import get_settings
from watchdog.core.exceptions import LiveTradingDisabledError
from watchdog.core.logging import configure_logging
from watchdog.db.init import init_db
from watchdog.db.models import MakerQuote, Market, MarketSnapshot, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.polymarket_cli import PolymarketCli
from watchdog.risk.vpin import TradeFlow, VPINCalculator
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


def run_market_maker_cycle(
    *,
    session,
    cli: PolymarketCli,
    settings,
    maker: AvellanedaStoikovPredictionMM,
    vpin_calc: VPINCalculator,
    max_markets: int,
    market_slug: str | None,
    dry_run: bool,
    quote_size: float,
) -> dict[str, int]:
    sync_markets_once(session, cli, limit=max(200, max_markets))
    markets = session.execute(
        select(Market)
        .where(Market.status.in_(["active", "open"]))
        .order_by(Market.id.asc())
        .limit(max_markets)
    ).scalars().all()
    if market_slug:
        markets = [m for m in markets if m.slug == market_slug]

    quoted = 0
    skipped = 0
    for market in markets:
        response = cli.orderbook(market.slug)
        metrics = _extract_orderbook_metrics(response.payload)
        mid = float(metrics.get("mid") or 0.0)
        if mid <= 0:
            skipped += 1
            LOGGER.info("Skipping %s: no mid-price", market.slug)
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

        near_resolution = False
        if market.resolution_time is not None:
            resolution_time = market.resolution_time
            if resolution_time.tzinfo is None:
                resolution_time = resolution_time.replace(tzinfo=UTC)
            remaining_hours = (resolution_time - datetime.now(UTC)).total_seconds() / 3600
            near_resolution = remaining_hours <= settings.near_resolution_hours

        if vpin >= settings.vpin_kill_threshold:
            skipped += 1
            LOGGER.info(
                "Skipping %s: VPIN %.4f >= threshold %.4f",
                market.slug,
                vpin,
                settings.vpin_kill_threshold,
            )
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

        if near_resolution:
            skipped += 1
            LOGGER.info("Skipping %s: within near_resolution_hours=%s", market.slug, settings.near_resolution_hours)
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

        base_quote = maker.compute_quotes(
            mid_price=mid,
            inventory=inventory_before,
            tau_hours=_hours_to_resolution(market.resolution_time),
        )
        half_spread = max(float(settings.backtest_spread_proxy), 0.01)
        bid_yes = float(max(0.01, min(0.99, mid - half_spread)))
        ask_yes = float(max(0.01, min(0.99, mid + half_spread)))
        ask_no = float(max(0.01, min(0.99, 1 - ask_yes)))

        yes_token = market.yes_token_id or market.slug
        no_token = market.no_token_id or market.slug

        if dry_run:
            bid_payload: dict[str, Any] = {"order_id": f"dry-bid-{market.id}"}
            ask_payload: dict[str, Any] = {"order_id": f"dry-ask-{market.id}"}
            LOGGER.info(
                "DRY-RUN quote market=%s bid_yes=%.4f ask_yes=%.4f half_spread=%.4f vpin=%.4f",
                market.slug,
                bid_yes,
                ask_yes,
                half_spread,
                vpin,
            )
        else:
            bid_resp = cli.create_limit_order(
                token_id=yes_token,
                side="YES",
                price=bid_yes,
                size=quote_size,
                post_only=True,
            )
            ask_resp = cli.create_limit_order(
                token_id=no_token,
                side="NO",
                price=ask_no,
                size=quote_size,
                post_only=True,
            )
            bid_payload = bid_resp.payload if isinstance(bid_resp.payload, dict) else {}
            ask_payload = ask_resp.payload if isinstance(ask_resp.payload, dict) else {}
            LOGGER.info(
                "Quote placed market=%s bid_yes=%.4f ask_yes=%.4f half_spread=%.4f",
                market.slug,
                bid_yes,
                ask_yes,
                half_spread,
            )

        session.add(
            MakerQuote(
                market_id=market.id,
                reservation_price=mid,
                optimal_spread=2 * half_spread,
                bid_price=bid_yes,
                ask_price=ask_yes,
                vpin_score=vpin,
                reward_eligible=base_quote.reward_eligible,
                bid_order_id=str(bid_payload.get("order_id") or bid_payload.get("id") or ""),
                ask_order_id=str(ask_payload.get("order_id") or ask_payload.get("id") or ""),
                inventory_before=inventory_before,
                inventory_after=inventory_before,
                canceled=False,
            )
        )
        quoted += 1

    session.commit()
    return {"quoted": quoted, "skipped": skipped}


async def main(
    interval: int = 30,
    max_markets: int = 30,
    quote_size: float = 1.0,
    market_slug: str | None = None,
    dry_run: bool = False,
    max_inventory: float | None = None,
) -> None:
    class Args:
        pass
    args = Args()
    args.interval = interval
    args.max_markets = max_markets
    args.quote_size = quote_size
    args.market_slug = market_slug
    args.dry_run = dry_run
    args.max_inventory = max_inventory

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
            counts = run_market_maker_cycle(
                session=session,
                cli=cli,
                settings=settings,
                maker=maker,
                vpin_calc=vpin_calc,
                max_markets=args.max_markets,
                market_slug=args.market_slug,
                dry_run=args.dry_run,
                quote_size=args.quote_size,
            )
            LOGGER.info("Cycle done quoted=%s skipped=%s", counts["quoted"], counts["skipped"])

        await asyncio.sleep(max(args.interval, 2))


if __name__ == "__main__":
    asyncio.run(main())
