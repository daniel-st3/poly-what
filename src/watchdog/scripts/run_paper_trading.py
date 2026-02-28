from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from dateutil import parser as dtparser
from sqlalchemy import select

from watchdog.core.config import get_settings
from watchdog.core.exceptions import PolymarketCliError
from watchdog.core.logging import configure_logging
from watchdog.db.init import init_db
from watchdog.db.models import Market, Signal, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.manifold_client import ManifoldAPIError, ManifoldClient
from watchdog.market_data.polymarket_cli import PolymarketCli
from watchdog.risk.vpin import TradeFlow, VPINCalculator
from watchdog.services.pipeline import _extract_orderbook_metrics, _hours_to_resolution
from watchdog.signals.calibration import CalibrationSurfaceService

LOGGER = logging.getLogger(__name__)


def _extract_markets_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("markets", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _parse_resolution_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = dtparser.parse(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _upsert_market(session, market_row: dict[str, Any]) -> Market:
    slug = str(market_row.get("slug") or "").strip()
    if not slug:
        raise ValueError("Market slug is required")

    existing = session.execute(select(Market).where(Market.slug == slug)).scalar_one_or_none()
    if existing is None:
        existing = Market(
            slug=slug,
            question=str(market_row.get("question") or slug),
            domain=str(market_row.get("domain") or "other"),
            yes_token_id=str(market_row.get("yes_token_id") or "") or None,
            no_token_id=str(market_row.get("no_token_id") or "") or None,
            resolution_time=_parse_resolution_time(market_row.get("resolution_time")),
            status=str(market_row.get("status") or "active"),
        )
        session.add(existing)
        session.flush()
        return existing

    existing.question = str(market_row.get("question") or existing.question)
    existing.domain = str(market_row.get("domain") or existing.domain)
    existing.yes_token_id = str(market_row.get("yes_token_id") or existing.yes_token_id or "") or None
    existing.no_token_id = str(market_row.get("no_token_id") or existing.no_token_id or "") or None
    existing.resolution_time = _parse_resolution_time(market_row.get("resolution_time")) or existing.resolution_time
    existing.status = str(market_row.get("status") or existing.status)
    return existing


async def _load_platform_markets(
    *,
    platform: str,
    max_markets: int,
    manifold: ManifoldClient | None,
    polymarket: PolymarketCli | None,
) -> list[dict[str, Any]]:
    if platform == "manifold":
        assert manifold is not None
        rows = await asyncio.to_thread(manifold.get_markets, max_markets)
        out: list[dict[str, Any]] = []
        for raw in rows:
            resolved = bool(raw.get("isResolved") or raw.get("resolved") or raw.get("resolution") is not None)
            if resolved:
                continue

            market_id = str(raw.get("id") or raw.get("marketId") or "").strip()
            slug = str(raw.get("slug") or market_id).strip()
            if not market_id or not slug:
                continue

            probability = float(raw.get("probability") or 0.5)
            probability = max(0.01, min(0.99, probability))

            out.append(
                {
                    "market_id": market_id,
                    "slug": slug,
                    "question": str(raw.get("question") or raw.get("text") or slug),
                    "domain": manifold._infer_domain(raw),
                    "resolution_time": _parse_resolution_time(raw.get("closeTime") or raw.get("close_time")),
                    "status": "active",
                    "probability": probability,
                    "yes_token_id": market_id,
                    "no_token_id": market_id,
                }
            )
        return out

    assert polymarket is not None
    response = await asyncio.to_thread(polymarket.list_markets, max_markets)
    rows = _extract_markets_payload(response.payload)
    out = []
    for row in rows:
        slug = str(row.get("slug") or "").strip()
        if not slug:
            continue

        status = str(row.get("status") or "active").lower()
        if status not in {"active", "open"}:
            continue

        orderbook = await asyncio.to_thread(polymarket.orderbook, slug)
        metrics = _extract_orderbook_metrics(orderbook.payload)
        mid = float(metrics.get("mid") or 0.0)
        if mid <= 0:
            continue

        out.append(
            {
                "market_id": slug,
                "slug": slug,
                "question": str(row.get("question") or row.get("title") or slug),
                "domain": str(row.get("domain") or row.get("category") or "other"),
                "resolution_time": _parse_resolution_time(row.get("endDate") or row.get("resolution_time")),
                "status": status,
                "probability": max(0.01, min(0.99, mid)),
                "yes_token_id": str(row.get("yesTokenId") or row.get("yes_token_id") or slug),
                "no_token_id": str(row.get("noTokenId") or row.get("no_token_id") or slug),
            }
        )

    return out


async def _running_pnl_estimate(
    *,
    platform: str,
    session_factory,
    manifold: ManifoldClient | None,
    polymarket: PolymarketCli | None,
) -> float:
    with session_factory() as session:
        open_trades = session.execute(
            select(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .where(Trade.is_paper.is_(True), Trade.status == "open")
        ).all()

    pnl_estimate = 0.0
    for trade, market in open_trades:
        mark_probability = None
        if platform == "manifold" and manifold is not None and market.yes_token_id:
            try:
                market_row = await asyncio.to_thread(manifold.get_market, market.yes_token_id)
                mark_probability = float(market_row.get("probability") or 0.5)
            except Exception:
                continue
        elif platform == "polymarket" and polymarket is not None:
            try:
                response = await asyncio.to_thread(polymarket.orderbook, market.slug)
                metrics = _extract_orderbook_metrics(response.payload)
                mark_probability = float(metrics.get("mid") or 0.5)
            except Exception:
                continue

        if mark_probability is None:
            continue

        mark_probability = max(0.01, min(0.99, mark_probability))
        mark_price = mark_probability if trade.side == "YES" else (1 - mark_probability)
        contracts = trade.size / max(trade.entry_price, 1e-9)
        pnl_estimate += contracts * (mark_price - trade.entry_price)

    return float(pnl_estimate)


async def run_paper_trading_loop(
    virtual_bankroll: float,
    platform: str = "manifold",
    max_markets: int = 80,
    iterations: int = 0,
) -> None:
    settings = get_settings()
    configure_logging(settings)

    if platform not in {"manifold", "polymarket"}:
        raise ValueError("platform must be one of: manifold, polymarket")

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    manifold = (
        ManifoldClient(base_url=settings.manifold_api_base_url, api_key=settings.manifold_api_key)
        if platform == "manifold"
        else None
    )
    polymarket = PolymarketCli(settings) if platform == "polymarket" else None
    if polymarket is not None:
        polymarket.startup_check()

    calibration = CalibrationSurfaceService()
    vpin_calc = VPINCalculator()
    bankroll = float(virtual_bankroll)
    iteration = 0

    LOGGER.warning("Paper trading mode started | platform=%s virtual_bankroll=%.2f", platform, bankroll)

    while True:
        iteration += 1
        markets_checked = 0
        signals_generated = 0
        bets_placed = 0

        with session_factory() as session:
            calibration.load_from_db(session)

        market_rows = await _load_platform_markets(
            platform=platform,
            max_markets=max_markets,
            manifold=manifold,
            polymarket=polymarket,
        )

        with session_factory() as session:
            for market_row in market_rows:
                markets_checked += 1
                db_market = _upsert_market(session, market_row)

                p_market = float(market_row["probability"])
                vpin_score: float | None = None
                if platform == "manifold" and manifold is not None:
                    try:
                        synthetic_book = await asyncio.to_thread(manifold.get_orderbook, market_row["market_id"])
                    except (ManifoldAPIError, ValueError) as exc:
                        LOGGER.warning(
                            "Skipping %s: failed synthetic orderbook fetch (%s)",
                            market_row["slug"],
                            exc,
                        )
                        continue

                    p_market = float(synthetic_book.get("mid") or p_market)
                    p_market = max(0.01, min(0.99, p_market))
                    bid_volume = float(synthetic_book.get("bid_volume") or 0.0)
                    ask_volume = float(synthetic_book.get("ask_volume") or 0.0)
                    vpin_score = vpin_calc.compute(
                        [
                            TradeFlow(side="SELL", volume=bid_volume),
                            TradeFlow(side="BUY", volume=ask_volume),
                        ]
                    )

                result = calibration.calibrate(
                    market_probability=p_market,
                    hours_to_resolution=_hours_to_resolution(db_market.resolution_time),
                    domain=db_market.domain,
                    sentiment_score=0.0,
                )
                divergence = abs(result.model_probability - p_market)

                signal = Signal(
                    market_id=db_market.id,
                    model_probability=result.model_probability,
                    market_probability=p_market,
                    divergence=divergence,
                    signal_type=f"paper_{platform}",
                    calibration_adjustments=result.adjustment,
                    vpin_score=vpin_score,
                    should_trade=False,
                    rationale="divergence_below_threshold",
                )

                if vpin_score is not None and vpin_score >= settings.vpin_kill_threshold:
                    signal.rationale = "vpin_kill_switch"
                    session.add(signal)
                    continue

                if divergence < settings.min_divergence_paper:
                    session.add(signal)
                    continue

                signals_generated += 1
                side = "YES" if result.model_probability > p_market else "NO"
                stake = min(bankroll * settings.max_position_per_market, bankroll * 0.10)
                stake = max(stake, 1.0)
                kelly_fraction = min(stake / max(bankroll, 1e-9), settings.max_position_per_market)
                entry_price = p_market if side == "YES" else (1 - p_market)

                signal.should_trade = True
                signal.rationale = "divergence_triggered"
                session.add(signal)
                session.flush()

                order_id = f"paper-{platform}-{market_row['market_id']}-{iteration}"
                if platform == "manifold" and manifold is not None:
                    try:
                        bet_response = await asyncio.to_thread(
                            manifold.place_bet,
                            market_row["market_id"],
                            side,
                            stake,
                        )
                        order_id = str(
                            bet_response.get("id")
                            or bet_response.get("betId")
                            or bet_response.get("orderId")
                            or order_id
                        )
                    except (ManifoldAPIError, ValueError) as exc:
                        LOGGER.warning(
                            "Manifold place_bet failed for %s (%s). Logging simulated paper bet.",
                            market_row["slug"],
                            exc,
                        )

                session.add(
                    Trade(
                        market_id=db_market.id,
                        signal_id=signal.id,
                        side=side,
                        size=stake,
                        entry_price=entry_price,
                        kelly_fraction=kelly_fraction,
                        confidence_score=min(1.0, divergence),
                        order_id=order_id,
                        is_paper=True,
                        status="open",
                        opened_at=datetime.now(UTC),
                    )
                )
                bets_placed += 1

            try:
                session.commit()
            except Exception:
                session.rollback()
                raise

        running_pnl_estimate = await _running_pnl_estimate(
            platform=platform,
            session_factory=session_factory,
            manifold=manifold,
            polymarket=polymarket,
        )

        LOGGER.info(
            "paper_summary markets_checked=%s signals_generated=%s bets_placed=%s running_pnl_estimate=%.4f",
            markets_checked,
            signals_generated,
            bets_placed,
            running_pnl_estimate,
        )
        print(
            "paper_summary "
            f"markets_checked={markets_checked} "
            f"signals_generated={signals_generated} "
            f"bets_placed={bets_placed} "
            f"running_pnl_estimate={running_pnl_estimate:.4f}"
        )

        if iterations > 0 and iteration >= iterations:
            break

        await asyncio.sleep(settings.paper_loop_seconds)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Watchdog paper-trading loop")
    parser.add_argument("--virtual-bankroll", type=float, default=500.0)
    parser.add_argument("--platform", choices=["manifold", "polymarket"], default="manifold")
    parser.add_argument("--max-markets", type=int, default=80)
    parser.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    args = parser.parse_args()

    await run_paper_trading_loop(
        virtual_bankroll=args.virtual_bankroll,
        platform=args.platform,
        max_markets=args.max_markets,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except PolymarketCliError as exc:
        print(f"Paper trading halted: {exc}")
        raise SystemExit(1) from None
