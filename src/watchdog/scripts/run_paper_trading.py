from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

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
from watchdog.risk.circuit_breaker import CircuitBreaker
from watchdog.risk.kelly import EmpiricalKellySizer
from watchdog.risk.vpin import TradeFlow, VPINCalculator
from watchdog.services.pipeline import _extract_orderbook_metrics, _hours_to_resolution
from watchdog.signals.calibration import CalibrationSurfaceService
from watchdog.strategies.arbitrage_detector import ArbitrageDetector
from watchdog.strategies.exit_manager import ExitManager

LOGGER = logging.getLogger(__name__)


def _raw_kelly(p_model: float, p_market: float, side: str) -> float:
    """Full Kelly fraction WITHOUT clipping to [0, 1].

    Negative value means negative expected value — trade should be rejected.
    """
    p_model = max(1e-6, min(1 - 1e-6, p_model))
    p_market = max(1e-6, min(1 - 1e-6, p_market))
    if side.upper() == "YES":
        q = p_model
        b = (1 - p_market) / p_market
    else:
        q = 1 - p_model
        p_no = 1 - p_market
        b = (1 - p_no) / p_no
    return (b * q - (1 - q)) / b


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
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    # Handle numeric timestamps (int or float) — Manifold sends ms since epoch
    if isinstance(value, (int, float)):
        ts = float(value)
        # If value > 1e10 it's in milliseconds, convert to seconds
        if ts > 1e10:
            ts /= 1000
        try:
            return datetime.fromtimestamp(ts, tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
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
            condition_id=str(market_row.get("condition_id") or "") or None,
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
    existing.condition_id = str(market_row.get("condition_id") or existing.condition_id or "") or None
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

            outcome_type = str(raw.get("outcomeType") or "BINARY").upper()
            if outcome_type != "BINARY":
                LOGGER.debug(
                    "Skipping non_binary_market: slug=%s outcomeType=%s",
                    raw.get("slug") or raw.get("id"),
                    outcome_type,
                )
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
                    # Use 24h volume if available, fall back to total volume
                    "volume": float(raw.get("volume24Hours") or raw.get("volume") or 0),
                }
            )
        return out

    # --- Polymarket via REST API (Gamma + CLOB) ---
    from watchdog.market_data.polymarket_rest import PolymarketRestClient

    rest = PolymarketRestClient()
    raw_markets = await asyncio.to_thread(rest.list_active_markets, max_markets)

    out = []
    for market in raw_markets:
        slug = market.get("slug") or ""
        if not slug:
            continue

        yes_token = market.get("yes_token_id")
        if not yes_token:
            continue

        try:
            orderbook = await asyncio.to_thread(rest.get_orderbook, yes_token)
        except Exception:
            LOGGER.debug("Failed to fetch orderbook for %s", slug)
            continue

        mid = float(orderbook.get("mid") or 0.0)
        if mid <= 0:
            continue

        out.append(
            {
                "market_id": slug,
                "slug": slug,
                "question": str(market.get("question") or slug),
                "domain": "other",
                "resolution_time": _parse_resolution_time(market.get("end_date")),
                "status": "active",
                "probability": max(0.01, min(0.99, mid)),
                "volume": float(market.get("volume_24h") or 0),
                "yes_token_id": yes_token,
                "no_token_id": market.get("no_token_id"),
                "condition_id": market.get("condition_id"),
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

    from watchdog.market_data.polymarket_rest import PolymarketRestClient
    polymarket_rest = PolymarketRestClient() if platform == "polymarket" else None

    calibration = CalibrationSurfaceService()
    vpin_calc = VPINCalculator()
    kelly_sizer = EmpiricalKellySizer(
        kelly_fraction=settings.kelly_fraction,
        max_drawdown_p95=settings.max_drawdown_p95,
    )
    circuit_breaker = CircuitBreaker(
        max_daily_loss_usd=settings.max_daily_loss_usd,
        cooldown_minutes=settings.circuit_breaker_cooldown_minutes,
        n8n_webhook_url=settings.n8n_webhook_url,
    )
    exit_manager = ExitManager(
        take_profit_pct=settings.take_profit_pct,
        take_profit_2_pct=settings.take_profit_2_pct,
        stop_loss_pct=settings.stop_loss_pct,
        stop_loss_2_pct=settings.stop_loss_2_pct,
        max_hold_hours=settings.max_hold_hours,
    )
    arb_detector = ArbitrageDetector(
        min_arb_spread=settings.min_arb_spread,
        max_arb_position_size=settings.max_arb_position_size,
    )
    bankroll = float(virtual_bankroll)
    iteration = 0

    LOGGER.warning("Paper trading mode started | platform=%s virtual_bankroll=%.2f", platform, bankroll)

    while True:
        iteration += 1
        markets_checked = 0
        signals_generated = 0
        bets_placed = 0
        exits_closed = 0

        with session_factory() as session:
            calibration.load_from_db(session)
            # --- Circuit breaker: halt loop if daily loss limit exceeded ---
            if not circuit_breaker.check(session):
                LOGGER.warning(
                    "CircuitBreaker active — skipping iteration %d. "
                    "Sleeping %d s before retry.",
                    iteration,
                    settings.paper_loop_seconds,
                )
                await asyncio.sleep(settings.paper_loop_seconds)
                continue

        market_rows = await _load_platform_markets(
            platform=platform,
            max_markets=max_markets,
            manifold=manifold,
            polymarket=polymarket,
        )

        # --- Check exits on open positions FIRST ---
        with session_factory() as session:
            current_prices: dict[int, float] = {}
            for mr in market_rows:
                db_m = session.execute(
                    select(Market).where(Market.slug == mr["slug"])
                ).scalar_one_or_none()
                if db_m is not None:
                    current_prices[db_m.id] = float(mr["probability"])

            exits_closed = exit_manager.check_exits(
                session,
                current_prices=current_prices,
                platform=platform,
                manifold=manifold,
                polymarket=polymarket,
                polymarket_rest=polymarket_rest,
            )
            try:
                session.commit()
            except Exception:
                session.rollback()
                raise

        # --- Main market scan ---
        with session_factory() as session:
            # Count positions already open (from prior iterations)
            open_position_count = session.execute(
                select(Trade).where(Trade.is_paper.is_(True), Trade.status == "open")
            ).scalars().all()
            open_count = len(open_position_count)
            deployed_capital = sum(t.size for t in open_position_count)
            free_capital = max(bankroll - deployed_capital, 0.0)

            if open_count >= settings.max_positions_simultaneous:
                LOGGER.info(
                    "Portfolio full (%d/%d positions, $%.2f deployed) — skipping new entries",
                    open_count, settings.max_positions_simultaneous, deployed_capital,
                )

            new_positions_this_iter = 0  # track within this session block

            for market_row in market_rows:
                markets_checked += 1
                db_market = _upsert_market(session, market_row)

                # --- Filter 1: Resolution time window ---
                if db_market.resolution_time:
                    hours_until = _hours_to_resolution(db_market.resolution_time)
                    if hours_until > settings.max_resolution_hours:
                        LOGGER.debug(
                            "Skipping %s: resolves in %.1f hours (max %d)",
                            market_row["slug"], hours_until, settings.max_resolution_hours,
                        )
                        continue
                    if hours_until < settings.min_resolution_hours:
                        LOGGER.debug(
                            "Skipping %s: resolves in %.1f hours (too soon, min %d)",
                            market_row["slug"], hours_until, settings.min_resolution_hours,
                        )
                        continue

                # --- Filter 2: Volume check ---
                volume_24h = float(
                    market_row.get("volume_24h")
                    or market_row.get("volume")
                    or 0,
                )
                if volume_24h < settings.min_volume_24h:
                    LOGGER.debug(
                        "Skipping %s: volume_24h=$%.0f (below $%.0f threshold)",
                        market_row["slug"], volume_24h, settings.min_volume_24h,
                    )
                    continue

                # --- Filter 3: Single-position guard (no duplicate trades) ---
                existing_open_trade = session.execute(
                    select(Trade).where(
                        Trade.market_id == db_market.id,
                        Trade.status == "open",
                        Trade.is_paper.is_(True),
                    )
                ).scalars().first()
                if existing_open_trade:
                    LOGGER.debug(
                        "Skipping %s: already have open position (trade_id=%d)",
                        market_row["slug"],
                        existing_open_trade.id,
                    )
                    continue

                # Volume multiplier: high-volume markets accept lower divergence
                volume_multiplier = 1.0
                if volume_24h > 10000:
                    volume_multiplier = 1.5

                p_market = float(market_row["probability"])

                # --- Check for arbitrage first (highest priority) ---
                if settings.enable_arbitrage:
                    arb = arb_detector.check_single_market_arb(
                        market_row["slug"],
                        yes_price=p_market,
                        no_price=1 - p_market,
                    )
                    if arb is not None:
                        stake_per_side = min(
                            bankroll * 0.05,
                            settings.max_arb_position_size,
                        )
                        session.add(Signal(
                            market_id=db_market.id,
                            model_probability=p_market,
                            market_probability=p_market,
                            divergence=arb["profit_pct"] / 100,
                            signal_type="arbitrage",
                            should_trade=True,
                            rationale=f"yes_no_sum={arb['total_cost']:.4f}",
                        ))
                        session.flush()
                        session.add(Trade(
                            market_id=db_market.id,
                            side="YES",
                            size=stake_per_side,
                            entry_price=p_market,
                            kelly_fraction=0.05,
                            order_id=f"arb-yes-{market_row['slug']}-{iteration}",
                            is_paper=True,
                            status="open",
                            opened_at=datetime.now(UTC),
                            strategy="arbitrage",
                        ))
                        session.add(Trade(
                            market_id=db_market.id,
                            side="NO",
                            size=stake_per_side,
                            entry_price=1 - p_market,
                            kelly_fraction=0.05,
                            order_id=f"arb-no-{market_row['slug']}-{iteration}",
                            is_paper=True,
                            status="open",
                            opened_at=datetime.now(UTC),
                            strategy="arbitrage",
                        ))
                        bets_placed += 2
                        signals_generated += 1
                        continue

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

                # Platform-specific divergence threshold
                base_min_div = (
                    settings.min_divergence_manifold
                    if platform == "manifold"
                    else settings.min_divergence_polymarket
                )
                effective_min_div = base_min_div / volume_multiplier
                if divergence < effective_min_div:
                    signal.rationale = f"divergence_below_threshold (min={effective_min_div:.4f})"
                    session.add(signal)
                    continue

                signals_generated += 1
                side = "YES" if result.model_probability > p_market else "NO"

                # --- Portfolio cap: stop opening new positions when full ---
                total_open = open_count + new_positions_this_iter
                if total_open >= settings.max_positions_simultaneous:
                    signal.rationale = (
                        f"portfolio_full ({total_open}/{settings.max_positions_simultaneous} positions)"
                    )
                    session.add(signal)
                    LOGGER.debug(
                        "Skipping %s: portfolio full (%d/%d open positions)",
                        market_row["slug"], total_open, settings.max_positions_simultaneous,
                    )
                    continue

                # --- Reject negative-Kelly trades ---
                raw_k = _raw_kelly(result.model_probability, p_market, side)
                if raw_k <= 0:
                    signal.rationale = f"negative_kelly ({raw_k:.4f})"
                    session.add(signal)
                    LOGGER.debug(
                        "Skipping %s: negative Kelly %.4f (no edge on this side)",
                        market_row["slug"], raw_k,
                    )
                    continue

                # --- Quarter-Kelly position sizing (sized against free capital) ---
                # free_capital tracks undeployed bankroll, shrinking as we place bets.
                if free_capital <= 1.0:
                    signal.rationale = "no_free_capital"
                    session.add(signal)
                    continue

                sizing = kelly_sizer.size(
                    p_model=result.model_probability,
                    p_market=p_market,
                    side=side,
                    historical_edge_estimates=np.array([]),
                    historical_trade_returns=np.array([]),
                )
                # When no historical data is available empirical_fraction collapses to 0
                # (cv_edge = 1.0 → uncertainty_scale = 0). Fall back to simple Quarter-Kelly.
                if sizing.empirical_fraction > 0:
                    kelly_fraction = sizing.empirical_fraction
                else:
                    kelly_fraction = sizing.full_kelly * settings.kelly_fraction
                kelly_fraction = min(kelly_fraction, settings.max_position_per_market)
                kelly_fraction = max(kelly_fraction, 0.001)
                # Size against free_capital so stakes shrink naturally as capital deploys.
                # Hard cap: never more than max_position_per_market × original bankroll.
                stake = free_capital * kelly_fraction
                stake = max(stake, 1.0)
                stake = min(stake, bankroll * settings.max_position_per_market)

                LOGGER.debug(
                    "Kelly sizing %s: full_k=%.4f empirical=%.4f stake=%.2f",
                    market_row["slug"], sizing.full_kelly, sizing.empirical_fraction, stake,
                )

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
                        strategy="calibration",
                        high_water_mark=entry_price,
                        remaining_fraction=1.0,
                    )
                )
                bets_placed += 1
                new_positions_this_iter += 1
                free_capital -= stake  # reduce available capital for the next candidate

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
            "paper_summary markets_checked=%s signals_generated=%s bets_placed=%s exits=%s running_pnl_estimate=%.4f",
            markets_checked,
            signals_generated,
            bets_placed,
            exits_closed,
            running_pnl_estimate,
        )
        print(
            "paper_summary "
            f"markets_checked={markets_checked} "
            f"signals_generated={signals_generated} "
            f"bets_placed={bets_placed} "
            f"exits={exits_closed} "
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
