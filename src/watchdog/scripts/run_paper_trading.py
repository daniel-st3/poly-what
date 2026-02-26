from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import select

from watchdog.backtest.metrics import compute_sharpe_ratio
from watchdog.core.config import get_settings
from watchdog.core.logging import configure_logging
from watchdog.db.init import init_db
from watchdog.db.models import Market, Signal, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.llm.executor import build_executor
from watchdog.llm.router import build_router
from watchdog.market_data.manifold_client import ManifoldClient, ManifoldMarket
from watchdog.news.models import NewsItem
from watchdog.risk.kelly import EmpiricalKellySizer
from watchdog.signals.calibration import CalibrationSurfaceService

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PaperPosition:
    trade_id: int
    manifold_market_id: str
    slug: str
    side: str
    entry_price: float
    stake: float


def _hours_to_resolution(close_time: datetime | None) -> float:
    if close_time is None:
        return 24.0
    now = datetime.now(UTC)
    return max((close_time - now).total_seconds() / 3600.0, 0.1)


def _market_domain(market: ManifoldMarket) -> str:
    return str(market.raw.get("groupSlug") or "other").lower()


def _load_historical_arrays(session, market_db_id: int) -> tuple[np.ndarray, np.ndarray]:
    signal_rows = session.execute(
        select(Signal.divergence).where(Signal.market_id == market_db_id, Signal.should_trade.is_(True)).limit(500)
    ).all()
    trade_rows = session.execute(
        select(Trade.pnl, Trade.entry_price)
        .where(Trade.market_id == market_db_id, Trade.pnl.is_not(None), Trade.entry_price > 0)
        .limit(500)
    ).all()
    edges = np.array([abs(float(row[0])) for row in signal_rows], dtype=np.float64)
    returns = np.array([float(pnl) / float(entry) for pnl, entry in trade_rows], dtype=np.float64)
    return edges, returns


def _upsert_market(session, market: ManifoldMarket, domain: str) -> Market:
    existing = session.execute(select(Market).where(Market.slug == market.slug)).scalar_one_or_none()
    if existing is not None:
        existing.question = market.question
        existing.domain = domain
        existing.resolution_time = market.close_time
        existing.status = "resolved" if market.is_resolved else "active"
        existing.resolution_outcome = str(market.outcome) if market.outcome is not None else None
        return existing

    row = Market(
        slug=market.slug,
        question=market.question,
        domain=domain,
        resolution_time=market.close_time,
        status="resolved" if market.is_resolved else "active",
        resolution_outcome=str(market.outcome) if market.outcome is not None else None,
    )
    session.add(row)
    session.flush()
    return row


async def run_paper_trading_loop(
    virtual_bankroll: float,
    platform: str = "manifold",
    max_markets: int = 80,
    iterations: int = 0,
) -> None:
    settings = get_settings()
    configure_logging(settings)

    if platform != "manifold":
        raise ValueError("Only platform='manifold' is currently supported in run_paper_trading")

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    manifold = ManifoldClient(
        settings.manifold_api_base_url,
        settings.manifold_api_key,
        session_factory=session_factory,
        enable_live_trading=settings.enable_live_trading,
    )
    router = build_router(settings)
    executor = build_executor(settings)
    calibration = CalibrationSurfaceService()
    sizer = EmpiricalKellySizer(
        kelly_fraction=settings.kelly_fraction,
        max_drawdown_p95=settings.max_drawdown_p95,
    )

    bankroll = float(virtual_bankroll)
    closed_returns: list[float] = []
    closed_pnls: list[float] = []
    open_positions: dict[str, PaperPosition] = {}
    iteration = 0

    with session_factory() as session:
        calibration.load_from_db(session)

    LOGGER.warning("Paper trading mode active on %s (virtual bankroll %.2f)", platform, bankroll)

    while True:
        iteration += 1

        active_markets = await manifold.get_active_markets(limit=max_markets)
        with session_factory() as session:
            calibration.load_from_db(session)

            for market in active_markets:
                if market.slug in open_positions:
                    continue

                domain = _market_domain(market)
                db_market = _upsert_market(session, market, domain)

                p_market = float(np.clip(market.probability, 0.01, 0.99))
                calib = calibration.calibrate(
                    market_probability=p_market,
                    hours_to_resolution=_hours_to_resolution(market.close_time),
                    domain=domain,
                    sentiment_score=0.0,
                )
                divergence = abs(calib.model_probability - p_market)

                signal = Signal(
                    market_id=db_market.id,
                    model_probability=calib.model_probability,
                    market_probability=p_market,
                    divergence=divergence,
                    signal_type="paper_manifold",
                    calibration_adjustments=calib.adjustment,
                    should_trade=False,
                    rationale="",
                )

                if divergence < settings.min_divergence_paper:
                    signal.rationale = "divergence_below_paper_threshold"
                    session.add(signal)
                    continue

                news_item = NewsItem(
                    headline=market.question,
                    source="paper:manifold",
                    url=f"https://manifold.markets/{market.slug}",
                    raw_text=market.question,
                    domain_tags=domain,
                    received_at=datetime.now(UTC),
                )

                route = await router.route(news_item, [market.slug])
                if not route.relevant:
                    signal.rationale = "router_irrelevant"
                    signal.router_confidence = route.confidence
                    session.add(signal)
                    continue

                decision = await executor.decide(
                    {
                        "market_slug": market.slug,
                        "market_question": market.question,
                        "market_probability": p_market,
                        "model_probability": calib.model_probability,
                        "divergence": divergence,
                        "min_divergence": settings.min_divergence_paper,
                    }
                )
                signal.router_confidence = route.confidence
                signal.executor_confidence = decision.confidence
                signal.rationale = decision.rationale

                if not decision.trade or decision.side not in {"YES", "NO"}:
                    session.add(signal)
                    continue

                hist_edges, hist_returns = _load_historical_arrays(session, db_market.id)
                sizing = sizer.size(
                    p_model=calib.model_probability,
                    p_market=p_market,
                    side=decision.side,
                    historical_edge_estimates=hist_edges,
                    historical_trade_returns=hist_returns,
                )

                size_fraction = min(sizing.empirical_fraction, settings.max_position_per_market)
                stake = bankroll * size_fraction
                if stake <= 0:
                    signal.rationale = f"{signal.rationale} | non_positive_size"
                    session.add(signal)
                    continue

                entry_price = p_market if decision.side == "YES" else (1 - p_market)
                signal.should_trade = True
                session.add(signal)
                session.flush()

                trade = Trade(
                    market_id=db_market.id,
                    signal_id=signal.id,
                    side=decision.side,
                    size=stake,
                    entry_price=entry_price,
                    kelly_fraction=size_fraction,
                    confidence_score=decision.confidence,
                    order_id=f"paper-manifold-{market.market_id}-{iteration}",
                    is_paper=True,
                    opened_at=datetime.now(UTC),
                    status="open",
                )
                session.add(trade)
                session.flush()

                open_positions[market.slug] = PaperPosition(
                    trade_id=trade.id,
                    manifold_market_id=market.market_id,
                    slug=market.slug,
                    side=decision.side,
                    entry_price=entry_price,
                    stake=stake,
                )

            session.commit()

        for slug, position in list(open_positions.items()):
            latest = await manifold.get_market(position.manifold_market_id)
            if not latest.is_resolved or latest.outcome is None:
                continue

            contracts = position.stake / max(position.entry_price, 1e-8)
            payoff = float(latest.outcome) if position.side == "YES" else float(1 - latest.outcome)
            pnl = contracts * (payoff - position.entry_price)

            bankroll_before = bankroll
            bankroll += pnl
            if bankroll_before > 0:
                closed_returns.append(pnl / bankroll_before)
            closed_pnls.append(pnl)

            with session_factory() as session:
                trade_row = session.get(Trade, position.trade_id)
                if trade_row is not None:
                    trade_row.exit_price = 1.0 if payoff > 0 else 0.0
                    trade_row.pnl = pnl
                    trade_row.closed_at = datetime.now(UTC)
                    trade_row.status = "closed"
                session.commit()

            del open_positions[slug]

        if iteration % settings.paper_summary_every == 0:
            win_rate = float(np.mean(np.asarray(closed_pnls) > 0)) if closed_pnls else 0.0
            sharpe = compute_sharpe_ratio(closed_returns, periods_per_year=365, risk_free=0.0)
            LOGGER.info(
                "paper_summary iter=%s bankroll=%.2f closed=%s win_rate=%.3f sharpe=%.3f open_positions=%s",
                iteration,
                bankroll,
                len(closed_pnls),
                win_rate,
                sharpe,
                len(open_positions),
            )

        if iterations > 0 and iteration >= iterations:
            break

        await asyncio.sleep(settings.paper_loop_seconds)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Watchdog paper-trading loop")
    parser.add_argument("--virtual-bankroll", type=float, default=500.0)
    parser.add_argument("--platform", type=str, default="manifold")
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
    asyncio.run(main())
