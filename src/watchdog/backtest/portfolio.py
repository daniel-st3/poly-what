from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from watchdog.db.models import MarketSnapshot, PortfolioSnapshot, Trade

KELLY_CAP = 0.20
MIN_POSITION = 1.0
# Strategies excluded from the normal-bot $100/$500 portfolio scope.
# btc_scalp is a separate Railway worker; intra_event_arb is excluded by convention.
_DEFAULT_EXCLUDED: frozenset[str] = frozenset({"intra_event_arb", "btc_scalp"})
EXCLUDED_STRATEGIES = _DEFAULT_EXCLUDED  # backward-compat alias
STARTING_CAPITALS = [100.0, 500.0]


class PortfolioState(NamedTuple):
    starting_capital: float
    current_balance: float
    peak_balance: float
    run_pnl: float
    total_return_pct: float
    drawdown_pct: float
    trades_today: int
    timestamp: datetime
    closed_count: int
    wins: int
    losses: int
    breakevens: int
    open_count: int
    realized_pnl: float
    deployed_capital: float
    free_capital: float
    open_pnl_estimate: float | None
    open_pnl_priced_count: int
    equity: float


def _fetch_closed_paper_trades(
    session: Session,
    excluded_strategies: frozenset[str] | None = None,
) -> list[Trade]:
    """Fetch all closed paper trades with non-null pnl, ordered by closed_at.

    Excludes strategies in excluded_strategies (defaults to _DEFAULT_EXCLUDED).
    Trade.strategy is nullable: NULL-strategy rows are included in the normal scope.
    """
    excl = excluded_strategies if excluded_strategies is not None else _DEFAULT_EXCLUDED
    return list(
        session.execute(
            select(Trade)
            .where(
                Trade.is_paper == True,  # noqa: E712
                Trade.status == "closed",
                Trade.pnl.is_not(None),
                or_(Trade.strategy.is_(None), Trade.strategy.not_in(excl)),
            )
            .order_by(Trade.closed_at.asc())
        ).scalars().all()
    )


def _fetch_open_paper_trades(
    session: Session,
    excluded_strategies: frozenset[str] | None = None,
) -> list[Trade]:
    """Fetch all open paper trades.

    Excludes strategies in excluded_strategies (defaults to _DEFAULT_EXCLUDED).
    Trade.strategy is nullable: NULL-strategy rows are included in the normal scope.
    """
    excl = excluded_strategies if excluded_strategies is not None else _DEFAULT_EXCLUDED
    return list(
        session.execute(
            select(Trade)
            .where(
                Trade.is_paper == True,  # noqa: E712
                Trade.status == "open",
                or_(Trade.strategy.is_(None), Trade.strategy.not_in(excl)),
            )
        ).scalars().all()
    )


def _compute_open_pnl_estimate(
    session: Session,
    open_trades: list[Trade],
) -> tuple[float | None, int, int]:
    """Estimate unrealized PnL for open trades using latest MarketSnapshot.mid.

    Returns (estimate, priced_count, total_open_count).
    - estimate is None if no trades have a usable mid price.
    - estimate is a partial sum if only some trades are priced.
    """
    total_open_count = len(open_trades)
    if total_open_count == 0:
        return None, 0, 0

    market_ids = {t.market_id for t in open_trades}

    # Build {market_id: latest non-null mid}.
    # Order by captured_at desc and keep first hit per market_id (Python-side dedup
    # so the query works on both SQLite and PostgreSQL).
    mid_by_market: dict[int, float] = {}
    for row in session.execute(
        select(MarketSnapshot.market_id, MarketSnapshot.mid, MarketSnapshot.captured_at)
        .where(
            MarketSnapshot.market_id.in_(market_ids),
            MarketSnapshot.mid.is_not(None),
        )
        .order_by(MarketSnapshot.captured_at.desc())
    ).all():
        market_id_val = row[0]
        if market_id_val not in mid_by_market:
            mid_by_market[market_id_val] = float(row[1])

    running_sum = 0.0
    priced_count = 0
    for t in open_trades:
        mid = mid_by_market.get(t.market_id)
        if mid is None:
            continue
        if t.side == "YES":
            running_sum += t.size * (mid - t.entry_price)
        else:
            running_sum += t.size * (t.entry_price - mid)
        priced_count += 1

    if priced_count == 0:
        return None, 0, total_open_count
    return running_sum, priced_count, total_open_count


def _portfolio_stats_from_trades(
    trades: list[Trade],
    cap: float,
    run_start: datetime,
) -> tuple[float, float, float, float, float, int, int, int, int, int]:
    """Compute all portfolio stats from actual Trade.pnl values.

    Returns (balance, peak, run_pnl, total_return_pct, drawdown_pct,
             closed_count, wins, losses, breakevens, run_count).
    """
    realized_pnl = sum(t.pnl for t in trades)  # type: ignore[misc]
    balance = cap + realized_pnl
    total_return_pct = (realized_pnl / cap) * 100

    # Peak from running cumulative balance through ordered trade history
    peak = cap
    running = cap
    for t in trades:
        running += t.pnl  # type: ignore[operator]
        if running > peak:
            peak = running
    drawdown_pct = (peak - balance) / peak * 100 if peak > 0 else 0.0

    # "This run" = trades closed since run_start
    run_trades = [t for t in trades if t.closed_at is not None and t.closed_at >= run_start]
    run_pnl = sum(t.pnl for t in run_trades)  # type: ignore[misc]

    closed_count = len(trades)
    wins = sum(1 for t in trades if (t.pnl or 0.0) > 0)
    losses = sum(1 for t in trades if (t.pnl or 0.0) < 0)
    breakevens = sum(1 for t in trades if (t.pnl or 0.0) == 0.0)
    run_count = len(run_trades)

    return balance, peak, run_pnl, total_return_pct, drawdown_pct, closed_count, wins, losses, breakevens, run_count


def update_portfolios(session: Session, run_window_hours: int = 6) -> dict[float, PortfolioState]:
    """Recompute portfolio stats from all closed paper trades and save new snapshots."""
    now = datetime.utcnow()
    run_start = now - timedelta(hours=run_window_hours)

    # Fetch once — shared across all portfolio sizes
    closed_trades = _fetch_closed_paper_trades(session)
    open_trades = _fetch_open_paper_trades(session)

    open_pnl_estimate, open_pnl_priced_count, total_open_count = _compute_open_pnl_estimate(
        session, open_trades
    )
    deployed_capital = sum(t.size * t.entry_price for t in open_trades)

    results: dict[float, PortfolioState] = {}

    for cap in STARTING_CAPITALS:
        balance, peak, run_pnl, total_return_pct, drawdown_pct, closed_count, wins, losses, breakevens, run_count = (
            _portfolio_stats_from_trades(closed_trades, cap, run_start)
        )

        realized_pnl = balance - cap
        free_capital = balance - deployed_capital
        equity = balance + (open_pnl_estimate or 0.0)

        print(
            f"📊 portfolio update ${cap:.0f}: closed_trades={closed_count}, "
            f"realized_pnl={realized_pnl:.2f}, balance={balance:.2f}"
        )

        snap = PortfolioSnapshot(
            timestamp=now,
            starting_capital=cap,
            current_balance=balance,
            peak_balance=peak,
            run_pnl=run_pnl,
            total_return_pct=total_return_pct,
            drawdown_pct=drawdown_pct,
            trades_today=run_count,
        )
        session.add(snap)
        results[cap] = PortfolioState(
            starting_capital=cap,
            current_balance=balance,
            peak_balance=peak,
            run_pnl=run_pnl,
            total_return_pct=total_return_pct,
            drawdown_pct=drawdown_pct,
            trades_today=run_count,
            timestamp=now,
            closed_count=closed_count,
            wins=wins,
            losses=losses,
            breakevens=breakevens,
            open_count=total_open_count,
            realized_pnl=realized_pnl,
            deployed_capital=deployed_capital,
            free_capital=free_capital,
            open_pnl_estimate=open_pnl_estimate,
            open_pnl_priced_count=open_pnl_priced_count,
            equity=equity,
        )

    session.commit()
    return results


def seed_portfolios(session: Session) -> dict[float, PortfolioState]:
    """Recompute portfolios from all historical closed trades (alias for update_portfolios).

    Safe to call multiple times. Always derives state from DB, never from hardcoded values.
    """
    return update_portfolios(session)


def get_current_portfolios(session: Session) -> dict[float, PortfolioState | None]:
    """Return the latest snapshot for each portfolio without writing anything.

    Counts (closed_count, wins, losses, breakevens, open_count) are computed
    live from DB trades so they are never stale zeros.
    """
    # Fetch trades once — shared across all portfolio sizes
    closed_trades = _fetch_closed_paper_trades(session)
    open_trades = _fetch_open_paper_trades(session)

    open_pnl_estimate, open_pnl_priced_count, total_open_count = _compute_open_pnl_estimate(
        session, open_trades
    )
    deployed_capital = sum(t.size * t.entry_price for t in open_trades)

    now = datetime.utcnow()
    run_start = now - timedelta(hours=6)

    results: dict[float, PortfolioState | None] = {}
    for cap in STARTING_CAPITALS:
        latest = session.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.starting_capital == cap)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest is None:
            results[cap] = None
            continue

        dd = (
            (latest.peak_balance - latest.current_balance) / latest.peak_balance * 100
            if latest.peak_balance > 0
            else 0.0
        )

        _, _, _, _, _, closed_count, wins, losses, breakevens, _ = _portfolio_stats_from_trades(
            closed_trades, cap, run_start
        )
        free_capital = latest.current_balance - deployed_capital
        equity = latest.current_balance + (open_pnl_estimate or 0.0)

        results[cap] = PortfolioState(
            starting_capital=cap,
            current_balance=latest.current_balance,
            peak_balance=latest.peak_balance,
            run_pnl=latest.run_pnl,
            total_return_pct=latest.total_return_pct,
            drawdown_pct=dd,
            trades_today=latest.trades_today,
            timestamp=latest.timestamp,
            closed_count=closed_count,
            wins=wins,
            losses=losses,
            breakevens=breakevens,
            open_count=total_open_count,
            realized_pnl=latest.current_balance - cap,
            deployed_capital=deployed_capital,
            free_capital=free_capital,
            open_pnl_estimate=open_pnl_estimate,
            open_pnl_priced_count=open_pnl_priced_count,
            equity=equity,
        )
    return results


def count_runs_today(session: Session) -> int:
    """Count how many update snapshots (non-seed) have been saved today for the $100 portfolio."""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = session.execute(
        select(PortfolioSnapshot).where(
            PortfolioSnapshot.starting_capital == 100.0,
            PortfolioSnapshot.timestamp >= today_start,
            PortfolioSnapshot.trades_today > 0,
        )
    ).scalars().all()
    # +1 for the snapshot about to be saved in this run
    return len(rows) + 1
