from __future__ import annotations

from datetime import datetime, timedelta
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from watchdog.db.models import PortfolioSnapshot, Trade

KELLY_CAP = 0.20
MIN_POSITION = 1.0
EXCLUDED_STRATEGIES = {"intra_event_arb"}
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


def _fetch_closed_paper_trades(session: Session) -> list[Trade]:
    """Fetch all closed paper trades with non-null pnl, ordered by closed_at."""
    return list(
        session.execute(
            select(Trade)
            .where(
                Trade.is_paper == True,  # noqa: E712
                Trade.status == "closed",
                Trade.pnl.is_not(None),
            )
            .order_by(Trade.closed_at.asc())
        ).scalars().all()
    )


def _portfolio_stats_from_trades(
    trades: list[Trade],
    cap: float,
    run_start: datetime,
) -> tuple[float, float, float, float, float, int, int, int, int]:
    """Compute all portfolio stats from actual Trade.pnl values.

    Returns (balance, peak, run_pnl, total_return_pct, drawdown_pct,
             closed_count, wins, losses, run_count).
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
    losses = sum(1 for t in trades if (t.pnl or 0.0) <= 0)
    run_count = len(run_trades)

    return balance, peak, run_pnl, total_return_pct, drawdown_pct, closed_count, wins, losses, run_count


def update_portfolios(session: Session, run_window_hours: int = 6) -> dict[float, PortfolioState]:
    """Recompute portfolio stats from all closed paper trades and save new snapshots."""
    now = datetime.utcnow()
    run_start = now - timedelta(hours=run_window_hours)

    # Fetch all closed paper trades once — same data for both portfolio sizes
    trades = _fetch_closed_paper_trades(session)

    results: dict[float, PortfolioState] = {}

    for cap in STARTING_CAPITALS:
        balance, peak, run_pnl, total_return_pct, drawdown_pct, closed_count, wins, losses, run_count = (
            _portfolio_stats_from_trades(trades, cap, run_start)
        )

        realized_pnl = balance - cap
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
        )

    session.commit()
    return results


def seed_portfolios(session: Session) -> dict[float, PortfolioState]:
    """Recompute portfolios from all historical closed trades (alias for update_portfolios).

    Safe to call multiple times. Always derives state from DB, never from hardcoded values.
    """
    return update_portfolios(session)


def get_current_portfolios(session: Session) -> dict[float, PortfolioState | None]:
    """Return the latest snapshot for each portfolio without writing anything."""
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
        results[cap] = PortfolioState(
            starting_capital=cap,
            current_balance=latest.current_balance,
            peak_balance=latest.peak_balance,
            run_pnl=latest.run_pnl,
            total_return_pct=latest.total_return_pct,
            drawdown_pct=dd,
            trades_today=latest.trades_today,
            timestamp=latest.timestamp,
            closed_count=0,  # not stored in snapshot; callers that need it use update_portfolios
            wins=0,
            losses=0,
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
