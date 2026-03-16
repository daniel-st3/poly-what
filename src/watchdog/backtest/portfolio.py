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


def _apply_trades(
    trades: list[Trade],
    starting_balance: float,
    starting_peak: float,
) -> tuple[float, float, float, int]:
    """Apply trades in order using Kelly sizing. Returns (balance, peak, total_pnl, count)."""
    balance = starting_balance
    peak = starting_peak
    total_pnl = 0.0
    count = 0

    for trade in trades:
        if trade.exit_price is None or trade.entry_price is None or trade.entry_price == 0:
            continue
        kelly = min(trade.kelly_fraction or 0.10, KELLY_CAP)
        position_size = balance * kelly
        if position_size < MIN_POSITION:
            continue

        pnl = position_size * (trade.exit_price - trade.entry_price) / trade.entry_price
        balance = max(0.0, balance + pnl)
        peak = max(peak, balance)
        total_pnl += pnl
        count += 1

    return balance, peak, total_pnl, count


def _fetch_closed_trades(
    session: Session,
    *,
    after: datetime | None = None,
    before: datetime | None = None,
) -> list[Trade]:
    q = (
        select(Trade)
        .where(
            Trade.status == "closed",
            Trade.entry_price.is_not(None),
            ~Trade.strategy.in_(list(EXCLUDED_STRATEGIES)),
        )
        .order_by(Trade.opened_at.asc())
    )
    if after is not None:
        q = q.where(Trade.closed_at > after)
    if before is not None:
        q = q.where(Trade.closed_at <= before)
    return list(session.execute(q).scalars().all())


def seed_portfolios(session: Session) -> dict[float, PortfolioState]:
    """Replay all existing closed trades to establish baseline balances for both portfolios.

    Saves a seed snapshot (run_pnl=0, trades_today=0) so incremental updates know
    where to start. Safe to call multiple times — only seeds if no snapshot exists yet.
    """
    now = datetime.utcnow()  # naive UTC — must match how closed_at is stored in SQLite
    results: dict[float, PortfolioState] = {}

    for cap in STARTING_CAPITALS:
        latest = session.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.starting_capital == cap)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest is not None and latest.current_balance != cap:
            # Latest snapshot has real data — return it directly.
            # If current_balance == starting capital the snapshot is likely from a
            # botched run (saved before any trades were replayed); fall through
            # and recalculate from trades so we recover the correct balance.
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
            )
            continue

        trades = _fetch_closed_trades(session)
        balance, peak, _, _ = _apply_trades(trades, cap, cap)
        total_return = (balance - cap) / cap * 100
        drawdown = (peak - balance) / peak * 100 if peak > 0 else 0.0

        snap = PortfolioSnapshot(
            timestamp=now,
            starting_capital=cap,
            current_balance=balance,
            peak_balance=peak,
            run_pnl=0.0,
            total_return_pct=total_return,
            drawdown_pct=drawdown,
            trades_today=0,
        )
        session.add(snap)
        results[cap] = PortfolioState(
            starting_capital=cap,
            current_balance=balance,
            peak_balance=peak,
            run_pnl=0.0,
            total_return_pct=total_return,
            drawdown_pct=drawdown,
            trades_today=0,
            timestamp=now,
        )

    session.commit()
    return results


def update_portfolios(session: Session, run_window_hours: int = 6) -> dict[float, PortfolioState]:
    """Apply trades closed since last snapshot and save new snapshots.

    Called after each bot run. The run_pnl field captures PnL from trades
    closed within the run_window_hours window for the Telegram display.
    """
    now = datetime.utcnow()  # naive UTC — must match how closed_at is stored in SQLite
    run_start = now - timedelta(hours=run_window_hours)
    results: dict[float, PortfolioState] = {}

    for cap in STARTING_CAPITALS:
        latest = session.execute(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.starting_capital == cap)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest is None:
            # Auto-seed if called before explicit seeding
            seeded = seed_portfolios(session)
            return seeded

        # All trades since last snapshot
        new_trades = _fetch_closed_trades(session, after=latest.timestamp)

        # "This run" = trades from within the run window (for display)
        run_trades = [t for t in new_trades if t.closed_at is not None and t.closed_at >= run_start]

        print(
            f"📊 portfolio update ${cap:.0f}: last_snapshot={latest.timestamp.strftime('%m-%d %H:%M')}, "
            f"new_trades={len(new_trades)}, run_trades={len(run_trades)}"
        )

        balance, peak, _new_pnl, _ = _apply_trades(
            new_trades, latest.current_balance, latest.peak_balance
        )
        _, _, run_pnl, run_count = _apply_trades(
            run_trades, latest.current_balance, latest.peak_balance
        )

        total_return = (balance - cap) / cap * 100
        drawdown = (peak - balance) / peak * 100 if peak > 0 else 0.0

        snap = PortfolioSnapshot(
            timestamp=now,
            starting_capital=cap,
            current_balance=balance,
            peak_balance=peak,
            run_pnl=run_pnl,
            total_return_pct=total_return,
            drawdown_pct=drawdown,
            trades_today=run_count,
        )
        session.add(snap)
        results[cap] = PortfolioState(
            starting_capital=cap,
            current_balance=balance,
            peak_balance=peak,
            run_pnl=run_pnl,
            total_return_pct=total_return,
            drawdown_pct=drawdown,
            trades_today=run_count,
            timestamp=now,
        )

    session.commit()
    return results


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
