from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from watchdog.backtest.metrics import (
    brier_score,
    compute_max_drawdown,
    compute_win_rate,
    evaluate_pass_fail,
)
from watchdog.db.models import Signal, Trade

LOGGER = logging.getLogger(__name__)


def check_go_live_gate(session: Session) -> tuple[bool, list[str]]:
    stmt = (
        select(Trade, Signal)
        .outerjoin(Signal, Trade.signal_id == Signal.id)
        .where(
            Trade.is_paper.is_(True),
            Trade.status == "closed",
            Trade.pnl.is_not(None),
            Trade.entry_price > 0,
        )
        .order_by(Trade.closed_at.desc())
        .limit(50)
    )

    rows = session.execute(stmt).all()
    n_trades = len(rows)

    if n_trades < 50:
        return False, [f"insufficient_trades: {n_trades} < 50"]

    # Equity curve needs chronological order
    rows = list(reversed(rows))

    outcomes: list[float] = []
    predictions: list[float] = []
    equity_curve: list[float] = [500.0]  # Standard bankroll
    current_equity = 500.0
    total_pnl = 0.0

    for trade, signal in rows:
        won = trade.pnl is not None and trade.pnl > 0
        y_true = (1.0 if won else 0.0) if trade.side == "YES" else (0.0 if won else 1.0)
        outcomes.append(y_true)

        if signal is not None:
            pred = signal.model_probability
        else:
            pred = trade.entry_price if trade.side == "YES" else 1.0 - trade.entry_price
        predictions.append(pred)

        pnl = trade.pnl or 0.0
        total_pnl += pnl
        current_equity += pnl
        equity_curve.append(max(current_equity, 1e-12))  # Ensure > 0 for drawdown math

    win_rate = compute_win_rate(outcomes, predictions)
    brier = brier_score(outcomes, predictions)
    max_dd = compute_max_drawdown(equity_curve)

    passed, reasons = evaluate_pass_fail(
        win_rate=win_rate, brier=brier, n_trades=n_trades, max_drawdown=max_dd
    )

    # Convert failed reasons list
    all_reasons = list(reasons)

    if total_pnl <= 0:
        all_reasons.append(f"total_pnl_below_threshold: {total_pnl:.4f} <= 0")
        passed = False

    LOGGER.info(
        "Gate metrics: trades=%d win_rate=%.4f brier=%.4f max_dd=%.4f pnl=%.4f (Pass=%s)",
        n_trades, win_rate, brier, max_dd, total_pnl, passed
    )

    return passed, all_reasons
