"""Reset the paper trading account by closing all open positions.

Trades are marked closed with close_reason='legacy_reset' — history is preserved.
Run with --confirm to execute; without it, prints a dry-run summary only.
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from sqlalchemy import select

# Make sure the package is importable when run from the repo root
sys.path.insert(0, "src")

from watchdog.core.config import get_settings
from watchdog.db.init import init_db
from watchdog.db.models import Market, Trade
from watchdog.db.session import build_engine, build_session_factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Close all open paper trades (legacy reset)")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually execute the reset (omit for dry-run preview)",
    )
    args = parser.parse_args()

    settings = get_settings()
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    now = datetime.now(UTC)

    with session_factory() as session:
        rows = session.execute(
            select(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .where(Trade.is_paper.is_(True), Trade.status == "open")
            .order_by(Trade.opened_at)
        ).all()

        if not rows:
            print("No open paper trades found. Nothing to reset.")
            return

        # --- Compute PnL for each open trade using entry price as exit price ---
        # (We have no live prices here; we record $0 PnL on positions we can't mark)
        total_committed = 0.0
        total_estimated_pnl = 0.0
        side_counts: dict[str, int] = {"YES": 0, "NO": 0}
        strategy_counts: dict[str, int] = {}

        print(f"\n{'─'*70}")
        print(f"  PAPER ACCOUNT RESET — {'DRY RUN' if not args.confirm else 'EXECUTING'}")
        print(f"{'─'*70}")
        print(f"  Open positions found : {len(rows)}")

        for trade, market in rows:
            total_committed += trade.size
            side_counts[trade.side] = side_counts.get(trade.side, 0) + 1
            strat = trade.strategy or "unknown"
            strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

        print(f"  Total committed      : ${total_committed:,.2f}")
        print(f"  Side breakdown       : YES={side_counts.get('YES',0)}  NO={side_counts.get('NO',0)}")
        print(f"  Strategy breakdown   : {strategy_counts}")
        print(f"  Close reason         : legacy_reset")
        print(f"  Exit price used      : entry_price (no live mark)")
        print(f"  PnL recorded         : $0.00 per position (no live data)")
        print(f"{'─'*70}")

        if not args.confirm:
            print("\n  DRY RUN — no changes made.")
            print("  Re-run with --confirm to execute the reset.\n")
            return

        # --- Execute ---
        closed = 0
        for trade, market in rows:
            trade.status = "closed"
            trade.exit_price = trade.entry_price   # mark at cost
            trade.pnl = 0.0                        # no live price → neutral
            trade.closed_at = now
            trade.close_reason = "legacy_reset"
            closed += 1

        session.commit()

        print(f"\n  Closed {closed} trades.")
        print(f"  All marked: status=closed, exit_price=entry_price, pnl=0.00")
        print(f"  History preserved — trades table unchanged except status fields.")
        print(f"\n  Bankroll is now free. New trades will use Quarter-Kelly sizing.")
        print(f"  Configured bankroll: ${settings.paper_bankroll_usdc:.2f}\n")


if __name__ == "__main__":
    main()
