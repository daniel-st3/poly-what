"""Read-only inspector for BTC scalp paper trades.

Shows open/closed counts, latest 20 trades, and any stale open trades
(open longer than 15 minutes without closing).

Usage:
    python scripts/btc_scalp_trades.py               # reads DATABASE_URL / .env
    python scripts/btc_scalp_trades.py --stale-min 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, "src")

from watchdog.core.config import get_settings
from watchdog.db.models import Market, Trade
from watchdog.db.session import build_engine, build_session_factory


def _fmt(v: object, fmt: str = "") -> str:
    if v is None:
        return "—"
    if fmt and isinstance(v, float):
        return format(v, fmt)
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only BTC scalp trade inspector")
    parser.add_argument(
        "--stale-min",
        type=int,
        default=15,
        help="Minutes after open before a trade is considered stale (default: 15)",
    )
    args = parser.parse_args()

    settings = get_settings()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    now = datetime.now(UTC)
    stale_cutoff = now - timedelta(minutes=args.stale_min)
    cutoff_24h = now - timedelta(hours=24)

    with session_factory() as session:
        # All BTC scalp paper trades
        all_trades = (
            session.query(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .filter(
                Trade.strategy == "btc_scalp",
                Trade.is_paper.is_(True),
            )
            .order_by(Trade.opened_at.desc())
            .all()
        )

        open_trades   = [(t, m) for t, m in all_trades if t.status == "open"]
        closed_24h    = [
            (t, m) for t, m in all_trades
            if t.status == "closed"
            and t.closed_at is not None
            and (
                t.closed_at if t.closed_at.tzinfo else t.closed_at.replace(tzinfo=UTC)
            ) >= cutoff_24h
        ]
        stale_trades  = [
            (t, m) for t, m in open_trades
            if t.opened_at is not None
            and (
                t.opened_at if t.opened_at.tzinfo else t.opened_at.replace(tzinfo=UTC)
            ) <= stale_cutoff
        ]

        latest_20 = all_trades[:20]

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BTC Scalp Trade Inspector   {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  DB: {settings.database_url}")
    print(f"{'='*72}")
    print(f"  Open trades (total):          {len(open_trades)}")
    print(f"  Closed trades (last 24h):     {len(closed_24h)}")
    print(f"  Stale open (>{args.stale_min}min):           {len(stale_trades)}")

    # ── Latest 20 trades ────────────────────────────────────────────────────
    col = "{:<6} {:<38} {:<5} {:<7} {:>7} {:>7} {:>7}  {:<19} {:<19}"
    print(f"\n── Latest 20 BTC scalp trades {'─'*42}")
    print(col.format("id", "slug", "side", "status", "entry", "exit", "pnl",
                     "opened_at (UTC)", "closed_at (UTC)"))
    print("─" * 130)
    for trade, market in latest_20:
        opened = (
            trade.opened_at.strftime("%Y-%m-%d %H:%M:%S")
            if trade.opened_at else "—"
        )
        closed = (
            trade.closed_at.strftime("%Y-%m-%d %H:%M:%S")
            if trade.closed_at else "—"
        )
        slug_short = (market.slug or "?")[-38:]
        print(col.format(
            trade.id,
            slug_short,
            trade.side or "?",
            trade.status or "?",
            _fmt(trade.entry_price, ".3f"),
            _fmt(trade.exit_price,  ".3f"),
            _fmt(trade.pnl,         "+.3f"),
            opened,
            closed,
        ))

    # ── Stale open trades ───────────────────────────────────────────────────
    print(f"\n── Stale open trades (open > {args.stale_min} min) {'─'*39}")
    if not stale_trades:
        print("  (none)")
    else:
        for trade, market in stale_trades:
            opened_dt = (
                trade.opened_at if trade.opened_at.tzinfo
                else trade.opened_at.replace(tzinfo=UTC)
            )
            age_min = (now - opened_dt).total_seconds() / 60
            print(
                f"  trade_id={trade.id:<6} slug={market.slug}"
                f"  side={trade.side}  entry={_fmt(trade.entry_price, '.3f')}"
                f"  open_for={age_min:.1f}min"
            )

    print(f"\n{'='*72}\n")


if __name__ == "__main__":
    main()
