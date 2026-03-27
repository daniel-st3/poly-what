"""Export BTC scalp historical dataset to CSV.

Joins BtcScanLog (every scan row) with Trade outcomes and BtcSignalLog to
produce one denormalized CSV suitable for offline analysis and model calibration.

Columns:
  timestamp_utc, slug, market_id, side_selected, start_price_proxy,
  current_price, minutes_left, momentum_60s, realized_vol,
  raw_up_bid, raw_up_ask, raw_down_bid, raw_down_ask,
  up_clob_available, down_clob_available,
  effective_bid, effective_ask, effective_spread,
  p_up_model, ev_up, ev_down, decision, skip_reason,
  trade_opened, trade_id, trade_entry_price, trade_side,
  trade_status, trade_pnl, trade_closed_at, outcome

Usage:
    python scripts/btc_scalp_export.py
    python scripts/btc_scalp_export.py --out data/btc_scalp_history.csv
    python scripts/btc_scalp_export.py --since 2026-03-01
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime

sys.path.insert(0, "src")

from sqlalchemy import select

from watchdog.core.config import get_settings
from watchdog.db.models import BtcScanLog, Market, Trade
from watchdog.db.session import build_engine, build_session_factory

DEFAULT_OUT = "data/btc_scalp_history.csv"

COLUMNS = [
    "timestamp_utc",
    "slug",
    "market_id",
    "side_selected",
    "start_price_proxy",
    "current_price",
    "minutes_left",
    "momentum_60s",
    "realized_vol",
    "raw_up_bid",
    "raw_up_ask",
    "raw_down_bid",
    "raw_down_ask",
    "up_clob_available",
    "down_clob_available",
    # effective_* = best executable quote (eff_ask = entry price used; may equal raw ask)
    "effective_bid",
    "effective_ask",
    "effective_spread",
    "p_up_model",
    "ev_up",
    "ev_down",
    "decision",
    "skip_reason",
    "trade_opened",
    "trade_id",
    "trade_entry_price",
    "trade_side",
    "trade_status",
    "trade_pnl",
    "trade_closed_at",
    # outcome: WIN / LOSS / open / None
    "outcome",
]


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BTC scalp history to CSV")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument(
        "--since",
        default=None,
        help="Only export rows from this date onward (YYYY-MM-DD UTC)",
    )
    args = parser.parse_args()

    settings = get_settings()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    since_dt: datetime | None = None
    if args.since:
        since_dt = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)

    with session_factory() as session:
        q = session.query(BtcScanLog, Market).outerjoin(
            Market, BtcScanLog.market_id == Market.id
        )
        if since_dt:
            q = q.filter(BtcScanLog.timestamp_utc >= since_dt)
        q = q.order_by(BtcScanLog.timestamp_utc.asc())
        scan_rows = q.all()

        # Build a lookup: market_id → list of trades (btc_scalp, paper).
        # Narrow projection to only the 7 columns used below; date-bound to
        # match the scan query so this never becomes an unbounded table scan.
        trades_stmt = select(
            Trade.id,
            Trade.market_id,
            Trade.side,
            Trade.status,
            Trade.pnl,
            Trade.entry_price,
            Trade.closed_at,
        ).where(
            Trade.strategy == "btc_scalp",
            Trade.is_paper.is_(True),
        )
        if since_dt:
            trades_stmt = trades_stmt.where(Trade.created_at >= since_dt)
        all_trades = session.execute(trades_stmt).all()
        trades_by_market: dict[int, list] = {}
        for t in all_trades:
            trades_by_market.setdefault(t.market_id, []).append(t)

    total_rows = 0
    total_trade_rows = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        for scan, _market in scan_rows:
            # Find the trade for this market (if any) — first one wins
            market_trades = trades_by_market.get(scan.market_id or -1, [])
            trade: Trade | None = market_trades[0] if market_trades else None

            if trade:
                if trade.status == "open":
                    outcome = "open"
                elif trade.pnl is not None:
                    outcome = "WIN" if trade.pnl > 0 else ("LOSS" if trade.pnl < 0 else "BE")
                else:
                    outcome = "closed_no_pnl"
                total_trade_rows += 1
            else:
                outcome = ""

            # Effective bid/ask: for scan rows we infer from available fields.
            # BtcScanLog stores raw CLOb top-of-book. The strategy overrides
            # side_entry_price with eff_ask from depth when stub is bypassed,
            # but that eff_ask is not stored in the scan log (only in live logs).
            # We approximate: effective_ask = entry_price from trade if traded,
            # else the best raw ask for the selected side.
            if trade and trade.entry_price is not None:
                eff_ask = trade.entry_price
            elif scan.selected_side == "Up":
                eff_ask = scan.up_best_ask
            elif scan.selected_side == "Down":
                eff_ask = scan.down_best_ask
            else:
                eff_ask = None

            eff_bid: float | None
            if scan.selected_side == "Up":
                eff_bid = scan.up_best_bid
            elif scan.selected_side == "Down":
                eff_bid = scan.down_best_bid
            else:
                eff_bid = None

            eff_spread: float | None = None
            if eff_bid is not None and eff_ask is not None:
                eff_spread = round(eff_ask - eff_bid, 6)

            row = {
                "timestamp_utc": _fmt_dt(scan.timestamp_utc),
                "slug": scan.slug,
                "market_id": scan.market_id,
                "side_selected": scan.selected_side or "",
                "start_price_proxy": scan.start_price_proxy,
                "current_price": scan.current_price,
                "minutes_left": scan.minutes_left,
                "momentum_60s": scan.momentum_60s,
                "realized_vol": scan.realized_vol,
                "raw_up_bid": scan.up_best_bid,
                "raw_up_ask": scan.up_best_ask,
                "raw_down_bid": scan.down_best_bid,
                "raw_down_ask": scan.down_best_ask,
                "up_clob_available": int(scan.up_clob_available),
                "down_clob_available": int(scan.down_clob_available),
                "effective_bid": eff_bid,
                "effective_ask": eff_ask,
                "effective_spread": eff_spread,
                "p_up_model": scan.p_up_model,
                "ev_up": scan.ev_up,
                "ev_down": scan.ev_down,
                "decision": scan.decision,
                "skip_reason": scan.skip_reason or "",
                "trade_opened": 1 if trade else 0,
                "trade_id": trade.id if trade else "",
                "trade_entry_price": trade.entry_price if trade else "",
                "trade_side": trade.side if trade else "",
                "trade_status": trade.status if trade else "",
                "trade_pnl": trade.pnl if trade else "",
                "trade_closed_at": _fmt_dt(trade.closed_at) if trade else "",
                "outcome": outcome,
            }
            writer.writerow(row)
            total_rows += 1

    print(f"Exported {total_rows} scan rows ({total_trade_rows} with trade) → {args.out}")


if __name__ == "__main__":
    main()
