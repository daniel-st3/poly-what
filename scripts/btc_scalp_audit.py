"""Raw-trade truth-table audit for post-b580925 closed BTC scalp trades.

Reads the CSV produced by btc_scalp_export.py and prints one row per
closed trade with every field needed to verify that side / final-outcome /
pnl are internally coherent.

Derived columns (not in the export CSV):
  exit_price        -- 1.0 for WIN, 0.0 for LOSS, blank for BE/open
  resolution_outcome -- derived from (side, won): Up+WIN->YES, Up+LOSS->NO,
                       Down+WIN->NO, Down+LOSS->YES
  derived_outcome   -- same value (mirrors _check_resolutions() path)
  won               -- outcome == "WIN"

Unavailable columns (not stored anywhere):
  opened_at     -- scan_ts (first trade-decision scan row) is the best proxy
  outcome_prices -- Gamma API field; never persisted to DB or CSV

Usage:
    python scripts/btc_scalp_audit.py
    python scripts/btc_scalp_audit.py --csv data/btc_scalp_history_after_b580925.csv
    python scripts/btc_scalp_audit.py --limit 40
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import sys

DEFAULT_CSV = "data/btc_scalp_history_after_b580925.csv"
DEFAULT_LIMIT = 20


def _f(v: object, width: int = 0) -> str:
    """Format a value for table output."""
    s = "—" if v is None or v == "" else str(v)
    return s if width == 0 else f"{s:<{width}}"


def _derive_resolution_outcome(side: str, outcome: str) -> str:
    """Mirror the resolution logic in _check_resolutions().

    Up bought YES token  → wins when market resolved YES
    Down bought NO token → wins when market resolved NO
    """
    if outcome == "WIN":
        return "YES" if side == "Up" else "NO"
    if outcome == "LOSS":
        return "NO" if side == "Up" else "YES"
    return "?"


def _derive_exit_price(outcome: str) -> str:
    if outcome == "WIN":
        return "1.0000"
    if outcome == "LOSS":
        return "0.0000"
    return "—"


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC scalp raw-trade truth-table audit")
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to btc_scalp_export CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max trades to show (default 20)",
    )
    args = parser.parse_args()

    try:
        with open(args.csv, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run:  python scripts/btc_scalp_export.py  first.")
        sys.exit(1)

    # Collect the first (entry-time) scan row for each closed trade,
    # deduped by trade_id in ascending timestamp order.
    # CSV is sorted timestamp_utc ASC so the first occurrence = entry scan.
    # Filter to decision=='trade' rows only: the export attaches a trade to ALL
    # scan rows for that market_id (including prior SKIP cycles), so without this
    # filter the first row per trade_id would be a SKIP scan with stale EV/bid/ask.
    seen_ids: set[str] = set()
    trade_rows: list[dict] = []
    for r in rows:
        if r.get("trade_status") != "closed":
            continue
        if not r.get("trade_pnl"):
            continue
        if r.get("decision") != "trade":
            continue
        tid = r.get("trade_id") or ""
        if not tid or tid in seen_ids:
            continue
        seen_ids.add(tid)
        trade_rows.append(r)

    trade_rows = trade_rows[: args.limit]

    if not trade_rows:
        print("No closed trades found. Re-export with btc_scalp_export.py first.")
        sys.exit(0)

    # ── Build audit records ───────────────────────────────────────────────────

    cols: list[tuple[str, int]] = [
        ("trade_id",           8),
        ("slug",              44),
        # opened_at is not in the export CSV; scan_ts (first 'trade' scan) is the proxy
        ("scan_ts",           22),
        ("closed_at",         22),
        ("side",               5),
        ("entry_price",       11),
        ("exit_price",        10),
        ("pnl",                9),
        ("status",             8),
        ("p_up",               7),
        ("ev_side",            8),
        ("eff_bid",            8),
        ("eff_ask",            8),
        ("eff_spread",        10),
        # resolution_outcome: derived from (side, outcome) — not stored in CSV
        ("resolution_outcome", 19),
        # outcome_prices: Gamma API field — never persisted
        ("outcome_prices",    22),
        # derived_outcome: mirrors _check_resolutions(); same as resolution_outcome here
        ("derived_outcome",   15),
        ("won",                5),
    ]

    sep = "  "
    header = sep.join(f"{name:<{w}}" for _, (name, w) in
                      [(None, c) for c in cols])

    # Re-derive header properly
    header = sep.join(f"{name:<{w}}" for name, w in cols)
    divider = "─" * len(header)

    print(f"\nBTC Scalp Raw-Trade Audit  |  CSV: {args.csv}")
    print(f"Closed trades shown: {len(trade_rows)} (limit={args.limit})\n")
    print("NOTE: 'opened_at' not in export CSV -- 'scan_ts' is the entry-scan timestamp proxy.")
    print("NOTE: 'outcome_prices' never persisted; always N/A.\n")
    print(header)
    print(divider)

    total_pnl = 0.0
    wins = 0
    losses = 0

    for r in trade_rows:
        tid         = r.get("trade_id", "")
        slug        = r.get("slug", "")
        scan_ts     = r.get("timestamp_utc", "")
        closed_at   = r.get("trade_closed_at", "")
        side        = r.get("trade_side") or r.get("side_selected") or ""
        entry_price = r.get("trade_entry_price", "")
        outcome_str = r.get("outcome", "")
        exit_price  = _derive_exit_price(outcome_str)
        pnl_raw     = r.get("trade_pnl", "")
        status      = r.get("trade_status", "")
        p_up        = r.get("p_up_model", "")
        ev_side     = r.get("ev_up", "") if side == "Up" else r.get("ev_down", "")
        eff_bid     = r.get("effective_bid", "")
        eff_ask     = r.get("effective_ask", "")
        eff_spread  = r.get("effective_spread", "")
        res_outcome = _derive_resolution_outcome(side, outcome_str)
        won_str     = ("True" if outcome_str == "WIN"
                       else "False" if outcome_str == "LOSS"
                       else "—")

        with contextlib.suppress(ValueError, TypeError):
            total_pnl += float(pnl_raw)
        if outcome_str == "WIN":
            wins += 1
        elif outcome_str == "LOSS":
            losses += 1

        vals = {
            "trade_id": tid,
            "slug": slug,
            "scan_ts": scan_ts,
            "closed_at": closed_at,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl_raw,
            "status": status,
            "p_up": p_up,
            "ev_side": ev_side,
            "eff_bid": eff_bid,
            "eff_ask": eff_ask,
            "eff_spread": eff_spread,
            "resolution_outcome": res_outcome,
            "outcome_prices": "N/A (not stored)",
            "derived_outcome": res_outcome,
            "won": won_str,
        }

        line = sep.join(_f(vals.get(col, ""), w) for col, w in cols)
        print(line)

    print(divider)
    n = len(trade_rows)
    print(
        f"\nSummary: {n} trade(s)  |  wins={wins}  losses={losses}"
        f"  win_rate={wins/n*100:.1f}%  total_pnl={total_pnl:+.4f}\n"
    )


if __name__ == "__main__":
    main()
