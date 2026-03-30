"""Offline edge evaluation for the BTC scalp strategy family.

Loads all available closed-trade data, deduplicates by trade_id (entry-time
snapshot), then answers one question:

    Does this strategy family have any real edge?

Outputs bucketed win-rate + avg-pnl tables for the six dimensions requested,
then prints a binary verdict: RETIRE or REBUILD_FOR_REGIME.

Usage:
    python scripts/btc_scalp_edge_eval.py
    python scripts/btc_scalp_edge_eval.py --csv data/btc_scalp_post_patch.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

# All CSVs to merge by default (deduped by trade_id).
DEFAULT_CSVS = [
    "data/btc_scalp_history.csv",
    "data/btc_scalp_post_patch.csv",
]

EDGE_WIN_THRESHOLD   = 0.55   # win rate must exceed this to claim edge
EDGE_PNL_THRESHOLD   = 0.01   # avg pnl must exceed this to claim edge
MIN_BUCKET_COUNT     = 5      # ignore buckets smaller than this


# ── helpers ──────────────────────────────────────────────────────────────────

def _f(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _bucket_drift(d: float | None) -> str:
    """BTC price drift from market-open proxy, in percent."""
    if d is None:
        return "unknown"
    if d < -0.50:
        return "<-0.50%"
    if d < -0.20:
        return "-0.50 to -0.20%"
    if d < -0.05:
        return "-0.20 to -0.05%"
    if d < 0.05:
        return "-0.05 to +0.05%"
    if d < 0.20:
        return "+0.05 to +0.20%"
    if d < 0.50:
        return "+0.20 to +0.50%"
    return ">+0.50%"


def _bucket_p_up(p: float | None) -> str:
    if p is None:
        return "unknown"
    if p < 0.35:
        return "<0.35"
    if p < 0.45:
        return "0.35-0.45"
    if p < 0.55:
        return "0.45-0.55"
    if p < 0.65:
        return "0.55-0.65"
    if p < 0.75:
        return "0.65-0.75"
    return ">=0.75"


def _bucket_minutes(ml: float | None) -> str:
    if ml is None:
        return "unknown"
    if ml <= 1.0:
        return "0-1m"
    if ml <= 2.0:
        return "1-2m"
    if ml <= 3.0:
        return "2-3m"
    if ml <= 4.0:
        return "3-4m"
    return "4m+"


def _bucket_entry(ep: float | None) -> str:
    """Binary contract entry price (0-1 scale)."""
    if ep is None:
        return "unknown"
    if ep < 0.10:
        return "<0.10"
    if ep < 0.20:
        return "0.10-0.20"
    if ep < 0.30:
        return "0.20-0.30"
    if ep < 0.40:
        return "0.30-0.40"
    if ep < 0.50:
        return "0.40-0.50"
    if ep < 0.60:
        return "0.50-0.60"
    if ep < 0.70:
        return "0.60-0.70"
    if ep < 0.80:
        return "0.70-0.80"
    if ep < 0.90:
        return "0.80-0.90"
    return "0.90-1.00"


def _stats(pnls: list[float]) -> tuple[int, int, float, float, float]:
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / n if n else 0.0
    avg_pnl = sum(pnls) / n if n else 0.0
    total_pnl = sum(pnls)
    return n, wins, win_rate, avg_pnl, total_pnl


def _print_table(
    title: str,
    buckets: dict[str, list[float]],
    bucket_order: list[str] | None = None,
) -> list[tuple[str, int, float, float]]:
    """Print and return (bucket, n, win_rate, avg_pnl) rows with enough data."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  {'Bucket':<22} {'N':>5} {'W':>5} {'L':>5} {'Win%':>7} {'Avg PnL':>9} {'Total':>10}")
    print(f"  {'─'*22} {'─'*5} {'─'*5} {'─'*5} {'─'*7} {'─'*9} {'─'*10}")

    keys = bucket_order if bucket_order else sorted(buckets.keys())
    results = []
    for label in keys:
        pnls = buckets.get(label, [])
        if not pnls:
            continue
        n, wins, win_rate, avg_pnl, total_pnl = _stats(pnls)
        losses = n - wins
        marker = ""
        if n >= MIN_BUCKET_COUNT:
            if win_rate > EDGE_WIN_THRESHOLD and avg_pnl > EDGE_PNL_THRESHOLD:
                marker = "  ◀ edge?"
            results.append((label, n, win_rate, avg_pnl))
        print(
            f"  {label:<22} {n:>5} {wins:>5} {losses:>5}"
            f" {win_rate*100:>6.1f}% {avg_pnl:>+9.4f} {total_pnl:>+10.4f}{marker}"
        )
    return results


# ── load & deduplicate ────────────────────────────────────────────────────────

def load_trade_rows(csv_paths: list[str]) -> list[dict]:
    """Load all CSVs, keep first entry-time row per trade_id for closed trades."""
    all_rows: list[dict] = []
    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            print(f"  [warn] not found, skipping: {path}", file=sys.stderr)
            continue
        with open(p, newline="") as f:
            all_rows.extend(csv.DictReader(f))

    # Keep only rows that have a closed trade with pnl
    candidates = [
        r for r in all_rows
        if r.get("trade_opened") == "1"
        and r.get("decision") == "trade"
        and r.get("trade_pnl") not in ("", None)
        and r.get("trade_status") != "open"
    ]

    # Sort by timestamp so first occurrence = entry time
    candidates.sort(key=lambda r: r.get("timestamp_utc", ""))

    # Deduplicate by trade_id
    seen: set[str] = set()
    trade_rows: list[dict] = []
    for r in candidates:
        tid = r.get("trade_id", "")
        if tid in seen:
            continue
        seen.add(tid)
        trade_rows.append(r)

    return trade_rows


# ── edge detection ────────────────────────────────────────────────────────────

def _has_edge(results: list[tuple[str, int, float, float]]) -> tuple[bool, list[str]]:
    """Return (has_edge, [regime_descriptions])."""
    regimes = []
    for label, n, win_rate, avg_pnl in results:
        if n >= MIN_BUCKET_COUNT and win_rate > EDGE_WIN_THRESHOLD and avg_pnl > EDGE_PNL_THRESHOLD:
            regimes.append(f"{label}  (n={n}, win={win_rate*100:.1f}%, avg={avg_pnl:+.4f})")
    return bool(regimes), regimes


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BTC scalp edge evaluation")
    parser.add_argument(
        "--csv", nargs="+", default=DEFAULT_CSVS,
        help="CSV file(s) to analyse (will be merged & deduped)",
    )
    args = parser.parse_args()

    trade_rows = load_trade_rows(args.csv)

    print(f"\n{'='*70}")
    print("  BTC Scalp Edge Evaluation  —  offline, no live code")
    print(f"  Sources: {', '.join(args.csv)}")
    print(f"{'='*70}")
    print(f"  Distinct closed trades analysed: {len(trade_rows)}")

    if len(trade_rows) < 20:
        print("\n  INSUFFICIENT DATA — fewer than 20 closed trades. Cannot evaluate edge.")
        print("\n  VERDICT: RETIRE  (insufficient sample)")
        return

    # ── build feature buckets ─────────────────────────────────────────────────
    by_drift:    dict[str, list[float]] = defaultdict(list)
    by_p_up:     dict[str, list[float]] = defaultdict(list)
    by_minutes:  dict[str, list[float]] = defaultdict(list)
    by_entry:    dict[str, list[float]] = defaultdict(list)

    for r in trade_rows:
        pnl = _f(r.get("trade_pnl", ""))
        if pnl is None:
            continue

        # 1. Drift: (current_price - start_price_proxy) / start_price_proxy * 100
        cur  = _f(r.get("current_price", ""))
        base = _f(r.get("start_price_proxy", ""))
        drift_pct: float | None = None
        if cur is not None and base and base != 0:
            drift_pct = (cur - base) / base * 100.0

        # 2. p_up at entry time
        p_up = _f(r.get("p_up_model", ""))

        # 3. minutes_left at entry time
        ml = _f(r.get("minutes_left", ""))

        # 4. entry price (binary contract price paid)
        ep = _f(r.get("trade_entry_price", ""))

        by_drift[_bucket_drift(drift_pct)].append(pnl)
        by_p_up[_bucket_p_up(p_up)].append(pnl)
        by_minutes[_bucket_minutes(ml)].append(pnl)
        by_entry[_bucket_entry(ep)].append(pnl)

    # ── overall baseline ──────────────────────────────────────────────────────
    all_pnls = [_f(r.get("trade_pnl", "")) for r in trade_rows]
    all_pnls = [p for p in all_pnls if p is not None]
    n_all, _wins_all, wr_all, avg_all, tot_all = _stats(all_pnls)
    print(f"\n  Overall: n={n_all}  win={wr_all*100:.1f}%  avg_pnl={avg_all:+.4f}  total={tot_all:+.4f}")

    # ── tables ────────────────────────────────────────────────────────────────
    drift_order = [
        "<-0.50%", "-0.50 to -0.20%", "-0.20 to -0.05%",
        "-0.05 to +0.05%", "+0.05 to +0.20%", "+0.20 to +0.50%", ">+0.50%",
        "unknown",
    ]
    p_order = ["<0.35", "0.35-0.45", "0.45-0.55", "0.55-0.65", "0.65-0.75", ">=0.75", "unknown"]
    min_order = ["0-1m", "1-2m", "2-3m", "3-4m", "4m+", "unknown"]
    entry_order = [
        "<0.10", "0.10-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
        "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00", "unknown",
    ]

    r_drift   = _print_table("Win Rate & Avg PnL by DRIFT bucket (% from open)", by_drift, drift_order)
    r_p_up    = _print_table("Win Rate & Avg PnL by P_UP_MODEL bucket", by_p_up, p_order)
    r_minutes = _print_table("Win Rate & Avg PnL by MINUTES_LEFT bucket", by_minutes, min_order)
    r_entry   = _print_table("Win Rate & Avg PnL by ENTRY PRICE bucket", by_entry, entry_order)

    # ── edge detection ────────────────────────────────────────────────────────
    edge_drift,   reg_drift   = _has_edge(r_drift)
    edge_p_up,    reg_p_up    = _has_edge(r_p_up)
    edge_minutes, reg_minutes = _has_edge(r_minutes)
    edge_entry,   reg_entry   = _has_edge(r_entry)

    any_edge = edge_drift or edge_p_up or edge_minutes or edge_entry
    all_regimes: list[str] = []
    if reg_drift:
        all_regimes += [f"  drift:      {x}" for x in reg_drift]
    if reg_p_up:
        all_regimes += [f"  p_up:       {x}" for x in reg_p_up]
    if reg_minutes:
        all_regimes += [f"  minutes:    {x}" for x in reg_minutes]
    if reg_entry:
        all_regimes += [f"  entry_price:{x}" for x in reg_entry]

    # ── also check side-level total pnl ──────────────────────────────────────
    side_pnl: dict[str, list[float]] = defaultdict(list)
    for r in trade_rows:
        pnl = _f(r.get("trade_pnl", ""))
        side = r.get("trade_side") or r.get("side_selected") or "unknown"
        if pnl is not None:
            side_pnl[side].append(pnl)

    print(f"\n{'─'*70}")
    print("  Side breakdown")
    print(f"{'─'*70}")
    for side in ("Up", "Down", "unknown"):
        pnls = side_pnl.get(side, [])
        if not pnls:
            continue
        n, _wins, wr, avg, tot = _stats(pnls)
        print(f"  {side:<8} n={n:>4}  win={wr*100:>5.1f}%  avg={avg:>+8.4f}  total={tot:>+9.4f}")

    # ── conclusion ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  CONCLUSION")
    print(f"{'='*70}")
    print(f"  Overall win rate : {wr_all*100:.1f}%  (threshold for edge: >{EDGE_WIN_THRESHOLD*100:.0f}%)")
    print(f"  Overall avg pnl  : {avg_all:+.4f}  (threshold: >{EDGE_PNL_THRESHOLD:+.2f})")
    print(f"  Total pnl        : {tot_all:+.4f}")
    print()

    if not any_edge and wr_all > EDGE_WIN_THRESHOLD and avg_all > EDGE_PNL_THRESHOLD:
        any_edge = True

    if any_edge:
        print("  Narrow edge detected in the following regimes:")
        for line in all_regimes:
            print(line)
        print()
        print(f"  These regimes show win rate > {EDGE_WIN_THRESHOLD*100:.0f}% AND avg pnl > {EDGE_PNL_THRESHOLD:.2f} on n >= {MIN_BUCKET_COUNT} trades.")
        print("  Strategy shows no broad edge but may be viable when filtered to these regimes.")
        print()
        print("  VERDICT: REBUILD_FOR_REGIME")
    else:
        print("  No bucket across any dimension meets both win-rate and avg-pnl thresholds.")
        print("  The strategy produces negative or neutral expectation in all measurable regimes.")
        print()
        print("  VERDICT: RETIRE")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
