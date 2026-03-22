"""Offline analysis of BTC scalp historical CSV.

Loads the CSV produced by btc_scalp_export.py and prints bucketed metrics:
  - by side (Up / Down)
  - by minutes_left bucket
  - by EV bucket (ev of the selected side)
  - by p_up bucket
  - by effective spread bucket

For each bucket: count, win rate, avg pnl, total pnl.

Optionally fits a logistic regression calibration model on p_up_model vs outcome
and prints recommended probability adjustment coefficients.

Usage:
    python scripts/btc_scalp_analysis.py
    python scripts/btc_scalp_analysis.py --csv data/btc_scalp_history.csv
    python scripts/btc_scalp_analysis.py --calibrate
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict

DEFAULT_CSV = "data/btc_scalp_history.csv"


# ── helpers ─────────────────────────────────────────────────────────────────

def _safe_float(v: str) -> float | None:
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


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


def _bucket_ev(ev: float | None) -> str:
    if ev is None:
        return "unknown"
    if ev < 0.00:
        return "<0"
    if ev < 0.02:
        return "0.00-0.02"
    if ev < 0.05:
        return "0.02-0.05"
    if ev < 0.10:
        return "0.05-0.10"
    return ">=0.10"


def _bucket_p(p: float | None) -> str:
    if p is None:
        return "unknown"
    if p < 0.40:
        return "<0.40"
    if p < 0.50:
        return "0.40-0.50"
    if p < 0.60:
        return "0.50-0.60"
    if p < 0.70:
        return "0.60-0.70"
    return ">=0.70"


def _bucket_spread(sp: float | None) -> str:
    if sp is None:
        return "unknown"
    if sp < 0.05:
        return "<0.05"
    if sp < 0.10:
        return "0.05-0.10"
    if sp < 0.20:
        return "0.10-0.20"
    return ">=0.20"


def _print_table(title: str, buckets: dict[str, list[float | None]]) -> None:
    """Print a summary table for a set of buckets.

    buckets: {bucket_label → [pnl, ...]}  (None entries = open / unknown)
    """
    print(f"\n{'─'*64}")
    print(f"  {title}")
    print(f"{'─'*64}")
    print(f"  {'Bucket':<18} {'Count':>6} {'W':>5} {'L':>5} {'Win%':>7} {'Avg PnL':>9} {'Total':>9}")
    print(f"  {'─'*18} {'─'*6} {'─'*5} {'─'*5} {'─'*7} {'─'*9} {'─'*9}")
    for label in sorted(buckets.keys()):
        pnls = [p for p in buckets[label] if p is not None]
        n = len(pnls)
        if n == 0:
            continue
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        win_rate = wins / n * 100
        avg_pnl = sum(pnls) / n
        total_pnl = sum(pnls)
        print(
            f"  {label:<18} {n:>6} {wins:>5} {losses:>5}"
            f" {win_rate:>6.1f}% {avg_pnl:>+9.4f} {total_pnl:>+9.4f}"
        )


# ── logistic regression (no scipy dependency) ────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logistic_fit(xs: list[float], ys: list[int], lr: float = 0.1, iters: int = 500) -> tuple[float, float]:
    """Fit logistic regression y ~ sigmoid(a*x + b) via gradient descent.

    Returns (a, b): slope, intercept.
    """
    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(iters):
        grad_a = grad_b = 0.0
        for x, y in zip(xs, ys, strict=True):
            p = _sigmoid(a * x + b)
            err = p - y
            grad_a += err * x
            grad_b += err
        a -= lr * grad_a / n
        b -= lr * grad_b / n
    return a, b


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse BTC scalp history CSV")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to history CSV")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit logistic calibration model and print coefficients",
    )
    args = parser.parse_args()

    try:
        with open(args.csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run:  python scripts/btc_scalp_export.py  first.")
        sys.exit(1)

    # Only analyse CLOSED trades with known pnl
    trade_rows = [r for r in rows if r.get("trade_opened") == "1" and r.get("trade_pnl") != ""]
    open_rows  = [r for r in rows if r.get("trade_opened") == "1" and r.get("trade_status") == "open"]
    skip_rows  = [r for r in rows if r.get("trade_opened") == "0"]

    print(f"\n{'='*64}")
    print("  BTC Scalp Historical Analysis")
    print(f"  CSV: {args.csv}")
    print(f"{'='*64}")
    print(f"  Total scan rows:    {len(rows)}")
    print(f"  Trade rows (closed):{len(trade_rows)}")
    print(f"  Trade rows (open):  {len(open_rows)}")
    print(f"  Skip rows:          {len(skip_rows)}")

    if not trade_rows:
        print("\n  No closed trades to analyse. Run more trades first.")
        return

    # Build per-bucket maps: {label → [pnl, ...]}
    by_side:   dict[str, list[float | None]] = defaultdict(list)
    by_min:    dict[str, list[float | None]] = defaultdict(list)
    by_ev:     dict[str, list[float | None]] = defaultdict(list)
    by_p:      dict[str, list[float | None]] = defaultdict(list)
    by_spread: dict[str, list[float | None]] = defaultdict(list)

    for r in trade_rows:
        pnl = _safe_float(r.get("trade_pnl") or "")
        side = r.get("side_selected") or r.get("trade_side") or "unknown"
        ml = _safe_float(r.get("minutes_left") or "")
        p_up = _safe_float(r.get("p_up_model") or "")
        sp = _safe_float(r.get("effective_spread") or "")

        # EV for selected side
        ev_side: float | None = None
        if side == "Up":
            ev_side = _safe_float(r.get("ev_up") or "")
        elif side == "Down":
            ev_side = _safe_float(r.get("ev_down") or "")
        # p for selected side
        p_side: float | None = None
        if p_up is not None:
            p_side = p_up if side == "Up" else (1.0 - p_up)

        by_side[side].append(pnl)
        by_min[_bucket_minutes(ml)].append(pnl)
        by_ev[_bucket_ev(ev_side)].append(pnl)
        by_p[_bucket_p(p_side)].append(pnl)
        by_spread[_bucket_spread(sp)].append(pnl)

    _print_table("Win Rate by Side", by_side)
    _print_table("Win Rate by minutes_left bucket", by_min)
    _print_table("Win Rate by EV bucket (selected side)", by_ev)
    _print_table("Win Rate by p_up / p_side bucket", by_p)
    _print_table("Win Rate by effective spread bucket", by_spread)

    # Skip reason breakdown
    skip_reasons: dict[str, int] = defaultdict(int)
    for r in skip_rows:
        skip_reasons[r.get("skip_reason") or "unknown"] += 1
    print(f"\n{'─'*64}")
    print("  Skip reason breakdown (scan rows that did NOT trade):")
    for reason, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        print(f"    {reason:<32} {cnt:>6}")

    # ── Calibration ─────────────────────────────────────────────────────────
    if args.calibrate:
        # Build (p_model, outcome) pairs for closed trades
        calib_xs: list[float] = []
        calib_ys: list[int] = []
        for r in trade_rows:
            side = r.get("side_selected") or r.get("trade_side") or ""
            p_up = _safe_float(r.get("p_up_model") or "")
            pnl = _safe_float(r.get("trade_pnl") or "")
            if p_up is None or pnl is None:
                continue
            p_side = p_up if side == "Up" else (1.0 - p_up)
            outcome = 1 if pnl > 0 else 0
            calib_xs.append(p_side)
            calib_ys.append(outcome)

        if len(calib_xs) < 10:
            print(f"\n  Calibration skipped — need ≥10 closed trades (have {len(calib_xs)}).")
        else:
            a, b = _logistic_fit(calib_xs, calib_ys)
            print(f"\n{'─'*64}")
            print("  Logistic calibration: P(win) = sigmoid(a * p_model + b)")
            print(f"    a (slope)    = {a:+.4f}")
            print(f"    b (intercept)= {b:+.4f}")
            print()
            print("  Probability adjustment table (p_model → calibrated estimate):")
            print(f"    {'p_model':>8}  {'p_calib':>8}  {'delta':>8}")
            for p in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
                p_cal = _sigmoid(a * p + b)
                print(f"    {p:>8.2f}  {p_cal:>8.4f}  {p_cal - p:>+8.4f}")
            print()
            print("  NOTE: Do NOT auto-apply these coefficients to live logic.")
            print("        Review and validate before any model change.")

    print(f"\n{'='*64}\n")


if __name__ == "__main__":
    main()
