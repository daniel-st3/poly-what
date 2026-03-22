"""BTC Scalp Performance Charts and Markdown Report.

Generates PNG charts and a markdown summary from the history CSV produced by
btc_scalp_export.py.  Writes everything to --out-dir (default: data/reports/).

Charts produced:
  1. equity_curve.png          — cumulative PnL over time
  2. pnl_by_trade_number.png   — per-trade PnL bar chart
  3. win_rate_by_ev_bucket.png — win rate (%) by EV bucket
  4. avg_pnl_by_minutes_left.png — avg PnL per minutes_left bucket
  5. entry_price_histogram.png — histogram of effective entry prices
  6. side_distribution.png     — Up vs Down pie / bar
  7. win_loss_by_hour.png      — win/loss count by UTC hour

Report: report.md — text summary with embedded chart links.

Dependencies: matplotlib (pip install matplotlib).

Usage:
    python scripts/btc_scalp_charts.py
    python scripts/btc_scalp_charts.py --csv data/btc_scalp_history.csv --out-dir data/reports
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime

DEFAULT_CSV = "data/btc_scalp_history.csv"
DEFAULT_OUT_DIR = "data/reports"


def _safe_float(v: str | None) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
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


def _parse_dt(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BTC scalp performance charts")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("ERROR: matplotlib not installed.  Run: pip install matplotlib")
        sys.exit(1)

    try:
        with open(args.csv, newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"ERROR: CSV not found: {args.csv}")
        print("Run:  python scripts/btc_scalp_export.py  first.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Filter to closed trades with pnl ────────────────────────────────────
    trade_rows = [
        r for r in rows
        if r.get("trade_opened") == "1"
        and r.get("trade_pnl") not in ("", None)
        and r.get("trade_status") == "closed"
    ]

    if not trade_rows:
        print("No closed trades found — charts will be empty placeholders.")

    # Sort by trade open time (timestamp_utc as proxy)
    def _sort_key(r: dict) -> str:
        return r.get("timestamp_utc") or ""

    trade_rows_sorted = sorted(trade_rows, key=_sort_key)

    pnls = [_safe_float(r["trade_pnl"]) or 0.0 for r in trade_rows_sorted]
    sides = [r.get("side_selected") or r.get("trade_side") or "?" for r in trade_rows_sorted]
    entry_prices = [_safe_float(r.get("effective_ask") or "") for r in trade_rows_sorted]
    open_times = [_parse_dt(r.get("timestamp_utc") or "") for r in trade_rows_sorted]
    minutes_lefts = [_safe_float(r.get("minutes_left") or "") for r in trade_rows_sorted]

    # ── 1. Equity curve ──────────────────────────────────────────────────────
    cumulative = [sum(pnls[: i + 1]) for i in range(len(pnls))]
    fig, ax = plt.subplots(figsize=(10, 4))
    if cumulative:
        ax.plot(range(1, len(cumulative) + 1), cumulative, marker=".", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.fill_between(
            range(1, len(cumulative) + 1),
            0,
            cumulative,
            alpha=0.15,
            color="green" if cumulative[-1] >= 0 else "red",
        )
    ax.set_title("BTC Scalp — Equity Curve (Cumulative PnL)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    fig.tight_layout()
    path1 = os.path.join(args.out_dir, "equity_curve.png")
    fig.savefig(path1, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ── 2. PnL by trade number ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["green" if p >= 0 else "red" for p in pnls]
    if pnls:
        ax.bar(range(1, len(pnls) + 1), pnls, color=colors, width=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("BTC Scalp — PnL by Trade Number")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL ($)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    fig.tight_layout()
    path2 = os.path.join(args.out_dir, "pnl_by_trade_number.png")
    fig.savefig(path2, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path2}")

    # ── 3. Win rate by EV bucket ─────────────────────────────────────────────
    ev_buckets: dict[str, list[float]] = defaultdict(list)
    for r, p in zip(trade_rows_sorted, pnls, strict=True):
        side = r.get("side_selected") or r.get("trade_side") or ""
        ev_val = _safe_float(r.get("ev_up") if side == "Up" else r.get("ev_down"))
        ev_buckets[_bucket_ev(ev_val)].append(p)

    bucket_order = ["<0", "0.00-0.02", "0.02-0.05", "0.05-0.10", ">=0.10", "unknown"]
    labels = [b for b in bucket_order if b in ev_buckets]
    win_rates = [
        sum(1 for p in ev_buckets[b] if p > 0) / len(ev_buckets[b]) * 100
        for b in labels
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, win_rates, color="steelblue")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--", label="50%")
    ax.set_title("BTC Scalp — Win Rate by EV Bucket")
    ax.set_xlabel("EV bucket")
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    path3 = os.path.join(args.out_dir, "win_rate_by_ev_bucket.png")
    fig.savefig(path3, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path3}")

    # ── 4. Avg PnL by minutes_left bucket ───────────────────────────────────
    min_buckets: dict[str, list[float]] = defaultdict(list)
    for ml, p in zip(minutes_lefts, pnls, strict=True):
        min_buckets[_bucket_minutes(ml)].append(p)

    ml_order = ["0-1m", "1-2m", "2-3m", "3-4m", "4m+", "unknown"]
    ml_labels = [b for b in ml_order if b in min_buckets]
    ml_avgs = [sum(min_buckets[b]) / len(min_buckets[b]) for b in ml_labels]
    ml_colors = ["green" if v >= 0 else "red" for v in ml_avgs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ml_labels, ml_avgs, color=ml_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("BTC Scalp — Avg PnL by minutes_left Bucket")
    ax.set_xlabel("Minutes left at entry")
    ax.set_ylabel("Avg PnL ($)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.3f"))
    fig.tight_layout()
    path4 = os.path.join(args.out_dir, "avg_pnl_by_minutes_left.png")
    fig.savefig(path4, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path4}")

    # ── 5. Histogram of effective entry prices ───────────────────────────────
    valid_prices = [p for p in entry_prices if p is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    if valid_prices:
        ax.hist(valid_prices, bins=20, color="steelblue", edgecolor="white")
    ax.set_title("BTC Scalp — Histogram of Effective Entry Prices")
    ax.set_xlabel("Entry price (Polymarket contract, 0-1)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path5 = os.path.join(args.out_dir, "entry_price_histogram.png")
    fig.savefig(path5, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path5}")

    # ── 6. Side distribution ─────────────────────────────────────────────────
    side_counts: dict[str, int] = defaultdict(int)
    for s in sides:
        side_counts[s] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    side_labels = list(side_counts.keys())
    side_vals = [side_counts[s] for s in side_labels]
    ax.bar(side_labels, side_vals, color=["#4c8dc0", "#e07b39", "#aaaaaa"])
    ax.set_title("BTC Scalp — Side Distribution (Up vs Down)")
    ax.set_xlabel("Side")
    ax.set_ylabel("Trade count")
    fig.tight_layout()
    path6 = os.path.join(args.out_dir, "side_distribution.png")
    fig.savefig(path6, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path6}")

    # ── 7. Win/loss by UTC hour ──────────────────────────────────────────────
    hour_wins:   dict[int, int] = defaultdict(int)
    hour_losses: dict[int, int] = defaultdict(int)
    for dt, p in zip(open_times, pnls, strict=True):
        if dt is None:
            continue
        h = dt.hour
        if p > 0:
            hour_wins[h] += 1
        elif p < 0:
            hour_losses[h] += 1

    hours = list(range(24))
    wins_per_hour = [hour_wins.get(h, 0) for h in hours]
    losses_per_hour = [hour_losses.get(h, 0) for h in hours]
    x = range(24)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, wins_per_hour, color="green", alpha=0.7, label="Wins")
    ax.bar(x, [-v for v in losses_per_hour], color="red", alpha=0.7, label="Losses")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("BTC Scalp — Win/Loss Count by UTC Hour")
    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Count (wins up / losses down)")
    ax.set_xticks(list(x))
    ax.legend()
    fig.tight_layout()
    path7 = os.path.join(args.out_dir, "win_loss_by_hour.png")
    fig.savefig(path7, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path7}")

    # ── Markdown report ──────────────────────────────────────────────────────
    total_trades = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    be = total_trades - wins - losses
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / total_trades if total_trades else 0.0
    win_rate = wins / total_trades * 100 if total_trades else 0.0

    up_count = sides.count("Up")
    down_count = sides.count("Down")

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    md = f"""# BTC Scalp Performance Report

*Generated: {generated_at}*
*Source CSV: {args.csv}*

---

## Summary

| Metric | Value |
|---|---|
| Total closed trades | {total_trades} |
| Wins / Losses / BE | {wins} / {losses} / {be} |
| Win rate | {win_rate:.1f}% |
| Total PnL | ${total_pnl:+.4f} |
| Avg PnL per trade | ${avg_pnl:+.4f} |
| Up trades | {up_count} |
| Down trades | {down_count} |

---

## Charts

### 1. Equity Curve
![Equity Curve](equity_curve.png)

**What to look for:** Consistent upward slope = edge. Flat or declining = model not working. Look for drawdown depth and recovery shape.

---

### 2. PnL by Trade Number
![PnL by Trade](pnl_by_trade_number.png)

**What to look for:** Random distribution of green/red bars = healthy variance. Clustering of losses at particular trade numbers may indicate time-of-day or market lifecycle effects.

---

### 3. Win Rate by EV Bucket
![Win Rate by EV](win_rate_by_ev_bucket.png)

**What to look for:** Higher EV buckets should have higher win rates if the model is calibrated. If win rate is flat across EV buckets, the EV calculation may not be predictive.

---

### 4. Avg PnL by minutes_left Bucket
![Avg PnL by minutes left](avg_pnl_by_minutes_left.png)

**What to look for:** Trades entered with more time left may have different expected value than late entries. This tells you the optimal entry window.

---

### 5. Entry Price Histogram
![Entry Price Histogram](entry_price_histogram.png)

**What to look for:** Concentration near 0.50 = fair market pricing. Concentration far from 0.50 = we're consistently entering tilted markets. Both can be fine if the edge is real.

---

### 6. Side Distribution
![Side Distribution](side_distribution.png)

**What to look for:** Strong imbalance (mostly Up or mostly Down) may indicate a momentum bias in the model rather than balanced edge.

---

### 7. Win/Loss by UTC Hour
![Win Loss by Hour](win_loss_by_hour.png)

**What to look for:** If certain hours have consistently higher loss rates, consider restricting trading to better-performing windows.

---

## Next Steps

1. Run `python scripts/btc_scalp_analysis.py --calibrate` for logistic calibration coefficients.
2. Review `[SignalAudit] anomaly=probability_book_mismatch` entries in Railway logs.
3. Once ≥50 closed trades exist, the bucketed metrics become statistically meaningful.
"""

    report_path = os.path.join(args.out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(md)
    print(f"  Saved: {report_path}")
    print(f"\nAll outputs written to: {args.out_dir}/")


if __name__ == "__main__":
    main()
