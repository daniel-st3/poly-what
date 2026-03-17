#!/usr/bin/env python3
"""Walk-forward ARB backtest using CLOB daily price history.

No external dataset download required. Data sources:
  1. watchdog.db          — market slugs + YES token IDs (read-only)
  2. gamma-api            — event → markets grouping (active + closed)
  3. clob.polymarket.com  — daily price history per token

Usage:
    python scripts/run_arb_backtest_clob.py
    python scripts/run_arb_backtest_clob.py --skip-fetch   # reuse existing CSV
    python scripts/run_arb_backtest_clob.py --max-events 500

Output:
    data/price_history.csv           — daily price per token
    data/skipped_tokens.txt          — tokens with empty CLOB history
    results/arb_backtest_monthly.csv — monthly performance
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from watchdog.db.base import Base  # noqa: E402
from watchdog.strategies.intra_event_arb import IntraEventArbScanner  # noqa: E402

# ── API endpoints ─────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH = PROJECT_ROOT / "watchdog.db"
PRICE_HISTORY_CSV = PROJECT_ROOT / "data" / "price_history.csv"
SKIPPED_TOKENS_TXT = PROJECT_ROOT / "data" / "skipped_tokens.txt"
MONTHLY_CSV = PROJECT_ROOT / "results" / "arb_backtest_monthly.csv"

# ── Backtest parameters ────────────────────────────────────────────────────────
RATE_LIMIT_SLEEP = 0.20          # 5 req/sec
FEE_RATE = 0.005                 # config: backtest_fee_rate
SLIPPAGE_RATE = 0.005            # config: backtest_slippage_rate
MIN_DAYS_REQUIRED = 30
STARTING_CAPITALS = [100.0, 500.0]

# Fix 1: cap yes_sum — anything above 1.10 is almost certainly a non-exclusive
# multi-outcome market (e.g. "who will win the NHL championship" with 32 teams)
# where the sum legitimately exceeds 1 and is NOT a real arb opportunity.
MAX_YES_SUM = 1.10

# Fix 2: require at least $50k combined lifetime volume across all legs before
# taking a position — eliminates illiquid bracket/award markets entirely.
MIN_EVENT_VOLUME_BACKTEST = 50_000.0

# Fix 3: fixed fractional sizing — NEVER compounds.
# $100 portfolio → max $1.00/trade, $500 portfolio → max $5.00/trade.
FIXED_POSITIONS: dict[float, float] = {100.0: 1.0, 500.0: 5.0}

# ── Fill uncertainty model ─────────────────────────────────────────────────────
# Real arb fills depend on latency and book depth you can't control.
FILL_RATE = 0.50             # 50% of signals actually fill
TRANSACTION_COST_RATE = 0.0002  # 0.02% per filled trade ($1 → $0.0002 cost)
ADVERSE_MOVE_RATE = 0.10     # 10% of fills get adverse price movement
ADVERSE_MOVE_LOSS = 0.05     # 5% of position lost on adverse fills
BACKTEST_SEED = 42           # fixed seed for reproducibility


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Load DB markets
# ═════════════════════════════════════════════════════════════════════════════

def load_db_markets() -> dict[str, dict[str, str]]:
    """Return {yes_token_id: {slug, condition_id}} for tokens in watchdog.db."""
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True)
    result: dict[str, dict[str, str]] = {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT slug, yes_token_id, condition_id "
                "FROM markets "
                "WHERE length(yes_token_id) > 30"
            )
        ).fetchall()
    for slug, token_id, cond_id in rows:
        if token_id:
            result[token_id] = {"slug": slug or "", "condition_id": cond_id or ""}
    print(f"  DB: {len(result)} markets with valid token IDs")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2 — Fetch Gamma API events (active + closed)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_clob_token_ids(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    return list(raw) if isinstance(raw, list) else []


def _fetch_events_page(closed: bool, offset: int, limit: int = 100) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if closed:
        params["closed"] = "true"
    else:
        params["active"] = "true"
        params["closed"] = "false"
    resp = httpx.get(f"{GAMMA_API}/events", params=params, timeout=20.0)
    resp.raise_for_status()
    return resp.json() or []


def fetch_gamma_events(max_events: int) -> list[dict[str, Any]]:
    """Paginate both active and closed events; keep those with 2+ markets."""
    events: list[dict[str, Any]] = []
    per_half = max_events // 2

    for closed, label in [(False, "active"), (True, "closed")]:
        offset = 0
        fetched = 0
        while fetched < per_half:
            try:
                batch = _fetch_events_page(closed=closed, offset=offset)
            except Exception as exc:
                print(f"  WARNING: Gamma events ({label}) fetch error at offset {offset}: {exc}")
                break
            if not batch:
                break

            for item in batch:
                raw_markets = item.get("markets") or []
                if len(raw_markets) < 2:
                    continue
                markets_out = []
                for m in raw_markets:
                    clob_ids = _parse_clob_token_ids(m.get("clobTokenIds"))
                    yes_token = clob_ids[0] if clob_ids else None
                    markets_out.append(
                        {
                            "slug": m.get("slug") or m.get("conditionId", ""),
                            "question": m.get("question") or "",
                            "condition_id": m.get("conditionId") or "",
                            "outcome_prices": m.get("outcomePrices") or [],
                            "volume": float(m.get("volume") or 0),
                            "yes_token_id": yes_token,
                        }
                    )
                if len(markets_out) >= 2:
                    events.append(
                        {
                            "event_slug": item.get("slug") or item.get("id", ""),
                            "title": item.get("title") or "",
                            "markets": markets_out,
                        }
                    )

            fetched += len(batch)
            offset += len(batch)
            if len(batch) < 100:
                break
            time.sleep(0.05)  # gentle on Gamma API

        print(f"  Gamma {label}: fetched {fetched} raw events")

    print(f"  Total multi-outcome events kept: {len(events)}")
    return events


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3 — Fetch CLOB price history
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_clob_history(token_id: str) -> dict[date, float]:
    """Return {date: price}. Returns {} on empty/error."""
    resp = httpx.get(
        f"{CLOB_API}/prices-history",
        params={"interval": "max", "fidelity": "1440", "market": token_id},
        timeout=15.0,
    )
    resp.raise_for_status()
    hist = resp.json().get("history") or []
    result: dict[date, float] = {}
    for entry in hist:
        ts = entry.get("t")
        price = entry.get("p")
        if ts is not None and price is not None:
            d = datetime.fromtimestamp(ts).date()
            result[d] = float(price)  # last price for each date wins
    return result


def fetch_price_history(
    token_ids: list[str],
    db_markets: dict[str, dict[str, str]],
    skip_fetch: bool,
) -> dict[str, dict[date, float]]:
    """Fetch CLOB price history for all tokens; save to data/price_history.csv."""
    PRICE_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    SKIPPED_TOKENS_TXT.parent.mkdir(parents=True, exist_ok=True)

    if skip_fetch and PRICE_HISTORY_CSV.exists():
        print(f"  --skip-fetch: loading existing {PRICE_HISTORY_CSV}")
        return _load_price_history_csv()

    # Deduplicate
    unique_tokens = list(dict.fromkeys(t for t in token_ids if t and len(t) > 30))
    print(f"  Fetching CLOB history for {len(unique_tokens)} unique tokens …")

    all_rows: list[dict[str, str]] = []
    skipped: list[str] = []
    price_cache: dict[str, dict[date, float]] = {}

    for i, token_id in enumerate(unique_tokens, 1):
        try:
            history = _fetch_clob_history(token_id)
        except Exception:
            history = {}

        if not history:
            skipped.append(token_id)
        else:
            slug = db_markets.get(token_id, {}).get("slug", "")
            price_cache[token_id] = history
            for d, p in sorted(history.items()):
                all_rows.append(
                    {
                        "market_slug": slug,
                        "token_id": token_id,
                        "date": d.isoformat(),
                        "price": str(p),
                    }
                )

        if i % 100 == 0:
            print(f"    {i}/{len(unique_tokens)} fetched, {len(skipped)} skipped so far")

        time.sleep(RATE_LIMIT_SLEEP)

    # Write CSV
    with PRICE_HISTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["market_slug", "token_id", "date", "price"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Write skipped log
    with SKIPPED_TOKENS_TXT.open("w", encoding="utf-8") as f:
        f.write("\n".join(skipped))

    print(
        f"  Done: {len(price_cache)} tokens with history, "
        f"{len(skipped)} skipped → {PRICE_HISTORY_CSV}"
    )
    return price_cache


def _load_price_history_csv() -> dict[str, dict[date, float]]:
    cache: dict[str, dict[date, float]] = {}
    with PRICE_HISTORY_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["token_id"]
            d = date.fromisoformat(row["date"])
            p = float(row["price"])
            cache.setdefault(tid, {})[d] = p
    print(f"  Loaded {len(cache)} tokens from {PRICE_HISTORY_CSV}")
    return cache


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4 — Build daily events from price history
# ═════════════════════════════════════════════════════════════════════════════

def build_daily_events(
    events: list[dict[str, Any]],
    price_cache: dict[str, dict[date, float]],
) -> dict[date, list[dict[str, Any]]]:
    """
    For each event x day where ALL markets have a CLOB price, build an event
    dict (outcome_prices replaced by that day's price) in the format that
    IntraEventArbScanner._process_event expects.

    No look-ahead: prices on day D come only from CLOB history entries for day D.
    """
    daily: dict[date, list[dict[str, Any]]] = defaultdict(list)
    events_used = 0

    for event in events:
        markets = event["markets"]
        # Only keep markets that have CLOB price history
        markets_with_history = [
            m for m in markets
            if m.get("yes_token_id") and m["yes_token_id"] in price_cache
        ]
        if len(markets_with_history) < 2:
            continue

        events_used += 1
        # Collect all dates where EVERY market in this event has a price
        date_sets = [set(price_cache[m["yes_token_id"]].keys()) for m in markets_with_history]
        common_dates = date_sets[0].intersection(*date_sets[1:])

        for d in common_dates:
            day_markets = [
                {
                    "slug": m["slug"],
                    "question": m["question"],
                    "condition_id": m["condition_id"],
                    "outcome_prices": [price_cache[m["yes_token_id"]][d]],
                    "volume": m["volume"],
                    "yes_token_id": m["yes_token_id"],
                }
                for m in markets_with_history
            ]
            daily[d].append(
                {"event_slug": event["event_slug"], "markets": day_markets}
            )

    all_dates = sorted(daily.keys())
    print(
        f"  {events_used} events with ≥2 markets in price cache, "
        f"{len(all_dates)} total unique days"
    )
    return dict(daily)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5 — Dual portfolio backtest
# ═════════════════════════════════════════════════════════════════════════════

def _make_memory_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _prefilter_events(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Apply yes_sum cap and volume floor before passing events to the scanner.

    Returns (kept_events, n_filtered_yes_sum, n_filtered_volume).
    """
    kept: list[dict[str, Any]] = []
    n_yes_sum = 0
    n_volume = 0
    for evt in events:
        markets = evt.get("markets") or []
        yes_prices: list[float] = []
        total_volume = 0.0
        for m in markets:
            op = m.get("outcome_prices") or []
            if isinstance(op, list) and op:
                try:
                    yes_prices.append(float(op[0]))
                except (TypeError, ValueError):
                    pass
            total_volume += float(m.get("volume") or 0)

        if not yes_prices:
            continue

        # Fix 1: reject non-exclusive outcome bundles
        if sum(yes_prices) > MAX_YES_SUM:
            n_yes_sum += 1
            continue

        # Fix 2: reject illiquid events
        if total_volume < MIN_EVENT_VOLUME_BACKTEST:
            n_volume += 1
            continue

        kept.append(evt)

    return kept, n_yes_sum, n_volume


def run_dual_portfolio_backtest(
    daily_events: dict[date, list[dict[str, Any]]],
    min_profit_cents: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Run the live IntraEventArbScanner with historical_mode=True for each day.
    Returns a list of daily records (one per day that had events), including
    portfolio balances for each starting capital.
    """
    scanner = IntraEventArbScanner()
    sorted_days = sorted(daily_events.keys())
    rng = random.Random(BACKTEST_SEED)  # seeded for reproducibility

    # Portfolio state per starting capital
    balances = {cap: cap for cap in STARTING_CAPITALS}
    peaks = {cap: cap for cap in STARTING_CAPITALS}

    daily_records: list[dict[str, Any]] = []

    for day in sorted_days:
        raw_events = daily_events[day]

        # Apply pre-filters (Fix 1 + Fix 2) before the scanner sees the events
        events, n_filtered_yes_sum, n_filtered_volume = _prefilter_events(raw_events)

        session = _make_memory_session()
        result = scanner.scan(
            session=session,
            rest_client=None,
            is_paper=False,
            events=events,
            historical_mode=True,
        )
        session.close()

        opps = result.get("opportunities_detail", [])
        n_opportunities = result["opportunities"]

        # Apply each opportunity to both portfolios (Fix 3: fixed position sizes)
        day_profit: dict[float, float] = dict.fromkeys(STARTING_CAPITALS, 0.0)
        trades_taken = 0
        n_below_min = 0   # rejected by --min-profit-cents
        n_micro = 0       # profit_cents < 5 (always tracked regardless of filter)

        for opp in opps:
            profit_cents = opp["profit_cents"]

            # Track micro-signals for distribution reporting (before any filter)
            if profit_cents < 5.0:
                n_micro += 1

            # --min-profit-cents filter
            if profit_cents < min_profit_cents:
                n_below_min += 1
                continue

            # Fill uncertainty model — fill decision is the same for both portfolios
            # (it's an external market event, not portfolio-specific).
            if rng.random() >= FILL_RATE:
                continue  # did not fill — latency/depth miss

            trades_taken += 1
            is_adverse = rng.random() < ADVERSE_MOVE_RATE  # market moved before fill

            for cap in STARTING_CAPITALS:
                position = FIXED_POSITIONS[cap]
                if is_adverse:
                    # Adverse fill: lose 5% of position size
                    net = -(position * ADVERSE_MOVE_LOSS)
                else:
                    gross = position * profit_cents / 100.0
                    cost = position * TRANSACTION_COST_RATE
                    net = gross - cost
                balances[cap] += net
                day_profit[cap] += net

        # Update peak
        for cap in STARTING_CAPITALS:
            if balances[cap] > peaks[cap]:
                peaks[cap] = balances[cap]

        # Drawdowns
        drawdowns = {
            cap: (peaks[cap] - balances[cap]) / peaks[cap]
            for cap in STARTING_CAPITALS
        }

        daily_records.append(
            {
                "date": day,
                "events_scanned": result["events_scanned"],
                "opportunities": n_opportunities,
                "trades_taken": trades_taken,
                "events_filtered_yes_sum": n_filtered_yes_sum,
                "events_filtered_volume": n_filtered_volume,
                "n_below_min_profit": n_below_min,
                "n_micro_signals": n_micro,
                **{f"profit_{cap}": day_profit[cap] for cap in STARTING_CAPITALS},
                **{f"balance_{cap}": balances[cap] for cap in STARTING_CAPITALS},
                **{f"drawdown_{cap}": drawdowns[cap] for cap in STARTING_CAPITALS},
            }
        )

    return daily_records


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6 — Monthly aggregation
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_monthly(daily_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    monthly: dict[str, dict[str, Any]] = {}

    for rec in daily_records:
        month = rec["date"].strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {
                "month": month,
                "opportunities": 0,
                "trades_taken": 0,
                "events_filtered_yes_sum": 0,
                "events_filtered_volume": 0,
                **{f"gross_profit_{cap}": 0.0 for cap in STARTING_CAPITALS},
                **{f"net_balance_{cap}": 0.0 for cap in STARTING_CAPITALS},
                **{f"max_drawdown_{cap}": 0.0 for cap in STARTING_CAPITALS},
                "best_day": rec["date"].isoformat(),
                "worst_day": rec["date"].isoformat(),
                "_best_profit": -math.inf,
                "_worst_profit": math.inf,
            }
        m = monthly[month]
        m["opportunities"] += rec["opportunities"]
        m["trades_taken"] += rec["trades_taken"]
        m["events_filtered_yes_sum"] += rec["events_filtered_yes_sum"]
        m["events_filtered_volume"] += rec["events_filtered_volume"]

        # Use $100 portfolio as the reference for best/worst day
        ref_cap = STARTING_CAPITALS[0]
        day_profit = rec[f"profit_{ref_cap}"]
        if day_profit > m["_best_profit"]:
            m["_best_profit"] = day_profit
            m["best_day"] = rec["date"].isoformat()
        if day_profit < m["_worst_profit"]:
            m["_worst_profit"] = day_profit
            m["worst_day"] = rec["date"].isoformat()

        for cap in STARTING_CAPITALS:
            m[f"gross_profit_{cap}"] += rec[f"profit_{cap}"]
            m[f"net_balance_{cap}"] = rec[f"balance_{cap}"]  # end-of-month balance
            m[f"max_drawdown_{cap}"] = max(
                m[f"max_drawdown_{cap}"], rec[f"drawdown_{cap}"]
            )

    # Compute return_pct using start-of-month balance
    rows: list[dict[str, Any]] = []
    prev_balance: dict[float, float] = {cap: cap for cap in STARTING_CAPITALS}

    for month_key in sorted(monthly.keys()):
        m = monthly[month_key]
        out: dict[str, Any] = {
            "month": m["month"],
            "opportunities": m["opportunities"],
            "trades_taken": m["trades_taken"],
            "events_filtered_yes_sum": m["events_filtered_yes_sum"],
            "events_filtered_volume": m["events_filtered_volume"],
            "best_day": m["best_day"],
            "worst_day": m["worst_day"],
        }
        for cap in STARTING_CAPITALS:
            end_bal = m[f"net_balance_{cap}"]
            start_bal = prev_balance[cap]
            ret_pct = (end_bal - start_bal) / start_bal * 100 if start_bal > 0 else 0.0
            label = str(int(cap))
            out[f"gross_profit_usd_{label}"] = round(m[f"gross_profit_{cap}"], 4)
            out[f"net_balance_{label}"] = round(end_bal, 4)
            out[f"return_pct_{label}"] = round(ret_pct, 4)
            out[f"max_drawdown_pct_{label}"] = round(m[f"max_drawdown_{cap}"] * 100, 4)
            prev_balance[cap] = end_bal

        rows.append(out)

    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7 — Sharpe + final stats
# ═════════════════════════════════════════════════════════════════════════════

def compute_sharpe(daily_records: list[dict[str, Any]], cap: float) -> float:
    returns = []
    prev = cap
    for rec in daily_records:
        bal = rec[f"balance_{cap}"]
        if prev > 0:
            returns.append((bal - prev) / prev)
        prev = bal
    if len(returns) < 2:
        return 0.0
    import statistics
    mean_r = statistics.mean(returns)
    std_r = statistics.stdev(returns)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(252)


def compute_max_drawdown(daily_records: list[dict[str, Any]], cap: float) -> float:
    return max((rec[f"drawdown_{cap}"] for rec in daily_records), default=0.0)


def find_best_worst_month(monthly_rows: list[dict[str, Any]], label: str) -> tuple[str, str]:
    best = max(monthly_rows, key=lambda r: r[f"gross_profit_usd_{label}"])
    worst = min(monthly_rows, key=lambda r: r[f"gross_profit_usd_{label}"])
    return best["month"], worst["month"]


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="ARB backtest using CLOB price history.")
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip CLOB API calls; reuse data/price_history.csv if it exists.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=2000,
        help="Max Gamma API events to fetch (split evenly active/closed). Default: 2000.",
    )
    parser.add_argument(
        "--min-profit-cents",
        type=float,
        default=0.0,
        help="Reject signals with profit_cents below this threshold. Default: 0 (no filter).",
    )
    args = parser.parse_args()

    MONTHLY_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Load DB markets ──────────────────────────────────────────────
    print("\n[1/6] Loading markets from watchdog.db …")
    if not DB_PATH.exists():
        print(f"ERROR: watchdog.db not found at {DB_PATH}")
        sys.exit(1)
    db_markets = load_db_markets()

    # ── Phase 2: Fetch Gamma events ───────────────────────────────────────────
    print(f"\n[2/6] Fetching Gamma events (max {args.max_events}) …")
    events = fetch_gamma_events(max_events=args.max_events)

    # Collect all token IDs: from DB + from events
    event_token_ids = [
        m["yes_token_id"]
        for evt in events
        for m in evt["markets"]
        if m.get("yes_token_id")
    ]
    all_token_ids = list(dict.fromkeys(list(db_markets.keys()) + event_token_ids))
    print(f"  Total unique token IDs to fetch: {len(all_token_ids)}")

    # ── Phase 3: Fetch CLOB price history ─────────────────────────────────────
    print(f"\n[3/6] Fetching CLOB price history ({'' if not args.skip_fetch else 'SKIP — '}5 req/sec) …")
    price_cache = fetch_price_history(
        token_ids=all_token_ids,
        db_markets=db_markets,
        skip_fetch=args.skip_fetch,
    )

    # ── Phase 4: Build daily event snapshots ─────────────────────────────────
    print("\n[4/6] Building daily event datasets …")
    daily_events = build_daily_events(events, price_cache)

    all_days = sorted(daily_events.keys())
    if len(all_days) < MIN_DAYS_REQUIRED:
        print(
            f"\nWARNING: Only {len(all_days)} days of data available "
            f"(minimum {MIN_DAYS_REQUIRED} required). Exiting."
        )
        sys.exit(0)

    print(f"  Date range: {all_days[0]} → {all_days[-1]} ({len(all_days)} days)")

    # ── Phase 5: Run backtest ─────────────────────────────────────────────────
    print("\n[5/6] Running dual portfolio backtest ($100 / $500) …")
    if args.min_profit_cents > 0:
        print(f"  --min-profit-cents={args.min_profit_cents} filter active")
    daily_records = run_dual_portfolio_backtest(
        daily_events, min_profit_cents=args.min_profit_cents
    )

    total_opps = sum(r["opportunities"] for r in daily_records)
    days_with_opps = sum(1 for r in daily_records if r["opportunities"] > 0)
    days_zero = len(daily_records) - days_with_opps
    total_filtered_yes_sum = sum(r["events_filtered_yes_sum"] for r in daily_records)
    total_filtered_volume = sum(r["events_filtered_volume"] for r in daily_records)
    total_raw = total_opps + total_filtered_yes_sum + total_filtered_volume
    total_micro = sum(r["n_micro_signals"] for r in daily_records)
    total_below_min = sum(r["n_below_min_profit"] for r in daily_records)
    pct_micro = total_micro / total_opps * 100 if total_opps else 0.0

    # ── Phase 6: Aggregate and write output ───────────────────────────────────
    print("\n[6/6] Aggregating monthly results …")
    monthly_rows = aggregate_monthly(daily_records)

    fieldnames = [
        "month", "opportunities", "trades_taken",
        "events_filtered_yes_sum", "events_filtered_volume",
        "gross_profit_usd_100", "net_balance_100", "return_pct_100", "max_drawdown_pct_100",
        "gross_profit_usd_500", "net_balance_500", "return_pct_500", "max_drawdown_pct_500",
        "best_day", "worst_day",
    ]
    with MONTHLY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(monthly_rows)

    # ── Console summary ───────────────────────────────────────────────────────
    for cap in STARTING_CAPITALS:
        label = str(int(cap))
        final_bal = daily_records[-1][f"balance_{cap}"]
        total_return = (final_bal - cap) / cap * 100
        sharpe = compute_sharpe(daily_records, cap)
        max_dd = compute_max_drawdown(daily_records, cap) * 100
        best_month, worst_month = find_best_worst_month(monthly_rows, label)
        best_profit = max(r[f"gross_profit_usd_{label}"] for r in monthly_rows)
        worst_profit = min(r[f"gross_profit_usd_{label}"] for r in monthly_rows)

        print(f"\n  ${cap:,.0f} Portfolio:")
        print(f"    Start:        ${cap:>10.2f}")
        print(f"    End:          ${final_bal:>10.2f}")
        sign = "+" if total_return >= 0 else ""
        print(f"    Total return: {sign}{total_return:.1f}%")
        sign = "+" if best_profit >= 0 else ""
        print(f"    Best month:   {best_month}  {sign}${best_profit:.2f}")
        sign = "+" if worst_profit >= 0 else ""
        print(f"    Worst month:  {worst_month}  {sign}${worst_profit:.2f}")
        print(f"    Max drawdown: -{max_dd:.1f}%")
        print(f"    Sharpe ratio: {sharpe:.2f}")

    avg_opps = total_opps / len(daily_records) if daily_records else 0.0
    pct_yes_sum = total_filtered_yes_sum / total_raw * 100 if total_raw else 0.0
    pct_volume  = total_filtered_volume  / total_raw * 100 if total_raw else 0.0
    pct_kept    = total_opps             / total_raw * 100 if total_raw else 0.0
    print(f"\n  Total days tested:     {len(daily_records)}")
    print(f"  Avg opportunities/day: {avg_opps:.1f}")
    print(f"  Days with 0 opps:      {days_zero}")
    print(f"\n  Signal quality (of {total_raw} raw candidates):")
    print(f"    Rejected by yes_sum > {MAX_YES_SUM}: {total_filtered_yes_sum:>6} ({pct_yes_sum:.1f}%)")
    print(f"    Rejected by volume  < ${MIN_EVENT_VOLUME_BACKTEST:,.0f}: {total_filtered_volume:>6} ({pct_volume:.1f}%)")
    print(f"    Kept (real signals): {total_opps:>6} ({pct_kept:.1f}%)")
    print(f"\n  Position sizing:  $100→${FIXED_POSITIONS[100.0]:.2f}/trade  $500→${FIXED_POSITIONS[500.0]:.2f}/trade (fixed)")
    total_filled = sum(r["trades_taken"] for r in daily_records)
    total_sigs = sum(r["opportunities"] for r in daily_records)
    print(f"  Fill model:       {FILL_RATE*100:.0f}% fill rate → {total_filled}/{total_sigs} filled  |  "
          f"{ADVERSE_MOVE_RATE*100:.0f}% adverse ({ADVERSE_MOVE_LOSS*100:.0f}% loss)  |  "
          f"{TRANSACTION_COST_RATE*100:.4f}% tx cost  (seed={BACKTEST_SEED})")
    print(f"\n  Profit distribution (of {total_opps} signals after yes_sum + volume filters):")
    print(f"    Micro (<5¢):     {total_micro:>5} ({pct_micro:.1f}%)")
    print(f"    Viable (≥5¢):    {total_opps - total_micro:>5} ({100-pct_micro:.1f}%)")
    if total_below_min > 0:
        print(f"    Filtered by --min-profit-cents {args.min_profit_cents}¢: {total_below_min}")
    print(f"\n  Monthly CSV → {MONTHLY_CSV}")


if __name__ == "__main__":
    print("=" * 50)
    print("  ARB BACKTEST — CLOB price history")
    print("=" * 50)
    main()
    print("\n" + "=" * 50)
