"""Walk-forward backtest for intra-event arb using Kaggle Polymarket CSV data.

Usage
-----
# Run with a local CSV file:
    python scripts/backtest_arb.py --csv /path/to/markets.csv

# Download from Kaggle first (requires kaggle CLI + API key):
    kaggle datasets download <dataset-slug> -p /tmp/kaggle --unzip
    python scripts/backtest_arb.py --csv /tmp/kaggle/markets.csv

# Override output path:
    python scripts/backtest_arb.py --csv markets.csv --out results/arb_backtest.csv

# Filter to a specific date range:
    python scripts/backtest_arb.py --csv markets.csv --start 2024-01-01 --end 2024-06-30

Field Mapping (Kaggle CSV → internal event dict)
-------------------------------------------------
Kaggle column    Internal field              Notes
───────────────  ──────────────────────────  ────────────────────────────────────────────
event_slug       event["event_slug"]         Rows with same event_slug are grouped into
                                             one event dict with a "markets" list.
condition_id     market["condition_id"]      Also used as market slug fallback.
outcome_prices   market["outcome_prices"]    JSON string "[0.65, 0.35]" or "[0.70]".
                                             _process_event handles json.loads internally.
volume           market["volume"]            Dollar volume; must be numeric.
slug             market["slug"]              Optional; falls back to condition_id.
question         market["question"]          Optional; falls back to slug.
end_date         used to assign day bucket   ISO datetime string; date portion used.

The backtest uses an in-memory SQLite database so it never touches watchdog.db.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

# ── SQLAlchemy in-memory DB setup ─────────────────────────────────────────────
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project src to path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from watchdog.db.base import Base
from watchdog.strategies.intra_event_arb import IntraEventArbScanner

# ── Constants ─────────────────────────────────────────────────────────────────
# Kaggle column names (adjust here if the dataset uses different headers).
COL_EVENT_SLUG = "event_slug"
COL_CONDITION_ID = "condition_id"
COL_OUTCOME_PRICES = "outcome_prices"
COL_VOLUME = "volume"
COL_SLUG = "slug"
COL_QUESTION = "question"
COL_END_DATE = "end_date"  # also accept "endDate" (normalised below)

# Fallback column aliases (Kaggle sometimes uses camelCase).
_ALIASES: dict[str, list[str]] = {
    COL_EVENT_SLUG: ["event_slug", "eventSlug", "groupItemSlug"],
    COL_CONDITION_ID: ["condition_id", "conditionId"],
    COL_OUTCOME_PRICES: ["outcome_prices", "outcomePrices"],
    COL_VOLUME: ["volume", "volumeNum"],
    COL_SLUG: ["slug"],
    COL_QUESTION: ["question"],
    COL_END_DATE: ["end_date", "endDate", "closedTime", "closed_time"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_columns(header: list[str]) -> dict[str, str]:
    """Return {canonical_name: actual_csv_column} for every recognised field."""
    header_set = set(header)
    mapping: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in header_set:
                mapping[canonical] = alias
                break
    return mapping


def _parse_date(value: str) -> date | None:
    """Best-effort ISO date parse; returns None on failure."""
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value[:len(fmt)], fmt).date()
        except ValueError:
            continue
    return None


def _get(row: dict[str, str], col_map: dict[str, str], canonical: str, default: str = "") -> str:
    actual = col_map.get(canonical)
    if actual is None:
        return default
    return row.get(actual, default)


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path: str) -> list[dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def build_event_list_for_day(
    rows: list[dict[str, str]],
    col_map: dict[str, str],
    day: date,
) -> list[dict]:
    """Group rows for a given day into the event dict format scan() expects."""
    # Collect markets that belong to this day.
    day_markets: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        raw_date = _get(row, col_map, COL_END_DATE)
        row_date = _parse_date(raw_date)
        if row_date != day:
            continue

        event_slug = _get(row, col_map, COL_EVENT_SLUG)
        if not event_slug:
            # Use first 60 chars of condition_id as a synthetic event slug.
            event_slug = _get(row, col_map, COL_CONDITION_ID)[:60]
        if not event_slug:
            continue

        market_dict: dict = {
            "slug": _get(row, col_map, COL_SLUG) or _get(row, col_map, COL_CONDITION_ID),
            "question": _get(row, col_map, COL_QUESTION),
            "condition_id": _get(row, col_map, COL_CONDITION_ID),
            "outcome_prices": _get(row, col_map, COL_OUTCOME_PRICES),
            "volume": _get(row, col_map, COL_VOLUME) or "0",
            "yes_token_id": None,
        }
        day_markets[event_slug].append(market_dict)

    events = [
        {"event_slug": slug, "markets": mkts}
        for slug, mkts in day_markets.items()
    ]
    return events


# ── In-memory SQLite session ──────────────────────────────────────────────────

def make_memory_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    csv_path: str,
    out_path: str,
    start_date: date | None,
    end_date: date | None,
) -> None:
    print(f"Loading {csv_path} …")
    rows = load_csv(csv_path)
    if not rows:
        print("ERROR: CSV is empty.")
        sys.exit(1)

    col_map = _resolve_columns(list(rows[0].keys()))
    print(f"Detected columns: {col_map}")

    required = [COL_EVENT_SLUG, COL_OUTCOME_PRICES, COL_VOLUME, COL_CONDITION_ID]
    missing = [c for c in required if c not in col_map]
    if missing:
        print(
            f"ERROR: required columns not found: {missing}\n"
            f"Available headers: {list(rows[0].keys())}"
        )
        sys.exit(1)

    # Collect all unique dates present in the CSV.
    all_dates: set[date] = set()
    for row in rows:
        raw = _get(row, col_map, COL_END_DATE)
        d = _parse_date(raw)
        if d:
            all_dates.add(d)

    if not all_dates:
        print("ERROR: no parseable dates found in end_date/endDate column.")
        sys.exit(1)

    sorted_dates = sorted(all_dates)
    if start_date:
        sorted_dates = [d for d in sorted_dates if d >= start_date]
    if end_date:
        sorted_dates = [d for d in sorted_dates if d <= end_date]

    print(
        f"Date range: {sorted_dates[0]} → {sorted_dates[-1]} "
        f"({len(sorted_dates)} days, {len(rows)} total rows)"
    )

    scanner = IntraEventArbScanner()
    output_rows: list[dict] = []

    for day in sorted_dates:
        session = make_memory_session()  # fresh isolated DB per day
        events = build_event_list_for_day(rows, col_map, day)

        if not events:
            session.close()
            continue

        result = scanner.scan(
            session=session,
            rest_client=None,
            is_paper=False,
            events=events,
            historical_mode=True,
        )
        session.close()

        opps = result.get("opportunities_detail", [])
        opportunities_found = result["opportunities"]
        estimated_profit_usd = result["est_daily_profit_cents"] / 100.0
        would_have_traded = opportunities_found > 0

        output_rows.append(
            {
                "date": day.isoformat(),
                "events_scanned": result["events_scanned"],
                "opportunities_found": opportunities_found,
                "estimated_profit_usd": round(estimated_profit_usd, 4),
                "would_have_traded": would_have_traded,
                "top_event_slug": opps[0]["event_slug"] if opps else "",
                "top_arb_type": opps[0]["arb_type"] if opps else "",
                "top_yes_sum": round(opps[0]["yes_sum"], 4) if opps else "",
                "top_profit_cents": round(opps[0]["profit_cents"], 2) if opps else "",
            }
        )

        if opportunities_found:
            print(
                f"  {day}  events={result['events_scanned']:>4}  "
                f"opps={opportunities_found}  "
                f"est_profit=${estimated_profit_usd:.4f}"
            )

    if not output_rows:
        print("No data to write — no events passed filters.")
        sys.exit(0)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    total_days_with_opps = sum(1 for r in output_rows if r["would_have_traded"])
    total_profit = sum(r["estimated_profit_usd"] for r in output_rows)
    print(
        f"\nDone. {len(output_rows)} days analysed, "
        f"{total_days_with_opps} days with opportunities, "
        f"total estimated profit ${total_profit:.4f}\n"
        f"Output: {out_path}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward arb backtest using Kaggle Polymarket CSV data."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the Kaggle Polymarket markets CSV file.",
    )
    parser.add_argument(
        "--out",
        default="results/arb_backtest.csv",
        help="Output CSV path (default: results/arb_backtest.csv).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date filter ISO format YYYY-MM-DD (inclusive).",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date filter ISO format YYYY-MM-DD (inclusive).",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    run_backtest(
        csv_path=args.csv,
        out_path=args.out,
        start_date=start,
        end_date=end,
    )


if __name__ == "__main__":
    main()
