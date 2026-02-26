from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from watchdog.backtest.historical_loader import BeckerHistoricalLoader
from watchdog.backtest.metrics import monte_carlo_drawdown_distribution
from watchdog.core.config import get_settings
from watchdog.risk.kelly import EmpiricalKellySizer


def _print_table(rows: list[dict], columns: list[str], max_rows: int = 30) -> None:
    subset = rows[:max_rows]
    if not subset:
        print("(empty)")
        return

    widths = {c: max(len(c), *(len(str(row.get(c, ""))) for row in subset)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(header)
    print(sep)
    for row in subset:
        print(" | ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Becker/RohOnChain analysis report")
    parser.add_argument("--platform", type=str, default="polymarket")
    parser.add_argument("--min-trades-per-bucket", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    loader = BeckerHistoricalLoader(settings.becker_dataset_path)

    report: dict[str, object] = {
        "platform": args.platform,
        "generated_at": datetime.now(UTC).isoformat(),
    }

    print("=== Becker Analysis ===")
    print(f"Platform: {args.platform}")

    print("\n[1/5] Longshot bias profile (<20% implied)")
    longshot = loader.get_longshot_bias_profile(args.platform)
    longshot_df = longshot.reset_index().rename(
        columns={"implied_probability": "implied_prob", "empirical_win_rate": "empirical_win_rate"}
    )
    longshot_df = longshot_df[longshot_df["implied_prob"] < 0.20].copy()
    longshot_df["mispricing_pct"] = (
        (longshot_df["empirical_win_rate"] - longshot_df["implied_prob"])
        / longshot_df["implied_prob"].clip(lower=1e-6)
    ) * 100
    _print_table(
        longshot_df.round(6).to_dict(orient="records"),
        ["implied_prob", "empirical_win_rate", "mispricing_pct"],
        max_rows=40,
    )
    report["longshot_bias_profile"] = longshot_df.to_dict(orient="records")

    print("\n[2/5] Maker vs taker excess return by price level")
    maker_taker = loader.get_maker_taker_stats(args.platform)
    maker_taker_rows = maker_taker.round(6).to_dict(orient="records")
    _print_table(
        maker_taker_rows,
        [
            "price_level",
            "maker_excess_return",
            "taker_excess_return",
            "maker_n",
            "taker_n",
        ],
        max_rows=40,
    )
    report["maker_taker_stats"] = maker_taker_rows

    print("\n[3/5] 2D calibration surface C(p,t)")
    surface = loader.build_calibration_surface_from_data(
        platform=args.platform,
        min_trades_per_bucket=args.min_trades_per_bucket,
    )
    surface_rows = surface.round(6).to_dict(orient="records")
    _print_table(surface_rows, ["p_bucket", "t_bucket", "domain", "empirical_win_rate", "n_trades"], max_rows=50)
    report["calibration_surface"] = surface_rows

    print("\n[4/5] Empirical Kelly Monte Carlo on strategy analogs")
    analogs = loader.get_strategy_analogs(entry_price_max=0.20, model_prob_min=0.55, platform=args.platform)
    returns = analogs["return_pct"].dropna().to_numpy(dtype=np.float64)
    median_dd, p95_dd, _ = monte_carlo_drawdown_distribution(returns, n_simulations=10_000, percentile=0.95)

    p_model = 0.60
    p_market = 0.20
    full_kelly = EmpiricalKellySizer.full_kelly_fraction(p_model=p_model, p_market=p_market, side="YES")
    print(f"Full Kelly fraction (illustrative): {full_kelly:.4f}")
    print(f"Monte Carlo median drawdown: {median_dd:.4f}")
    print(f"Monte Carlo p95 drawdown: {p95_dd:.4f}")

    report["empirical_kelly"] = {
        "analogs_n": int(analogs.shape[0]),
        "full_kelly_fraction": full_kelly,
        "median_drawdown": median_dd,
        "p95_drawdown": p95_dd,
    }

    print("\n[5/5] Save report")
    out_dir = Path("./becker_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
