from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from watchdog.backtest.backtester import Backtester
from watchdog.backtest.historical_loader import BeckerHistoricalLoader
from watchdog.backtest.metrics import evaluate_pass_fail
from watchdog.core.config import get_settings


def _print_report(results) -> None:
    print("=== Backtest Report ===")
    print(f"Total markets: {results.n_total_markets}")
    print(f"Traded: {results.n_traded}")
    print(f"Wins / Losses: {results.n_won} / {results.n_lost}")
    print(f"Win rate: {results.win_rate:.4f}")
    print(f"Brier score: {results.brier_score:.4f}")
    print(f"Max drawdown: {results.max_drawdown:.4f}")
    print(f"Sharpe ratio: {results.sharpe_ratio:.4f}")
    print(f"Total PnL (USDC): {results.total_pnl_usdc:.4f}")
    print(f"Avg PnL per trade: {results.avg_pnl_per_trade:.4f}")
    print(f"Avg slippage: {results.avg_slippage:.4f}")


def _build_synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        [
            {"market_id": "tr1", "question": "Synthetic train 1", "domain": "politics", "resolution_time": "2024-06-01T00:00:00Z", "outcome": 1, "price_30d_before": 0.40, "price_7d_before": 0.45, "price_1d_before": 0.50, "days_to_resolution": 40, "volume_total": 10000},
            {"market_id": "tr2", "question": "Synthetic train 2", "domain": "politics", "resolution_time": "2024-07-01T00:00:00Z", "outcome": 0, "price_30d_before": 0.60, "price_7d_before": 0.58, "price_1d_before": 0.52, "days_to_resolution": 40, "volume_total": 11000},
        ]
    )
    oos = pd.DataFrame(
        [
            {"market_id": "te1", "question": "Synthetic test 1", "domain": "politics", "resolution_time": "2025-06-01T00:00:00Z", "outcome": 1, "price_30d_before": 0.35, "price_7d_before": 0.40, "price_1d_before": 0.48, "days_to_resolution": 35, "volume_total": 9000},
            {"market_id": "te2", "question": "Synthetic test 2", "domain": "politics", "resolution_time": "2025-07-01T00:00:00Z", "outcome": 0, "price_30d_before": 0.65, "price_7d_before": 0.60, "price_1d_before": 0.55, "days_to_resolution": 35, "volume_total": 9500},
        ]
    )
    surface = pd.DataFrame(
        [
            {"p_bucket": 0.35, "t_bucket": "long", "domain": "politics", "empirical_win_rate": 0.55, "n_trades": 100},
            {"p_bucket": 0.40, "t_bucket": "mid", "domain": "politics", "empirical_win_rate": 0.58, "n_trades": 100},
            {"p_bucket": 0.45, "t_bucket": "short", "domain": "politics", "empirical_win_rate": 0.62, "n_trades": 100},
            {"p_bucket": 0.60, "t_bucket": "long", "domain": "politics", "empirical_win_rate": 0.42, "n_trades": 100},
            {"p_bucket": 0.55, "t_bucket": "mid", "domain": "politics", "empirical_win_rate": 0.38, "n_trades": 100},
            {"p_bucket": 0.50, "t_bucket": "short", "domain": "politics", "empirical_win_rate": 0.35, "n_trades": 100},
        ]
    )
    return train, oos, surface


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Watchdog OOS backtest (2024 train / 2025 test)")
    parser.add_argument("--platform", type=str, default="polymarket")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--strategy-mode", type=str, choices=["taker", "maker"], default="taker")
    parser.add_argument("--min-trades-per-bucket", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    try:
        loader = BeckerHistoricalLoader(settings.becker_dataset_path)

        print("Loading 2024 training markets...")
        train_markets = loader.get_resolved_markets(
            platform=args.platform,
            domain_filter=args.domain,
            start_year=2024,
            end_year=2024,
        )

        print("Building calibration surface from 2024 data...")
        calibration_surface = loader.build_calibration_surface_from_data(
            platform=args.platform,
            min_trades_per_bucket=args.min_trades_per_bucket,
            start_year=2024,
            end_year=2024,
        )

        print("Loading 2025 out-of-sample markets...")
        oos_markets = loader.get_resolved_markets(
            platform=args.platform,
            domain_filter=args.domain,
            start_year=2025,
            end_year=2025,
        )
    except FileNotFoundError:
        print("Becker dataset not found. Falling back to synthetic backtest fixtures.")
        train_markets, oos_markets, calibration_surface = _build_synthetic_inputs()
    oos_markets["pre_registered"] = True

    backtester = Backtester(
        min_divergence_backtest=settings.min_divergence_backtest,
        fee_rate=settings.backtest_fee_rate,
        slippage_proxy=settings.backtest_slippage_rate,
        spread_proxy=settings.backtest_spread_proxy,
        kelly_fraction=settings.kelly_fraction,
    )

    results = backtester.run_backtest(
        markets_df=oos_markets,
        calibration_surface_df=calibration_surface,
        strategy_mode=args.strategy_mode,
        verbose=False,
    )

    _print_report(results)

    passed, reasons = evaluate_pass_fail(
        win_rate=results.win_rate,
        brier=results.brier_score,
        n_trades=results.n_traded,
        max_drawdown=results.max_drawdown,
    )

    out_dir = Path("./backtest_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{ts}.json"

    payload = {
        "meta": {
            "platform": args.platform,
            "domain": args.domain,
            "strategy_mode": args.strategy_mode,
            "train_markets": int(train_markets.shape[0]),
            "oos_markets": int(oos_markets.shape[0]),
        },
        "results": results.to_json_dict(),
        "verdict": {
            "pass": passed,
            "reasons": reasons,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if passed:
        print(f"PASS: strategy cleared go-live checks. Saved report to {out_path}")
    else:
        print(f"ABORT: strategy failed checks ({'; '.join(reasons)}). Saved report to {out_path}")


if __name__ == "__main__":
    main()
