from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from watchdog.backtest.backtester import Backtester
from watchdog.backtest.historical_loader import BeckerHistoricalLoader


def _write_synthetic_becker_parquet(base_dir: Path) -> Path:
    rows: list[dict] = []
    markets = [
        ("m1", "Will policy X pass?", "politics", datetime(2024, 6, 1, tzinfo=UTC), 1),
        ("m2", "Will BTC > 100k?", "crypto", datetime(2024, 8, 1, tzinfo=UTC), 0),
        ("m3", "Will policy Y pass?", "politics", datetime(2025, 6, 1, tzinfo=UTC), 1),
        ("m4", "Will ETH > 8k?", "crypto", datetime(2025, 9, 1, tzinfo=UTC), 0),
        ("m5", "Will team A win?", "sports", datetime(2025, 10, 1, tzinfo=UTC), 1),
    ]

    for market_id, question, domain, resolution_time, outcome in markets:
        start_ts = resolution_time - timedelta(days=45)
        for i in range(20):
            ts = start_ts + timedelta(days=i * 2)
            base_price = 0.15 + (i / 60)
            if outcome == 0:
                base_price = 1 - base_price
            rows.append(
                {
                    "market_id": market_id,
                    "question": question,
                    "domain": domain,
                    "resolution_time": resolution_time,
                    "outcome": outcome,
                    "price": max(0.01, min(0.99, base_price)),
                    "timestamp": ts,
                    "volume": 100 + i,
                    "taker_direction": "buy" if i % 2 == 0 else "sell",
                    "platform": "polymarket",
                    "liquidity_role": "taker" if i % 3 else "maker",
                }
            )

    df = pd.DataFrame(rows)
    out_file = base_dir / "synthetic_trades.parquet"
    df.to_parquet(out_file, index=False)
    return out_file


def test_loader_and_backtester_with_synthetic_parquet(tmp_path: Path) -> None:
    _write_synthetic_becker_parquet(tmp_path)
    loader = BeckerHistoricalLoader(tmp_path)

    train = loader.get_resolved_markets("polymarket", None, 2024, 2024)
    oos = loader.get_resolved_markets("polymarket", None, 2025, 2025)
    surface = loader.build_calibration_surface_from_data(
        "polymarket",
        min_trades_per_bucket=1,
        start_year=2024,
        end_year=2024,
    )

    assert not train.empty
    assert not oos.empty
    assert not surface.empty

    oos["pre_registered"] = True
    backtester = Backtester(min_divergence_backtest=0.15)
    results = backtester.run_backtest(oos, surface, strategy_mode="taker", verbose=False)

    assert results.n_total_markets == int(oos.shape[0])
    assert 0 <= results.win_rate <= 1


def test_backtester_enforces_year_split(tmp_path: Path) -> None:
    _write_synthetic_becker_parquet(tmp_path)
    loader = BeckerHistoricalLoader(tmp_path)
    mixed = loader.get_resolved_markets("polymarket", None, 2024, 2025)
    surface = loader.build_calibration_surface_from_data("polymarket", min_trades_per_bucket=1)

    mixed["pre_registered"] = True
    backtester = Backtester(min_divergence_backtest=0.15)

    with pytest.raises(ValueError, match="mixed"):
        backtester.run_backtest(mixed, surface, strategy_mode="taker", verbose=False)


def test_pass_go_live_false_if_insufficient_trades(tmp_path: Path) -> None:
    _write_synthetic_becker_parquet(tmp_path)
    loader = BeckerHistoricalLoader(tmp_path)

    oos = loader.get_resolved_markets("polymarket", None, 2025, 2025)
    surface = loader.build_calibration_surface_from_data("polymarket", min_trades_per_bucket=1)

    oos["pre_registered"] = True
    results = Backtester(min_divergence_backtest=0.15).run_backtest(oos, surface, strategy_mode="taker")
    assert results.n_traded < 50
    assert results.pass_go_live is False
