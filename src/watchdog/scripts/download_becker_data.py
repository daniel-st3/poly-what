from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/RupertMa/polymarket-analysis/main/data/polymarket_trades.parquet"
DOWNLOAD_FILENAME = "polymarket_trades.parquet"
SYNTHETIC_FILENAME = "polymarket_trades_synthetic.parquet"


def _generate_synthetic_parquet(path: Path, n_rows: int = 500) -> None:
    rng = np.random.default_rng(42)
    domains = np.array(["politics", "crypto", "sports", "macro", "tech", "other"])

    now = datetime.now(UTC)
    resolution_times = [now + timedelta(days=int(days)) for days in rng.integers(3, 180, size=n_rows)]

    prices_30d = rng.uniform(0.01, 0.99, size=n_rows)
    prices_7d = np.clip(prices_30d + rng.normal(0.0, 0.05, size=n_rows), 0.01, 0.99)
    prices_1d = np.clip(prices_7d + rng.normal(0.0, 0.05, size=n_rows), 0.01, 0.99)
    outcomes = (rng.uniform(0, 1, size=n_rows) < prices_1d).astype(int)

    df = pd.DataFrame(
        {
            "market_id": [f"synthetic_{i:04d}" for i in range(n_rows)],
            "question": [f"Synthetic market question {i}" for i in range(n_rows)],
            "domain": rng.choice(domains, size=n_rows),
            "resolution_time": resolution_times,
            "outcome": outcomes,
            "price_30d_before": prices_30d,
            "price_7d_before": prices_7d,
            "price_1d_before": prices_1d,
            "days_to_resolution": rng.integers(1, 365, size=n_rows),
            "volume_total": rng.uniform(100.0, 100000.0, size=n_rows),
            "platform": ["polymarket"] * n_rows,
            "year": rng.choice([2024, 2025], size=n_rows),
        }
    )

    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Becker dataset parquet with synthetic fallback")
    parser.add_argument("--output-dir", type=str, default="./data/becker")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_parquets = sorted(output_dir.glob("*.parquet"))
    if existing_parquets:
        print(f"Found existing parquet files in {output_dir}; skipping download.")
        for path in existing_parquets:
            print(f" - {path}")
        return

    download_path = output_dir / DOWNLOAD_FILENAME
    synthetic_path = output_dir / SYNTHETIC_FILENAME

    print(f"Attempting download: {DATA_URL}")
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(DATA_URL)
            if response.status_code < 200 or response.status_code >= 300:
                raise RuntimeError(f"HTTP {response.status_code}")
            download_path.write_bytes(response.content)
        print(f"Download successful: {download_path}")
        return
    except Exception as exc:
        print(f"WARNING: Download failed ({exc}). Generating synthetic fallback dataset.")

    _generate_synthetic_parquet(synthetic_path, n_rows=500)
    print(f"Synthetic dataset generated: {synthetic_path}")


if __name__ == "__main__":
    main()
