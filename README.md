# Watchdog

Watchdog is a lean, dual-agent prediction market intelligence system focused on:

- Calibration-driven mispricing detection (`C(p, t)` surfaces)
- Maker-side structural positioning
- News-triggered routing and selective execution
- Full telemetry for latency and edge validation

## Core Design Constraints

- All Polymarket market data + order interactions route through the official CLI wrapper.
- Non-negotiable database schema includes `markets`, `market_snapshots`, `news_events`, `signals`, `trades`, `telemetry`, `calibration_surface`, `maker_quotes`.
- Calibration and backtests are DuckDB-over-Parquet, not full-frame pandas loads.
- Risk controls include empirical Kelly sizing, VPIN kill switch, and near-resolution maker halt.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env
watchdog init-db
watchdog healthcheck
```

## Main Commands

```bash
watchdog init-db
watchdog healthcheck
watchdog build-calibration --dataset-path /path/to/trades.parquet
watchdog ingest-news-once
watchdog run-pipeline-once
```

## Notes

- This repository defaults to SQLite for development and supports PostgreSQL via `DATABASE_URL`.
- Trading is paper by default. Live mode requires explicit configuration.
