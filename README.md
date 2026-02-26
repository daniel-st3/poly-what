# Watchdog

![Python](https://img.shields.io/badge/python-3.12-blue)
![SQLAlchemy](https://img.shields.io/badge/ORM-SQLAlchemy-green)
![DuckDB](https://img.shields.io/badge/data-DuckDB-orange)
![LLM](https://img.shields.io/badge/agents-dual--LLM-black)
![CI](https://img.shields.io/badge/CI-GitHub%20Actions-purple)

Watchdog: Lean dual-agent prediction market intelligence system with calibrated mispricing detection, maker-side structural positioning, and full telemetry for edge validation.

## Architecture

```text
+--------------------------+
| News Sources             |
| GDELT / RSS / Reddit     |
+------------+-------------+
             |
             v
+--------------------------+
| Router LLM (cheap)       |
| relevance + market link  |
+------------+-------------+
             |
             v
+--------------------------+
| Calibration Layer        |
| C(p,t) + domain bias     |
+------------+-------------+
             |
             v
+--------------------------+
| Executor LLM (selective) |
| trade / no-trade + risk  |
+------------+-------------+
             |
             v
+--------------------------+
| Order Executor           |
| pre-checks + sizing gate |
+------------+-------------+
             |
             v
+--------------------------+
| Polymarket CLI (Rust)    |
| orderbook + orders       |
+------------+-------------+
             |
             v
+--------------------------+
| Polygon / USDC settlement|
+--------------------------+
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
watchdog init-db && watchdog healthcheck
```

## Three Edges

1. Calibration bias edge
- Build and query 2D calibration surface `C(p,t)` from Becker historical trades.
- Trade only when model/market divergence survives spread + fee + slippage.

2. Maker structural edge
- Use bounded Avellaneda-Stoikov quoting with VPIN kill-switch.
- Target spread capture while controlling adverse selection and inventory risk.

3. News-latency edge
- Route high-volume event flow through cheap router + selective executor.
- Telemetry measures whether market moved after or before signal completion.

## $50 Validation Experiment

Goal: validate latency and slippage, not maximize PnL.

- Hard bankroll cap: `$50`
- Position sizing: max `5` simultaneous positions at `$10` each
- Stricter trigger: divergence `> 20%`
- Telemetry captures:
  - `ts_news_received`
  - `ts_router_completed`
  - `ts_calibration_completed`
  - `ts_executor_completed`
  - `ts_order_submitted`
  - `market_price_at_signal`, `market_price_1m`, `market_price_5m`

## Abort Conditions

| Condition | Action |
|---|---|
| Geoblock check fails | Halt all trading operations |
| VPIN above threshold | Withdraw/widen maker quotes |
| Live trading disabled | Force paper mode |
| Drawdown / win-rate guard fails | Abort live execution |
| API or execution pre-check failure | Skip trade and log reason |

## Main Commands

```bash
watchdog init-db
watchdog build-calibration --dataset-path /path/to/becker.parquet
watchdog run-paper-trading --platform manifold --virtual-bankroll 500 --iterations 1
python -m watchdog.scripts.run_backtest --platform polymarket
python -m watchdog.scripts.run_live_validation --experiment-id feb2026_v1 --bankroll 50 --platform manifold
```

## Scripts

- `python -m watchdog.scripts.run_snapshot_collector`
- `python -m watchdog.scripts.run_paper_trading --virtual-bankroll 500 --platform manifold`
- `python -m watchdog.scripts.run_live_validation --experiment-id feb2026_v1 --bankroll 50 --platform polymarket`
- `python -m watchdog.scripts.run_backtest --platform polymarket --domain politics`
- `python -m watchdog.scripts.run_market_maker --dry-run`
- `python -m watchdog.scripts.run_becker_analysis --platform polymarket`
- `python -m watchdog.scripts.download_becker_data --output-dir ./data/becker`
