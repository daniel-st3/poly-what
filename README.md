# Watchdog

![Python](https://img.shields.io/badge/python-3.12-blue)
![SQLAlchemy](https://img.shields.io/badge/ORM-SQLAlchemy-green)
![DuckDB](https://img.shields.io/badge/data-DuckDB-orange)
![LLM](https://img.shields.io/badge/agents-dual--LLM-black)
![Daily Run](https://github.com/daniel-st3/poly-agent/actions/workflows/daily_run.yml/badge.svg)

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

## Telegram Alerts

1. Create a bot via @BotFather → copy `TELEGRAM_BOT_TOKEN`
2. Get your chat ID by messaging @userinfobot → copy `TELEGRAM_CHAT_ID`
3. Add to `.env`: `TELEGRAM_BOT_TOKEN=<token>` and `TELEGRAM_CHAT_ID=<chat_id>`
4. Test: `watchdog test-telegram`

Daily summaries are sent automatically at the end of each `scripts/daily_validation.sh` run. GitHub Actions also sends a failure alert if the run fails.

**Required GitHub Secrets** (for GitHub Actions):
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `MANIFOLD_API_KEY`, `MANIFOLD_USER_ID`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## Main Commands

```bash
watchdog init-db
watchdog build-calibration --dataset-path /path/to/becker.parquet
watchdog run-paper-trading --platform manifold --virtual-bankroll 500 --iterations 1
watchdog go-live-check
watchdog run-snapshot-collector
watchdog run-backtest --platform polymarket --domain politics
watchdog run-market-maker --dry-run
watchdog ingest-news-loop --interval-seconds 30
watchdog run-pipeline-loop --iterations 0 --interval-seconds 60
watchdog run-live-validation --experiment-id feb2026_v1 --bankroll 50
```

## Retired Strategies

| Strategy | Retired | Reason | Reference |
|----------|---------|--------|-----------|
| btc_scalp | 2026-03-29 | No profitable regime found across 332 paper trades | [BTC_SCALP_RETROSPECTIVE.md](BTC_SCALP_RETROSPECTIVE.md) |

`enable_btc_scalp` defaults to `false` in Settings. `railway.json` still lists
`watchdog run-btc-scalp` as the start command — update or disable the Railway
service if it is still running.

## Scripts

- `python -m watchdog.scripts.run_snapshot_collector`
- `python -m watchdog.scripts.run_paper_trading --virtual-bankroll 500 --platform manifold`
- `python -m watchdog.scripts.run_live_validation --experiment-id feb2026_v1 --bankroll 50 --platform polymarket`
- `python -m watchdog.scripts.run_backtest --platform polymarket --domain politics`
- `python -m watchdog.scripts.run_market_maker --dry-run`
- `python -m watchdog.scripts.run_becker_analysis --platform polymarket`
- `python -m watchdog.scripts.download_becker_data --output-dir ./data/becker`
