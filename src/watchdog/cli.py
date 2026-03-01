from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy import text

from watchdog.core.config import Settings, get_settings
from watchdog.core.exceptions import GeoblockError, PolymarketCliError
from watchdog.core.logging import configure_logging
from watchdog.db.base import Base
from watchdog.db.session import build_engine, build_session_factory
from watchdog.llm.executor import build_executor
from watchdog.llm.router import build_router
from watchdog.market_data.polymarket_cli import PolymarketCli
from watchdog.news.ingest import ingest_news_once_sync
from watchdog.risk.kelly import EmpiricalKellySizer
from watchdog.risk.vpin import VPINCalculator
from watchdog.scripts.run_paper_trading import run_paper_trading_loop
from watchdog.services.market_sync import sync_markets_once
from watchdog.services.pipeline import PipelineRunner
from watchdog.signals.calibration import CalibrationSurfaceService
from watchdog.signals.telegram_bot import TelegramAlerter

app = typer.Typer(help="Watchdog command-line interface")
LOGGER = logging.getLogger(__name__)


def _build_runtime() -> tuple[Settings, PolymarketCli]:
    settings = get_settings()
    configure_logging(settings)
    return settings, PolymarketCli(settings)


@app.command("init-db")
def init_db_command() -> None:
    settings, _ = _build_runtime()
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    typer.echo("Initialized database schema")


@app.command("healthcheck")
def healthcheck_command(
    mode: Annotated[str, typer.Option(help="Select check mode")] = "paper"
) -> None:
    if mode not in ("paper", "live"):
        typer.echo(f"Invalid mode: {mode}")
        raise typer.Exit(code=1)

    settings = get_settings()
    configure_logging(settings)
    failed = False

    typer.echo(f"[INFO] LIVE_TRADING={settings.enable_live_trading}")

    engine = build_engine(settings)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        typer.echo("[OK]  DB connection")
    except Exception as exc:
        typer.echo(f"[FAIL] DB connection ({exc})")
        failed = True

    if settings.anthropic_api_key or settings.openai_api_key:
        typer.echo("[OK]  LLM API key configured")
    else:
        typer.echo("[FAIL] missing LLM API key (set ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        failed = True

    if mode == "live":
        cli = PolymarketCli(settings)
        try:
            version = cli.check_version()
            typer.echo(f"[OK]  polymarket CLI version {version}")
        except PolymarketCliError as exc:
            message = str(exc).lower()
            if "not found" in message:
                typer.echo("[FAIL] polymarket CLI not found")
            else:
                typer.echo(f"[FAIL] polymarket CLI version check failed ({exc})")
            failed = True
    else:
        # paper mode
        if settings.manifold_api_key:
            typer.echo("[OK]  MANIFOLD_API_KEY configured")
        else:
            typer.echo("[FAIL] missing MANIFOLD_API_KEY")
            failed = True

        if settings.manifold_user_id:
            typer.echo("[OK]  MANIFOLD_USER_ID configured")
        else:
            typer.echo("[FAIL] missing MANIFOLD_USER_ID")
            failed = True

    if settings.telegram_bot_token:
        typer.echo("[OK]  TELEGRAM_BOT_TOKEN configured")
    else:
        typer.echo("[WARN] TELEGRAM_BOT_TOKEN not set (alerts disabled)")

    if failed:
        raise typer.Exit(code=1)


@app.command("sync-markets")
def sync_markets_command(limit: Annotated[int, typer.Option(min=1, max=1000)] = 200) -> None:
    settings, cli = _build_runtime()
    try:
        cli.startup_check()
    except (PolymarketCliError, GeoblockError) as exc:
        typer.echo(f"Sync halted: {exc}")
        raise typer.Exit(code=1) from None

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        count = sync_markets_once(session, cli, limit=limit)
    typer.echo(f"Synced {count} markets")


@app.command("build-calibration")
def build_calibration_command(
    dataset_path: Annotated[Path, typer.Option(exists=True, file_okay=True, dir_okay=False)],
    dataset_source: Annotated[str, typer.Option()] = "becker",
    price_column: Annotated[str, typer.Option()] = "price",
    outcome_column: Annotated[str, typer.Option()] = "outcome",
    hours_to_resolution_column: Annotated[str, typer.Option()] = "hours_to_resolution",
    domain_column: Annotated[str, typer.Option()] = "domain",
) -> None:
    settings, _ = _build_runtime()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    calibration = CalibrationSurfaceService()

    with session_factory() as session:
        rows = calibration.build_from_becker_parquet(
            session=session,
            dataset_path=str(dataset_path),
            dataset_source=dataset_source,
            price_column=price_column,
            outcome_column=outcome_column,
            hours_to_resolution_column=hours_to_resolution_column,
            domain_column=domain_column,
        )
    typer.echo(f"Built calibration surface rows: {rows}")


@app.command("ingest-news-once")
def ingest_news_once_command() -> None:
    settings, _ = _build_runtime()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    inserted = ingest_news_once_sync(settings, session_factory)
    typer.echo(f"Inserted news events: {inserted}")


@app.command("run-pipeline-once")
def run_pipeline_once_command(max_news: Annotated[int, typer.Option(min=1, max=100)] = 10) -> None:
    settings, cli = _build_runtime()
    try:
        cli.startup_check()
    except (PolymarketCliError, GeoblockError) as exc:
        typer.echo(f"Pipeline halted: {exc}")
        raise typer.Exit(code=1) from None

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    router = build_router(settings)
    executor = build_executor(settings)
    calibration = CalibrationSurfaceService()
    alerter = TelegramAlerter(settings)
    sizer = EmpiricalKellySizer(kelly_fraction=settings.kelly_fraction, max_drawdown_p95=settings.max_drawdown_p95)
    vpin_calc = VPINCalculator()

    runner = PipelineRunner(
        settings=settings,
        session_factory=session_factory,
        cli=cli,
        router=router,
        executor=executor,
        calibration=calibration,
        sizer=sizer,
        vpin_calc=vpin_calc,
        alerter=alerter,
    )

    try:
        stats = asyncio.run(runner.run_once(max_news=max_news))
    except PolymarketCliError as exc:
        typer.echo(f"Pipeline failed: {exc}")
        raise typer.Exit(code=1) from None

    typer.echo(
        f"Pipeline done | processed_news={stats.processed_news} generated_signals={stats.generated_signals} executed_trades={stats.executed_trades}"
    )


@app.command("run-paper-trading")
def run_paper_trading_command(
    platform: Annotated[str, typer.Option()] = "manifold",
    virtual_bankroll: Annotated[float, typer.Option()] = 500.0,
    max_markets: Annotated[int, typer.Option(min=1, max=500)] = 80,
    iterations: Annotated[int, typer.Option(min=0, max=100000)] = 1,
) -> None:
    asyncio.run(
        run_paper_trading_loop(
            virtual_bankroll=virtual_bankroll,
            platform=platform,
            max_markets=max_markets,
            iterations=iterations,
        )
    )


@app.command("cancel-all")
def cancel_all_command() -> None:
    settings, cli = _build_runtime()
    if not settings.enable_live_trading:
        typer.echo("Live trading disabled; cancel-all skipped")
        raise typer.Exit(code=1)

    try:
        cli.startup_check()
        resp = cli.cancel_all()
    except (PolymarketCliError, GeoblockError) as exc:
        typer.echo(f"Cancel-all failed: {exc}")
        raise typer.Exit(code=1) from None

    typer.echo(str(resp.payload))


@app.command("go-live-check")
def go_live_check_command() -> None:
    """Check whether paper trading results pass the go-live gate."""
    settings, _ = _build_runtime()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    from watchdog.backtest.go_live_gate import check_go_live_gate

    with session_factory() as session:
        passed, reasons = check_go_live_gate(session)

    if passed:
        typer.echo("[PASS] Go-live gate passed. Ready for live trading.")
    else:
        typer.echo("[FAIL] Go-live gate failed:")
        for r in reasons:
            typer.echo(f"  - {r}")
        raise typer.Exit(code=1)


@app.command("run-snapshot-collector")
def run_snapshot_collector_command() -> None:
    """Run the continuous orderbook snapshot collector daemon."""
    import asyncio

    from watchdog.scripts.run_snapshot_collector import main as _snapshot_main

    try:
        asyncio.run(_snapshot_main())
    except KeyboardInterrupt:
        typer.echo("Snapshot collector stopped.")


@app.command("run-backtest")
def run_backtest_command(
    platform: Annotated[str, typer.Option()] = "polymarket",
    domain: Annotated[str, typer.Option()] = "",
    output_json: Annotated[Path, typer.Option()] = Path("backtest_results.json"),
) -> None:
    """Run backtester against Becker historical data."""
    from watchdog.scripts.run_backtest import main as _backtest_main
    
    _backtest_main(
        platform=platform,
        domain=domain or None,
        strategy_mode="taker",
        min_trades_per_bucket=100
    )


@app.command("run-market-maker")
def run_market_maker_command(
    dry_run: Annotated[bool, typer.Option("--dry-run/--live")] = True,
) -> None:
    """Run the Avellaneda-Stoikov market maker loop."""
    import asyncio

    from watchdog.scripts.run_market_maker import main as _mm_main

    try:
        asyncio.run(_mm_main(dry_run=dry_run))
    except KeyboardInterrupt:
        typer.echo("Market maker stopped.")


@app.command("ingest-news-loop")
def ingest_news_loop_command(
    interval_seconds: Annotated[int, typer.Option(min=5, max=3600)] = 30,
) -> None:
    """Run continuous news ingestion daemon."""
    import time

    from watchdog.news.ingest import ingest_news_once_sync

    settings, _ = _build_runtime()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    typer.echo(f"Starting news loop every {interval_seconds}s. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.perf_counter()
            inserted = ingest_news_once_sync(settings, session_factory)
            elapsed = time.perf_counter() - t0
            typer.echo(f"Ingested {inserted} items in {elapsed:.2f}s")
            sleep_for = max(0.0, interval_seconds - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        typer.echo("News loop stopped.")


@app.command("run-pipeline-loop")
def run_pipeline_loop_command(
    max_news: Annotated[int, typer.Option(min=1, max=100)] = 10,
    interval_seconds: Annotated[int, typer.Option(min=5, max=3600)] = 60,
    iterations: Annotated[int, typer.Option(min=0)] = 0,
) -> None:
    """Run the full pipeline loop continuously (0 iterations = infinite)."""
    import asyncio
    import time

    settings, cli = _build_runtime()
    try:
        cli.startup_check()
    except (PolymarketCliError, GeoblockError) as exc:
        typer.echo(f"Pipeline loop halted: {exc}")
        raise typer.Exit(code=1) from None

    engine = build_engine(settings)
    session_factory = build_session_factory(engine)
    router = build_router(settings)
    executor = build_executor(settings)
    calibration = CalibrationSurfaceService()
    alerter = TelegramAlerter(settings)
    sizer = EmpiricalKellySizer(
        kelly_fraction=settings.kelly_fraction,
        max_drawdown_p95=settings.max_drawdown_p95,
    )
    vpin_calc = VPINCalculator()
    runner = PipelineRunner(
        settings=settings,
        session_factory=session_factory,
        cli=cli,
        router=router,
        executor=executor,
        calibration=calibration,
        sizer=sizer,
        vpin_calc=vpin_calc,
        alerter=alerter,
    )

    n = 0
    typer.echo(f"Pipeline loop started. iterations={'∞' if iterations == 0 else iterations}")
    try:
        while iterations == 0 or n < iterations:
            t0 = time.perf_counter()
            try:
                stats = asyncio.run(runner.run_once(max_news=max_news))
            except PolymarketCliError as exc:
                typer.echo(f"Pipeline iteration failed: {exc}")
                stats = None
            if stats:
                typer.echo(
                    f"[iter {n+1}] news={stats.processed_news} "
                    f"signals={stats.generated_signals} trades={stats.executed_trades}"
                )
            elapsed = time.perf_counter() - t0
            n += 1
            sleep_for = max(0.0, interval_seconds - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        typer.echo("Pipeline loop stopped.")


@app.command("run-live-validation")
def run_live_validation_command(
    experiment_id: Annotated[str, typer.Option()] = "",
    bankroll: Annotated[float, typer.Option(min=1.0)] = 50.0,
    platform: Annotated[str, typer.Option()] = "polymarket",
) -> None:
    """Run the $50 live validation experiment."""
    import asyncio

    from watchdog.scripts.run_live_validation import main as _lv_main

    try:
        asyncio.run(
            _lv_main(
                experiment_id=experiment_id or "default_exp",
                bankroll=bankroll,
                platform=platform,
            )
        )
    except KeyboardInterrupt:
        typer.echo("Live validation stopped.")
    except Exception as exc:
        typer.echo(f"Live validation failed: {exc}")
        raise typer.Exit(code=1) from None

@app.command("analyze-paper-trades")
def analyze_paper_trades_command() -> None:
    """Analyze paper trading performance by strategy."""
    import math
    from statistics import mean, stdev

    from sqlalchemy import select

    from watchdog.db.models import Market, Trade

    settings = get_settings()
    engine = build_engine(settings)
    SessionFactory = build_session_factory(engine)

    with SessionFactory() as session:
        rows = session.execute(
            select(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .where(Trade.is_paper.is_(True))
        ).all()

    if not rows:
        typer.echo("No paper trades found.")
        raise typer.Exit()

    # Group by strategy
    strategies: dict[str, list[dict]] = {}
    for trade, market in rows:
        strat = trade.strategy or "calibration"
        if strat not in strategies:
            strategies[strat] = []

        pnl = trade.pnl or 0.0
        strategies[strat].append({
            "slug": market.slug,
            "domain": market.domain,
            "side": trade.side,
            "size": trade.size,
            "entry": trade.entry_price,
            "exit": trade.exit_price,
            "pnl": pnl,
            "status": trade.status,
        })

    # Summary table header
    typer.echo("")
    header = f"{'Strategy':<20} {'Trades':>6} {'Open':>5} {'Closed':>6} {'Win%':>6} {'Avg PnL':>9} {'Sharpe':>7} {'Max DD':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for strat, trades in sorted(strategies.items()):
        total = len(trades)
        closed = [t for t in trades if t["status"] == "closed"]
        open_trades = [t for t in trades if t["status"] == "open"]

        if closed:
            pnls = [t["pnl"] for t in closed]
            wins = sum(1 for p in pnls if p > 0)
            win_pct = (wins / len(pnls)) * 100
            avg_pnl = mean(pnls)
            sharpe = (mean(pnls) / stdev(pnls) * math.sqrt(252)) if len(pnls) > 1 and stdev(pnls) > 0 else 0.0

            # Max drawdown
            cumulative = 0.0
            peak = 0.0
            max_dd = 0.0
            for p in pnls:
                cumulative += p
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd

            typer.echo(
                f"{strat:<20} {total:>6} {len(open_trades):>5} {len(closed):>6} "
                f"{win_pct:>5.1f}% ${avg_pnl:>7.2f} {sharpe:>7.2f} -${max_dd:>6.2f}"
            )
        else:
            typer.echo(
                f"{strat:<20} {total:>6} {len(open_trades):>5} {len(closed):>6} "
                f"{'  -':>6} {'  -':>9} {'  -':>7} {'  -':>8}"
            )

    # Domain breakdown
    typer.echo("")
    typer.echo("Domain Breakdown:")
    domains: dict[str, int] = {}
    for trades in strategies.values():
        for t in trades:
            d = t["domain"] or "unknown"
            domains[d] = domains.get(d, 0) + 1

    for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
        typer.echo(f"  {domain:<20} {count} trades")

    typer.echo("")


@app.command("find-quick-markets")
def find_quick_markets_command(
    platform: Annotated[str, typer.Option()] = "manifold",
    limit: Annotated[int, typer.Option(min=1, max=500)] = 80,
) -> None:
    """Find short-term markets suitable for fast trading."""
    import asyncio

    from watchdog.market_data.manifold_client import ManifoldClient
    from watchdog.services.pipeline import _hours_to_resolution

    settings = get_settings()

    async def _find() -> None:
        manifold = None
        polymarket = None

        if platform == "manifold":
            manifold = ManifoldClient(
                base_url=settings.manifold_api_base_url,
                api_key=settings.manifold_api_key,
            )
        elif platform == "polymarket":
            from watchdog.market_data.polymarket_cli import PolymarketCli
            polymarket = PolymarketCli(settings)
        else:
            typer.echo(f"Platform '{platform}' not supported.")
            raise typer.Exit(code=1)

        from watchdog.scripts.run_paper_trading import _load_platform_markets

        typer.echo(f"Fetching up to {limit} markets from {platform}...")
        unified_markets = await _load_platform_markets(
            platform=platform,
            max_markets=limit,
            manifold=manifold,
            polymarket=polymarket,
        )

        candidates: list[dict] = []
        for m in unified_markets:
            res_time = m.get("resolution_time")
            if not res_time:
                continue

            hours = _hours_to_resolution(res_time)
            if hours < settings.min_resolution_hours or hours > settings.max_resolution_hours:
                continue

            vol = float(m.get("volume") or m.get("volume24Hours") or m.get("volume_24h") or 0)
            if vol < settings.min_volume_24h:
                continue

            candidates.append({
                "question": m["question"][:55],
                "hours": hours,
                "volume": vol,
                "prob": m["probability"],
                "domain": m["domain"],
                "slug": m["slug"],
            })

        candidates.sort(key=lambda x: -x["volume"])
        top = candidates[:10]

        if not top:
            typer.echo("No markets match the 6-168h resolution + volume filters.")
            raise typer.Exit()

        typer.echo("")
        header = f"{'Rank':>4}  {'Market':<57} {'Resolves':>10} {'Vol 24h':>12} {'Domain':<12}"
        typer.echo(header)
        typer.echo("-" * len(header))

        for i, c in enumerate(top, 1):
            hours_str = f"{c['hours']:.0f}h"
            vol_str = f"${c['volume']:,.0f}"
            typer.echo(
                f"{i:>4}  {c['question']:<57} {hours_str:>10} {vol_str:>12} {c['domain']:<12}"
            )

        typer.echo(f"\n{len(candidates)} total markets match filters ({len(unified_markets)} scanned)")

    asyncio.run(_find())


if __name__ == "__main__":
    app()
