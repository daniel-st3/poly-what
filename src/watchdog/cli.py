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


if __name__ == "__main__":
    app()
