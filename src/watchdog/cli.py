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
    import os

    settings, _ = _build_runtime()
    engine = build_engine(settings)
    Base.metadata.create_all(engine)
    # Print the resolved absolute DB path so CI logs show exactly which file is used.
    db_url = str(engine.url)
    if db_url.startswith("sqlite:///"):
        rel = db_url[len("sqlite:///"):]
        abs_path = os.path.abspath(rel) if rel and not rel.startswith("/") else rel or ":memory:"
        typer.echo(f"DB path: {abs_path}  (url={db_url})")
    else:
        typer.echo(f"DB url: {db_url}")
    typer.echo("Initialized database schema")


@app.command("restore-baseline")
def restore_baseline_command() -> None:
    """Seed DB from db/seed_data.sql if fewer than 100 trades exist (idempotent)."""
    import os

    settings = get_settings()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        trade_count = session.execute(text("SELECT COUNT(*) FROM trades")).scalar_one()

    if trade_count >= 100:
        typer.echo(f"restore-baseline: skipped — DB already has {trade_count} trades")
        return

    # Locate seed file — CI path first, then editable-install and CWD fallbacks.
    _here = Path(__file__).parent
    candidates = [
        Path(os.environ.get("GITHUB_WORKSPACE", "")) / "db" / "seed_data.sql",  # CI
        _here.parent.parent / "db" / "seed_data.sql",   # editable local install
        Path(os.getcwd()) / "db" / "seed_data.sql",     # fallback
        Path("/home/runner/work/poly-what/poly-what/db/seed_data.sql"),  # hardcoded runner
    ]
    seed_path = next((p for p in candidates if p.exists()), None)
    if seed_path is None:
        typer.echo("restore-baseline: seed file not found — checked: " + ", ".join(str(p) for p in candidates), err=True)
        raise typer.Exit(code=1)

    sql = seed_path.read_text()
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))

    with session_factory() as session:
        new_count = session.execute(text("SELECT COUNT(*) FROM trades")).scalar_one()

    typer.echo(f"restore-baseline: seeded from {seed_path} — {trade_count} → {new_count} trades")


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
    session_factory = build_session_factory(engine)

    arb_strategies = {"intra_event_arb", "pair_cost_arb"}

    with session_factory() as session:
        rows = session.execute(
            select(Trade, Market)
            .join(Market, Trade.market_id == Market.id)
            .where(Trade.is_paper.is_(True))
        ).all()

    if not rows:
        typer.echo("No paper trades found.")
        raise typer.Exit()

    # Separate arb trades from core bot trades
    arb_rows = [(t, m) for t, m in rows if (t.strategy or "") in arb_strategies]
    core_rows = [(t, m) for t, m in rows if (t.strategy or "") not in arb_strategies]

    if not core_rows:
        typer.echo("No core (non-arb) paper trades found.")
        raise typer.Exit()

    rows = core_rows  # main dashboard uses core trades only

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

    # Summary table header (extended with Sortino, Brier, F1, Sig?)
    typer.echo("")
    header = (
        f"{'Strategy':<20} {'Trades':>6} {'Win%':>6} {'Avg PnL':>9} "
        f"{'Sharpe':>7} {'Sortino':>8} {'Brier':>7} {'F1':>6}  {'Sig?'}"
    )
    typer.echo(header)
    typer.echo("-" * (len(header) + 10))

    for strat, trades in sorted(strategies.items()):
        closed = [t for t in trades if t["status"] == "closed"]
        [t for t in trades if t["status"] == "open"]

        if not closed:
            typer.echo(
                f"{strat:<20} {len(trades):>6}   {'  -':>5} {'  -':>9} "
                f"{'  -':>7} {'  -':>8} {'  -':>7} {'  -':>6}  -"
            )
            continue

        pnls = [t["pnl"] for t in closed]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        win_pct = (wins / n) * 100

        avg_pnl = mean(pnls)
        sharpe = (mean(pnls) / stdev(pnls) * math.sqrt(252)) if n > 1 and stdev(pnls) > 0 else 0.0

        # Sortino Ratio (downside deviation only)
        neg = [p for p in pnls if p < 0]
        if len(neg) > 1:
            d_std = stdev(neg)
        elif len(neg) == 1:
            d_std = abs(neg[0])
        else:
            d_std = 1e-9
        sortino = (mean(pnls) / d_std * math.sqrt(252)) if d_std > 0 else 0.0

        # Brier Score (entry_price as predicted prob; outcome=1 if pnl>0)
        brier = mean(
            (t["entry"] - (1.0 if t["pnl"] > 0 else 0.0)) ** 2 for t in closed
        )

        # F1 Score (YES trade → predicted positive; pnl > 0 → actual positive)
        tp = sum(1 for t in closed if t["side"] == "YES" and t["pnl"] > 0)
        fp = sum(1 for t in closed if t["side"] == "YES" and t["pnl"] <= 0)
        fn = sum(1 for t in closed if t["side"] == "NO" and t["pnl"] > 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        # Wilson 95% CI significance
        z = 1.96
        p_hat = wins / n
        center = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
        margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        ci_low = center - margin
        is_sig = ci_low > 0.5
        if is_sig:
            sig_str = "✅"
        elif abs(p_hat - 0.5) > 1e-4:
            n_needed = math.ceil(z**2 / (4 * (p_hat - 0.5) ** 2))
            sig_str = f"❌ (need {n_needed})"
        else:
            sig_str = "❌"

        typer.echo(
            f"{strat:<20} {n:>6} {win_pct:>5.1f}% ${avg_pnl:>7.2f} "
            f"{sharpe:>7.2f} {sortino:>8.2f} {brier:>7.3f} {f1:>6.3f}  {sig_str}"
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

    # Arb strategy summary (excluded from main P&L to avoid distortion)
    if arb_rows:
        typer.echo("")
        typer.echo("Arb Strategy Trades (excluded from main P&L):")
        arb_by_strat: dict[str, list] = {}
        for t, _m in arb_rows:
            s = t.strategy or "arb"
            arb_by_strat.setdefault(s, []).append(t)
        for s, trades_list in sorted(arb_by_strat.items()):
            open_c = sum(1 for t in trades_list if t.status == "open")
            closed_c = sum(1 for t in trades_list if t.status == "closed")
            pnl_sum = sum((t.pnl or 0.0) for t in trades_list if t.status == "closed")
            typer.echo(f"  {s:<22} total={len(trades_list)} open={open_c} closed={closed_c} realized_pnl=${pnl_sum:.2f}")

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
            typer.echo(f"No markets match the {settings.min_resolution_hours}-{settings.max_resolution_hours}h resolution + volume filters.")
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


@app.command("run-intra-event-arb")
def run_intra_event_arb_command() -> None:
    """📊 Module 1: Scan Polymarket multi-outcome events for YES-sum arbitrage."""
    from watchdog.db.init import init_db
    from watchdog.market_data.polymarket_rest import PolymarketRestClient
    from watchdog.strategies.intra_event_arb import IntraEventArbScanner

    settings = get_settings()
    configure_logging(settings)

    if not settings.enable_intra_event_arb:
        typer.echo("📊 ARB SCAN disabled via config (enable_intra_event_arb=False)")
        return

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    rest_client = PolymarketRestClient()
    scanner = IntraEventArbScanner()

    with session_factory() as session:
        scanner.scan(session, rest_client, is_paper=True)


@app.command("run-pair-cost-scan")
def run_pair_cost_scan_command() -> None:
    """🔄 Module 2: Scan binary markets for YES+NO pair cost arbitrage."""
    from watchdog.db.init import init_db
    from watchdog.market_data.polymarket_rest import PolymarketRestClient
    from watchdog.strategies.pair_cost_scanner import PairCostScanner

    settings = get_settings()
    configure_logging(settings)

    if not settings.enable_pair_cost_scan:
        typer.echo("🔄 PAIR COST disabled via config (enable_pair_cost_scan=False)")
        return

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    rest_client = PolymarketRestClient()
    scanner = PairCostScanner()

    with session_factory() as session:
        scanner.scan(session, rest_client, is_paper=True)


@app.command("run-resolution-check")
def run_resolution_check_command() -> None:
    """⏰ Module 3: Flag open trades approaching resolution and act on near-certain/high-risk ones."""
    import logging as _logging

    from watchdog.db.init import init_db
    from watchdog.db.models import Market, Trade
    from watchdog.market_data.polymarket_rest import PolymarketRestClient
    from watchdog.strategies.resolution_filter import ResolutionProximityFilter

    settings = get_settings()
    configure_logging(settings)

    if not settings.enable_resolution_filter:
        typer.echo("⏰ RESOLUTION CHECK disabled via config (enable_resolution_filter=False)")
        return

    _log = _logging.getLogger(__name__)
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    rest_client = PolymarketRestClient()

    with session_factory() as session:
        # Collect unique markets for open paper trades that have yes_token_id
        open_trades = session.query(Trade).filter(Trade.status == "open", Trade.is_paper.is_(True)).all()
        market_ids: set[int] = {t.market_id for t in open_trades}

        current_prices: dict[int, float] = {}
        for market_id in market_ids:
            market = session.get(Market, market_id)
            if market is None or not market.yes_token_id:
                continue
            try:
                book = rest_client.get_orderbook(market.yes_token_id)
                current_prices[market_id] = book["mid"]
            except Exception as exc:
                _log.warning(
                    "RESOLUTION CHECK: failed to fetch price for market_id=%d (%s) — %s",
                    market_id,
                    market.slug,
                    exc,
                )
                # Skip this market — never pass None

        resolution_filter = ResolutionProximityFilter()
        resolution_filter.run(session, current_prices, is_paper=True)


@app.command("run-whale-watch")
def run_whale_watch_command() -> None:
    """🐋 Module 4: Detect large trades on tracked Polymarket markets (observation only)."""
    from watchdog.db.init import init_db
    from watchdog.strategies.whale_detector import WhaleFlowDetector

    settings = get_settings()
    configure_logging(settings)

    if not settings.enable_whale_detector:
        typer.echo("🐋 WHALE WATCH disabled via config (enable_whale_detector=False)")
        return

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    detector = WhaleFlowDetector()

    with session_factory() as session:
        detector.run(session)


@app.command("run-daily-summary")
def run_daily_summary_command() -> None:
    """Print a one-line summary of today's module results."""
    from datetime import UTC, datetime, timedelta

    from watchdog.db.init import init_db
    from watchdog.db.models import ArbOpportunity, Trade, WhaleActivity

    settings = get_settings()
    configure_logging(settings)
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    with session_factory() as session:
        arb_count = (
            session.query(ArbOpportunity)
            .filter(ArbOpportunity.timestamp >= today_start)
            .count()
        )
        pair_cost_count = (
            session.query(ArbOpportunity)
            .filter(
                ArbOpportunity.timestamp >= today_start,
                ArbOpportunity.strategy == "pair_cost_arb",
            )
            .count()
        )
        near_certain_count = (
            session.query(Trade)
            .filter(
                Trade.close_reason == "resolution_near_certain",
                Trade.closed_at >= today_start,
            )
            .count()
        )
        high_risk_count = (
            session.query(Trade)
            .filter(
                Trade.resolution_flag == "high_risk_close",
                Trade.is_paper.is_(True),
            )
            .count()
        )
        whale_count = (
            session.query(WhaleActivity)
            .filter(WhaleActivity.timestamp >= today_start - timedelta(hours=6))
            .count()
        )

    typer.echo(
        f"Today: {arb_count} arb opps found, "
        f"{pair_cost_count} pair cost opps, "
        f"{near_certain_count} near-certain exit(s), "
        f"{high_risk_count} high-risk flagged, "
        f"{whale_count} whale alert(s)"
    )


@app.command("run-ofi-scan")
def run_ofi_scan_command() -> None:
    """📈 Module 5: Compute Order Flow Imbalance for tracked markets."""
    from watchdog.db.init import init_db
    from watchdog.market_data.polymarket_rest import PolymarketRestClient
    from watchdog.strategies.ofi_signal import OFIScanner

    settings = get_settings()
    configure_logging(settings)

    if not settings.enable_ofi_signal:
        typer.echo("📈 OFI SCAN disabled via config (enable_ofi_signal=False)")
        return

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    rest_client = PolymarketRestClient()
    scanner = OFIScanner()

    with session_factory() as session:
        scanner.scan(session, rest_client)


@app.command("run-ensemble-scan")
def run_ensemble_scan_command() -> None:
    """🧠 Module 6: Fetch Metaculus + Manifold ensemble probabilities for tracked markets."""
    from watchdog.db.init import init_db
    from watchdog.market_data.polymarket_rest import PolymarketRestClient
    from watchdog.strategies.ensemble_signal import EnsembleScanner

    settings = get_settings()
    configure_logging(settings)

    if not settings.enable_ensemble_signal:
        typer.echo("🧠 ENSEMBLE SCAN disabled via config (enable_ensemble_signal=False)")
        return

    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)
    rest_client = PolymarketRestClient()
    scanner = EnsembleScanner()

    with session_factory() as session:
        scanner.run(session, rest_client)


@app.command("test-telegram")
def test_telegram_command() -> None:
    """Send a daily and weekly sample Telegram message to verify setup."""
    from watchdog.notifications.telegram import send_telegram

    settings = get_settings()
    configure_logging(settings)

    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        typer.echo("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing)")
        return

    send_telegram(
        "🤖 poly-agent test — Telegram is working!",
        settings.telegram_bot_token,
        settings.telegram_chat_id,
    )
    typer.echo("✅ Daily test message sent")

    weekly_sample = (
        "📅 WEEKLY SUMMARY — week of Mar 10-16\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 Trades closed:     12\n"
        "🏆 Best trade:        +$3.47 (calibration)\n"
        "💀 Worst trade:       -$1.22 (calibration)\n"
        "✅ Win rate:          67%\n"
        "\n"
        "💼 Portfolio performance\n"
        "   $100 → $104.18  (+4.2% this week)\n"
        "   $500 → $520.90  (+4.2% this week)\n"
        "\n"
        "🔍 Signal activity\n"
        "   ARB: avg 3 opps/run, est $0.42/run\n"
        "   OFI: 5 bullish, 2 bearish signals\n"
        "   ENSEMBLE: 4 divergences\n"
        "   WHALES: 1 smart money moves\n"
        "\n"
        "📈 Outlook: 8 open positions tracking\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "[SAMPLE — not real data]"
    )
    send_telegram(weekly_sample, settings.telegram_bot_token, settings.telegram_chat_id)
    typer.echo("✅ Weekly sample message sent")


@app.command("send-daily-telegram")
def send_daily_telegram_command() -> None:
    """Send formatted daily summary to Telegram."""
    from datetime import UTC, datetime

    from watchdog.backtest.portfolio import count_runs_today, update_portfolios
    from watchdog.db.init import init_db
    from watchdog.db.models import ArbOpportunity, EnsembleSignal, OFISignal, Trade, WhaleActivity
    from watchdog.notifications.telegram import send_telegram

    settings = get_settings()
    configure_logging(settings)
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    with session_factory() as session:
        # Arb opportunities today
        arb_count = (
            session.query(ArbOpportunity)
            .filter(ArbOpportunity.timestamp >= today_start)
            .count()
        ) or 0

        arb_profit_sum = (
            session.query(ArbOpportunity)
            .filter(ArbOpportunity.timestamp >= today_start)
            .all()
        )
        daily_est_cents = sum((r.profit_cents or 0.0) for r in arb_profit_sum)

        # OFI signals today
        ofi_bullish = (
            session.query(OFISignal)
            .filter(OFISignal.scanned_at >= today_start, OFISignal.signal == "OFI_BULLISH")
            .count()
        ) or 0
        ofi_bearish = (
            session.query(OFISignal)
            .filter(OFISignal.scanned_at >= today_start, OFISignal.signal == "OFI_BEARISH")
            .count()
        ) or 0

        # Ensemble divergences today
        ens_divs = (
            session.query(EnsembleSignal)
            .filter(EnsembleSignal.scanned_at >= today_start, EnsembleSignal.divergence > 0.05)
            .count()
        ) or 0

        # Whale smart money signals today
        whale_signals = (
            session.query(WhaleActivity)
            .filter(
                WhaleActivity.timestamp >= today_start,
                WhaleActivity.whale_signal == "smart_money_present",
            )
            .count()
        ) or 0

        # Resolution flags today
        resolution_flagged = (
            session.query(Trade)
            .filter(
                Trade.closed_at >= today_start,
                Trade.close_reason.in_(["resolution_near_certain", "resolution_high_risk_partial"]),
            )
            .count()
        ) or 0

        # Net realized PnL (core, non-arb closed trades)
        _arb_strats = {"intra_event_arb", "pair_cost_arb"}
        closed_core = (
            session.query(Trade)
            .filter(Trade.status == "closed", Trade.is_paper.is_(True))
            .all()
        )
        net_pnl = sum(
            (t.pnl or 0.0)
            for t in closed_core
            if (t.strategy or "") not in _arb_strats
        )

        # Portfolio tracker — update both portfolios and get current state
        portfolios = update_portfolios(session)
        run_n = count_runs_today(session)

    def _pnl_str(pnl: float) -> str:
        if pnl == 0.0:
            return "$0.00 (no closed positions)"
        sign = "+" if pnl >= 0 else ""
        return f"{sign}${pnl:.2f}"

    p100 = portfolios.get(100.0)
    p500 = portfolios.get(500.0)

    portfolio_block = (
        "\n💼 PORTFOLIO TRACKER\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        f"💵 $100 Portfolio\n"
        f"   Balance:      ${p100.current_balance:,.2f}\n"
        f"   This run:     {_pnl_str(p100.run_pnl)}\n"
        f"   Total return: {p100.total_return_pct:+.1f}%\n"
        f"   Drawdown:     -{p100.drawdown_pct:.1f}% from peak\n"
        f"\n"
        f"💵 $500 Portfolio\n"
        f"   Balance:      ${p500.current_balance:,.2f}\n"
        f"   This run:     {_pnl_str(p500.run_pnl)}\n"
        f"   Total return: {p500.total_return_pct:+.1f}%\n"
        f"   Drawdown:     -{p500.drawdown_pct:.1f}% from peak\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📅 Run {run_n} of 4 today"
    ) if p100 and p500 else ""

    msg = (
        f"🤖 poly-agent run complete — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"📊 ARB: {arb_count} opps, est ${daily_est_cents / 100:.2f}\n"
        f"📈 OFI: {ofi_bullish} bullish, {ofi_bearish} bearish\n"
        f"🧠 ENSEMBLE: {ens_divs} divergences\n"
        f"🐋 WHALES: {whale_signals} smart money signals\n"
        f"⏰ RESOLUTION: {resolution_flagged} flagged\n"
        f"💰 Net realized PnL: ${net_pnl:.2f}"
        f"{portfolio_block}"
    )

    send_telegram(msg, settings.telegram_bot_token, settings.telegram_chat_id)
    typer.echo(msg)


@app.command("portfolio-status")
def portfolio_status_command() -> None:
    """Print the current state of the $100 and $500 simulated portfolios."""
    from watchdog.backtest.portfolio import (
        STARTING_CAPITALS,
        get_current_portfolios,
        seed_portfolios,
    )
    from watchdog.db.init import init_db

    settings = get_settings()
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        portfolios = get_current_portfolios(session)

        # Auto-seed if either portfolio has no snapshot yet
        needs_seed = any(portfolios.get(cap) is None for cap in STARTING_CAPITALS)
        if needs_seed:
            typer.echo("No portfolio snapshots found — seeding from historical trades...")
            portfolios = seed_portfolios(session)
            typer.echo("Seeding complete.")
            typer.echo("")

    typer.echo("Portfolio Status")
    typer.echo("=" * 40)
    for cap in STARTING_CAPITALS:
        p = portfolios.get(cap)
        if p is None:
            typer.echo(f"  ${cap:.0f} portfolio: no data")
            continue
        run_str = (
            f"+${p.run_pnl:.2f}" if p.run_pnl > 0
            else f"-${abs(p.run_pnl):.2f}" if p.run_pnl < 0
            else "$0.00 (no closed positions)"
        )
        typer.echo(f"  ${cap:.0f} Portfolio")
        typer.echo(f"    Balance:      ${p.current_balance:,.2f}")
        typer.echo(f"    Peak:         ${p.peak_balance:,.2f}")
        typer.echo(f"    Last run PnL: {run_str}")
        typer.echo(f"    Total return: {p.total_return_pct:+.1f}%")
        typer.echo(f"    Drawdown:     -{p.drawdown_pct:.1f}% from peak")
        typer.echo(f"    Last updated: {p.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        typer.echo("")


@app.command("simulate-capital")
def simulate_capital_command(
    capital: Annotated[float, typer.Option(min=1.0, help="Starting capital in USD")] = 100.0,
    strategy: Annotated[str, typer.Option(help="Filter by strategy name (optional)")] = "",
) -> None:
    """Replay closed paper trades through a capital-constrained backtest simulation."""
    import csv
    import math
    from datetime import UTC, datetime

    from sqlalchemy import select

    from watchdog.db.models import Trade

    kelly_cap = 0.20
    min_position = 1.0
    excluded_strategies = {"intra_event_arb"}

    settings = get_settings()
    engine = build_engine(settings)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        q = (
            select(Trade)
            .where(
                Trade.status == "closed",
                Trade.exit_price.is_not(None),
                Trade.entry_price.is_not(None),
            )
            .order_by(Trade.opened_at.asc())
        )
        if strategy:
            q = q.where(Trade.strategy == strategy)
        else:
            q = q.where(
                ~Trade.strategy.in_(list(excluded_strategies))
            )
        trades = session.execute(q).scalars().all()

    if not trades:
        typer.echo("No closed trades found.")
        raise typer.Exit()

    balance = capital
    peak = capital
    max_dd = 0.0
    trades_taken = 0
    trades_skipped = 0
    equity_curve = [capital]
    monthly_start: dict[str, float] = {}
    monthly_end: dict[str, float] = {}
    pnl_list: list[float] = []
    trade_log: list[dict] = []

    for trade in trades:
        kelly = min(trade.kelly_fraction or 0.10, kelly_cap)
        position_size = balance * kelly

        if position_size < min_position:
            trades_skipped += 1
            continue

        entry = trade.entry_price
        exit_ = trade.exit_price
        pnl = position_size * (exit_ - entry) / entry

        prev_balance = balance
        balance = max(0.0, balance + pnl)
        trades_taken += 1
        pnl_list.append(pnl)
        equity_curve.append(balance)

        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        ts = trade.closed_at or trade.opened_at or datetime.now(UTC)
        month_key = ts.strftime("%b %Y")
        if month_key not in monthly_start:
            monthly_start[month_key] = prev_balance
        monthly_end[month_key] = balance

        trade_log.append({
            "trade_id": trade.id,
            "date": ts.strftime("%Y-%m-%d"),
            "strategy": trade.strategy or "calibration",
            "position_size": round(position_size, 4),
            "pnl": round(pnl, 4),
            "running_balance": round(balance, 4),
        })

    total_return = (balance - capital) / capital * 100 if capital > 0 else 0.0
    avg_pnl = sum(pnl_list) / len(pnl_list) if pnl_list else 0.0
    avg_pos = sum(r["position_size"] for r in trade_log) / len(trade_log) if trade_log else 0.0

    neg_pnl = [p for p in pnl_list if p < 0]
    if len(neg_pnl) > 1:
        import statistics
        d_std = statistics.stdev(neg_pnl)
    elif len(neg_pnl) == 1:
        d_std = abs(neg_pnl[0])
    else:
        d_std = 1e-9
    sortino = (avg_pnl / d_std * math.sqrt(252)) if d_std > 0 else 0.0

    typer.echo("")
    typer.echo(f"Capital Simulation — Starting balance: ${capital:.2f}")
    typer.echo("=" * 48)
    typer.echo(f"Trades replayed:     {trades_taken}")
    typer.echo(f"Trades skipped:      {trades_skipped}")
    typer.echo(f"Final balance:       ${balance:.2f}")
    typer.echo(f"Total return:        {total_return:+.1f}%")
    typer.echo(f"Peak balance:        ${peak:.2f}")
    typer.echo(f"Max drawdown:        -{max_dd * 100:.1f}%")
    typer.echo(f"Sortino (live):      {sortino:.2f}")
    typer.echo(f"Avg position size:   ${avg_pos:.2f}")
    typer.echo(f"Avg trade PnL:       ${avg_pnl:.2f}")
    typer.echo("")
    typer.echo("Equity curve (by month):")

    sorted_months = sorted(monthly_start.keys(), key=lambda m: datetime.strptime(m, "%b %Y"))
    for month in sorted_months:
        m_start = monthly_start[month]
        m_end = monthly_end[month]
        m_ret = (m_end - m_start) / m_start * 100 if m_start > 0 else 0.0
        sign = "+" if m_ret >= 0 else ""
        typer.echo(f"  {month}:  ${m_start:.2f} → ${m_end:.2f}  ({sign}{m_ret:.0f}%)")

    typer.echo("")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    today_str = datetime.now(UTC).strftime("%Y%m%d")
    capital_str = str(int(capital))
    csv_path = reports_dir / f"simulation_{capital_str}_{today_str}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["trade_id", "date", "strategy", "position_size", "pnl", "running_balance"],
        )
        writer.writeheader()
        writer.writerows(trade_log)

    typer.echo(f"CSV saved to {csv_path}")


@app.command("send-weekly-telegram")
def send_weekly_telegram_command() -> None:
    """Send a weekly performance summary to Telegram (fires every Sunday)."""
    from datetime import UTC, datetime, timedelta

    from sqlalchemy import select

    from watchdog.backtest.portfolio import STARTING_CAPITALS
    from watchdog.db.init import init_db
    from watchdog.db.models import (
        ArbOpportunity,
        EnsembleSignal,
        OFISignal,
        PortfolioSnapshot,
        Trade,
        WhaleActivity,
    )
    from watchdog.notifications.telegram import send_telegram

    settings = get_settings()
    configure_logging(settings)
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    now = datetime.now(UTC)
    week_start = now - timedelta(days=7)

    # Header dates — "Mar 10-16" or "Mar 29 - Apr 4"
    if week_start.month == now.month:
        date_range = f"{week_start.strftime('%b')} {week_start.day}-{now.day}"
    else:
        date_range = f"{week_start.strftime('%b')} {week_start.day} - {now.strftime('%b')} {now.day}"

    with session_factory() as session:
        # ── Trades closed this week ───────────────────────────────────────
        closed_week = (
            session.query(Trade)
            .filter(
                Trade.status == "closed",
                Trade.is_paper.is_(True),
                Trade.closed_at >= week_start,
            )
            .all()
        )
        total_closed = len(closed_week)
        wins = sum(1 for t in closed_week if (t.pnl or 0.0) > 0)
        win_rate = round(wins / total_closed * 100) if total_closed > 0 else 0

        best = max(closed_week, key=lambda t: t.pnl or 0.0, default=None)
        worst = min(closed_week, key=lambda t: t.pnl or 0.0, default=None)

        def _trade_str(t: Trade | None, sign: str) -> str:
            if t is None:
                return f"{sign}$0.00 (—)"
            pnl = t.pnl or 0.0
            strat = t.strategy or "calibration"
            return f"{'+' if pnl >= 0 else ''}${pnl:.2f} ({strat})"

        # ── Portfolio balances: 7-days-ago vs now ─────────────────────────
        port_lines: list[str] = []
        for cap in STARTING_CAPITALS:
            latest = session.execute(
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.starting_capital == cap)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .limit(1)
            ).scalar_one_or_none()

            oldest_this_week = session.execute(
                select(PortfolioSnapshot)
                .where(
                    PortfolioSnapshot.starting_capital == cap,
                    PortfolioSnapshot.timestamp >= week_start,
                )
                .order_by(PortfolioSnapshot.timestamp.asc())
                .limit(1)
            ).scalar_one_or_none()

            now_bal = latest.current_balance if latest else cap
            then_bal = oldest_this_week.current_balance if oldest_this_week else cap
            week_chg = (now_bal - then_bal) / then_bal * 100 if then_bal > 0 else 0.0
            sign = "+" if week_chg >= 0 else ""
            port_lines.append(
                f"   ${cap:.0f} → ${now_bal:,.2f}  ({sign}{week_chg:.1f}% this week)"
            )

        # ── Run count this week (each run saves a PortfolioSnapshot) ──────
        run_count = (
            session.query(PortfolioSnapshot)
            .filter(
                PortfolioSnapshot.starting_capital == 100.0,
                PortfolioSnapshot.timestamp >= week_start,
            )
            .count()
        ) or 1  # avoid div-by-zero if DB just seeded

        # ── ARB signals ───────────────────────────────────────────────────
        arb_rows = (
            session.query(ArbOpportunity)
            .filter(ArbOpportunity.timestamp >= week_start)
            .all()
        )
        arb_total = len(arb_rows)
        arb_value_usd = sum((r.profit_cents or 0.0) for r in arb_rows) / 100
        arb_per_run = arb_total / run_count
        arb_val_per_run = arb_value_usd / run_count

        # ── OFI signals ───────────────────────────────────────────────────
        ofi_bullish = (
            session.query(OFISignal)
            .filter(OFISignal.scanned_at >= week_start, OFISignal.signal == "OFI_BULLISH")
            .count()
        ) or 0
        ofi_bearish = (
            session.query(OFISignal)
            .filter(OFISignal.scanned_at >= week_start, OFISignal.signal == "OFI_BEARISH")
            .count()
        ) or 0

        # ── Ensemble divergences ──────────────────────────────────────────
        ens_divs = (
            session.query(EnsembleSignal)
            .filter(EnsembleSignal.scanned_at >= week_start, EnsembleSignal.divergence > 0.05)
            .count()
        ) or 0

        # ── Whale smart money moves ───────────────────────────────────────
        whale_moves = (
            session.query(WhaleActivity)
            .filter(
                WhaleActivity.timestamp >= week_start,
                WhaleActivity.whale_signal == "smart_money_present",
            )
            .count()
        ) or 0

        # ── Open positions ────────────────────────────────────────────────
        open_positions = (
            session.query(Trade)
            .filter(Trade.status == "open", Trade.is_paper.is_(True))
            .count()
        ) or 0

    divider = "━━━━━━━━━━━━━━━━━━━━━━━━━━"
    msg = (
        f"📅 WEEKLY SUMMARY — week of {date_range}\n"
        f"{divider}\n"
        f"📊 Trades closed:     {total_closed}\n"
        f"🏆 Best trade:        {_trade_str(best, '+')}\n"
        f"💀 Worst trade:       {_trade_str(worst, '-')}\n"
        f"✅ Win rate:          {win_rate}%\n"
        f"\n"
        f"💼 Portfolio performance\n"
        f"{chr(10).join(port_lines)}\n"
        f"\n"
        f"🔍 Signal activity\n"
        f"   ARB: avg {arb_per_run:.0f} opps/run, est ${arb_val_per_run:.2f}/run\n"
        f"   OFI: {ofi_bullish} bullish, {ofi_bearish} bearish signals\n"
        f"   ENSEMBLE: {ens_divs} divergences\n"
        f"   WHALES: {whale_moves} smart money moves\n"
        f"\n"
        f"📈 Outlook: {open_positions} open positions tracking\n"
        f"{divider}"
    )

    send_telegram(msg, settings.telegram_bot_token, settings.telegram_chat_id)
    typer.echo(msg)


if __name__ == "__main__":
    app()
