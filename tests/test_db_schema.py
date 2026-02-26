from __future__ import annotations

from sqlalchemy import inspect

from watchdog.core.config import Settings
from watchdog.db.init import init_db
from watchdog.db.session import build_engine

REQUIRED_TABLES = {
    "markets",
    "market_snapshots",
    "news_events",
    "signals",
    "trades",
    "telemetry",
    "calibration_surface",
    "maker_quotes",
}


def test_required_tables_exist() -> None:
    settings = Settings(database_url="sqlite:///:memory:")
    engine = build_engine(settings)
    init_db(engine)

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert REQUIRED_TABLES.issubset(tables)
