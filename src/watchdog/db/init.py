from __future__ import annotations

import contextlib
import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

from watchdog.db import models  # noqa: F401
from watchdog.db.base import Base

LOGGER = logging.getLogger(__name__)

# Columns added after initial schema — ALTER TABLE adds them to existing DBs.
_MIGRATIONS = [
    "ALTER TABLE trades ADD COLUMN high_water_mark REAL",
    "ALTER TABLE trades ADD COLUMN remaining_fraction REAL DEFAULT 1.0",
    "ALTER TABLE trades ADD COLUMN resolution_flag TEXT",
]


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)
    _run_migrations(engine)


def _run_migrations(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in _MIGRATIONS:
            with contextlib.suppress(Exception):
                conn.execute(text(stmt))
