from __future__ import annotations

from sqlalchemy.engine import Engine

from watchdog.db import models  # noqa: F401
from watchdog.db.base import Base


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)
