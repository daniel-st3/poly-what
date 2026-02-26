from __future__ import annotations

import logging

from watchdog.core.config import Settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(level=settings.log_level.upper(), format=LOG_FORMAT)
