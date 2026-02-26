from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class NewsItem:
    headline: str
    source: str
    url: str | None
    raw_text: str | None
    domain_tags: str | None
    received_at: datetime
