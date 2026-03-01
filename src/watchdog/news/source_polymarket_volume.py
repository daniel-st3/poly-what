from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx

from watchdog.core.config import Settings
from watchdog.news.models import NewsItem

LOGGER = logging.getLogger(__name__)

GAMMA_API_URL = (
    "https://gamma-api.polymarket.com/events"
)

VOLUME_SPIKE_MULTIPLIER = 2.0


async def fetch_polymarket_volume_spikes(
    settings: Settings, limit: int = 20
) -> list[NewsItem]:
    """Scrape the Polymarket gamma API for top markets by 24h volume.

    When a market's 24h volume is > 2x its 7-day average, flag it as a
    'whale activity' signal and create a synthetic NewsEvent.

    No API key required.
    """
    if not settings.polymarket_volume_spike_enabled:
        return []

    params = {
        "active": "true",
        "closed": "false",
        "limit": str(limit),
        "order": "volume24hr",
        "ascending": "false",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(GAMMA_API_URL, params=params)
        response.raise_for_status()
        events = response.json()

    if not isinstance(events, list):
        LOGGER.warning("Polymarket gamma API returned unexpected format: %s", type(events))
        return []

    out: list[NewsItem] = []
    now = datetime.now(UTC)

    for event in events:
        if not isinstance(event, dict):
            continue

        title = str(event.get("title") or event.get("question") or "").strip()
        slug = str(event.get("slug") or "").strip()
        if not title:
            continue

        # Extract volume metrics
        volume_24h = _safe_float(event.get("volume24hr", 0))
        volume_total = _safe_float(event.get("volume", 0))

        # Estimate 7-day average daily volume from total and creation date
        # Some events expose volume7d directly; otherwise approximate
        volume_7d = _safe_float(event.get("volume7d", 0))
        if volume_7d > 0:
            avg_daily_7d = volume_7d / 7.0
        elif volume_total > 0:
            # Rough approximation: assume market has been active at least 7 days
            avg_daily_7d = volume_total / max(7.0, 30.0)
        else:
            avg_daily_7d = 0.0

        # Detect spike: 24h volume > 2x the 7-day daily average
        is_spike = (
            volume_24h > 0
            and avg_daily_7d > 0
            and volume_24h > VOLUME_SPIKE_MULTIPLIER * avg_daily_7d
        )

        if not is_spike:
            continue

        spike_ratio = volume_24h / avg_daily_7d if avg_daily_7d > 0 else 0.0

        raw_text = (
            f"VOLUME SPIKE: {slug or title} | "
            f"24h_vol=${volume_24h:,.0f} | "
            f"7d_avg_daily=${avg_daily_7d:,.0f} | "
            f"spike_ratio={spike_ratio:.1f}x"
        )

        LOGGER.warning(
            "VOLUME SPIKE DETECTED: %s 24h_vol=%.0f spike_ratio=%.1fx",
            slug or title,
            volume_24h,
            spike_ratio,
        )

        out.append(
            NewsItem(
                headline=f"Volume spike: {title}",
                source="polymarket_volume_spike",
                url=f"https://polymarket.com/event/{slug}" if slug else None,
                raw_text=raw_text,
                domain_tags="volume_spike",
                received_at=now,
            )
        )

    LOGGER.info("Polymarket volume scan: %d spike(s) detected out of %d events", len(out), len(events))
    return out


def _safe_float(value: object) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
