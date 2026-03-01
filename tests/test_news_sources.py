from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from watchdog.core.config import Settings
from watchdog.news.source_gdelt_gkg import fetch_gdelt_gkg
from watchdog.news.source_marketaux import fetch_marketaux
from watchdog.news.source_polymarket_volume import fetch_polymarket_volume_spikes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data: dict | list, status_code: int = 200) -> httpx.Response:
    """Build a fake httpx.Response for mocking."""
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "https://mock"),
    )


def _base_settings(**overrides) -> Settings:
    defaults = dict(
        database_url="sqlite:///:memory:",
        marketaux_enabled=True,
        marketaux_api_key="test-key-123",
        polymarket_volume_spike_enabled=True,
        gdelt_gkg_enabled=True,
        gdelt_enabled=False,
        rss_enabled=False,
        reddit_enabled=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Marketaux tests
# ---------------------------------------------------------------------------

MARKETAUX_RESPONSE = {
    "data": [
        {
            "title": "Bitcoin ETF approved by SEC",
            "description": "The SEC approved the bitcoin ETF application.",
            "url": "https://example.com/btc-etf",
            "entities": [
                {"symbol": "BTC", "sentiment_score": 0.85},
                {"symbol": "ETH", "sentiment_score": 0.42},
            ],
        },
        {
            "title": "Fed holds rates steady",
            "description": "Federal Reserve maintained interest rates.",
            "url": "https://example.com/fed-rates",
            "entities": [],
        },
        {
            "title": "",  # empty title — should be skipped
            "description": "No title",
            "url": "https://example.com/empty",
        },
    ]
}


@pytest.mark.asyncio
async def test_marketaux_source_parses_response() -> None:
    settings = _base_settings()

    with patch("watchdog.news.source_marketaux.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(MARKETAUX_RESPONSE)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        items = await fetch_marketaux(settings)

    assert len(items) == 2
    assert items[0].headline == "Bitcoin ETF approved by SEC"
    assert items[0].source == "marketaux"
    assert "BTC:0.85" in items[0].raw_text
    assert items[1].headline == "Fed holds rates steady"
    assert items[1].source == "marketaux"


@pytest.mark.asyncio
async def test_marketaux_source_disabled() -> None:
    settings = _base_settings(marketaux_enabled=False)
    items = await fetch_marketaux(settings)
    assert items == []


@pytest.mark.asyncio
async def test_marketaux_source_no_api_key() -> None:
    settings = _base_settings(marketaux_api_key=None)
    items = await fetch_marketaux(settings)
    assert items == []


# ---------------------------------------------------------------------------
# Polymarket volume spike tests
# ---------------------------------------------------------------------------

VOLUME_SPIKE_RESPONSE = [
    {
        "title": "Will Trump win 2026?",
        "slug": "will-trump-win-2026",
        "volume24hr": 500000,
        "volume7d": 700000,  # avg daily = 100k, spike = 5x
        "volume": 5000000,
    },
    {
        "title": "Will ETH reach 10k?",
        "slug": "will-eth-reach-10k",
        "volume24hr": 10000,
        "volume7d": 350000,  # avg daily = 50k, no spike (10k < 100k)
        "volume": 1000000,
    },
    {
        "title": "Normal market",
        "slug": "normal-market",
        "volume24hr": 5000,
        "volume7d": 70000,  # avg daily = 10k, 5k < 20k, no spike
        "volume": 500000,
    },
]


@pytest.mark.asyncio
async def test_polymarket_volume_spike_detects_spike() -> None:
    settings = _base_settings()

    with patch("watchdog.news.source_polymarket_volume.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(VOLUME_SPIKE_RESPONSE)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        items = await fetch_polymarket_volume_spikes(settings)

    # Only the first event should be flagged (500k > 2 * 100k)
    assert len(items) == 1
    assert items[0].source == "polymarket_volume_spike"
    assert "will-trump-win-2026" in (items[0].raw_text or "")
    assert items[0].domain_tags == "volume_spike"


@pytest.mark.asyncio
async def test_polymarket_volume_no_spike() -> None:
    """All markets with normal volume → no spike items."""
    settings = _base_settings()
    normal_data = [
        {
            "title": "Normal market",
            "slug": "normal",
            "volume24hr": 1000,
            "volume7d": 70000,  # avg daily = 10k, 1k < 20k
            "volume": 500000,
        }
    ]

    with patch("watchdog.news.source_polymarket_volume.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(normal_data)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        items = await fetch_polymarket_volume_spikes(settings)

    assert len(items) == 0


@pytest.mark.asyncio
async def test_polymarket_volume_spike_disabled() -> None:
    settings = _base_settings(polymarket_volume_spike_enabled=False)
    items = await fetch_polymarket_volume_spikes(settings)
    assert items == []


# ---------------------------------------------------------------------------
# GDELT GKG tests
# ---------------------------------------------------------------------------

GDELT_GKG_RESPONSE = {
    "articles": [
        {
            "title": "Inflation hits new high",
            "url": "https://example.com/inflation",
            "tone": -3.5,
            "seendate": "20260301T000000Z",
            "domain": "reuters.com",
        },
        {
            "title": "Bitcoin surges past 100k",
            "url": "https://example.com/btc-surge",
            "tone": 7.2,
            "seendate": "20260301T010000Z",
            "domain": "coindesk.com",
        },
        {
            "title": "",  # empty title — skip
            "url": "https://example.com/empty",
            "tone": 0.0,
        },
    ]
}


@pytest.mark.asyncio
async def test_gdelt_gkg_parses_articles() -> None:
    settings = _base_settings()

    with patch("watchdog.news.source_gdelt_gkg.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(GDELT_GKG_RESPONSE)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        items = await fetch_gdelt_gkg(settings, keywords=["inflation"])

    assert len(items) == 2
    assert items[0].headline == "Inflation hits new high"
    assert items[0].source == "gdelt_gkg"
    assert "tone=-3.50" in items[0].raw_text
    assert items[1].headline == "Bitcoin surges past 100k"
    assert "tone=7.20" in items[1].raw_text


@pytest.mark.asyncio
async def test_gdelt_gkg_disabled() -> None:
    settings = _base_settings(gdelt_gkg_enabled=False)
    items = await fetch_gdelt_gkg(settings)
    assert items == []


@pytest.mark.asyncio
async def test_gdelt_gkg_dedupes_by_url() -> None:
    """Duplicate URLs across keywords should be deduped."""
    settings = _base_settings()
    duplicate_response = {
        "articles": [
            {
                "title": "Same article",
                "url": "https://example.com/same",
                "tone": 1.0,
            }
        ]
    }

    with patch("watchdog.news.source_gdelt_gkg.httpx.AsyncClient") as MockClient:
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(duplicate_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        items = await fetch_gdelt_gkg(settings, keywords=["crypto", "bitcoin"])

    # Same URL from both keywords → only 1 item
    assert len(items) == 1
