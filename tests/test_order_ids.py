from __future__ import annotations

from watchdog.core.order_ids import normalize_platform_order_id


def test_normalize_platform_order_id_wraps_raw_exchange_ids() -> None:
    assert normalize_platform_order_id(
        platform="manifold",
        provider_order_id="abc123",
        fallback_order_id="paper-manifold-fallback",
    ) == "paper-manifold-abc123"


def test_normalize_platform_order_id_preserves_fallback_mode_prefix() -> None:
    assert normalize_platform_order_id(
        platform="polymarket",
        provider_order_id="live-123",
        fallback_order_id="live-polymarket-fallback",
    ) == "live-123"

    assert normalize_platform_order_id(
        platform="polymarket",
        provider_order_id="book-456",
        fallback_order_id="live-polymarket-fallback",
    ) == "live-polymarket-book-456"
