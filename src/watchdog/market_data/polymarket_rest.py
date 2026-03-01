"""Lightweight Polymarket REST API client using public endpoints.

Uses Gamma API for market discovery and CLOB API for orderbook data.
No authentication required — all endpoints are public.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class PolymarketRestClient:
    """Fetch active Polymarket markets and orderbook data via public REST APIs."""

    def __init__(self, timeout: float = 15.0) -> None:
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Market discovery (Gamma API)
    # ------------------------------------------------------------------

    def list_active_markets(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return active, open markets sorted by 24h volume descending."""
        params: dict[str, str] = {
            "limit": str(limit),
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        }
        resp = httpx.get(
            f"{GAMMA_BASE}/markets",
            params=params,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        raw: list[dict[str, Any]] = resp.json()

        markets: list[dict[str, Any]] = []
        for item in raw:
            # clobTokenIds can be a JSON string or a list
            clob_ids = item.get("clobTokenIds") or []
            if isinstance(clob_ids, str):
                try:
                    clob_ids = json.loads(clob_ids)
                except (json.JSONDecodeError, TypeError):
                    clob_ids = []

            if not clob_ids or not item.get("endDate"):
                continue

            markets.append(
                {
                    "slug": item.get("slug") or item.get("conditionId", ""),
                    "question": item.get("question") or "",
                    "end_date": item.get("endDate"),
                    "volume_24h": float(item.get("volume24hr") or 0),
                    "liquidity": float(item.get("liquidity") or 0),
                    "yes_token_id": clob_ids[0] if len(clob_ids) > 0 else None,
                    "no_token_id": clob_ids[1] if len(clob_ids) > 1 else None,
                    "image": item.get("image"),
                }
            )

        return markets

    # ------------------------------------------------------------------
    # Orderbook (CLOB API)
    # ------------------------------------------------------------------

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        """Fetch top-of-book for a given outcome token."""
        resp = httpx.get(
            f"{CLOB_BASE}/book",
            params={"token_id": token_id},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        book = resp.json()

        bids = book.get("bids") or []
        asks = book.get("asks") or []

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 1.0
        mid = (best_bid + best_ask) / 2

        return {
            "bid": best_bid,
            "ask": best_ask,
            "mid": mid,
            "spread": best_ask - best_bid,
            "probability": mid,
            "bid_volume": sum(float(b.get("size", 0)) for b in bids[:5]),
            "ask_volume": sum(float(a.get("size", 0)) for a in asks[:5]),
        }
