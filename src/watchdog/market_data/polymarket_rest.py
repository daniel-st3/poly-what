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
        markets: list[dict[str, Any]] = []
        offset = 0
        batch_size = 100

        while len(markets) < limit:
            fetch_limit = min(batch_size, limit - len(markets))
            params: dict[str, str] = {
                "limit": str(fetch_limit),
                "offset": str(offset),
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
            
            if not raw:
                break

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
                        "condition_id": item.get("conditionId"),
                        "end_date": item.get("endDate"),
                        "volume_24h": float(item.get("volume24hr") or 0),
                        "liquidity": float(item.get("liquidity") or 0),
                        "yes_token_id": clob_ids[0] if len(clob_ids) > 0 else None,
                        "no_token_id": clob_ids[1] if len(clob_ids) > 1 else None,
                        "image": item.get("image"),
                    }
                )
            
            offset += len(raw)

        return markets[:limit]

    # ------------------------------------------------------------------
    # Orderbook (CLOB API)
    # ------------------------------------------------------------------

    def get_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return active multi-outcome events from the Gamma API."""
        events: list[dict[str, Any]] = []
        offset = 0
        batch_size = 100

        while len(events) < limit:
            fetch_limit = min(batch_size, limit - len(events))
            params: dict[str, str] = {
                "limit": str(fetch_limit),
                "offset": str(offset),
                "active": "true",
                "closed": "false",
            }
            resp = httpx.get(
                f"{GAMMA_BASE}/events",
                params=params,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw: list[dict[str, Any]] = resp.json()

            if not raw:
                break

            for item in raw:
                markets = item.get("markets") or []
                if len(markets) < 2:
                    continue
                events.append(
                    {
                        "event_slug": item.get("slug") or item.get("id", ""),
                        "title": item.get("title") or "",
                        "markets": [
                            {
                                "slug": m.get("slug") or m.get("conditionId", ""),
                                "question": m.get("question") or "",
                                "condition_id": m.get("conditionId"),
                                "outcome_prices": m.get("outcomePrices") or [],
                                "volume": float(m.get("volume") or 0),
                                "yes_token_id": (
                                    json.loads(m["clobTokenIds"])[0]
                                    if isinstance(m.get("clobTokenIds"), str) and m.get("clobTokenIds")
                                    else (m.get("clobTokenIds") or [None])[0]
                                ),
                            }
                            for m in markets
                        ],
                    }
                )

            offset += len(raw)
            if len(raw) < fetch_limit:
                break

        return events[:limit]

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

    def get_clob_executable_price(self, token_id: str, side: str = "BUY") -> float | None:
        """Return the real executable price from CLOB (best_ask for BUY, best_bid for SELL).

        Logs a warning if the executable price differs from mid by more than 2¢.
        Returns None on any failure — callers should fall back to mid price.
        """
        try:
            book = self.get_orderbook(token_id)
            mid = book.get("mid", 0.5)
            price = book["ask"] if side.upper() == "BUY" else book["bid"]
            if abs(price - mid) > 0.02:
                LOGGER.warning(
                    "CLOB executable %.4f vs mid %.4f >2¢ (token=%s side=%s)",
                    price,
                    mid,
                    token_id[:12],
                    side,
                )
            return float(price)
        except Exception as exc:
            LOGGER.warning("get_clob_executable_price failed for %s: %s", token_id[:12], exc)
            return None

    def get_orderbook_levels(self, token_id: str, levels: int = 10) -> dict[str, list]:
        """Return sorted bids/asks lists (up to `levels` each) for OFI computation."""
        try:
            resp = httpx.get(
                f"{CLOB_BASE}/book",
                params={"token_id": token_id},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            bids = sorted(data.get("bids", []), key=lambda x: -float(x["price"]))[:levels]
            asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))[:levels]
            return {"bids": bids, "asks": asks}
        except Exception as exc:
            LOGGER.warning("get_orderbook_levels failed for %s: %s", token_id[:12], exc)
            return {"bids": [], "asks": []}
