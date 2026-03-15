"""Module 4: Smart Whale Watchlist via Polymarket Holders API.

Strategy:
1. Fetch top active markets from gamma-api (by volume)
2. For each market, GET https://data-api.polymarket.com/holders?market=<condition_id>
3. Flag any holder with position > WHALE_THRESHOLD_USD as WHALE_ALERT

Note: The /rankings endpoint (used previously) no longer exists (HTTP 404).
This implementation detects large positions directly via the holders endpoint.

This module is OBSERVATION ONLY — no paper trades are created.
Full try/except at every HTTP call — graceful degradation if any endpoint fails.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy.orm import Session

from watchdog.db.models import Market, SmartWallet, Trade, WhaleActivity

LOGGER = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
HOLDERS_URL = "https://data-api.polymarket.com/holders"
REQUEST_TIMEOUT = 15.0
TOP_N_MARKETS = 30       # fetch this many top markets from gamma-api
WHALE_THRESHOLD_USD = 5_000.0  # minimum position size to flag as whale


def _fetch_top_condition_ids(n: int = TOP_N_MARKETS) -> list[dict[str, Any]]:
    """Return top-N active markets from gamma-api sorted by 24h CLOB volume.

    Each dict has keys: conditionId, slug, question, yes_token_id, volume24hr.
    """
    try:
        resp = httpx.get(
            f"{GAMMA_API}/markets",
            params={"closed": "false", "limit": n, "order": "volume24hrClob", "ascending": "false"},
            timeout=REQUEST_TIMEOUT,
            verify=False,
        )
        if resp.status_code != 200:
            return []
        markets = resp.json()
        if not isinstance(markets, list):
            return []
        result = []
        for m in markets:
            cid = m.get("conditionId") or ""
            if not cid:
                continue
            clob_ids_raw = m.get("clobTokenIds") or "[]"
            try:
                clob_ids = json.loads(clob_ids_raw) if isinstance(clob_ids_raw, str) else clob_ids_raw
            except Exception:
                clob_ids = []
            yes_token = clob_ids[0] if clob_ids else None
            result.append({
                "conditionId": cid,
                "slug": m.get("slug", ""),
                "question": m.get("question", ""),
                "yes_token_id": yes_token,
                "volume24hr": float(m.get("volume24hrClob") or 0.0),
            })
        return result
    except Exception as exc:
        LOGGER.warning("WHALE WATCH: gamma-api fetch failed — %s", exc)
        return []


class WhaleFlowDetector:
    """Detect large positions on active Polymarket markets via the holders API."""

    def run(
        self,
        session: Session,
        threshold_usd: float = WHALE_THRESHOLD_USD,
    ) -> dict[str, Any]:
        """Fetch top markets, check holder sizes, flag whale-level positions."""
        print("\n🐋 WHALE WATCH")

        # ── Step 1: Get top active markets ────────────────────────────────────
        top_markets = _fetch_top_condition_ids()
        if not top_markets:
            print("🐋 WHALE WATCH done: could not fetch active markets from gamma-api")
            return {"whales_found": 0, "on_tracked_markets": 0}

        print(f"🐋 Scanning {len(top_markets)} active markets for whale positions (>${threshold_usd:,.0f})")

        # ── Step 2: For each market, check holders ─────────────────────────────
        whale_signals = 0
        markets_with_whales = 0
        now = datetime.now(UTC)

        for mkt in top_markets:
            cid = mkt["conditionId"]
            slug = mkt["slug"]
            question = mkt["question"]

            try:
                holders_resp = httpx.get(
                    HOLDERS_URL,
                    params={"market": cid},
                    timeout=REQUEST_TIMEOUT,
                )
                if holders_resp.status_code != 200:
                    LOGGER.debug(
                        "WHALE WATCH: holders returned %d for %s — skipping",
                        holders_resp.status_code, slug,
                    )
                    continue
                data = holders_resp.json()
                # Response: [{"token": "...", "holders": [{"proxyWallet": "...", "amount": ...}, ...]}, ...]
                if not isinstance(data, list):
                    continue
            except Exception as exc:
                LOGGER.warning("WHALE WATCH: holders fetch failed for %s — %s", slug, exc)
                continue

            market_flagged = False
            for token_data in data:
                for holder in token_data.get("holders", []):
                    wallet = str(holder.get("proxyWallet") or "").strip()
                    amount = float(holder.get("amount") or 0.0)
                    name = str(holder.get("name") or holder.get("pseudonym") or wallet[:10])

                    if amount < threshold_usd:
                        continue

                    whale_signals += 1
                    if not market_flagged:
                        markets_with_whales += 1
                        market_flagged = True

                    print(f"🐋 WHALE: {question[:50]!r} — {name} holds ${amount:,.0f}")

                    # Upsert SmartWallet (repurposed as large-position tracker)
                    existing = (
                        session.query(SmartWallet)
                        .filter(SmartWallet.proxy_wallet == wallet)
                        .first()
                    )
                    if existing:
                        existing.pnl = amount   # amount treated as proxy for "size"
                        existing.volume = amount
                        existing.updated_at = now
                    else:
                        session.add(
                            SmartWallet(
                                proxy_wallet=wallet,
                                pnl=amount,
                                volume=amount,
                                rank=0,
                                updated_at=now,
                            )
                        )

                    # Log to whale_activity
                    session.add(
                        WhaleActivity(
                            market_slug=slug,
                            direction="YES",
                            amount=amount,
                            timestamp=now,
                            market_probability_at_time=None,
                            whale_signal="large_position",
                        )
                    )

        session.commit()
        print(
            f"🐋 WHALE WATCH done: {len(top_markets)} markets checked, "
            f"{markets_with_whales} with whale positions, {whale_signals} whale alert(s)"
        )
        return {"whales_found": whale_signals, "on_tracked_markets": markets_with_whales}
