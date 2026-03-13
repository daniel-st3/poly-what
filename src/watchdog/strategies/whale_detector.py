"""Module 4: Smart Whale Watchlist via Polymarket Leaderboard.

Replaces the broken /activity endpoint with:
1. GET https://data-api.polymarket.com/rankings — fetch top-20 wallets by PnL
2. Upsert into smart_wallets table
3. For each tracked market, GET https://data-api.polymarket.com/holders?market=<condition_id>
4. Flag SMART_MONEY_PRESENT if any top-20 wallet holds a position

This module is OBSERVATION ONLY — no paper trades are created.
Full try/except at every HTTP call — graceful degradation if any endpoint fails.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy.orm import Session

from watchdog.db.models import Market, SmartWallet, Trade, WhaleActivity

LOGGER = logging.getLogger(__name__)

RANKINGS_URL = "https://data-api.polymarket.com/rankings"
HOLDERS_URL = "https://data-api.polymarket.com/holders"
REQUEST_TIMEOUT = 15.0
TOP_N_WALLETS = 20


class WhaleFlowDetector:
    """Detect smart-money positions via Polymarket leaderboard and holders API."""

    def run(
        self,
        session: Session,
        threshold_usd: float = 2_000.0,  # kept for API compatibility, unused
    ) -> dict[str, Any]:
        """Fetch leaderboard, upsert smart wallets, detect positions in tracked markets."""
        print("\n🐋 WHALE WATCH")

        # ── Step 1: Fetch leaderboard ─────────────────────────────────────────
        top_wallets: dict[str, int] = {}  # proxy_wallet → rank
        try:
            resp = httpx.get(RANKINGS_URL, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                LOGGER.warning("WHALE WATCH: rankings returned %d — skipping", resp.status_code)
                print(f"🐋 WHALE WATCH done: rankings unavailable (HTTP {resp.status_code})")
                return {"whales_found": 0, "on_tracked_markets": 0}
            raw = resp.json()
            if not isinstance(raw, list):
                LOGGER.warning("WHALE WATCH: unexpected rankings format — skipping")
                print("🐋 WHALE WATCH done: unexpected rankings format")
                return {"whales_found": 0, "on_tracked_markets": 0}

            now = datetime.now(UTC)
            for rank, item in enumerate(raw[:TOP_N_WALLETS], start=1):
                wallet = str(item.get("proxyWallet") or item.get("address") or "").strip()
                if not wallet:
                    continue
                pnl = float(item.get("pnl") or item.get("profit") or 0.0)
                volume = float(item.get("volume") or 0.0)
                top_wallets[wallet] = rank

                # Upsert into smart_wallets table (merge on proxy_wallet)
                existing = (
                    session.query(SmartWallet)
                    .filter(SmartWallet.proxy_wallet == wallet)
                    .first()
                )
                if existing:
                    existing.pnl = pnl
                    existing.volume = volume
                    existing.rank = rank
                    existing.updated_at = now
                else:
                    session.add(
                        SmartWallet(
                            proxy_wallet=wallet,
                            pnl=pnl,
                            volume=volume,
                            rank=rank,
                            updated_at=now,
                        )
                    )
            session.flush()

        except Exception as exc:
            LOGGER.warning("WHALE WATCH: failed to fetch rankings — %s", exc)
            print("🐋 WHALE WATCH done: rankings fetch failed, skipping")
            return {"whales_found": 0, "on_tracked_markets": 0}

        # ── Step 2: Check holders on tracked markets ──────────────────────────
        open_market_ids: set[int] = {
            t.market_id
            for t in session.query(Trade).filter(Trade.status == "open").all()
        }

        smart_money_signals = 0

        for market_id in open_market_ids:
            market: Market | None = session.get(Market, market_id)
            if market is None or not market.condition_id:
                continue

            try:
                holders_resp = httpx.get(
                    HOLDERS_URL,
                    params={"market": market.condition_id},
                    timeout=REQUEST_TIMEOUT,
                )
                if holders_resp.status_code != 200:
                    LOGGER.debug(
                        "WHALE WATCH: holders returned %d for %s — skipping",
                        holders_resp.status_code, market.slug,
                    )
                    continue
                holders = holders_resp.json()
                if not isinstance(holders, list):
                    continue
            except Exception as exc:
                LOGGER.warning("WHALE WATCH: holders fetch failed for %s — %s", market.slug, exc)
                continue

            for holder in holders:
                wallet = str(holder.get("proxyWallet") or holder.get("address") or "").strip()
                if not wallet or wallet not in top_wallets:
                    continue

                rank = top_wallets[wallet]
                smart_money_signals += 1
                print(f"🐋 SMART MONEY: {market.slug} — wallet rank {rank}")

                # Log to whale_activity
                session.add(
                    WhaleActivity(
                        market_slug=market.slug,
                        direction="YES",
                        amount=float(holder.get("amount") or holder.get("value") or 0.0),
                        timestamp=datetime.now(UTC),
                        market_probability_at_time=None,
                        whale_signal="smart_money_present",
                    )
                )

        session.commit()
        print(
            f"🐋 WHALE WATCH done: {len(top_wallets)} wallets tracked, "
            f"{smart_money_signals} smart money signal(s) found"
        )
        return {"whales_found": len(top_wallets), "on_tracked_markets": smart_money_signals}
