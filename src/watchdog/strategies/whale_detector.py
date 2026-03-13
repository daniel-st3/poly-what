"""Module 4: Whale Flow Detector (Observation Only).

Fetches recent large trades from Polymarket's public activity API.
Filters for trades > $2,000 on markets we currently track.
Logs to whale_activity table and prints alerts.

This module is OBSERVATION ONLY — no paper trades are created.
After 2 weeks of data collection, the signal quality will be evaluated.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy.orm import Session

from watchdog.db.models import Market, Trade, WhaleActivity

LOGGER = logging.getLogger(__name__)

ACTIVITY_URL = "https://data-api.polymarket.com/activity"
DEFAULT_THRESHOLD_USD = 2_000.0
WHALE_ALERT_WINDOW_HOURS = 6
REQUEST_TIMEOUT = 15.0


class WhaleFlowDetector:
    """Detect and log large trades on tracked Polymarket markets."""

    def run(
        self,
        session: Session,
        threshold_usd: float = DEFAULT_THRESHOLD_USD,
    ) -> dict[str, Any]:
        """Fetch recent activity, log whale trades on tracked markets."""
        print("\n🐋 WHALE WATCH")

        # Fetch activity — fail gracefully on any error
        raw_activity: list[dict[str, Any]] = []
        try:
            resp = httpx.get(
                ACTIVITY_URL,
                params={"limit": "50", "offset": "0"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                LOGGER.warning(
                    "WHALE WATCH: endpoint returned %d — skipping", resp.status_code
                )
                print(f"🐋 WHALE WATCH done: endpoint unavailable (HTTP {resp.status_code})")
                return {"whales_found": 0, "on_tracked_markets": 0}
            raw_activity = resp.json()
            if not isinstance(raw_activity, list):
                LOGGER.warning("WHALE WATCH: unexpected response format — skipping")
                print("🐋 WHALE WATCH done: unexpected response format")
                return {"whales_found": 0, "on_tracked_markets": 0}
        except Exception as exc:
            LOGGER.warning("WHALE WATCH: fetch failed — %s", exc)
            print("🐋 WHALE WATCH done: fetch failed, skipping")
            return {"whales_found": 0, "on_tracked_markets": 0}

        # Build set of slugs we currently track (open positions)
        tracked_slugs: set[str] = set()
        open_market_ids: set[int] = set()
        open_trades = session.query(Trade).filter(Trade.status == "open").all()
        for t in open_trades:
            open_market_ids.add(t.market_id)

        tracked_markets: dict[str, Market] = {}
        for market_id in open_market_ids:
            m = session.get(Market, market_id)
            if m:
                tracked_slugs.add(m.slug)
                tracked_markets[m.slug] = m
                if m.condition_id:
                    tracked_markets[m.condition_id] = m

        whales_found = 0
        on_tracked = 0
        now = datetime.now(UTC)
        alert_cutoff = now - timedelta(hours=WHALE_ALERT_WINDOW_HOURS)

        for item in raw_activity:
            try:
                amount = float(item.get("size") or item.get("amount") or 0)
            except (TypeError, ValueError):
                continue

            if amount < threshold_usd:
                continue

            whales_found += 1

            # Resolve market slug from asset field
            asset = str(item.get("asset") or item.get("conditionId") or item.get("market") or "")
            market_obj = tracked_markets.get(asset)
            market_slug = market_obj.slug if market_obj else asset

            # Determine direction
            side_raw = str(item.get("side") or item.get("outcome") or "BUY").upper()
            direction = "YES" if side_raw in ("BUY", "YES") else "NO"
            whale_signal = "buy_pressure" if direction == "YES" else "sell_pressure"

            # Parse timestamp
            ts_raw = item.get("timestamp") or item.get("createdAt") or item.get("time")
            try:
                if isinstance(ts_raw, (int, float)):
                    ts = datetime.fromtimestamp(ts_raw, tz=UTC)
                elif isinstance(ts_raw, str):
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                else:
                    ts = now
            except Exception:
                ts = now

            prob = None
            try:
                prob = float(item.get("price") or item.get("probability") or 0)
            except (TypeError, ValueError):
                pass

            # Log to whale_activity
            record = WhaleActivity(
                market_slug=market_slug,
                direction=direction,
                amount=amount,
                timestamp=ts,
                market_probability_at_time=prob,
                whale_signal=whale_signal,
            )
            session.add(record)

            # Alert if this is on a tracked market and within the 6h window
            if market_obj is not None:
                on_tracked += 1
                if ts >= alert_cutoff:
                    prob_str = f"prob={prob:.3f}" if prob is not None else "prob=unknown"
                    print(
                        f"🐋 WHALE ALERT: {market_slug} — {direction} ${amount:,.0f} at {prob_str}"
                    )

        session.commit()
        print(
            f"🐋 WHALE WATCH done: {whales_found} whale trade(s) found, "
            f"{on_tracked} on tracked market(s)"
        )
        return {"whales_found": whales_found, "on_tracked_markets": on_tracked}
