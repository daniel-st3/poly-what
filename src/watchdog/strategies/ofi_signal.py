"""Module 5: Order Flow Imbalance (OFI) Signal.

Fetches YES token order book depth from the CLOB API and computes
Order Book Imbalance (OBI) across top 10 price levels.

OBI_shares = (bid_depth - ask_depth) / (bid_depth + ask_depth)
OBI_notional = (bid_notional - ask_notional) / (bid_notional + ask_notional)

Signals:
  OFI_BULLISH  — OBI_shares > 0.5  (bid-heavy 3:1 ratio)
  OFI_BEARISH  — OBI_shares < -0.5 (ask-heavy 3:1 ratio)
  neutral      — otherwise

Used as an entry filter in the calibration pipeline:
  - YES trades require OBI_shares > 0.2
  - NO trades require OBI_shares < -0.2
  - If OFI data is unavailable, entry is allowed (graceful degradation)

Stores results in the ofi_signals table.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from watchdog.db.models import Market, OFISignal, Trade

if TYPE_CHECKING:
    from watchdog.market_data.polymarket_rest import PolymarketRestClient

LOGGER = logging.getLogger(__name__)

OFI_DEDUP_MINUTES = 60  # skip re-scan within this window
OFI_TOP_N_MARKETS = 25  # how many top-volume markets to scan each run
GAMMA_API = "https://gamma-api.polymarket.com"


def _top_market_ids(session: Session, n: int = OFI_TOP_N_MARKETS) -> list[int]:
    """Return up to n market IDs to scan, prioritising top 24h volume from gamma-api.

    Falls back to any DB markets with yes_token_id if gamma-api is unreachable.
    Open-trade markets are always included first.
    """
    # Always include markets with open trades
    open_market_ids: list[int] = [
        t.market_id
        for t in session.query(Trade).filter(Trade.status == "open").all()
    ]

    # Fetch top markets from gamma-api by 24h CLOB volume
    gamma_ids: list[int] = []
    try:
        resp = httpx.get(
            f"{GAMMA_API}/markets",
            params={"closed": "false", "limit": n, "order": "volume24hrClob", "ascending": "false"},
            timeout=10.0,
            verify=False,
        )
        if resp.status_code == 200:
            for m in resp.json():
                cid = m.get("conditionId") or ""
                slug = m.get("slug") or ""
                if not cid and not slug:
                    continue
                # Match to local DB by condition_id or slug
                db_market = None
                if cid:
                    db_market = session.query(Market).filter(Market.condition_id == cid).first()
                if db_market is None and slug:
                    db_market = session.query(Market).filter(Market.slug == slug).first()
                if db_market is not None and db_market.yes_token_id:
                    gamma_ids.append(db_market.id)
    except Exception as exc:
        LOGGER.warning("OFI SCAN: gamma-api fetch failed — %s", exc)

    # Merge: open-trade markets first, then gamma top markets, deduplicated
    seen: set[int] = set()
    result: list[int] = []
    for mid in open_market_ids + gamma_ids:
        if mid not in seen:
            seen.add(mid)
            result.append(mid)
    return result[:n]


class OFIScanner:
    """Compute Order Book Imbalance for Polymarket markets."""

    def compute_obi(
        self,
        token_id: str,
        rest_client: PolymarketRestClient,
    ) -> tuple[float, float, str] | None:
        """Compute OBI for a single token.

        Returns (obi_shares, obi_notional, signal) or None on failure.
        Failure → graceful degradation (allow entry anyway).
        """
        book = rest_client.get_orderbook_levels(token_id, levels=10)
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        if not bids or not asks:
            return None

        try:
            bid_sz = sum(float(b["size"]) for b in bids)
            ask_sz = sum(float(a["size"]) for a in asks)
            total_sz = bid_sz + ask_sz
            obi_shares = (bid_sz - ask_sz) / total_sz if total_sz > 0 else 0.0

            bid_notional = sum(float(b["price"]) * float(b["size"]) for b in bids)
            ask_notional = sum(float(a["price"]) * float(a["size"]) for a in asks)
            total_notional = bid_notional + ask_notional
            obi_notional = (
                (bid_notional - ask_notional) / total_notional
                if total_notional > 0
                else 0.0
            )

            if obi_shares > 0.5:
                signal = "OFI_BULLISH"
            elif obi_shares < -0.5:
                signal = "OFI_BEARISH"
            else:
                signal = "neutral"

            return obi_shares, obi_notional, signal
        except Exception as exc:
            LOGGER.warning("OFI compute_obi failed for token %s: %s", token_id[:12], exc)
            return None

    def scan(
        self,
        session: Session,
        rest_client: PolymarketRestClient,
    ) -> dict[str, Any]:
        """Scan top active markets by volume. Persist OFISignal rows. Return summary."""
        print("\n📈 OFI SCAN")

        market_ids = _top_market_ids(session)

        scanned = 0
        bullish = 0
        bearish = 0
        now = datetime.now(UTC)
        dedup_cutoff = now - timedelta(minutes=OFI_DEDUP_MINUTES)

        for market_id in market_ids:
            market: Market | None = session.get(Market, market_id)
            if market is None or not market.yes_token_id:
                continue

            # Deduplication: skip if scanned within last 60 min
            recent = session.execute(
                select(OFISignal).where(
                    OFISignal.market_id == market_id,
                    OFISignal.scanned_at >= dedup_cutoff,
                )
            ).first()
            if recent:
                continue

            result = self.compute_obi(market.yes_token_id, rest_client)
            if result is None:
                LOGGER.warning("OFI SCAN: no OBI data for %s — skipping", market.slug)
                continue

            obi_shares, obi_notional, signal = result
            scanned += 1

            if signal == "OFI_BULLISH":
                bullish += 1
            elif signal == "OFI_BEARISH":
                bearish += 1

            ofi_row = OFISignal(
                market_id=market_id,
                token_id=market.yes_token_id,
                obi_shares=obi_shares,
                obi_notional=obi_notional,
                signal=signal,
                scanned_at=now,
            )
            session.add(ofi_row)

        session.commit()
        print(
            f"📈 OFI SCAN done: {scanned} markets scanned, "
            f"{bullish} bullish, {bearish} bearish"
        )
        return {"scanned": scanned, "bullish": bullish, "bearish": bearish}
