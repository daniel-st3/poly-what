"""Module 6: Metaculus + Manifold Ensemble Signal.

Fetches external forecasts for open-trade markets and computes an ensemble
probability combining Metaculus (56% weight) and Manifold (44% weight).

P_ensemble = 0.56 * P_metaculus + 0.44 * P_manifold

Flags ENSEMBLE_DIVERGENCE when |P_ensemble - P_polymarket| > 0.05.

Used as a Kelly boost in the calibration pipeline:
  - When ensemble confirms trade direction -> Kelly x 1.2, capped at 0.35

Stores results in the ensemble_signals table.
"""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from sqlalchemy.orm import Session

from watchdog.db.models import EnsembleSignal, Market, Trade

if TYPE_CHECKING:
    from watchdog.market_data.polymarket_rest import PolymarketRestClient

LOGGER = logging.getLogger(__name__)

METACULUS_API = "https://www.metaculus.com/api2/questions/"
MANIFOLD_API = "https://api.manifold.markets/v0/search-markets"
DIVERGENCE_THRESHOLD = 0.05
WORD_OVERLAP_MIN = 0.3
REQUEST_TIMEOUT = 10.0
FETCH_SLEEP_SECONDS = 0.5
CACHE_HOURS = 24
ENSEMBLE_TOP_N_MARKETS = 20  # how many top-volume markets to scan each run
GAMMA_API = "https://gamma-api.polymarket.com"


def _top_market_ids_ensemble(session: Session, n: int = ENSEMBLE_TOP_N_MARKETS) -> list[int]:
    """Return up to n market IDs to scan, prioritising top 24h volume from gamma-api.

    Falls back to any DB markets with a question set.
    Open-trade markets are always included first.
    """
    open_market_ids: list[int] = [
        t.market_id
        for t in session.query(Trade).filter(Trade.status == "open").all()
    ]

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
                db_market = None
                if cid:
                    db_market = session.query(Market).filter(Market.condition_id == cid).first()
                if db_market is None and slug:
                    db_market = session.query(Market).filter(Market.slug == slug).first()
                if db_market is not None and db_market.question:
                    gamma_ids.append(db_market.id)
    except Exception as exc:
        LOGGER.warning("ENSEMBLE SCAN: gamma-api fetch failed — %s", exc)

    seen: set[int] = set()
    result: list[int] = []
    for mid in open_market_ids + gamma_ids:
        if mid not in seen:
            seen.add(mid)
            result.append(mid)
    return result[:n]


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


class EnsembleScanner:
    """Fetch external forecasts and compute ensemble probabilities."""

    def fetch_metaculus(self, title: str) -> float | None:
        """Search Metaculus for a question matching title. Return community probability."""
        try:
            resp = httpx.get(
                METACULUS_API,
                params={"search": title, "format": "json", "limit": "5"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            results = data.get("results") or []
            if not results:
                return None

            # Find best word-overlap match
            best_score = 0.0
            best_prob: float | None = None
            for item in results:
                q_title = item.get("title") or item.get("question", {}).get("title", "")
                score = _word_overlap(title, q_title)
                if score < WORD_OVERLAP_MIN or score <= best_score:
                    continue
                best_score = score
                # Extract community prediction with fallback chain
                cp = item.get("community_prediction")
                if isinstance(cp, dict):
                    full = cp.get("full") or {}
                    p = full.get("q2") or full.get("mean") or cp.get("q2") or cp.get("mean")
                elif isinstance(cp, (int, float)):
                    p = float(cp)
                else:
                    p = None
                if p is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        best_prob = float(p)

            return best_prob
        except Exception as exc:
            LOGGER.debug("fetch_metaculus failed for %r: %s", title[:40], exc)
            return None

    def fetch_manifold(self, title: str) -> float | None:
        """Search Manifold for a market matching title. Return probability."""
        try:
            resp = httpx.get(
                MANIFOLD_API,
                params={"term": title, "limit": "3"},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                return None
            results = resp.json()
            if not isinstance(results, list) or not results:
                return None

            best_score = 0.0
            best_prob: float | None = None
            for item in results:
                q_title = item.get("question") or item.get("slug", "")
                score = _word_overlap(title, q_title)
                if score < WORD_OVERLAP_MIN or score <= best_score:
                    continue
                best_score = score
                p = item.get("probability")
                if p is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        best_prob = float(p)

            return best_prob
        except Exception as exc:
            LOGGER.debug("fetch_manifold failed for %r: %s", title[:40], exc)
            return None

    def compute_ensemble(
        self,
        p_meta: float | None,
        p_mani: float | None,
        p_poly: float,
    ) -> tuple[float, float] | None:
        """Compute weighted ensemble probability and divergence.

        Returns (p_ensemble, divergence) or None if no external data available.
        """
        if p_meta is not None and p_mani is not None:
            p_ens = 0.56 * p_meta + 0.44 * p_mani
        elif p_meta is not None:
            p_ens = p_meta
        elif p_mani is not None:
            p_ens = p_mani
        else:
            return None  # no external data

        divergence = abs(p_ens - p_poly)
        return p_ens, divergence

    def run(
        self,
        session: Session,
        rest_client: PolymarketRestClient,
    ) -> dict[str, Any]:
        """Scan top active markets by volume. Fetch ensemble. Persist EnsembleSignal rows."""
        print("\n🧠 ENSEMBLE SCAN")

        market_ids = _top_market_ids_ensemble(session)

        checked = 0
        divergences = 0
        total_div = 0.0
        now = datetime.now(UTC)

        for market_id in market_ids:
            market: Market | None = session.get(Market, market_id)
            if market is None or not market.question:
                continue

            title = market.question

            # Fetch external forecasts (500ms sleep between markets to avoid rate limits)
            time.sleep(FETCH_SLEEP_SECONDS)
            p_meta = self.fetch_metaculus(title)
            p_mani = self.fetch_manifold(title)

            # Get current CLOB mid as polymarket probability
            p_poly: float | None = None
            if market.yes_token_id:
                try:
                    book = rest_client.get_orderbook(market.yes_token_id)
                    p_poly = book.get("mid")
                except Exception:
                    pass
            if p_poly is None:
                continue

            result = self.compute_ensemble(p_meta, p_mani, p_poly)
            if result is None:
                continue  # no external data for this market

            p_ens, divergence = result
            checked += 1

            if divergence > DIVERGENCE_THRESHOLD:
                divergences += 1
                total_div += divergence
                LOGGER.info(
                    "ENSEMBLE DIVERGENCE: %s p_ens=%.3f p_poly=%.3f div=%.3f",
                    market.slug, p_ens, p_poly, divergence,
                )

            row = EnsembleSignal(
                market_id=market_id,
                p_metaculus=p_meta,
                p_manifold=p_mani,
                p_ensemble=p_ens,
                p_polymarket=p_poly,
                divergence=divergence,
                scanned_at=now,
            )
            session.add(row)

        session.commit()

        avg_div_cents = (total_div / divergences * 100) if divergences > 0 else 0.0
        print(
            f"🧠 ENSEMBLE SCAN done: {checked} markets checked, "
            f"{divergences} divergences found, avg {avg_div_cents:.1f}¢"
        )
        return {
            "checked": checked,
            "divergences": divergences,
            "avg_divergence_cents": avg_div_cents,
        }

    def get_cached_ensemble(
        self,
        session: Session,
        market_id: int,
    ) -> float | None:
        """Return the most recent ensemble probability for a market (within 24h)."""
        cutoff = datetime.now(UTC) - timedelta(hours=CACHE_HOURS)
        row = (
            session.query(EnsembleSignal)
            .filter(
                EnsembleSignal.market_id == market_id,
                EnsembleSignal.scanned_at >= cutoff,
            )
            .order_by(EnsembleSignal.scanned_at.desc())
            .first()
        )
        return row.p_ensemble if row is not None else None
