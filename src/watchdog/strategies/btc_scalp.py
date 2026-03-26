"""
BTC intraday scalp strategy — V2.

Connects to Coinbase Advanced Trade WebSocket for live BTC price,
polls Polymarket Gamma API every 5s for near-expiry BTC markets,
and paper-trades using a fee-aware EV model anchored to a start-price proxy.

Price feed priority:
  1. Coinbase Advanced Trade WebSocket (wss://advanced-trade-ws.coinbase.com) — no geo-block
  2. CoinGecko HTTP fallback (30s intervals) — if WebSocket fails 3 consecutive times

V2 probability model:
  z = log(current_price / start_price_proxy)
      / max(vol_per_sqrt_min * sqrt(time_left_min / 5.0), vol_floor)
  p_up = normal_cdf(z) + momentum * weight   (clamped 0.01-0.99)
  EV_buy_up = p_up - ask_up                  (per $1-payout contract)

NOTE: start_price_proxy is the BTC/USD price captured when the market first
entered the active trading window — NOT the Chainlink on-chain reference used
by Polymarket for final settlement. See BtcScanLog docstring.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import random
import re
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

import aiohttp
import httpx
import websockets

from watchdog.core.config import get_settings
from watchdog.db.init import init_db
from watchdog.db.models import BtcScanLog, BtcSignalLog, Market, Trade
from watchdog.db.session import build_engine, build_session_factory
from watchdog.market_data.polymarket_rest import PolymarketRestClient

LOGGER = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
WS_MAX_FAILURES = 3
HTTP_FALLBACK_INTERVAL_S = 30
POLL_INTERVAL_S = 5
RESOLUTION_CHECK_INTERVAL_S = 60
PAPER_POSITION_SIZE = 2.0
SIGNAL_COOLDOWN_S = 90.0
CLOB_FETCH_TIMEOUT = 4.0   # seconds for synchronous CLOB call via asyncio.to_thread
# V1 legacy constants — kept for backward-compat with _compute_signal() and backtest helpers
_V1_MIN_EDGE = 0.10
_V1_CERTAINTY_THRESHOLD = 0.90
_V1_EDGE_THRESHOLD = 0.08
# Minimum acceptable effective ask recovered from depth (strategy safety invariant).
# Based on observed anomalous depth-recovery behavior in production (stale micro-orders
# at 0.01-0.02 that pass _first_usable_quote but produce implausibly large EV).
# Not a claim that such prices are universally impossible on Polymarket.
# Calibrated to the BTC 5-minute scalp trading range (normal range: 0.30-0.70).
_EFF_ASK_FLOOR = 0.10


class BtcScalpStrategy:
    def __init__(self) -> None:
        self._btc_price: float | None = None
        self._price_history: deque[float] = deque(maxlen=60)
        self._signals: int = 0
        self._trades_opened: int = 0
        self._trades_closed: int = 0
        self._pnl: float = 0.0
        self._has_sent_online_ping: bool = False
        self._fired_signals: dict[str, float] = {}  # signal_key → timestamp
        self._open_risk_cap_notified: bool = False
        # Per-window start prices: slug → (btc_price, captured_at_utc)
        self._window_start_prices: dict[str, tuple[float, datetime]] = {}
        # Single reused CLOB client — do NOT reinstantiate per scan cycle
        self._clob_client = PolymarketRestClient(timeout=CLOB_FETCH_TIMEOUT)
        _s = get_settings()
        self._tg_token: str | None = _s.telegram_bot_token
        self._tg_chat: str | None = _s.telegram_chat_id
        # Ensure all tables (trades, markets, …) exist before any DB access
        _engine = build_engine(_s)
        init_db(_engine)
        print("[BTC Scalp] DB initialised ✅", flush=True)

    def _notify(self, msg: str) -> None:
        """Fire-and-forget Telegram notification. Swallows all errors."""
        from watchdog.notifications.telegram import send_telegram
        try:
            send_telegram(msg, self._tg_token, self._tg_chat)
            print("[Notify] Telegram send succeeded", flush=True)
        except Exception as exc:
            print(f"[Notify] Telegram send failed: {exc}", flush=True)
            LOGGER.warning("Telegram notification failed: %s", exc)

    # ── Paper stats ─────────────────────────────────────────────────────────

    def _btc_paper_stats(self) -> dict[str, Any]:
        """Query DB for btc_scalp paper trade stats (all-time and today UTC).

        Semantics:
        - alltime_closed / today_closed: count of ALL closed trades regardless of pnl
        - wins/losses/be: only from closed trades where pnl is not None
          (pnl > 0 → win, pnl < 0 → loss, pnl == 0.0 → breakeven)
        - pnl sums: only from closed trades where pnl is not None
        """
        settings = get_settings()
        engine = build_engine(settings)
        session_factory = build_session_factory(engine)
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

        with session_factory() as session:
            closed = (
                session.query(Trade)
                .filter(
                    Trade.strategy == "btc_scalp",
                    Trade.is_paper.is_(True),
                    Trade.status == "closed",
                )
                .all()
            )
            open_trades = (
                session.query(Trade)
                .filter(
                    Trade.strategy == "btc_scalp",
                    Trade.is_paper.is_(True),
                    Trade.status == "open",
                )
                .all()
            )

        def _as_utc(dt: datetime) -> datetime:
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

        today_closed_list = [
            t for t in closed
            if t.closed_at is not None and _as_utc(t.closed_at) >= today_start
        ]

        def _tally(rows: list[Trade]) -> tuple[float, int, int, int]:
            """Return (pnl_sum, wins, losses, be) for rows with non-null pnl."""
            pnl_rows = [t for t in rows if t.pnl is not None]
            pnl_sum = sum(t.pnl for t in pnl_rows)  # type: ignore[misc]
            wins = sum(1 for t in pnl_rows if t.pnl > 0)
            losses = sum(1 for t in pnl_rows if t.pnl < 0)
            be = sum(1 for t in pnl_rows if t.pnl == 0.0)
            return pnl_sum, wins, losses, be

        at_pnl, at_wins, at_losses, at_be = _tally(closed)
        td_pnl, td_wins, td_losses, td_be = _tally(today_closed_list)

        return {
            "alltime_pnl": at_pnl,
            "alltime_closed": len(closed),
            "alltime_wins": at_wins,
            "alltime_losses": at_losses,
            "alltime_be": at_be,
            "today_pnl": td_pnl,
            "today_closed": len(today_closed_list),
            "today_wins": td_wins,
            "today_losses": td_losses,
            "today_be": td_be,
            "open_count": len(open_trades),
            "open_stake": sum(t.size for t in open_trades),
        }

    def _bankroll_block(self, stats: dict[str, Any]) -> str:
        """Format the bankroll section for Telegram close notifications."""
        at_pnl = stats["alltime_pnl"]
        open_label = "trade" if stats["open_count"] == 1 else "trades"
        b50_bal = 50.0 + at_pnl
        b100_bal = 100.0 + at_pnl
        b50_pct = (at_pnl / 50.0) * 100
        b100_pct = (at_pnl / 100.0) * 100
        return (
            f"📊 Today: {stats['today_closed']} closed"
            f" | W{stats['today_wins']}/L{stats['today_losses']}/B{stats['today_be']}"
            f" | Net: ${stats['today_pnl']:+.2f}\n"
            f"   Open: {stats['open_count']} {open_label} (${stats['open_stake']:.2f} at risk)\n"
            f"── Paper bankroll (@$2 stake, hypothetical) ──\n"
            f"  $50  → ${b50_bal:.2f} ({b50_pct:+.1f}%)\n"
            f"  $100 → ${b100_bal:.2f} ({b100_pct:+.1f}%)"
        )

    def _check_open_risk_cap(self, session: Any) -> bool:
        """Return True if opening a new trade would exceed BTC_SCALP_MAX_OPEN_RISK_USD.

        Logs a warning and sends a Telegram notification (at most once per session)
        when the cap is hit. Returns False (no cap) if env var is unset.
        """
        cap = get_settings().btc_scalp_max_open_risk_usd
        if cap is None:
            return False
        open_trades_q = (
            session.query(Trade)
            .filter(
                Trade.strategy == "btc_scalp",
                Trade.is_paper.is_(True),
                Trade.status == "open",
            )
            .all()
        )
        open_stake_total = sum(t.size for t in open_trades_q)
        if open_stake_total + PAPER_POSITION_SIZE > cap:
            LOGGER.warning(
                "BTC SCALP: skipping — open risk $%.2f + $%.2f would exceed cap $%.2f",
                open_stake_total,
                PAPER_POSITION_SIZE,
                cap,
            )
            if not self._open_risk_cap_notified:
                self._notify(
                    f"⚠️ BTC SCALP: trade skipped — open risk cap ${cap:.0f} reached"
                    f" (${open_stake_total:.2f} open)"
                )
                self._open_risk_cap_notified = True
            return True
        return False

    # ── Price feed ─────────────────────────────────────────────────────────

    async def _fetch_btc_price_http(self) -> float | None:
        """Fallback: CoinGecko simple price API — no API key, no geo-block, ~30s interval."""
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(COINGECKO_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp,
            ):
                data = await resp.json()
                return float(data["bitcoin"]["usd"])
        except Exception as exc:
            LOGGER.warning("CoinGecko HTTP fallback failed: %s", exc)
            return None

    async def _run_price_feed(self) -> None:
        """Connect to Coinbase Advanced Trade WebSocket for live BTC price.

        Falls back to CoinGecko HTTP polling (30s) after 3 consecutive WS failures.
        """
        ws_failures = 0

        while True:
            if ws_failures >= WS_MAX_FAILURES:
                # HTTP fallback mode — poll CoinGecko every 30s
                LOGGER.warning(
                    "WS failed %d times — switching to CoinGecko HTTP fallback (30s interval)",
                    ws_failures,
                )
                self._notify(
                    "⚠️ BTC Scalp: Coinbase WS failed all retries — switched to CoinGecko fallback"
                )
                while True:
                    price = await self._fetch_btc_price_http()
                    if price:
                        self._btc_price = price
                        LOGGER.debug("CoinGecko price update: $%.0f", price)
                    await asyncio.sleep(HTTP_FALLBACK_INTERVAL_S)

            try:
                async with websockets.connect(COINBASE_WS_URL) as ws:
                    subscribe_msg = {
                        "type": "subscribe",
                        "product_ids": ["BTC-USD"],
                        "channel": "ticker",
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    LOGGER.info("BTC price feed connected to Coinbase Advanced Trade")
                    ws_failures = 0  # reset on successful connect
                    print("Coinbase WS reconnected ✅", flush=True)

                    async for raw in ws:
                        msg = json.loads(raw)
                        for event in msg.get("events", []):
                            for ticker in event.get("tickers", []):
                                price = float(ticker.get("price", 0))
                                if price > 0:
                                    self._btc_price = price

            except Exception as exc:
                ws_failures += 1
                LOGGER.warning(
                    "Coinbase WS error (%d/%d): %s — reconnecting in 5s",
                    ws_failures,
                    WS_MAX_FAILURES,
                    exc,
                )
                await asyncio.sleep(5)

    # ── Signal computation ──────────────────────────────────────────────────

    def _extract_strike(self, question: str) -> float | None:
        """Extract strike from e.g. 'Will BTC be above $83,500 at 14:05?' → 83500.0"""
        match = re.search(r"\$([0-9,]+)", question)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        return None

    def _compute_certainty(self, current_price: float, strike: float) -> tuple[float, str]:
        if current_price > strike:
            score = min(0.99, 0.50 + (current_price - strike) / strike * 15)
            side = "YES"
        else:
            score = max(0.01, 0.50 - (strike - current_price) / strike * 15)
            side = "NO"
        return score, side

    def _get_momentum(self) -> float:
        """Return momentum score -1.0..1.0 from last 60s of BTC prices (5s intervals)."""
        if len(self._price_history) < 2:
            return 0.0
        recent = list(self._price_history)[-12:]  # last ~60s at 5s intervals
        change = (recent[-1] - recent[0]) / recent[0]
        return max(-1.0, min(1.0, change * 100))  # scale % change to -1..1

    # ── V2 probability and EV helpers ───────────────────────────────────────

    def _get_realized_vol(self) -> float:
        """Return a per-sqrt-minute local vol proxy from recent BTC price history.

        Computes log-return std dev from price_history (5s samples), scales to
        per-sqrt-minute. NOT annualized — a short-horizon proxy only.
        Floors at btc_scalp_vol_floor to prevent division-by-zero in z-score.
        """
        prices = list(self._price_history)
        if len(prices) < 3:
            return get_settings().btc_scalp_vol_floor
        log_returns = [
            math.log(prices[i] / prices[i - 1])
            for i in range(1, len(prices))
            if prices[i - 1] > 0
        ]
        if len(log_returns) < 2:
            return get_settings().btc_scalp_vol_floor
        n = len(log_returns)
        mean = sum(log_returns) / n
        variance = sum((r - mean) ** 2 for r in log_returns) / (n - 1)
        vol_per_5s = math.sqrt(variance)
        vol_per_minute = vol_per_5s * math.sqrt(12)  # 12 x 5s intervals = 1 minute
        return max(vol_per_minute, get_settings().btc_scalp_vol_floor)

    @staticmethod
    def _normal_cdf_approx(z: float) -> float:
        """Approximate standard normal CDF using Abramowitz & Stegun (formula 26.2.17).

        Accurate to ~1.5e-7. No scipy dependency.
        """
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        cdf = 1.0 - pdf * poly
        return cdf if z >= 0 else 1.0 - cdf

    def _p_up_model(
        self,
        start_price_proxy: float,
        current_price: float,
        time_left_min: float,
        realized_vol: float,
    ) -> float:
        """Estimate P(BTC >= start_price_proxy at expiry) via a diffusion-style z-score proxy.

        z = log(current / start) / (vol_per_sqrt_min * sqrt(time_left / 5.0))

        This is a diffusion-style z-score heuristic, NOT a true Brownian bridge
        (we do not condition on terminal settlement structure). start_price_proxy
        is a captured-price proxy — see BtcScanLog docstring for limitations.

        Structurally compatible with exact Chainlink integration: replace
        start_price_proxy with the true Chainlink reference price when available.
        """
        settings = get_settings()
        if start_price_proxy <= 0 or current_price <= 0:
            return 0.5
        log_ret = math.log(current_price / start_price_proxy)
        time_factor = math.sqrt(max(time_left_min, 0.1) / 5.0)
        denominator = max(realized_vol * time_factor, settings.btc_scalp_vol_floor)
        z = log_ret / denominator
        p_up = self._normal_cdf_approx(z)
        momentum = self._get_momentum()
        weight = settings.btc_scalp_momentum_adjust_weight
        return max(0.01, min(0.99, p_up + momentum * weight))

    @staticmethod
    def _ev_buy_yes(prob: float, ask: float) -> float:
        """EV per $1-payout contract of buying YES at `ask` given win probability `prob`.

        EV = prob * (1 - ask) - (1 - prob) * ask = prob - ask

        Per-contract, not per dollar of stake. Compare against
        btc_scalp_min_ev_per_contract (not against PAPER_POSITION_SIZE).
        """
        return prob - ask

    @staticmethod
    def _select_side(ev_up: float, ev_down: float) -> str:
        """Return 'Up' if ev_up >= ev_down, else 'Down'."""
        return "Up" if ev_up >= ev_down else "Down"

    @staticmethod
    def _should_skip_for_spread(spread: float | None, max_spread: float) -> bool:
        """Return True if spread exceeds threshold.

        None spread means CLOB was unavailable — that is not a spread skip.
        """
        return spread is not None and spread > max_spread

    @staticmethod
    def _is_stub_book(bid: float | None, ask: float | None, min_best_bid: float) -> bool:
        """Return True when the book is effectively non-tradable / stubbed.

        Criteria:
          - bid or ask is None (no data)
          - bid < min_best_bid (placeholder stub quote, e.g. 0.01)
          - ask > 0.95 (stub ask on the far side)
        """
        if bid is None or ask is None:
            return True
        if bid < min_best_bid:
            return True
        return ask > 0.95

    @staticmethod
    def _is_eff_ask_usable(eff_ask: float, floor: float) -> bool:
        """Return True if eff_ask meets the minimum quality floor for this strategy.

        Production safety guard applied after _first_usable_quote recovers a price
        from depth. Rejects prices that are anomalously low based on observed
        depth-recovery behavior (stale / ghost micro-orders), not on a claim that
        such prices are universally invalid on Polymarket.

        Accepts prices >= floor (price == floor is usable).
        Rejects prices strictly below floor.
        """
        return eff_ask >= floor

    @staticmethod
    def _first_usable_quote(
        bids: list, asks: list, min_bid: float
    ) -> tuple[float | None, float | None]:
        """Return (first_usable_bid, first_usable_ask) from sorted depth levels.

        first_usable_ask: lowest ask price strictly below 0.95 (stub threshold)
        first_usable_bid: highest bid price strictly above min_bid

        Levels are expected as dicts with a 'price' field (str or float).
        Lists should be sorted: asks ascending by price, bids descending by price.
        Returns (None, None) if no usable level found on either side.
        """
        eff_ask: float | None = None
        for level in asks:
            try:
                p = float(level["price"])
                if p < 0.95:
                    eff_ask = p
                    break
            except (KeyError, ValueError, TypeError):
                continue
        eff_bid: float | None = None
        for level in bids:
            try:
                p = float(level["price"])
                if p > min_bid:
                    eff_bid = p
                    break
            except (KeyError, ValueError, TypeError):
                continue
        return eff_bid, eff_ask

    def _parse_clob_token_ids(self, m: dict[str, Any]) -> tuple[str | None, str | None]:
        """Extract yes/no CLOB token IDs from a Gamma market dict.

        TOKEN MAPPING (explicit):
          clobTokenIds[0] → NO  token → "Down" side (BTC ends BELOW reference price)
          clobTokenIds[1] → YES token → "Up" side  (BTC ends ABOVE reference price)

        Polymarket btc-updown-5m markets have outcomes ordered [No, Yes] (index 0 = No/Down,
        index 1 = Yes/Up). outcomePrices and clobTokenIds share this same ordering.

        Buying YES = betting BTC goes UP.
        Buying NO  = betting BTC goes DOWN (we buy NO when side == "Down").

        Handles: JSON string, list, empty string, malformed JSON, non-list decoded value.
        Returns (yes_token_id, no_token_id) — either may be None.
        """
        raw = m.get("clobTokenIds") or []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                raw = []
        if not isinstance(raw, list):
            raw = []
        yes_id = str(raw[1]) if len(raw) > 1 and raw[1] else None
        no_id = str(raw[0]) if len(raw) > 0 and raw[0] else None
        return yes_id, no_id

    @staticmethod
    def _validate_market_mapping(
        m: dict[str, Any],
        yes_token_id: str | None,
        no_token_id: str | None,
        slug: str,
    ) -> str | None:
        """Validate that the market dict has a consistent, well-formed mapping.

        Returns an error string if validation fails, None if OK.

        Checks:
          1. question contains "BTC" (sanity — this is a BTC market)
          2. slug looks like a btc-updown-5m-* slug
          3. yes_token_id and no_token_id are both present (both sides needed)
          4. conditionId is present
          5. question contains "above" or "below" (directional — not generic)
        """
        question = (m.get("question") or "").strip()
        condition_id = m.get("conditionId") or ""

        if "BTC" not in question.upper() and "bitcoin" not in question.lower():
            return f"question missing BTC reference: {question!r}"

        if not slug.startswith("btc-updown-5m-") and "btc" not in slug.lower():
            return f"slug does not look like a btc-updown-5m market: {slug!r}"

        if yes_token_id is None or no_token_id is None:
            return (
                f"missing token IDs: yes_token={yes_token_id!r} no_token={no_token_id!r}"
            )

        if not condition_id:
            return "missing conditionId"

        if not any(w in question.lower() for w in ("above", "below", "higher", "lower", "up", "down")):
            return f"question has no directional keyword: {question!r}"

        return None

    def _compute_signal(
        self, market: dict[str, Any]
    ) -> tuple[str, float, float, float] | None:
        """Compute Up/Down signal from momentum vs market price.

        Returns (side, edge, our_prob, market_price) or None if no edge.
        side is 'Up' or 'Down'.
        """
        try:
            outcome_prices = json.loads(market.get("outcomePrices") or '["0.5","0.5"]')
            up_price = float(outcome_prices[1])    # index 1 = Yes/Up price
            down_price = float(outcome_prices[0])  # index 0 = No/Down price
        except (ValueError, IndexError, TypeError):
            return None

        momentum = self._get_momentum()
        our_up_prob = max(0.01, min(0.99, 0.5 + momentum * 0.4))
        our_down_prob = 1.0 - our_up_prob

        up_edge = our_up_prob - up_price
        down_edge = our_down_prob - down_price

        if up_edge > _V1_MIN_EDGE:
            return ("Up", up_edge, our_up_prob, up_price)
        if down_edge > _V1_MIN_EDGE:
            return ("Down", down_edge, our_down_prob, down_price)
        return None

    # ── Market scanner ──────────────────────────────────────────────────────

    async def _scan_markets(self) -> None:
        """Poll Gamma API for near-expiry BTC markets and paper-trade strong signals."""
        if self._btc_price is None:
            return

        btc = self._btc_price
        now = datetime.now(UTC)

        # ── Generate slugs from current timestamp (btc-updown-5m-[unix_5min_boundary]) ──
        def _btc_updown_slugs() -> list[str]:
            ts = int(time.time())
            return [f"btc-updown-5m-{(ts // 300 + offset) * 300}" for offset in range(4)]

        events_url = f"{GAMMA_API_BASE}/events"
        slugs = _btc_updown_slugs()
        events: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=10.0) as client:
            for slug in slugs:
                try:
                    resp = await client.get(events_url, params={"slug": slug})
                    if resp.status_code == 200:
                        data = resp.json()
                        if data:
                            events.extend(data if isinstance(data, list) else [data])
                            print(f"[BTC Scalp] Found event: {slug}", flush=True)
                            # Dump nested markets structure on first hit
                            first = data[0] if isinstance(data, list) else data
                            print(
                                f"[DEBUG] Event markets: {json.dumps(first.get('markets', []))[:500]}",
                                flush=True,
                            )
                except Exception as exc:
                    LOGGER.warning("Gamma /events fetch failed for %s: %s", slug, exc)

        print(f"[BTC Scalp] Active btc-updown events: {len(events)}", flush=True)

        # Markets nested inside each event
        btc_markets: list[dict[str, Any]] = []
        for event in events:
            for m in event.get("markets", []):
                m.setdefault("_event_slug", event.get("slug", ""))
                btc_markets.append(m)

        btc_markets = sorted(btc_markets, key=lambda m: m.get("endDate", ""), reverse=False)

        if not btc_markets:
            print("[BTC Scalp] No btc-updown markets found this cycle", flush=True)
            return

        settings = get_settings()
        engine = build_engine(settings)
        init_db(engine)
        session_factory = build_session_factory(engine)

        # Record current price into history for momentum calculation
        if self._btc_price is not None:
            self._price_history.append(self._btc_price)

        qualified_count = 0
        skipped_cooldown = 0
        stub_books_seen = 0
        real_books_seen = 0
        wide_spread_seen = 0
        no_ev_seen = 0
        trade_candidates_seen = 0
        invalid_mapping_seen = 0
        already_traded_seen = 0
        no_effective_ask_seen = 0
        eff_ask_floor_seen = 0
        up_side_disabled_seen = 0
        realized_vol = self._get_realized_vol()

        for m in btc_markets:
            question = m.get("question") or ""

            # btc-updown events use "endDate"; fall back to ISO variants
            end_date_str = (
                m.get("endDate") or m.get("endDateIso") or m.get("end_date_iso") or ""
            )
            if not end_date_str:
                continue
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=UTC)
            except ValueError:
                continue

            minutes_left = round((end_dt - now).total_seconds() / 60, 3)
            if not (settings.btc_scalp_min_minutes_left <= minutes_left
                    <= settings.btc_scalp_max_minutes_left):
                print(
                    f"[Scan] skip slug={m.get('slug') or m.get('conditionId') or '?'}"
                    f" minutes_left={minutes_left} (outside window)",
                    flush=True,
                )
                continue

            # ── Per-market variables — all initialized before any branch ──────
            _slug_raw = m.get("slug") or m.get("conditionId") or "?"
            skip_reason: str | None = None
            decision = "skip"
            selected_side: str | None = None

            # Capture start_price_proxy when market first enters window
            if _slug_raw not in self._window_start_prices and self._btc_price is not None:
                self._window_start_prices[_slug_raw] = (self._btc_price, now)
                print(
                    f"[Scan] start_price_proxy captured slug={_slug_raw}"
                    f" price={self._btc_price:,.2f}",
                    flush=True,
                )
            start_entry = self._window_start_prices.get(_slug_raw)
            start_price_proxy: float | None = start_entry[0] if start_entry else None
            start_price_captured_at: datetime | None = start_entry[1] if start_entry else None

            # Parse Gamma mid prices
            try:
                outcome_prices = json.loads(m.get("outcomePrices") or '["0.5","0.5"]')
                up_price_mid = float(outcome_prices[1])    # index 1 = Yes/Up price
            except (ValueError, IndexError, TypeError):
                up_price_mid = 0.5
            # Binary market identity: down = 1 - up (mutually exclusive binary outcomes)
            down_price_mid = 1.0 - up_price_mid

            # CLOB fetch (per side, independent availability)
            yes_token_id, no_token_id = self._parse_clob_token_ids(m)

            # Phase 1 — validate market mapping before any CLOB work
            mapping_err = self._validate_market_mapping(m, yes_token_id, no_token_id, _slug_raw)
            if mapping_err:
                print(
                    f"[Scan] skip slug={_slug_raw} reason=invalid_market_mapping detail={mapping_err!r}",
                    flush=True,
                )
                skip_reason = "invalid_market_mapping"
                # Still write a BtcScanLog below — fall through, do NOT continue here.

            up_best_ask: float | None = None
            up_best_bid: float | None = None
            up_spread: float | None = None
            up_clob_available = False
            up_bid_volume: float = 0.0
            up_ask_volume: float = 0.0
            down_best_ask: float | None = None
            down_best_bid: float | None = None
            down_spread: float | None = None
            down_clob_available = False
            down_bid_volume: float = 0.0
            down_ask_volume: float = 0.0

            if yes_token_id:
                try:
                    ob_up = await asyncio.to_thread(
                        self._clob_client.get_orderbook, yes_token_id
                    )
                    up_best_ask = ob_up["ask"]
                    up_best_bid = ob_up["bid"]
                    up_spread = ob_up["spread"]
                    up_bid_volume = ob_up.get("bid_volume", 0.0) or 0.0
                    up_ask_volume = ob_up.get("ask_volume", 0.0) or 0.0
                    up_clob_available = True
                except Exception as exc:
                    LOGGER.warning("CLOB fetch failed up token %s: %s", yes_token_id[:12], exc)

            if no_token_id:
                try:
                    ob_down = await asyncio.to_thread(
                        self._clob_client.get_orderbook, no_token_id
                    )
                    down_best_ask = ob_down["ask"]
                    down_best_bid = ob_down["bid"]
                    down_spread = ob_down["spread"]
                    down_bid_volume = ob_down.get("bid_volume", 0.0) or 0.0
                    down_ask_volume = ob_down.get("ask_volume", 0.0) or 0.0
                    down_clob_available = True
                except Exception as exc:
                    LOGGER.warning("CLOB fetch failed down token %s: %s", no_token_id[:12], exc)

            # Resolve entry price per side (side-specific CLOB requirement)
            if up_clob_available:
                up_entry_price: float | None = up_best_ask
            elif not settings.btc_scalp_require_clob:
                up_entry_price = up_price_mid   # permissive midpoint fallback (paper only)
                LOGGER.warning("Using up midpoint fallback for %s", _slug_raw)
            else:
                up_entry_price = None

            if down_clob_available:
                down_entry_price: float | None = down_best_ask
            elif not settings.btc_scalp_require_clob:
                down_entry_price = down_price_mid
                LOGGER.warning("Using down midpoint fallback for %s", _slug_raw)
            else:
                down_entry_price = None

            # EV model — compute for available sides; float("-inf") makes unavailable side lose
            if start_price_proxy is None:
                skip_reason = "missing_start_price_proxy"
                # p_up_model returns 0.5 on None, but we skip rather than silently use 0.5
                p_up = 0.5
                ev_up = float("-inf")
                ev_down = float("-inf")
            else:
                p_up = self._p_up_model(start_price_proxy, btc, minutes_left, realized_vol)
                ev_up = (
                    self._ev_buy_yes(p_up, up_entry_price)
                    if up_entry_price is not None
                    else float("-inf")
                )
                ev_down = (
                    self._ev_buy_yes(1.0 - p_up, down_entry_price)
                    if down_entry_price is not None
                    else float("-inf")
                )

            # Skip gate — evaluated on the provisional winning side only
            provisional_side: str | None = None
            provisional_ev = float("-inf")
            side_spread: float | None = None
            side_entry_price: float | None = None
            is_stub = False
            side_bid: float | None = None
            side_ask: float | None = None
            side_bid_vol: float = 0.0
            side_ask_vol: float = 0.0
            # Effective executable quotes from depth (None = depth not fetched / no usable level)
            eff_bid: float | None = None
            eff_ask: float | None = None

            if skip_reason is None:
                if up_entry_price is None and down_entry_price is None:
                    skip_reason = "missing_clob"
                else:
                    provisional_side = self._select_side(ev_up, ev_down)
                    if provisional_side == "Up":
                        provisional_ev = ev_up
                        side_spread = up_spread
                        side_entry_price = up_entry_price
                    else:
                        provisional_ev = ev_down
                        side_spread = down_spread
                        side_entry_price = down_entry_price

                    side_bid = up_best_bid if provisional_side == "Up" else down_best_bid
                    side_ask = up_best_ask if provisional_side == "Up" else down_best_ask
                    side_bid_vol = up_bid_volume if provisional_side == "Up" else down_bid_volume
                    side_ask_vol = up_ask_volume if provisional_side == "Up" else down_ask_volume
                    is_stub = self._is_stub_book(side_bid, side_ask, settings.btc_scalp_min_best_bid)
                    print(
                        f"[Scan] book_check slug={_slug_raw} side={provisional_side}"
                        f" minutes_left={minutes_left}"
                        f" bid={side_bid} ask={side_ask} spread={side_spread} stub={is_stub}"
                        f" bid_vol={side_bid_vol:.4f} ask_vol={side_ask_vol:.4f}",
                        flush=True,
                    )
                    if (not is_stub) or (side_bid_vol > 0) or (side_ask_vol > 0):
                        side_token_id = yes_token_id if provisional_side == "Up" else no_token_id
                        if side_token_id:
                            try:
                                levels = await asyncio.to_thread(
                                    self._clob_client.get_orderbook_levels, side_token_id, 10
                                )
                                all_bids = levels.get("bids", [])
                                all_asks = levels.get("asks", [])
                                bids_top3 = all_bids[:3]
                                asks_top3 = all_asks[:3]
                                n_bids = len(all_bids)
                                n_asks = len(all_asks)
                                # Values used as-is from API — may be str or float
                                top_bids_fmt = [(x.get("price"), x.get("size")) for x in bids_top3]
                                top_asks_fmt = [(x.get("price"), x.get("size")) for x in asks_top3]
                                print(
                                    f"[Scan] depth slug={_slug_raw} side={provisional_side}"
                                    f" stub={is_stub} minutes_left={minutes_left}"
                                    f" n_bids={n_bids} n_asks={n_asks}"
                                    f" top_bids={top_bids_fmt} top_asks={top_asks_fmt}",
                                    flush=True,
                                )
                                # Derive effective executable quote from depth (stub top-of-book only)
                                if is_stub:
                                    eff_bid, eff_ask = self._first_usable_quote(
                                        all_bids, all_asks, settings.btc_scalp_min_best_bid
                                    )
                                    eff_spread = (
                                        round(eff_ask - eff_bid, 6)
                                        if (eff_bid is not None and eff_ask is not None)
                                        else None
                                    )
                                    print(
                                        f"[Scan] effective_book slug={_slug_raw} side={provisional_side}"
                                        f" raw_bid={side_bid} raw_ask={side_ask}"
                                        f" eff_bid={eff_bid} eff_ask={eff_ask} eff_spread={eff_spread}",
                                        flush=True,
                                    )
                                    if eff_ask is not None:
                                        if not self._is_eff_ask_usable(eff_ask, _EFF_ASK_FLOOR):
                                            # Anomalously low price — stale / ghost order in depth.
                                            # Log, mark as floor violation, and nullify eff_ask so
                                            # the stub gate below classifies correctly.
                                            print(
                                                f"[Scan] eff_ask_rejected slug={_slug_raw}"
                                                f" side={provisional_side}"
                                                f" eff_ask={eff_ask} floor={_EFF_ASK_FLOOR}"
                                                f" eff_bid={eff_bid} eff_spread={eff_spread}",
                                                flush=True,
                                            )
                                            skip_reason = "eff_ask_below_floor"
                                            eff_ask = None
                                        else:
                                            # Override with executable depth price — not a stub skip
                                            is_stub = False
                                            side_entry_price = eff_ask
                                            side_spread = eff_spread
                                            provisional_ev = self._ev_buy_yes(
                                                p_up if provisional_side == "Up" else 1.0 - p_up,
                                                eff_ask,
                                            )
                                            # Sync ev_up / ev_down so trade_candidate log and
                                            # BtcScanLog reflect the effective ask, not the stale
                                            # raw stub ask that was used for initial EV computation.
                                            if provisional_side == "Up":
                                                ev_up = provisional_ev
                                            else:
                                                ev_down = provisional_ev
                            except Exception as exc:
                                LOGGER.warning("Depth fetch failed %s: %s", _slug_raw, exc)
                    if is_stub:
                        if skip_reason is None:
                            # Classify the stub skip more precisely using depth recovery outcome:
                            #   no_effective_ask — depth fetched, usable bid found, no usable ask
                            #                      (one-sided market: buyers present, no sellers at <0.95)
                            #   stub_book        — no usable depth at all (empty book, fetch failed,
                            #                      or both eff_bid and eff_ask are None)
                            if eff_bid is not None and eff_ask is None:
                                skip_reason = "no_effective_ask"
                                print(
                                    f"[Scan] skip slug={_slug_raw} reason=no_effective_ask"
                                    f" side={provisional_side}"
                                    f" raw_bid={side_bid} raw_ask={side_ask}"
                                    f" eff_bid={eff_bid} eff_ask=None",
                                    flush=True,
                                )
                            else:
                                skip_reason = "stub_book"
                    elif self._should_skip_for_spread(side_spread, settings.btc_scalp_max_spread):
                        skip_reason = "spread_too_wide"
                    elif provisional_ev < settings.btc_scalp_min_ev_per_contract:
                        skip_reason = "no_ev"
                    elif provisional_side == "Up" and settings.btc_scalp_disable_up_entries:
                        skip_reason = "up_side_disabled"

            if skip_reason:
                print(
                    f"[Scan] skip slug={_slug_raw} reason={skip_reason}"
                    f" p_up={p_up:.4f} ev_up={ev_up:.4f} ev_down={ev_down:.4f}"
                    f" up_bid={up_best_bid} up_ask={up_best_ask} up_spread={up_spread}"
                    f" down_bid={down_best_bid} down_ask={down_best_ask} down_spread={down_spread}",
                    flush=True,
                )
                # decision stays 'skip', selected_side stays None
            else:
                # Passed all gates — prepare trade
                decision = "trade"
                selected_side = provisional_side
                assert provisional_side is not None  # guaranteed by gate logic above
                assert side_entry_price is not None

            # ── Per-scan classification counters (after gate fully resolves) ─
            if skip_reason == "stub_book":
                stub_books_seen += 1
            elif skip_reason == "no_effective_ask":
                no_effective_ask_seen += 1
            elif skip_reason == "eff_ask_below_floor":
                eff_ask_floor_seen += 1
            elif skip_reason == "spread_too_wide":
                wide_spread_seen += 1
            elif skip_reason == "no_ev":
                no_ev_seen += 1
            elif skip_reason == "invalid_market_mapping":
                invalid_mapping_seen += 1
            elif skip_reason == "up_side_disabled":
                up_side_disabled_seen += 1
            elif decision == "trade":
                trade_candidates_seen += 1
            if provisional_side is not None and not is_stub:
                real_books_seen += 1

            # ── Single-point BtcScanLog write (every in-window candidate) ────
            # Added to the existing session; committed with the normal scan flow.
            # No early continue above this point.
            with session_factory() as session:
                # Ensure Market row exists (needed for market_id FK)
                slug = m.get("slug") or m.get("conditionId") or f"btc-scalp-{int(now.timestamp())}"
                market_row = session.query(Market).filter_by(slug=slug).first()
                if market_row is None:
                    market_row = Market(
                        slug=slug,
                        question=question,
                        domain="crypto",
                        condition_id=m.get("conditionId"),
                        status="active",
                    )
                    session.add(market_row)
                    session.flush()

                session.add(BtcScanLog(
                    timestamp_utc=now,
                    slug=slug,
                    market_id=market_row.id,
                    minutes_left=minutes_left,
                    start_price_proxy=start_price_proxy,
                    start_price_captured_at=start_price_captured_at,
                    current_price=btc,
                    realized_vol=realized_vol,
                    momentum_60s=self._get_momentum(),
                    up_price_mid=up_price_mid,
                    up_best_ask=up_best_ask,
                    up_best_bid=up_best_bid,
                    up_spread=up_spread,
                    up_clob_available=up_clob_available,
                    down_price_mid=down_price_mid,
                    down_best_ask=down_best_ask,
                    down_best_bid=down_best_bid,
                    down_spread=down_spread,
                    down_clob_available=down_clob_available,
                    p_up_model=p_up,
                    ev_up=ev_up if ev_up != float("-inf") else -999.0,
                    ev_down=ev_down if ev_down != float("-inf") else -999.0,
                    decision=decision,
                    selected_side=selected_side,  # None on skip, 'Up'/'Down' on trade
                    skip_reason=skip_reason,
                ))

                if decision == "skip":
                    session.commit()
                    continue  # only continue AFTER the log write

                # ── Trade candidate — announce and gate ──────────────────────
                assert provisional_side is not None
                assert side_entry_price is not None
                side = provisional_side
                entry_price = side_entry_price
                liquidity = float(m.get("liquidity") or 0)

                print(
                    f"[Scan] trade_candidate slug={slug} side={side}"
                    f" p_up={p_up:.4f} ev_up={ev_up:.4f} ev_down={ev_down:.4f}"
                    f" ask={entry_price:.4f} spread={side_spread}",
                    flush=True,
                )

                # Cooldown check
                signal_key = f"{m.get('conditionId', slug)}-{side}"
                now_ts = time.time()
                self._fired_signals = {
                    k: v for k, v in self._fired_signals.items()
                    if now_ts - v < SIGNAL_COOLDOWN_S
                }
                if signal_key in self._fired_signals:
                    skipped_cooldown += 1
                    session.commit()
                    continue
                self._fired_signals[signal_key] = now_ts

                # Phase 2 — hard block: one trade per slug, any status (open or closed)
                existing = (
                    session.query(Trade)
                    .filter(
                        Trade.market_id == market_row.id,
                        Trade.strategy == "btc_scalp",
                    )
                    .first()
                )
                if existing:
                    already_traded_seen += 1
                    print(
                        f"[Scan] skip slug={slug} reason=already_traded"
                        f" existing_id={existing.id} existing_status={existing.status}",
                        flush=True,
                    )
                    session.commit()
                    continue

                if self._check_open_risk_cap(session):
                    session.commit()
                    continue

                # ── Phase 1: EntryCheck — confirm mapping before commit ───────
                # Up  = YES token (clobTokenIds[0]): BTC ends ABOVE reference → buy YES
                # Down = NO token (clobTokenIds[1]): BTC ends BELOW reference → buy NO
                chosen_token = yes_token_id if side == "Up" else no_token_id
                print(
                    f"[EntryCheck] slug={slug} side={side}"
                    f" yes_token={yes_token_id} no_token={no_token_id}"
                    f" chosen_token={chosen_token}"
                    f' question="{question}"',
                    flush=True,
                )

                # ── Phase 3: SignalAudit — EV uses effective executable ask ──
                # EV = p_side - eff_ask (side_entry_price is eff_ask if stub was
                # overridden by depth; raw top-of-book ask otherwise).
                print(
                    f"[SignalAudit] slug={slug} side={side}"
                    f" start_price={start_price_proxy} current_price={btc}"
                    f" minutes_left={minutes_left}"
                    f" p_up={p_up:.4f} p_down={1.0 - p_up:.4f}"
                    f" eff_bid={side_bid} eff_ask={side_entry_price}"
                    f" ev={provisional_ev:.4f}",
                    flush=True,
                )
                # Anomaly: model probability is extreme but the book disagrees.
                # Extreme p_up (>0.90 or <0.10) should roughly correspond to the
                # market's ask being far from 0.50.  If p_up > 0.90 the Up ask
                # should be close to 1.0 (expensive), not cheap.  A cheap ask
                # while p_up is extreme suggests a model/book mismatch worth flagging.
                up_ask_for_check = up_best_ask or side_entry_price or 0.5
                down_ask_for_check = down_best_ask or (1.0 - (up_best_ask or 0.5))
                if (p_up > 0.90 and up_ask_for_check < 0.70) or (
                    p_up < 0.10 and down_ask_for_check < 0.70
                ):
                    print(
                        f"[SignalAudit] anomaly=probability_book_mismatch"
                        f" p_up={p_up:.4f} up_ask={up_ask_for_check}"
                        f" down_ask={down_ask_for_check}",
                        flush=True,
                    )

                # ── Open paper trade ─────────────────────────────────────────
                session.add(BtcSignalLog(
                    logged_at=now,
                    market_slug=slug,
                    question=question,
                    side=side,
                    edge=provisional_ev,
                    time_left_min=minutes_left,
                    our_estimate=p_up if side == "Up" else 1.0 - p_up,
                    market_price=entry_price,
                    momentum=self._get_momentum(),
                    btc_price=btc,
                    liquidity=liquidity,
                    outcome=None,
                ))

                trade = Trade(
                    market_id=market_row.id,
                    signal_id=None,
                    side=side,
                    size=PAPER_POSITION_SIZE,
                    entry_price=entry_price,
                    kelly_fraction=0.0,
                    confidence_score=p_up if side == "Up" else 1.0 - p_up,
                    is_paper=True,
                    status="open",
                    opened_at=now,
                    strategy="btc_scalp",
                    order_id=f"btcscalp_{market_row.id}_{int(now.timestamp())}",
                )
                session.add(trade)
                session.commit()
                self._signals += 1
                self._trades_opened += 1
                qualified_count += 1
                self._open_risk_cap_notified = False
                LOGGER.info(
                    "BTC SCALP: paper trade opened %s side=%s entry=%.4f p_up=%.3f ev=%.4f",
                    slug, side, entry_price, p_up, provisional_ev,
                )
                self._notify(
                    f"✅ BTC TRADE OPENED\n"
                    f"Market: {question}\n"
                    f"Side: {side} | Entry: {entry_price:.2f}\n"
                    f"EV: {provisional_ev:.4f} | p_up: {p_up:.3f}\n"
                    f"Time left: {minutes_left:.1f} min"
                )

        # Evict stale window start prices (slugs no longer in current scan)
        active_slugs = {m.get("slug") or m.get("conditionId") or "?" for m in btc_markets}
        self._window_start_prices = {
            k: v for k, v in self._window_start_prices.items() if k in active_slugs
        }
        # Note: if Gamma transiently omits a market, proxy may be evicted early.
        # Acceptable V2 limitation; evict-by-age can be added later.

        print(
            f"[BTC Scalp] Scan done — BTC=${btc:,.0f} | momentum={self._get_momentum():.3f}"
            f" | vol={realized_vol:.5f} | markets={len(btc_markets)}"
            f" | signals={qualified_count} | cooldown_skipped={skipped_cooldown}"
            f" | stub={stub_books_seen} real={real_books_seen}"
            f" | wide_spread={wide_spread_seen} no_ev={no_ev_seen}"
            f" | trade_candidates={trade_candidates_seen}"
            f" | invalid_mapping={invalid_mapping_seen}"
            f" | already_traded={already_traded_seen}"
            f" | no_eff_ask={no_effective_ask_seen}"
            f" | eff_ask_floor={eff_ask_floor_seen}"
            f" | up_disabled={up_side_disabled_seen}",
            flush=True,
        )

    # ── Resolution check ────────────────────────────────────────────────────

    async def _check_resolutions(self) -> None:
        """Close resolved btc_scalp paper trades at 1.0 (win) or 0.0 (loss)."""
        settings = get_settings()
        engine = build_engine(settings)
        session_factory = build_session_factory(engine)

        with session_factory() as session:
            open_trades = (
                session.query(Trade)
                .filter(
                    Trade.strategy == "btc_scalp",
                    Trade.status == "open",
                    Trade.is_paper.is_(True),
                )
                .all()
            )
            n_open = len(open_trades)
            if not open_trades:
                print("[Resolution] Checked 0 open positions, 0 resolved and closed", flush=True)
                return

            market_ids = {t.market_id for t in open_trades}
            now = datetime.now(UTC)
            closed_this_run = 0
            n_missing_cid: int = 0
            n_gamma_queried: int = 0
            n_unresolved: int = 0
            n_missing_outcome: int = 0
            n_notify_fail: int = 0
            n_close_msgs_queued: int = 0
            n_telegram_attempted: int = 0
            n_telegram_sent: int = 0
            pending_notifications: list[tuple[str, int, str]] = []  # (message, trade_id, slug)

            for market_id in market_ids:
                exit_price = 0.0
                market = session.get(Market, market_id)
                if market is None:
                    continue

                market_trades = [t for t in open_trades if t.market_id == market_id]
                sides = [t.side for t in market_trades]
                _trade_ids = [t.id for t in market_trades]
                print(
                    f"[Resolution] inspect trade_id={_trade_ids}"
                    f" slug={market.slug} side={sides}"
                    f" condition_id={market.condition_id or 'MISSING'}"
                    f" resolution_time={market.resolution_time}"
                    f" now={now.isoformat()}",
                    flush=True,
                )
                _cid = market.condition_id or "MISSING"
                _cached = (market.resolution_outcome or "").upper() or "none"
                _resolution_age = ""
                if market.resolution_time:
                    _rt = (
                        market.resolution_time
                        if market.resolution_time.tzinfo
                        else market.resolution_time.replace(tzinfo=UTC)
                    )
                    _resolution_age = f" resolution_time_age={(now - _rt).total_seconds():.0f}s"
                LOGGER.info(
                    "BTC SCALP resolution | slug=%s market_id=%d trade_ids=%s side=%s"
                    " condition_id=%s cached_outcome=%s%s",
                    market.slug, market_id, [t.id for t in market_trades],
                    sides, _cid, _cached, _resolution_age,
                )
                outcome = (market.resolution_outcome or "").upper()

                if outcome not in ("YES", "NO") and not market.condition_id:
                    LOGGER.warning(
                        "BTC SCALP: trade stuck — condition_id is NULL for slug=%s market_id=%d"
                        " (cannot query Gamma; trade will never auto-close)",
                        market.slug, market_id,
                    )
                    n_missing_cid += 1
                    continue

                if outcome not in ("YES", "NO") and market.condition_id:
                    try:
                        print(
                            f"[Gamma] querying slug={market.slug}"
                            f" condition_id={market.condition_id}{_resolution_age}",
                            flush=True,
                        )
                        async with httpx.AsyncClient(timeout=8.0) as client:
                            resp = await client.get(
                                f"{GAMMA_API_BASE}/markets",
                                params={"conditionId": market.condition_id},
                            )
                        n_gamma_queried += 1
                        print(
                            f"[Gamma] response slug={market.slug}"
                            f" condition_id={market.condition_id} status={resp.status_code}",
                            flush=True,
                        )
                        LOGGER.info(
                            "BTC SCALP Gamma query | slug=%s condition_id=%s status=%d",
                            market.slug, market.condition_id, resp.status_code,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            items = data if isinstance(data, list) else [data]

                            # Log raw Gamma fields before parsing so we can diagnose failures
                            if items:
                                _first = items[0]
                                print(
                                    f"[Resolution] gamma_raw slug={market.slug}"
                                    f" resolved={_first.get('resolved')!r}"
                                    f" resolutionOutcome={_first.get('resolutionOutcome')!r}"
                                    f" outcomePrices={_first.get('outcomePrices')!r}"
                                    f" winner={_first.get('winner')!r}"
                                    f" n_items={len(items)}",
                                    flush=True,
                                )

                            for item in items:
                                _is_resolved = item.get("resolved")
                                _raw_outcome = (item.get("resolutionOutcome") or "").upper()

                                if _is_resolved:
                                    if _raw_outcome in ("YES", "NO"):
                                        # Primary path: resolved=True + standard resolutionOutcome
                                        outcome = _raw_outcome
                                        market.resolution_outcome = outcome
                                        market.status = "resolved"
                                        print(
                                            f"[Gamma] resolved slug={market.slug}"
                                            f" resolved=True resolutionOutcome={_raw_outcome}",
                                            flush=True,
                                        )
                                        LOGGER.info(
                                            "BTC SCALP Gamma resolved | slug=%s"
                                            " resolved=True resolutionOutcome=%s",
                                            market.slug, _raw_outcome,
                                        )
                                        break
                                    else:
                                        # resolved=True but outcome is unexpected — log and fall through
                                        print(
                                            f"[Resolution] gamma_unexpected_outcome slug={market.slug}"
                                            f" resolved={_is_resolved!r}"
                                            f" resolutionOutcome={_raw_outcome!r}"
                                            f" — falling back to outcomePrices",
                                            flush=True,
                                        )

                                # Fallback: derive outcome from outcomePrices when prices are definitive.
                                # Gamma sets outcomePrices=["1","0"] for NO-win or ["0","1"] for YES-win
                                # even before or instead of setting resolutionOutcome.
                                # btc-updown-5m markets have outcomes ordered [No, Yes]:
                                #   outcomePrices[0] = No/Down price → ["1","0"] means NO-win
                                #   outcomePrices[1] = Yes/Up  price → ["0","1"] means YES-win
                                _prices_raw = item.get("outcomePrices")
                                if _prices_raw and outcome not in ("YES", "NO"):
                                    try:
                                        _prices = (
                                            json.loads(_prices_raw)
                                            if isinstance(_prices_raw, str)
                                            else _prices_raw
                                        )
                                        _p0 = float(_prices[0])
                                        _p1 = float(_prices[1])
                                        if _p0 >= 0.99:
                                            # outcomePrices[0] = No/Down → No/Down wins
                                            outcome = "NO"
                                            market.resolution_outcome = outcome
                                            market.status = "resolved"
                                            print(
                                                f"[Gamma] resolved_via_prices slug={market.slug}"
                                                f" outcomePrices=[{_p0},{_p1}] → outcome=NO",
                                                flush=True,
                                            )
                                            break
                                        elif _p1 >= 0.99:
                                            # outcomePrices[1] = Yes/Up → Yes/Up wins
                                            outcome = "YES"
                                            market.resolution_outcome = outcome
                                            market.status = "resolved"
                                            print(
                                                f"[Gamma] resolved_via_prices slug={market.slug}"
                                                f" outcomePrices=[{_p0},{_p1}] → outcome=YES",
                                                flush=True,
                                            )
                                            break
                                    except Exception as _pe:
                                        print(
                                            f"[Resolution] prices_parse_fail slug={market.slug}"
                                            f" err={_pe}",
                                            flush=True,
                                        )
                            else:
                                n_unresolved += 1
                                print(
                                    f"[Gamma] unresolved slug={market.slug}"
                                    f" condition_id={market.condition_id}"
                                    f" — no resolved item found in {len(items)} item(s)",
                                    flush=True,
                                )
                                LOGGER.info(
                                    "BTC SCALP Gamma not resolved | slug=%s"
                                    " no resolved item found",
                                    market.slug,
                                )
                    except Exception as exc:
                        LOGGER.warning(
                            "BTC SCALP resolution check failed for %s: %s",
                            market.slug,
                            exc,
                        )

                if outcome not in ("YES", "NO"):
                    n_missing_outcome += 1
                    print(
                        f"[Gamma] action=missing_outcome slug={market.slug}"
                        f" condition_id={market.condition_id}"
                        f" outcome_value={outcome!r}{_resolution_age}",
                        flush=True,
                    )
                    LOGGER.info(
                        "BTC SCALP trade disposition | slug=%s action=skipped_missing_outcome"
                        " outcome_value=%r",
                        market.slug, outcome,
                    )
                    continue

                print(
                    f"[Resolution] parsed slug={market.slug} outcome={outcome}"
                    f" trades={[t.id for t in market_trades]}",
                    flush=True,
                )

                for trade in market_trades:
                    # "Up" side bought the YES token → wins when outcome=="YES"
                    # "Down" side bought the NO token → wins when outcome=="NO"
                    if trade.side == "Up":
                        exit_price = 1.0 if outcome == "YES" else 0.0
                    elif trade.side == "Down":
                        exit_price = 1.0 if outcome == "NO" else 0.0
                    else:
                        exit_price = 0.0  # unknown side — conservative
                    won = exit_price == 1.0
                    pnl = (exit_price - trade.entry_price) * trade.size
                    print(
                        f"[Resolution] parsed trade_id={trade.id} slug={market.slug}"
                        f" side={trade.side} outcome={outcome} won={won} pnl={pnl:.4f}",
                        flush=True,
                    )
                    trade.exit_price = exit_price
                    trade.pnl = pnl
                    trade.status = "closed"
                    trade.closed_at = now
                    trade.close_reason = "resolution"
                    self._pnl += pnl
                    self._trades_closed += 1
                    closed_this_run += 1
                    LOGGER.info(
                        "BTC SCALP trade disposition | trade_id=%d slug=%s side=%s outcome=%s"
                        " action=%s pnl=%.4f",
                        trade.id, market.slug, trade.side, outcome,
                        "closed_win" if won else "closed_loss",
                        pnl,
                    )
                    if won:
                        pending_notifications.append((
                            f"🏆 BTC TRADE CLOSED — WIN\n"
                            f"Market: {market.question}\n"
                            f"Side: {trade.side} | Entry: {trade.entry_price:.2f} → Exit: 1.00\n"
                            f"PnL: +${pnl:.2f} on $2.00 stake\n"
                            f"Session: {self._trades_closed} closed | Net PnL: ${self._pnl:+.2f}",
                            trade.id,
                            market.slug,
                        ))
                    else:
                        pending_notifications.append((
                            f"❌ BTC TRADE CLOSED — LOSS\n"
                            f"Market: {market.question}\n"
                            f"Side: {trade.side} | Entry: {trade.entry_price:.2f} → Exit: 0.00\n"
                            f"PnL: -${abs(pnl):.2f} on $2.00 stake\n"
                            f"Session: {self._trades_closed} closed | Net PnL: ${self._pnl:+.2f}",
                            trade.id,
                            market.slug,
                        ))
                    n_close_msgs_queued += 1
                    print(
                        f"[Resolution] close_msg_queued trade_id={trade.id}"
                        f" slug={market.slug} pnl={pnl:.4f}",
                        flush=True,
                    )
                    LOGGER.info(
                        "BTC SCALP: closed %s side=%s outcome=%s pnl=%.4f",
                        market.slug,
                        trade.side,
                        outcome,
                        pnl,
                    )

                # Fill outcome into signal log rows for this market
                result_str = "WIN" if exit_price >= 1.0 else "LOSS"
                pending_logs = (
                    session.query(BtcSignalLog)
                    .filter(
                        BtcSignalLog.market_slug == market.slug,
                        BtcSignalLog.outcome.is_(None),
                    )
                    .all()
                )
                for log_row in pending_logs:
                    log_row.outcome = result_str
                if pending_logs:
                    LOGGER.info(
                        "BTC SCALP: filled %d signal log rows for %s → %s",
                        len(pending_logs),
                        market.slug,
                        result_str,
                    )

            try:
                session.commit()
                for t in open_trades:
                    if t.status == "closed":
                        print(
                            f"[Resolution] db_closed trade_id={t.id}"
                            f" slug={session.get(Market, t.market_id).slug if t.market_id else '?'}"
                            f" status=closed pnl={t.pnl:.4f}",
                            flush=True,
                        )
            except Exception as _commit_exc:
                print(f"[Resolution] db_commit_failed: {_commit_exc}", flush=True)
                LOGGER.error("BTC SCALP resolution commit failed: %s", _commit_exc)
                session.rollback()
                closed_this_run = 0  # reset so summary reflects reality
            LOGGER.info(
                "BTC SCALP resolution cycle | open_checked=%d missing_condition_id=%d"
                " gamma_queried=%d unresolved=%d missing_outcome=%d closed=%d",
                n_open, n_missing_cid, n_gamma_queried, n_unresolved, n_missing_outcome, closed_this_run,
            )

        if pending_notifications:
            from watchdog.notifications.telegram import send_telegram
            try:
                stats = self._btc_paper_stats()
                bankroll = self._bankroll_block(stats)
            except Exception as exc:
                LOGGER.warning("BTC SCALP: could not build bankroll block: %s", exc)
                bankroll = None
            for msg, trade_id, slug in pending_notifications:
                try:
                    full_msg = f"{msg}\n{bankroll}" if bankroll else msg
                    n_telegram_attempted += 1
                    print(
                        f"[Resolution] telegram_close_attempt trade_id={trade_id} slug={slug}",
                        flush=True,
                    )
                    # Call send_telegram directly so exceptions propagate to this try/except.
                    # _notify() swallows all errors (fire-and-forget), making telegram_close_failed
                    # unreachable. send_telegram now raises on HTTP errors so we can detect failures.
                    send_telegram(full_msg, self._tg_token, self._tg_chat)
                    n_telegram_sent += 1
                    print(f"[Resolution] telegram_close_sent trade_id={trade_id}", flush=True)
                    # Avoid Telegram rate limit (30 msg/min). Open notifications fire one per
                    # scan cycle (5s apart) so they are not affected.
                    await asyncio.sleep(0.3)
                except Exception as exc:
                    n_notify_fail += 1
                    print(
                        f"[Resolution] telegram_close_failed trade_id={trade_id} error={exc}",
                        flush=True,
                    )
                    LOGGER.warning("Close notification failed trade_id=%d: %s", trade_id, exc)

        n_parsed = n_open - n_missing_cid - n_unresolved - n_missing_outcome
        print(
            f"[Resolution] Summary inspected={n_open} parsed={n_parsed}"
            f" closed={closed_this_run} parse_fail={n_missing_outcome}"
            f" missing_cid={n_missing_cid} unresolved={n_unresolved}"
            f" close_msgs_queued={n_close_msgs_queued}"
            f" telegram_attempted={n_telegram_attempted}"
            f" telegram_sent={n_telegram_sent}"
            f" telegram_failed={n_notify_fail}",
            flush=True,
        )
        LOGGER.info(
            "[Resolution] open=%d closed=%d missing_cid=%d gamma_queried=%d"
            " unresolved=%d parsed=%d close_msgs_queued=%d"
            " telegram_attempted=%d telegram_sent=%d telegram_failed=%d",
            n_open, closed_this_run, n_missing_cid, n_gamma_queried,
            n_unresolved, n_parsed, n_close_msgs_queued,
            n_telegram_attempted, n_telegram_sent, n_notify_fail,
        )

    # ── Main loop ───────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        while True:
            # Wait up to 10s for initial price
            for _ in range(20):
                if self._btc_price is not None:
                    break
                await asyncio.sleep(0.5)

            if self._btc_price is None:
                LOGGER.warning("BTC price not available after 10s — retrying in 15s")
                await asyncio.sleep(15)
                continue

            print(
                f"⚡ BTC Scalp Worker starting...\n"
                f"   Current BTC price: ${self._btc_price:,.0f}\n"
                f"   Scanning for near-expiry markets every 5s\n"
                f"   Paper trading only — ENABLE_LIVE_TRADING=false",
                flush=True,
            )
            if not self._has_sent_online_ping:
                self._notify(
                    f"⚡ BTC Scalp Worker online 🟢\n"
                    f"BTC: ${self._btc_price:,.0f}\n"
                    f"Scanning Polymarket every 5s\n"
                    f"Paper mode: ON"
                )
                self._has_sent_online_ping = True

            last_resolution_check = datetime.now(UTC)
            while True:
                # ── Scan ────────────────────────────────────────────────
                try:
                    await self._scan_markets()
                except Exception as exc:
                    print(f"[BTC Scalp] Scan error: {exc} — continuing", flush=True)
                    LOGGER.warning("BTC scalp scan error: %s", exc)
                    await asyncio.sleep(10)
                    # intentional fall-through: resolution gate still runs below

                # ── Resolution gate ─────────────────────────────────────
                elapsed = (datetime.now(UTC) - last_resolution_check).total_seconds()
                if elapsed >= RESOLUTION_CHECK_INTERVAL_S:
                    print(
                        f"[Resolution] Scheduler firing (elapsed={elapsed:.0f}s)",
                        flush=True,
                    )
                    try:
                        await self._check_resolutions()
                    except Exception as exc:
                        print(f"[BTC Scalp] Resolution check error: {exc}", flush=True)
                        LOGGER.error("BTC scalp resolution check error: %s", exc)
                    last_resolution_check = datetime.now(UTC)  # always reset, success OR exception

                await asyncio.sleep(POLL_INTERVAL_S)

    async def run_forever(self) -> None:
        """Run price feed + scanner forever with crash recovery."""
        price_task = asyncio.create_task(self._run_price_feed())
        await asyncio.sleep(2)  # let price feed establish

        while True:
            try:
                await self._run_loop()
            except Exception as exc:
                print(f"[BTC Scalp] Loop error: {exc} — restarting in 10s", flush=True)
                LOGGER.error("BTC scalp loop crashed: %s", exc)
                await asyncio.sleep(10)

        price_task.cancel()  # unreachable but satisfies linters

    def get_summary(self) -> dict[str, Any]:
        base: dict[str, Any] = {
            "signals": self._signals,
            "trades_opened": self._trades_opened,
            "trades_closed": self._trades_closed,
            "pnl": self._pnl,
        }
        with contextlib.suppress(Exception):
            base.update(self._btc_paper_stats())
        return base


# ── Backtest sanity check ────────────────────────────────────────────────────


def btc_scalp_backtest() -> None:
    """Synthetic walk-forward: 1000 BTC price steps, ±0.3% per step.

    Tests ±3% strike offsets per step — a -3% offset (BTC above strike) yields
    certainty ≈ 0.96, representative of real near-expiry BTC binary markets.
    """
    rng = random.Random(42)
    price = 83_000.0
    prices: list[float] = [price]
    for _ in range(999):
        price *= 1.0 + rng.uniform(-0.003, 0.003)
        prices.append(price)

    signals = 0
    edges: list[float] = []
    wins = 0
    total_pnl = 0.0

    _strategy = object.__new__(BtcScalpStrategy)  # skip __init__

    # ±3% offset: BTC well above/below strike gives certainty ~0.96, crossing threshold
    strike_offsets = [-0.03, +0.03]

    for i, btc in enumerate(prices):
        for offset in strike_offsets:
            strike = btc * (1.0 + offset)
            score, side = _strategy._compute_certainty(btc, strike)
            if score <= _V1_CERTAINTY_THRESHOLD:
                continue

            # Synthetic ask: market prices slightly below certainty (realistic taker spread)
            synthetic_ask = score - rng.uniform(0.06, 0.14)
            edge = score - synthetic_ask
            if edge < _V1_EDGE_THRESHOLD:
                continue

            signals += 1
            edges.append(edge)

            # Outcome: does BTC hold position relative to strike in the next step?
            if i + 1 < len(prices):
                next_price = prices[i + 1]
                won = (next_price > strike) if side == "YES" else (next_price < strike)
            else:
                won = score > 0.5

            exit_price = 1.0 if won else 0.0
            pnl = (exit_price - synthetic_ask) * PAPER_POSITION_SIZE
            total_pnl += pnl
            if won:
                wins += 1

    avg_edge = (sum(edges) / len(edges) * 100) if edges else 0.0
    win_rate = (wins / signals * 100) if signals else 0.0

    print(
        f"\n📊 BTC SCALP BACKTEST (1000 synthetic price steps from $83,000, ±0.3%/step)\n"
        f"   Strike offsets:      ±3% from current price\n"
        f"   Total signals:       {signals}\n"
        f"   Avg edge:            {avg_edge:.1f}¢\n"
        f"   Est. win rate:       {win_rate:.1f}%\n"
        f"   Est. PnL @$2/trade:  ${total_pnl:.2f}\n",
        flush=True,
    )


if __name__ == "__main__":
    btc_scalp_backtest()
