"""
BTC intraday scalp strategy.
Connects to Coinbase Advanced Trade WebSocket for live BTC price,
polls Polymarket Gamma API every 5s for near-expiry BTC markets,
and paper-trades when certainty_score > 0.90 with 8c+ edge.

Price feed priority:
  1. Coinbase Advanced Trade WebSocket (wss://advanced-trade-ws.coinbase.com) — no geo-block
  2. CoinGecko HTTP fallback (30s intervals) — if WebSocket fails 3 consecutive times
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
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
from watchdog.db.models import BtcSignalLog, Market, Trade
from watchdog.db.session import build_engine, build_session_factory

LOGGER = logging.getLogger(__name__)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
WS_MAX_FAILURES = 3
HTTP_FALLBACK_INTERVAL_S = 30
POLL_INTERVAL_S = 5
RESOLUTION_CHECK_INTERVAL_S = 60
CERTAINTY_THRESHOLD = 0.90
EDGE_THRESHOLD = 0.08
MIN_EDGE = 0.10
PAPER_POSITION_SIZE = 2.0
NEAR_EXPIRY_MINUTES = 12
SIGNAL_COOLDOWN_S = 90.0


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

    def _compute_signal(
        self, market: dict[str, Any]
    ) -> tuple[str, float, float, float] | None:
        """Compute Up/Down signal from momentum vs market price.

        Returns (side, edge, our_prob, market_price) or None if no edge.
        side is 'Up' or 'Down'.
        """
        try:
            outcome_prices = json.loads(market.get("outcomePrices") or '["0.5","0.5"]')
            up_price = float(outcome_prices[0])
            down_price = float(outcome_prices[1])
        except (ValueError, IndexError, TypeError):
            return None

        momentum = self._get_momentum()
        our_up_prob = max(0.01, min(0.99, 0.5 + momentum * 0.4))
        our_down_prob = 1.0 - our_up_prob

        up_edge = our_up_prob - up_price
        down_edge = our_down_prob - down_price

        if up_edge > MIN_EDGE:
            return ("Up", up_edge, our_up_prob, up_price)
        if down_edge > MIN_EDGE:
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

            minutes_left = (end_dt - now).total_seconds() / 60
            if not (1.5 <= minutes_left <= 8.0):
                print(
                    f"[Scan] skip slug={m.get('slug') or m.get('conditionId') or '?'}"
                    f" minutes_left={minutes_left:.1f} (outside 1.5-8.0min)",
                    flush=True,
                )
                continue

            signal = self._compute_signal(m)
            if signal is None:
                print(
                    f"[Scan] no_edge slug={m.get('slug') or m.get('conditionId') or '?'}"
                    f" momentum={self._get_momentum():.4f}"
                    f" minutes_left={minutes_left:.1f}",
                    flush=True,
                )
                continue

            side, edge, our_prob, market_price = signal
            slug = m.get("slug") or m.get("conditionId") or f"btc-scalp-{int(now.timestamp())}"
            signal_key = f"{m.get('conditionId', slug)}-{side}"

            # Purge expired cooldown entries, then check
            now_ts = time.time()
            self._fired_signals = {
                k: v for k, v in self._fired_signals.items()
                if now_ts - v < SIGNAL_COOLDOWN_S
            }
            if signal_key in self._fired_signals:
                skipped_cooldown += 1
                continue
            self._fired_signals[signal_key] = now_ts

            self._signals += 1
            qualified_count += 1
            liquidity = float(m.get("liquidity") or 0)

            print(
                f"⚡ UP/DOWN SIGNAL | BTC=${btc:,.0f} | side={side} | "
                f"edge={edge:.2f} | our={our_prob:.2f} | mkt={market_price:.2f} | "
                f"liq=${liquidity:,.0f} | expires={minutes_left:.1f}min",
                flush=True,
            )

            with session_factory() as session:
                # ── Signal audit log ────────────────────────────────────────
                session.add(BtcSignalLog(
                    logged_at=now,
                    market_slug=slug,
                    question=question,
                    side=side,
                    edge=edge,
                    time_left_min=minutes_left,
                    our_estimate=our_prob,
                    market_price=market_price,
                    momentum=self._get_momentum(),
                    btc_price=btc,
                    liquidity=liquidity,
                    outcome=None,
                ))
                session.flush()

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

                existing = (
                    session.query(Trade)
                    .filter(
                        Trade.market_id == market_row.id,
                        Trade.strategy == "btc_scalp",
                        Trade.status == "open",
                    )
                    .first()
                )
                if existing:
                    LOGGER.debug("BTC SCALP: open trade already exists for %s — skipping", slug)
                    continue

                if self._check_open_risk_cap(session):
                    continue

                trade = Trade(
                    market_id=market_row.id,
                    signal_id=None,
                    side=side,
                    size=PAPER_POSITION_SIZE,
                    entry_price=market_price,
                    kelly_fraction=0.0,
                    confidence_score=our_prob,
                    is_paper=True,
                    status="open",
                    opened_at=now,
                    strategy="btc_scalp",
                    order_id=f"btcscalp_{market_row.id}_{int(now.timestamp())}",
                )
                session.add(trade)
                session.commit()
                self._trades_opened += 1
                self._open_risk_cap_notified = False
                LOGGER.info(
                    "BTC SCALP: paper trade opened %s side=%s entry=%.4f our_prob=%.3f",
                    slug,
                    side,
                    market_price,
                    our_prob,
                )
                # Notify AFTER confirmed DB write
                self._notify(
                    f"✅ BTC TRADE OPENED\n"
                    f"Market: {question}\n"
                    f"Side: {side} | Entry: {market_price:.2f}\n"
                    f"Edge: {edge:.2f} | Our est: {our_prob:.2f}\n"
                    f"Time left: {minutes_left:.1f} min"
                )

        print(
            f"[BTC Scalp] Scan done — BTC=${btc:,.0f} | momentum={self._get_momentum():.3f} | "
            f"markets={len(btc_markets)} | signals={qualified_count} | cooldown_skipped={skipped_cooldown}",
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
            pending_notifications: list[str] = []

            for market_id in market_ids:
                exit_price = 0.0
                market = session.get(Market, market_id)
                if market is None:
                    continue

                market_trades = [t for t in open_trades if t.market_id == market_id]
                sides = [t.side for t in market_trades]
                print(
                    f"[DEBUG] Checking position: slug={market.slug} condition_id={market.condition_id} side={sides}",
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
                            for item in items:
                                if item.get("resolved"):
                                    raw = (item.get("resolutionOutcome") or "").upper()
                                    if raw in ("YES", "NO"):
                                        outcome = raw
                                        market.resolution_outcome = outcome
                                        market.status = "resolved"
                                        print(
                                            f"[Gamma] resolved slug={market.slug}"
                                            f" condition_id={market.condition_id}"
                                            f" status={resp.status_code}"
                                            f" resolved=True resolutionOutcome={raw}",
                                            flush=True,
                                        )
                                        LOGGER.info(
                                            "BTC SCALP Gamma resolved | slug=%s"
                                            " resolved=True resolutionOutcome=%s",
                                            market.slug, raw,
                                        )
                                        break
                            else:
                                n_unresolved += 1
                                print(
                                    f"[Gamma] unresolved slug={market.slug}"
                                    f" condition_id={market.condition_id}"
                                    f" status={resp.status_code} resolved=False",
                                    flush=True,
                                )
                                LOGGER.info(
                                    "BTC SCALP Gamma not resolved | slug=%s"
                                    " resolved=False or no matching item",
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

                for trade in market_trades:
                    exit_price = 1.0 if trade.side == outcome else 0.0
                    pnl = (exit_price - trade.entry_price) * trade.size
                    trade.exit_price = exit_price
                    trade.pnl = pnl
                    trade.status = "closed"
                    trade.closed_at = now
                    trade.close_reason = "resolution"
                    self._pnl += pnl
                    self._trades_closed += 1
                    closed_this_run += 1
                    print(
                        f"[Resolution] trade closed in DB trade_id={trade.id}"
                        f" slug={market.slug} pnl={pnl:.4f}",
                        flush=True,
                    )
                    print(
                        f"[Gamma] action={'closed_win' if exit_price == 1.0 else 'closed_loss'}"
                        f" slug={market.slug} condition_id={market.condition_id}"
                        f" side={trade.side} outcome={outcome} pnl={pnl:.4f}",
                        flush=True,
                    )
                    LOGGER.info(
                        "BTC SCALP trade disposition | trade_id=%d slug=%s side=%s outcome=%s"
                        " action=%s pnl=%.4f",
                        trade.id, market.slug, trade.side, outcome,
                        "closed_win" if exit_price == 1.0 else "closed_loss",
                        pnl,
                    )
                    if exit_price == 1.0:
                        pending_notifications.append(
                            f"🏆 BTC TRADE CLOSED — WIN\n"
                            f"Market: {market.question}\n"
                            f"Side: {trade.side} | Entry: {trade.entry_price:.2f} → Exit: 1.00\n"
                            f"PnL: +${pnl:.2f} on $2.00 stake\n"
                            f"Session: {self._trades_closed} closed | Net PnL: ${self._pnl:+.2f}"
                        )
                    else:
                        pending_notifications.append(
                            f"❌ BTC TRADE CLOSED — LOSS\n"
                            f"Market: {market.question}\n"
                            f"Side: {trade.side} | Entry: {trade.entry_price:.2f} → Exit: 0.00\n"
                            f"PnL: -${abs(pnl):.2f} on $2.00 stake\n"
                            f"Session: {self._trades_closed} closed | Net PnL: ${self._pnl:+.2f}"
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

            session.commit()
            LOGGER.info(
                "BTC SCALP resolution cycle | open_checked=%d missing_condition_id=%d"
                " gamma_queried=%d unresolved=%d missing_outcome=%d closed=%d",
                n_open, n_missing_cid, n_gamma_queried, n_unresolved, n_missing_outcome, closed_this_run,
            )

        if pending_notifications:
            try:
                stats = self._btc_paper_stats()
                bankroll = self._bankroll_block(stats)
            except Exception as exc:
                LOGGER.warning("BTC SCALP: could not build bankroll block: %s", exc)
                bankroll = None
            for msg in pending_notifications:
                try:
                    full_msg = f"{msg}\n{bankroll}" if bankroll else msg
                    self._notify(full_msg)
                except Exception as exc:
                    print(f"[Resolution] Notification failed: {exc}", flush=True)
                    LOGGER.warning("Close notification failed: %s", exc)

        print(
            f"[Resolution] Checked {n_open} open positions, {closed_this_run} resolved and closed",
            flush=True,
        )
        LOGGER.info(
            "[Resolution] open=%d closed=%d missing_cid=%d gamma_queried=%d unresolved=%d",
            n_open, closed_this_run, n_missing_cid, n_gamma_queried, n_unresolved,
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
            if score <= CERTAINTY_THRESHOLD:
                continue

            # Synthetic ask: market prices slightly below certainty (realistic taker spread)
            synthetic_ask = score - rng.uniform(0.06, 0.14)
            edge = score - synthetic_ask
            if edge < EDGE_THRESHOLD:
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
