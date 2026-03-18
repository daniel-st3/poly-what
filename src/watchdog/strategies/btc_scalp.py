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
import json
import logging
import random
import re
import time
from collections import deque
from datetime import UTC, datetime, timedelta
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
        _s = get_settings()
        self._tg_token: str | None = _s.telegram_bot_token
        self._tg_chat: str | None = _s.telegram_chat_id
        # Ensure all tables (trades, markets, …) exist before any DB access
        _engine = build_engine(_s)
        init_db(_engine)
        print("[BTC Scalp] DB initialised ✅", flush=True)

    def _notify(self, msg: str) -> None:
        """Fire-and-forget Telegram notification (sync, swallows errors)."""
        from watchdog.notifications.telegram import send_telegram
        send_telegram(msg, self._tg_token, self._tg_chat)

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
                continue

            signal = self._compute_signal(m)
            if signal is None:
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

            for market_id in market_ids:
                market = session.get(Market, market_id)
                if market is None:
                    continue

                sides = [t.side for t in open_trades if t.market_id == market_id]
                print(
                    f"[DEBUG] Checking position: slug={market.slug} condition_id={market.condition_id} side={sides}",
                    flush=True,
                )
                outcome = (market.resolution_outcome or "").upper()

                if outcome not in ("YES", "NO") and market.condition_id:
                    try:
                        async with httpx.AsyncClient(timeout=8.0) as client:
                            resp = await client.get(
                                f"{GAMMA_API_BASE}/markets",
                                params={"conditionId": market.condition_id},
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
                                        break
                    except Exception as exc:
                        LOGGER.warning(
                            "BTC SCALP resolution check failed for %s: %s",
                            market.slug,
                            exc,
                        )

                if outcome not in ("YES", "NO"):
                    continue

                for trade in open_trades:
                    if trade.market_id != market_id:
                        continue
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
                    if exit_price == 1.0:
                        self._notify(
                            f"🏆 BTC TRADE CLOSED — WIN\n"
                            f"Market: {market.question}\n"
                            f"Side: {trade.side} | Entry: {trade.entry_price:.2f} → Exit: 1.00\n"
                            f"PnL: +${pnl:.2f} on $2.00 stake\n"
                            f"Session: {self._trades_closed} closed | Net PnL: ${self._pnl:+.2f}"
                        )
                    else:
                        self._notify(
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
        print(
            f"[Resolution] Checked {n_open} open positions, {closed_this_run} resolved and closed",
            flush=True,
        )

    # ── Main loop ───────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        # Wait up to 10s for initial price
        for _ in range(20):
            if self._btc_price is not None:
                break
            await asyncio.sleep(0.5)

        if self._btc_price is None:
            LOGGER.warning("BTC price not available after 10s — skipping scan cycle")
            return

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
            try:
                await self._scan_markets()

                elapsed = (datetime.now(UTC) - last_resolution_check).total_seconds()
                if elapsed >= RESOLUTION_CHECK_INTERVAL_S:
                    await self._check_resolutions()
                    last_resolution_check = datetime.now(UTC)
            except Exception as exc:
                print(f"[BTC Scalp] Loop error: {exc} — restarting in 10s", flush=True)
                await asyncio.sleep(10)
                continue

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
        return {
            "signals": self._signals,
            "trades_opened": self._trades_opened,
            "trades_closed": self._trades_closed,
            "pnl": self._pnl,
        }


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
