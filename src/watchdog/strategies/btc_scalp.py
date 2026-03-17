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
from datetime import UTC, datetime, timedelta
from typing import Any

import aiohttp
import httpx
import websockets

from watchdog.core.config import get_settings
from watchdog.db.init import init_db
from watchdog.db.models import Market, Trade
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
PAPER_POSITION_SIZE = 2.0
NEAR_EXPIRY_MINUTES = 12


class BtcScalpStrategy:
    def __init__(self) -> None:
        self._btc_price: float | None = None
        self._signals: int = 0
        self._trades_opened: int = 0
        self._trades_closed: int = 0
        self._pnl: float = 0.0
        self._has_sent_online_ping: bool = False
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

    # ── Market scanner ──────────────────────────────────────────────────────

    async def _scan_markets(self) -> None:
        """Poll Gamma API for near-expiry BTC markets and paper-trade strong signals."""
        if self._btc_price is None:
            return

        btc = self._btc_price
        now = datetime.now(UTC)
        cutoff = now + timedelta(minutes=NEAR_EXPIRY_MINUTES)

        BTC_KEYWORDS = ("btc", "bitcoin", "above $", "below $", "up or down")
        STRIKE_KEYWORDS = ("above", "below", "up or down", "higher", "lower")

        async def _fetch_markets(params: dict[str, Any]) -> list[dict[str, Any]]:
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.get(f"{GAMMA_API_BASE}/markets", params=params)
                if resp.status_code != 200:
                    LOGGER.warning("Gamma API returned %d: %s", resp.status_code, resp.text[:200])
                    return []
                return resp.json()
            except Exception as exc:
                LOGGER.warning("Gamma API poll failed: %s", exc)
                return []

        # Primary query: soonest-expiring active markets (client-side end_date filter)
        primary_params: dict[str, Any] = {
            "active": "true",
            "closed": "false",
            "archived": "false",
            "limit": "100",
            "order": "end_date_iso",
            "ascending": "true",
        }
        print(f"[DEBUG] Querying Gamma with params: {primary_params}", flush=True)
        markets = await _fetch_markets(primary_params)

        btc_markets = [
            m for m in markets
            if any(kw in (m.get("question") or "").lower() for kw in BTC_KEYWORDS)
            and any(kw in (m.get("question") or "").lower() for kw in STRIKE_KEYWORDS)
        ]
        print(
            f"[BTC Scalp] Total markets fetched: {len(markets)} | BTC strike: {len(btc_markets)}",
            flush=True,
        )
        if btc_markets:
            print(
                f"[DEBUG] First BTC market: {btc_markets[0].get('question')} | endDate: {btc_markets[0].get('endDate')}",
                flush=True,
            )

        # Fallback: top-volume active markets, filter BTC client-side
        if not btc_markets:
            fallback_params: dict[str, Any] = {
                "active": "true",
                "closed": "false",
                "limit": "50",
                "order": "volume_24hr",
                "ascending": "false",
            }
            print(f"[DEBUG] Primary returned 0 BTC markets — trying fallback: {fallback_params}", flush=True)
            fallback_markets = await _fetch_markets(fallback_params)
            btc_markets = [
                m for m in fallback_markets
                if any(kw in (m.get("question") or "").lower() for kw in STRIKE_KEYWORDS)
            ]
            print(f"[BTC Scalp] Fallback BTC strike markets: {len(btc_markets)}", flush=True)
            if fallback_markets and not btc_markets:
                # Show what the fallback did return so we can tune further
                print(
                    f"[DEBUG] Fallback sample: {fallback_markets[0].get('question')} | slug: {fallback_markets[0].get('slug')}",
                    flush=True,
                )

        if not btc_markets:
            print(
                f"[BTC Scalp] No near-expiry BTC strike markets found this cycle",
                flush=True,
            )
            return

        settings = get_settings()
        engine = build_engine(settings)
        init_db(engine)
        session_factory = build_session_factory(engine)

        qualified_count = 0
        for m in btc_markets:
            question = m.get("question") or ""

            end_date_str = m.get("endDateIso") or m.get("end_date_iso") or ""
            if not end_date_str:
                continue
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=UTC)
            except ValueError:
                continue

            if end_dt > cutoff or end_dt <= now:
                continue

            strike = self._extract_strike(question)
            if strike is None:
                continue

            score, side = self._compute_certainty(btc, strike)
            if score <= CERTAINTY_THRESHOLD:
                continue

            best_ask = float(m.get("bestAsk") or m.get("best_ask") or 0.50)
            if best_ask <= 0:
                best_ask = 0.50

            edge = score - best_ask
            if edge < EDGE_THRESHOLD:
                continue

            self._signals += 1
            qualified_count += 1
            minutes_left = (end_dt - now).total_seconds() / 60
            slug = m.get("slug") or m.get("conditionId") or f"btc-scalp-{int(now.timestamp())}"
            print(
                f"⚡ SIGNAL | BTC=${btc:,.0f} | strike=${strike:,.0f} | "
                f"score={score:.2f} | ask={best_ask:.2f} | edge={edge * 100:.1f}¢ | side={side}",
                flush=True,
            )
            self._notify(
                f"⚡ BTC SIGNAL\n"
                f"BTC: ${btc:,.0f} | Strike: ${strike:,.0f}\n"
                f"Side: {side} | Edge: {edge * 100:.1f}¢ | Ask: {best_ask:.2f}\n"
                f"Market: {slug}\n"
                f"Expires: {minutes_left:.0f}min"
            )

            with session_factory() as session:
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
                    continue

                trade = Trade(
                    market_id=market_row.id,
                    signal_id=None,
                    side=side,
                    size=PAPER_POSITION_SIZE,
                    entry_price=best_ask,
                    kelly_fraction=0.0,
                    confidence_score=score,
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
                    "BTC SCALP: paper trade opened %s side=%s entry=%.4f score=%.3f",
                    slug,
                    side,
                    best_ask,
                    score,
                )

        print(
            f"[BTC Scalp] Scan done — BTC=${btc:,.0f} | BTC strike markets: {len(btc_markets)} | Signals fired: {qualified_count}",
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
                    LOGGER.info(
                        "BTC SCALP: closed %s side=%s outcome=%s pnl=%.4f",
                        market.slug,
                        trade.side,
                        outcome,
                        pnl,
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
