from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from watchdog.core.config import Settings

LOGGER = logging.getLogger(__name__)


class TelegramAlerter:
    def __init__(self, settings: Settings) -> None:
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)
        self._tasks: set[asyncio.Task] = set()

        if not self.enabled:
            LOGGER.warning("Telegram alerts disabled: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")

    async def _send(self, text: str) -> None:
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network failure safety
            LOGGER.warning("Telegram send failed: %s", exc)

    def _dispatch(self, text: str) -> None:
        task = asyncio.create_task(self._send(text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def send_signal_alert(self, signal: Any, market: Any, snapshot: Any) -> None:
        mode = "PAPER" if getattr(signal, "is_paper", False) else "LIVE"
        text = (
            f"[{mode}] Watchdog Signal\n"
            f"Market: {getattr(market, 'question', 'unknown')}\n"
            f"p_market: {getattr(signal, 'market_probability', 'n/a')}\n"
            f"p_model: {getattr(signal, 'model_probability', 'n/a')}\n"
            f"divergence: {getattr(signal, 'divergence', 'n/a')}\n"
            f"direction: {'YES' if getattr(signal, 'model_probability', 0.5) >= getattr(signal, 'market_probability', 0.5) else 'NO'}\n"
            f"confidence: {getattr(signal, 'executor_confidence', 'n/a')}\n"
            f"kelly_size: {getattr(signal, 'kelly_fraction', 'n/a')}"
        )
        self._dispatch(text)

    def send_abort_alert(self, reason: str) -> None:
        self._dispatch(f"[CRITICAL] Watchdog abort\nReason: {reason}")

    def send_daily_summary(self, stats: dict[str, Any]) -> None:
        text = (
            "Watchdog Daily Summary\n"
            f"trades_today: {stats.get('trades_today')}\n"
            f"paper_pnl: {stats.get('paper_pnl')}\n"
            f"api_costs: {stats.get('api_costs')}\n"
            f"signal_count: {stats.get('signal_count')}\n"
            f"win_rate_7d: {stats.get('win_rate_7d')}"
        )
        self._dispatch(text)

    def send_latency_alert(self, experiment_id: str, signal_ms: int, price_moved: float) -> None:
        text = (
            "[LATENCY ALERT]\n"
            f"experiment_id: {experiment_id}\n"
            f"signal_ms: {signal_ms}\n"
            f"price_moved: {price_moved}"
        )
        self._dispatch(text)
