from __future__ import annotations

import logging

import httpx

LOGGER = logging.getLogger(__name__)


def send_telegram(message: str, token: str | None, chat_id: str | None) -> None:
    if not token or not chat_id:
        LOGGER.debug("Telegram disabled — TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return
    r = httpx.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=12.0,
    )
    r.raise_for_status()
