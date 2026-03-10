from __future__ import annotations

import logging
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from watchdog.db.models import Trade

LOGGER = logging.getLogger(__name__)


class CircuitBreaker:
    """Daily loss kill-switch.

    Queries realized P&L for paper trades closed today. If the loss exceeds
    ``max_daily_loss_usd`` the breaker trips, blocks new trades, and fires an
    optional n8n webhook alert.  The breaker resets automatically after
    ``cooldown_minutes`` to allow trading to resume.
    """

    def __init__(
        self,
        max_daily_loss_usd: float = 50.0,
        cooldown_minutes: int = 60,
        n8n_webhook_url: str | None = None,
    ) -> None:
        self.max_daily_loss_usd = max_daily_loss_usd
        self.cooldown_minutes = cooldown_minutes
        self.n8n_webhook_url = n8n_webhook_url

        self._tripped_at: datetime | None = None
        self._notified: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_open(self, session: Session) -> bool:
        """Return True if trading is ALLOWED (breaker not tripped)."""
        if self._tripped_at is None:
            return True

        elapsed = datetime.now(UTC) - self._tripped_at
        if elapsed >= timedelta(minutes=self.cooldown_minutes):
            LOGGER.warning(
                "CircuitBreaker: cooldown expired after %d min — resetting",
                self.cooldown_minutes,
            )
            self._tripped_at = None
            self._notified = False
            return True

        remaining = self.cooldown_minutes - elapsed.total_seconds() / 60
        LOGGER.warning(
            "CircuitBreaker: TRIPPED — trading blocked for another %.0f min",
            remaining,
        )
        return False

    def check(self, session: Session) -> bool:
        """Evaluate today's P&L and trip the breaker if threshold exceeded.

        Returns True if trading is allowed, False if blocked.
        """
        if not self.is_open(session):
            return False

        daily_pnl = self._get_daily_pnl(session)

        if daily_pnl <= -self.max_daily_loss_usd:
            self._trip(daily_pnl)
            return False

        LOGGER.debug(
            "CircuitBreaker: daily_pnl=%.2f (threshold=-%.2f) — OK",
            daily_pnl,
            self.max_daily_loss_usd,
        )
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_daily_pnl(self, session: Session) -> float:
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        rows = session.execute(
            select(Trade.pnl).where(
                Trade.is_paper.is_(True),
                Trade.status == "closed",
                Trade.closed_at >= today_start,
                Trade.pnl.is_not(None),
            )
        ).scalars().all()
        return float(sum(rows))

    def _trip(self, daily_pnl: float) -> None:
        self._tripped_at = datetime.now(UTC)
        LOGGER.warning(
            "CircuitBreaker: TRIPPED — daily_pnl=%.2f exceeded threshold -%.2f. "
            "Trading halted for %d min.",
            daily_pnl,
            self.max_daily_loss_usd,
            self.cooldown_minutes,
        )
        if not self._notified:
            self._send_alert(daily_pnl)
            self._notified = True

    def _send_alert(self, daily_pnl: float) -> None:
        if not self.n8n_webhook_url:
            return
        import json

        payload: dict[str, Any] = {
            "event": "circuit_breaker_tripped",
            "daily_pnl_usd": round(daily_pnl, 2),
            "threshold_usd": self.max_daily_loss_usd,
            "tripped_at": datetime.now(UTC).isoformat(),
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.n8n_webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                LOGGER.info("CircuitBreaker: webhook fired (status=%d)", resp.status)
        except Exception as exc:
            LOGGER.warning("CircuitBreaker: webhook failed: %s", exc)
