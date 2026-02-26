from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from watchdog.core.config import Settings
from watchdog.db.models import Signal, Telemetry, Trade
from watchdog.market_data.polymarket_cli import PolymarketCli

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionSnapshot:
    market_id: int
    market_slug: str
    ask: float
    mid: float
    volume: float
    liquidity: float


class OrderExecutor:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker[Session],
        polymarket_cli: PolymarketCli,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.polymarket_cli = polymarket_cli

    @staticmethod
    def _extract_position_fraction(kelly_result: Any) -> float:
        for attr in ("position_fraction", "empirical_fraction", "fraction"):
            if hasattr(kelly_result, attr):
                return float(getattr(kelly_result, attr))
            if isinstance(kelly_result, dict) and attr in kelly_result:
                return float(kelly_result[attr])
        return 0.0

    @staticmethod
    def _extract_bankroll(kelly_result: Any) -> float:
        if hasattr(kelly_result, "bankroll"):
            return float(kelly_result.bankroll)
        if isinstance(kelly_result, dict) and "bankroll" in kelly_result:
            return float(kelly_result["bankroll"])
        return 0.0

    @staticmethod
    def _side_from_signal(signal: Signal) -> str:
        return "YES" if signal.model_probability >= signal.market_probability else "NO"

    def _check_preconditions(
        self,
        session: Session,
        signal: Signal,
        snapshot: ExecutionSnapshot,
        position_fraction: float,
        bankroll: float,
    ) -> tuple[bool, str]:
        if snapshot.volume < self.settings.min_market_volume_usdc:
            return False, "volume_below_minimum"
        if snapshot.liquidity < self.settings.min_market_liquidity_usdc:
            return False, "liquidity_below_minimum"

        confidence = float(signal.executor_confidence or 0.0)
        if confidence < self.settings.executor_confidence_threshold:
            return False, "executor_confidence_below_threshold"

        if position_fraction <= 0:
            return False, "non_positive_position_fraction"

        open_positions = session.execute(select(func.count()).select_from(Trade).where(Trade.status == "open")).scalar_one()
        if int(open_positions) >= self.settings.max_positions_simultaneous:
            return False, "max_open_positions_reached"

        current_exposure = session.execute(
            select(func.coalesce(func.sum(Trade.size * Trade.entry_price), 0.0)).where(Trade.status == "open")
        ).scalar_one()
        if bankroll > 0 and float(current_exposure) >= bankroll * self.settings.max_position_fraction:
            return False, "open_exposure_limit_reached"

        return True, "ok"

    def execute(
        self,
        signal: Signal,
        snapshot: ExecutionSnapshot,
        kelly_result: Any,
        is_paper: bool,
    ) -> dict[str, Any]:
        position_fraction = self._extract_position_fraction(kelly_result)
        bankroll = self._extract_bankroll(kelly_result)

        requested_live = not is_paper
        live_enabled = self.settings.enable_live_trading
        live_mode = requested_live and live_enabled
        if requested_live and not live_enabled:
            LOGGER.warning("Live execution requested but ENABLE_LIVE_TRADING is false. Forcing paper mode.")

        side = self._side_from_signal(signal)
        fill_price = float(snapshot.ask)
        slippage = 0.005
        notional = bankroll * min(position_fraction, self.settings.max_position_fraction) if bankroll > 0 else position_fraction
        size = notional / max(fill_price, 1e-9)

        with self.session_factory() as session:
            try:
                allowed, reason = self._check_preconditions(session, signal, snapshot, position_fraction, bankroll)
                if not allowed:
                    return {
                        "executed": False,
                        "reason": reason,
                        "is_paper": True,
                    }

                order_id = f"paper-{signal.id}-{datetime.now(UTC).strftime('%H%M%S%f')}"
                if live_mode:
                    response = self.polymarket_cli.create_limit_order(
                        token_id=str(snapshot.market_id),
                        side=side,
                        price=fill_price,
                        size=size,
                        post_only=True,
                    )
                    payload = response.payload if isinstance(response.payload, dict) else {}
                    order_id = str(payload.get("order_id") or payload.get("id") or order_id)

                trade = Trade(
                    market_id=snapshot.market_id,
                    signal_id=signal.id,
                    side=side,
                    size=size,
                    entry_price=fill_price,
                    slippage=slippage,
                    kelly_fraction=position_fraction,
                    confidence_score=signal.executor_confidence,
                    order_id=order_id,
                    is_paper=not live_mode,
                    status="open",
                    opened_at=datetime.now(UTC),
                )
                session.add(trade)
                session.commit()

                return {
                    "executed": True,
                    "trade_id": trade.id,
                    "order_id": order_id,
                    "is_paper": trade.is_paper,
                    "fill_price": fill_price,
                    "size": size,
                }
            except Exception:
                session.rollback()
                raise

    def cancel_all(self, reason: str) -> dict[str, Any]:
        response = self.polymarket_cli.cancel_all()

        with self.session_factory() as session:
            try:
                session.execute(
                    Trade.__table__.update().where(Trade.status == "open").values(status="canceled", closed_at=datetime.now(UTC))
                )
                session.add(
                    Telemetry(
                        pipeline_id="order-executor",
                        ts_order_submitted=datetime.now(UTC),
                        total_latency_ms=0,
                    )
                )
                session.commit()
            except Exception:
                session.rollback()
                raise

        LOGGER.warning("Emergency cancel_all triggered: %s", reason)
        return {
            "reason": reason,
            "cancel_response": response.payload,
        }

    def close_position(self, trade_id: int, resolution_outcome: float) -> dict[str, Any]:
        outcome = float(resolution_outcome)

        with self.session_factory() as session:
            try:
                trade = session.get(Trade, trade_id)
                if trade is None:
                    return {"closed": False, "reason": "trade_not_found"}

                payout = outcome if trade.side == "YES" else (1 - outcome)
                pnl = trade.size * (payout - trade.entry_price)

                trade.exit_price = outcome
                trade.pnl = pnl
                trade.status = "closed"
                trade.closed_at = datetime.now(UTC)
                session.commit()

                return {
                    "closed": True,
                    "trade_id": trade_id,
                    "pnl_usdc": pnl,
                }
            except Exception:
                session.rollback()
                raise
