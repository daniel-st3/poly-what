from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from watchdog.db.base import Base


class Market(Base):
    __tablename__ = "markets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    question: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String(64), index=True)
    yes_token_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    no_token_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    condition_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    resolution_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    resolution_outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    snapshots: Mapped[list[MarketSnapshot]] = relationship(back_populates="market")
    signals: Mapped[list[Signal]] = relationship(back_populates="market")
    trades: Mapped[list[Trade]] = relationship(back_populates="market")
    maker_quotes: Mapped[list[MakerQuote]] = relationship(back_populates="market")


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    bid: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid: Mapped[float | None] = mapped_column(Float, nullable=True)
    spread: Mapped[float | None] = mapped_column(Float, nullable=True)
    bid_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    ask_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    cli_latency_ms: Mapped[int] = mapped_column(Integer)
    raw_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    market: Mapped[Market] = relationship(back_populates="snapshots")


class NewsEvent(Base):
    __tablename__ = "news_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    headline: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str | None] = mapped_column(String(1024), nullable=True, index=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    domain_tags: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    processed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    market_id: Mapped[int | None] = mapped_column(ForeignKey("markets.id"), nullable=True, index=True)
    news_event_id: Mapped[int | None] = mapped_column(ForeignKey("news_events.id"), nullable=True, index=True)
    model_probability: Mapped[float] = mapped_column(Float)
    market_probability: Mapped[float] = mapped_column(Float)
    divergence: Mapped[float] = mapped_column(Float)
    signal_type: Mapped[str] = mapped_column(String(64), index=True)
    router_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibration_adjustments: Mapped[float | None] = mapped_column(Float, nullable=True)
    vpin_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    executor_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    experiment_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    should_trade: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    market: Mapped[Market | None] = relationship(back_populates="signals")


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True, index=True)
    side: Mapped[str] = mapped_column(String(8))
    size: Mapped[float] = mapped_column(Float)
    entry_price: Mapped[float] = mapped_column(Float)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    slippage: Mapped[float | None] = mapped_column(Float, nullable=True)
    pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    kelly_fraction: Mapped[float] = mapped_column(Float)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    is_paper: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), default="open", nullable=False, index=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    strategy: Mapped[str | None] = mapped_column(String(32), nullable=True, default="calibration", index=True)
    close_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    high_water_mark: Mapped[float | None] = mapped_column(Float, nullable=True)
    remaining_fraction: Mapped[float | None] = mapped_column(Float, nullable=True, default=1.0)
    resolution_flag: Mapped[str | None] = mapped_column(String(32), nullable=True)  # normal / near_certain / high_risk_close

    market: Mapped[Market] = relationship(back_populates="trades")


class Telemetry(Base):
    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pipeline_id: Mapped[str] = mapped_column(String(64), index=True)
    market_id: Mapped[int | None] = mapped_column(ForeignKey("markets.id"), nullable=True, index=True)
    news_event_id: Mapped[int | None] = mapped_column(ForeignKey("news_events.id"), nullable=True, index=True)
    ts_news_received: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ts_router_completed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ts_calibration_completed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ts_executor_completed: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ts_order_submitted: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    market_price_at_signal: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_price_1m: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_price_5m: Mapped[float | None] = mapped_column(Float, nullable=True)
    router_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    calibration_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    executor_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    order_submission_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class CalibrationSurface(Base):
    __tablename__ = "calibration_surface"
    __table_args__ = (
        UniqueConstraint(
            "price_bucket", "time_bucket_hours", "domain", "dataset_source", name="uq_calibration_cell"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    price_bucket: Mapped[int] = mapped_column(Integer, index=True)
    time_bucket_hours: Mapped[int] = mapped_column(Integer, index=True)
    domain: Mapped[str] = mapped_column(String(64), index=True)
    dataset_source: Mapped[str] = mapped_column(String(128), index=True)
    sample_size: Mapped[int] = mapped_column(Integer)
    empirical_outcome_rate: Mapped[float] = mapped_column(Float)
    model_adjustment: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MakerQuote(Base):
    __tablename__ = "maker_quotes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[int] = mapped_column(ForeignKey("markets.id"), index=True)
    quoted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    reservation_price: Mapped[float] = mapped_column(Float)
    optimal_spread: Mapped[float] = mapped_column(Float)
    bid_price: Mapped[float] = mapped_column(Float)
    ask_price: Mapped[float] = mapped_column(Float)
    vpin_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    reward_eligible: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    bid_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ask_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    inventory_before: Mapped[float | None] = mapped_column(Float, nullable=True)
    inventory_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    canceled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    market: Mapped[Market] = relationship(back_populates="maker_quotes")


class ArbOpportunity(Base):
    __tablename__ = "arb_opportunities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_slug: Mapped[str] = mapped_column(String(255), index=True)
    type: Mapped[str] = mapped_column(String(32))  # underpriced / overpriced / buy_merge / split_sell
    sum_of_yes: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_cents: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    was_actioned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    strategy: Mapped[str] = mapped_column(String(32), index=True)  # intra_event_arb / pair_cost_arb


class WhaleActivity(Base):
    __tablename__ = "whale_activity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_slug: Mapped[str] = mapped_column(String(255), index=True)
    direction: Mapped[str] = mapped_column(String(8))  # YES / NO
    amount: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    market_probability_at_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    whale_signal: Mapped[str] = mapped_column(String(32))  # buy_pressure / sell_pressure / smart_money_present


class OFISignal(Base):
    __tablename__ = "ofi_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[int | None] = mapped_column(ForeignKey("markets.id"), nullable=True, index=True)
    token_id: Mapped[str] = mapped_column(String(128))
    obi_shares: Mapped[float] = mapped_column(Float)
    obi_notional: Mapped[float] = mapped_column(Float)
    signal: Mapped[str] = mapped_column(String(32))  # OFI_BULLISH / OFI_BEARISH / neutral
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class EnsembleSignal(Base):
    __tablename__ = "ensemble_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[int | None] = mapped_column(ForeignKey("markets.id"), nullable=True, index=True)
    p_metaculus: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_manifold: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_ensemble: Mapped[float] = mapped_column(Float)
    p_polymarket: Mapped[float] = mapped_column(Float)
    divergence: Mapped[float] = mapped_column(Float)
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class SmartWallet(Base):
    __tablename__ = "smart_wallets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    proxy_wallet: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    pnl: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    rank: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
