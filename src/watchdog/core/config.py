from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    watchdog_env: Literal["dev", "prod", "test"] = "dev"
    database_url: str = "sqlite:///watchdog.db"
    log_level: str = "INFO"
    becker_dataset_path: str = "./data/becker"

    polymarket_cli_path: str = "polymarket"
    polymarket_expected_version: str = "0.1.4"
    polymarket_country_code: str = "CO"
    snapshot_interval_seconds: int = Field(default=30, ge=5, le=3600)
    snapshot_retry_max: int = Field(default=5, ge=1, le=20)
    snapshot_retry_base_seconds: int = Field(default=5, ge=1, le=120)

    manifold_api_base_url: str = "https://api.manifold.markets/v0"
    manifold_api_key: str | None = None
    manifold_user_id: str | None = None
    paper_loop_seconds: int = Field(default=60, ge=5, le=3600)
    paper_summary_every: int = Field(default=10, ge=1, le=1000)

    enable_live_trading: bool = False
    max_position_per_market: float = Field(default=0.20, ge=0.01, le=1.0)
    kelly_fraction: float = Field(default=0.25, ge=0.01, le=1.0)
    max_drawdown_p95: float = Field(default=0.30, ge=0.01, le=0.95)
    vpin_kill_threshold: float = Field(default=0.70, ge=0.5, le=1.0)
    min_divergence: float = Field(default=0.15, ge=0.01, le=0.50)
    min_divergence_backtest: float = Field(default=0.15, ge=0.01, le=0.50)
    min_divergence_paper: float = Field(default=0.17, ge=0.01, le=0.60)
    min_divergence_live: float = Field(default=0.20, ge=0.01, le=0.60)
    near_resolution_hours: int = Field(default=2, ge=1, le=72)
    near_resolution_fraction: float = Field(default=0.05, ge=0.01, le=0.5)
    backtest_fee_rate: float = Field(default=0.005, ge=0.0, le=0.05)
    backtest_slippage_rate: float = Field(default=0.005, ge=0.0, le=0.05)
    backtest_spread_proxy: float = Field(default=0.01, ge=0.0, le=0.20)
    live_validation_max_positions: int = Field(default=5, ge=1, le=50)
    live_validation_position_size_usdc: float = Field(default=10.0, ge=1.0, le=100.0)
    min_market_volume_usdc: float = Field(default=1000.0, ge=0.0)
    min_market_liquidity_usdc: float = Field(default=500.0, ge=0.0)
    executor_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_positions_simultaneous: int = Field(default=5, ge=1, le=100)
    max_position_fraction: float = Field(default=0.20, ge=0.01, le=1.0)

    router_provider: Literal["mock", "openai"] = "mock"
    router_model: str = "gpt-4o-mini"
    executor_provider: Literal["mock", "anthropic"] = "mock"
    executor_model: str = "claude-sonnet-4"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    gdelt_enabled: bool = True
    gdelt_poll_seconds: int = Field(default=30, ge=5, le=600)
    rss_enabled: bool = True
    reddit_enabled: bool = False
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_user_agent: str = "watchdog/0.1"

    # New high-signal sources
    marketaux_enabled: bool = True
    marketaux_api_key: str | None = None
    polymarket_volume_spike_enabled: bool = True
    gdelt_gkg_enabled: bool = True

    brave_api_key: str | None = None
    deepseek_api_key: str | None = None
    gemini_api_key: str | None = None
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    polymarket_private_key: str | None = None
    live_bankroll_usdc: float = Field(default=50.0, ge=1.0, le=1000000.0)
    paper_bankroll_usdc: float = Field(default=500.0, ge=1.0, le=1000000.0)
    experiment_id: str = "feb2026_v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
