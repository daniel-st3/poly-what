"""Market data adapters."""

from watchdog.market_data.manifold_client import ManifoldClient, ManifoldMarket
from watchdog.market_data.polymarket_cli import CliResponse, PolymarketCli

__all__ = ["CliResponse", "ManifoldClient", "ManifoldMarket", "PolymarketCli"]
