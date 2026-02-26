"""Market data adapters."""

from watchdog.market_data.manifold_client import ManifoldAPIError, ManifoldClient, ManifoldMarket
from watchdog.market_data.polymarket_cli import CliResponse, PolymarketCli

__all__ = ["CliResponse", "ManifoldAPIError", "ManifoldClient", "ManifoldMarket", "PolymarketCli"]
