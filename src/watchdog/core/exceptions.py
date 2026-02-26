class WatchdogError(Exception):
    """Base application exception."""


class PolymarketCliError(WatchdogError):
    """Raised when the Polymarket CLI fails or is missing."""


class GeoblockError(WatchdogError):
    """Raised when startup geoblock checks fail."""


class LiveTradingDisabledError(WatchdogError):
    """Raised when live trading path is invoked while disabled."""
