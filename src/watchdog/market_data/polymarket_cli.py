from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from watchdog.core.config import Settings
from watchdog.core.exceptions import GeoblockError, PolymarketCliError

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CliResponse:
    payload: Any
    latency_ms: int


class PolymarketCli:
    def __init__(self, settings: Settings) -> None:
        self._bin = settings.polymarket_cli_path
        self._expected_version = settings.polymarket_expected_version
        self._country_code = settings.polymarket_country_code.upper()

    def startup_check(self) -> None:
        version = self.check_version()
        LOGGER.info("Polymarket CLI version detected: %s", version)
        self.check_geoblock()

    def _run(self, args: list[str], expect_json: bool = True, timeout_sec: int = 12) -> CliResponse:
        cmd = [self._bin, *args]
        if expect_json and "-o" not in args:
            cmd.extend(["-o", "json"])

        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
        except FileNotFoundError as exc:
            raise PolymarketCliError(
                f"Polymarket CLI not found at '{self._bin}'. Install from https://github.com/Polymarket/polymarket-cli"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PolymarketCliError(f"CLI command timed out: {' '.join(cmd)}") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)

        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise PolymarketCliError(f"CLI command failed ({result.returncode}): {message}")

        stdout = result.stdout.strip()
        if not expect_json:
            return CliResponse(payload=stdout, latency_ms=latency_ms)

        if not stdout:
            return CliResponse(payload={}, latency_ms=latency_ms)

        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise PolymarketCliError(f"Expected JSON output, got: {stdout[:250]}") from exc

        return CliResponse(payload=parsed, latency_ms=latency_ms)

    def check_version(self) -> str:
        out = self._run(["--version"], expect_json=False)
        version = out.payload.strip().split()[-1]
        if self._expected_version not in version:
            raise PolymarketCliError(
                f"Unexpected CLI version '{version}'. Expected to include '{self._expected_version}'."
            )
        return version

    def check_geoblock(self) -> None:
        response = self._run(["geoblock"], expect_json=True)
        payload = response.payload

        blocked_countries: list[str] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("blocked_countries"), list):
                blocked_countries = [str(code).upper() for code in payload["blocked_countries"]]
            elif isinstance(payload.get("countries"), list):
                blocked_countries = [str(code).upper() for code in payload["countries"]]

        if self._country_code in blocked_countries:
            raise GeoblockError(
                f"Trading halted: country {self._country_code} is geoblocked according to CLI geoblock check."
            )

    def list_markets(self, limit: int = 50) -> CliResponse:
        return self._run(["markets", "list", "--limit", str(limit)])

    def search_markets(self, query: str, limit: int = 20) -> CliResponse:
        return self._run(["markets", "search", query, "--limit", str(limit)])

    def orderbook(self, market_slug: str) -> CliResponse:
        return self._run(["orderbook", "get", market_slug])

    def price_history(self, market_slug: str, interval: str = "1m") -> CliResponse:
        return self._run(["price", "history", market_slug, "--interval", interval])

    def balance(self) -> CliResponse:
        return self._run(["balance"])

    def portfolio(self) -> CliResponse:
        return self._run(["portfolio", "value"])

    def place_limit_order(self, market_slug: str, side: str, price: float, size: float) -> CliResponse:
        return self._run(
            [
                "order",
                "limit",
                market_slug,
                "--side",
                side,
                "--price",
                f"{price:.4f}",
                "--size",
                f"{size:.4f}",
            ]
        )

    def create_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        post_only: bool = True,
    ) -> CliResponse:
        args = [
            "order",
            "limit",
            token_id,
            "--side",
            side,
            "--price",
            f"{price:.4f}",
            "--size",
            f"{size:.4f}",
        ]
        if post_only:
            args.append("--post-only")
        return self._run(args)

    def place_market_order(self, market_slug: str, side: str, size: float) -> CliResponse:
        return self._run(
            [
                "order",
                "market",
                market_slug,
                "--side",
                side,
                "--size",
                f"{size:.4f}",
            ]
        )

    def post_batch_orders(self, orders_path: str) -> CliResponse:
        return self._run(["order", "batch", "--file", orders_path])

    def cancel_all(self) -> CliResponse:
        return self._run(["order", "cancel-all"])

    def cancel_all_orders(self) -> CliResponse:
        return self.cancel_all()
