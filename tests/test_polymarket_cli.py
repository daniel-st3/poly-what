from __future__ import annotations

import pytest

from watchdog.core.config import Settings
from watchdog.core.exceptions import PolymarketCliError
from watchdog.market_data.polymarket_cli import CliResponse, PolymarketCli


def _settings() -> Settings:
    return Settings(
        polymarket_cli_path="/tmp/polymarket",
        polymarket_expected_version="0.1.4",
        polymarket_country_code="US",
    )


def test_check_geoblock_prefers_clob_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = PolymarketCli(_settings())
    calls: list[list[str]] = []

    def fake_run(args: list[str], expect_json: bool = True, timeout_sec: int = 12) -> CliResponse:
        calls.append(args)
        return CliResponse(payload={"blocked_countries": ["CA"]}, latency_ms=1)

    monkeypatch.setattr(cli, "_run", fake_run)

    cli.check_geoblock()

    assert calls == [["clob", "geoblock"]]


def test_check_geoblock_falls_back_to_legacy_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = PolymarketCli(_settings())
    calls: list[list[str]] = []

    def fake_run(args: list[str], expect_json: bool = True, timeout_sec: int = 12) -> CliResponse:
        calls.append(args)
        if args == ["clob", "geoblock"]:
            raise PolymarketCliError("unsupported")
        return CliResponse(payload={"blocked_countries": ["CA"]}, latency_ms=1)

    monkeypatch.setattr(cli, "_run", fake_run)

    cli.check_geoblock()

    assert calls == [["clob", "geoblock"], ["geoblock"]]
