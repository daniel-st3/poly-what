from __future__ import annotations

import pytest

from watchdog.market_data.manifold_client import ManifoldAPIError, ManifoldClient


class _FakeResponse:
    def __init__(self, status_code: int, payload, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        return self._payload


def test_get_markets_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    def _request(self, method, url, params=None, json=None, headers=None):
        assert method == "GET"
        assert url.endswith("/markets")
        assert params == {"limit": 2}
        return _FakeResponse(200, [{"id": "m1", "probability": 0.42}])

    monkeypatch.setattr("httpx.Client.request", _request)

    client = ManifoldClient(base_url="https://api.manifold.markets/v0")
    rows = client.get_markets(limit=2)

    assert len(rows) == 1
    assert rows[0]["id"] == "m1"


def test_place_bet_uses_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    def _request(self, method, url, params=None, json=None, headers=None):
        assert method == "POST"
        assert url.endswith("/bet")
        assert headers is not None
        assert headers.get("Authorization") == "Key test-key"
        assert json == {"marketId": "mkt-1", "outcome": "YES", "amount": 10.0}
        return _FakeResponse(200, {"betId": "b1", "status": "ok"})

    monkeypatch.setattr("httpx.Client.request", _request)

    client = ManifoldClient(base_url="https://api.manifold.markets/v0", api_key="test-key")
    result = client.place_bet("mkt-1", "YES", 10.0)

    assert result["betId"] == "b1"


def test_non_2xx_raises_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _request(self, method, url, params=None, json=None, headers=None):
        return _FakeResponse(500, {"error": "boom"}, text="boom")

    monkeypatch.setattr("httpx.Client.request", _request)

    client = ManifoldClient(base_url="https://api.manifold.markets/v0")

    with pytest.raises(ManifoldAPIError):
        client.get_markets(limit=1)


def test_get_orderbook_returns_fixed_synthetic_depths(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ManifoldClient(base_url="https://api.manifold.markets/v0")
    monkeypatch.setattr(client, "get_market", lambda market_id: {"id": market_id, "probability": 0.55})

    book = client.get_orderbook("mkt-123")

    assert book["mid"] == 0.55
    assert book["bid_volume"] == 500.0
    assert book["ask_volume"] == 500.0
    assert book["ask"] > book["bid"]
