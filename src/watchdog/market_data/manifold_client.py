from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from dateutil import parser as dtparser


class ManifoldAPIError(RuntimeError):
    """Raised when the Manifold API returns an error response."""


@dataclass(slots=True)
class ManifoldMarket:
    market_id: str
    slug: str
    question: str
    probability: float
    close_time: datetime | None
    is_resolved: bool
    outcome: int | None
    volume: float
    raw: dict[str, Any]


class ManifoldClient:
    def __init__(
        self,
        base_url: str = "https://api.manifold.markets/v0",
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _headers(self, include_auth: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if include_auth:
            if not self.api_key:
                raise ManifoldAPIError("MANIFOLD_API_KEY is required for place_bet")
            headers["Authorization"] = f"Key {self.api_key}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        include_auth: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=self._headers(include_auth=include_auth),
                )
        except httpx.HTTPError as exc:
            raise ManifoldAPIError(f"Manifold request failed for {method} {path}: {exc}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            body = response.text.strip()
            if len(body) > 250:
                body = f"{body[:250]}..."
            raise ManifoldAPIError(
                f"Manifold API error {response.status_code} for {method} {path}: {body}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise ManifoldAPIError(f"Invalid JSON response for {method} {path}") from exc

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 10_000_000_000:
                ts /= 1000
            return datetime.fromtimestamp(ts, tz=UTC)

        try:
            parsed = dtparser.parse(str(value))
        except Exception:
            return None

        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed

    @staticmethod
    def _normalize_outcome(value: Any) -> int | None:
        if value is None:
            return None

        normalized = str(value).strip().lower()
        if normalized in {"yes", "true", "1"}:
            return 1
        if normalized in {"no", "false", "0"}:
            return 0
        return None

    @staticmethod
    def _to_market_model(raw: dict[str, Any]) -> ManifoldMarket:
        probability = ManifoldClient._to_float(raw.get("probability"), default=0.5)
        probability = max(0.0, min(1.0, probability))
        resolution = raw.get("resolution") or raw.get("outcome")

        return ManifoldMarket(
            market_id=str(raw.get("id") or raw.get("marketId") or ""),
            slug=str(raw.get("slug") or raw.get("id") or ""),
            question=str(raw.get("question") or raw.get("text") or ""),
            probability=probability,
            close_time=ManifoldClient._parse_time(raw.get("closeTime") or raw.get("close_time")),
            is_resolved=bool(raw.get("isResolved") or raw.get("resolved") or resolution is not None),
            outcome=ManifoldClient._normalize_outcome(resolution),
            volume=ManifoldClient._to_float(raw.get("volume"), default=0.0),
            raw=raw,
        )

    @staticmethod
    def _infer_domain(market: dict[str, Any]) -> str:
        text = " ".join(
            [
                str(market.get("question") or market.get("text") or ""),
                str(market.get("groupSlug") or ""),
                str(market.get("slug") or ""),
            ]
        ).lower()

        mapping = [
            ("politics", ["election", "president", "congress", "senate", "politic"]),
            ("crypto", ["bitcoin", "crypto", "eth", "sol", "token"]),
            ("macro", ["fed", "inflation", "rates", "gdp", "cpi"]),
            ("sports", ["nba", "nfl", "soccer", "world cup", "game"]),
            ("tech", ["ai", "openai", "google", "apple", "microsoft"]),
            ("geo", ["war", "ukraine", "china", "taiwan", "israel", "gaza"]),
            ("entertainment", ["movie", "music", "oscar", "netflix"]),
            ("science", ["space", "nasa", "trial", "climate", "research"]),
        ]
        for domain, keywords in mapping:
            if any(keyword in text for keyword in keywords):
                return domain
        return "other"

    def get_markets(self, limit: int = 100) -> list[dict[str, Any]]:
        data = self._request("GET", "/markets", params={"limit": int(limit)})
        if not isinstance(data, list):
            raise ManifoldAPIError("Unexpected response for GET /markets; expected list")
        return [item for item in data if isinstance(item, dict)]

    def get_market(self, market_id: str) -> dict[str, Any]:
        data = self._request("GET", f"/market/{market_id}")
        if not isinstance(data, dict):
            raise ManifoldAPIError(f"Unexpected response for GET /market/{market_id}; expected object")
        return data

    def get_orderbook(self, market_id: str) -> dict[str, float]:
        market = self.get_market(market_id)
        mid = max(0.01, min(0.99, self._to_float(market.get("probability"), default=0.5)))
        bid = max(0.01, min(0.99, mid - 0.02))
        ask = max(0.01, min(0.99, mid + 0.02))
        if ask <= bid:
            ask = min(0.99, bid + 0.01)

        # Synthetic LMSR depth proxy used for paper-mode VPIN parity.
        bid_volume = 500.0
        ask_volume = 500.0

        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": ask - bid,
            "probability": mid,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
        }

    def place_bet(self, market_id: str, outcome: str, amount: float) -> dict[str, Any]:
        side = outcome.strip().upper()
        if side not in {"YES", "NO"}:
            raise ValueError("outcome must be YES or NO")
        if amount <= 0:
            raise ValueError("amount must be positive")

        payload = {
            "marketId": market_id,
            "outcome": side,
            "amount": float(amount),
        }
        data = self._request("POST", "/bet", json_body=payload, include_auth=True)
        if not isinstance(data, dict):
            raise ManifoldAPIError("Unexpected response for POST /bet; expected object")
        return data

    def get_positions(self, user_id: str) -> list[dict[str, Any]]:
        data = self._request("GET", "/bets", params={"userId": user_id, "limit": 500})
        if not isinstance(data, list):
            raise ManifoldAPIError("Unexpected response for GET /bets; expected list")
        return [item for item in data if isinstance(item, dict)]

    # Compatibility helpers used by existing scripts.
    def fetch_markets(self, limit: int = 100, filter_type: str = "active") -> list[dict[str, Any]]:
        rows = self.get_markets(limit=limit)
        normalized: list[dict[str, Any]] = []
        for raw in rows:
            market = self._to_market_model(raw)
            if filter_type == "active" and market.is_resolved:
                continue
            if filter_type == "resolved" and not market.is_resolved:
                continue
            normalized.append(
                {
                    "market_id": market.market_id,
                    "slug": market.slug,
                    "question": market.question,
                    "domain": self._infer_domain(raw),
                    "yes_token_id": market.market_id,
                    "resolution_time": market.close_time,
                    "status": "resolved" if market.is_resolved else "active",
                    "probability": market.probability,
                    "raw": raw,
                }
            )
        return normalized

    def get_current_prob(self, market_id: str) -> float:
        row = self.get_market(market_id)
        return max(0.0, min(1.0, self._to_float(row.get("probability"), default=0.5)))

    def get_price_history(self, market_id: str) -> list[dict[str, Any]]:
        row = self.get_market(market_id)
        history = row.get("probabilityHistory") or row.get("history") or []
        if not isinstance(history, list):
            return []

        points: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            ts = self._parse_time(item.get("time") or item.get("timestamp") or item.get("createdTime"))
            if ts is None:
                continue
            points.append(
                {
                    "timestamp": ts.isoformat(),
                    "probability": self._to_float(item.get("probability") or item.get("p"), default=0.5),
                }
            )
        return points

    def get_resolved_markets(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.fetch_markets(limit=limit, filter_type="resolved")

    async def get_active_markets(self, limit: int = 100) -> list[ManifoldMarket]:
        rows = await asyncio.to_thread(self.get_markets, limit)
        markets = [self._to_market_model(row) for row in rows]
        return [market for market in markets if not market.is_resolved]

    async def get_market_by_slug(self, slug: str) -> ManifoldMarket:
        data = await asyncio.to_thread(self._request, "GET", f"/slug/{slug}")
        if not isinstance(data, dict):
            raise ManifoldAPIError(f"Unexpected response for GET /slug/{slug}")
        return self._to_market_model(data)
