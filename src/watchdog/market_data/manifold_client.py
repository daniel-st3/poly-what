from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx
from dateutil import parser as dtparser
from sqlalchemy import select

from watchdog.db.models import Market, Trade


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
        base_url: str,
        api_key: str | None = None,
        timeout_sec: float = 20.0,
        session_factory=None,
        enable_live_trading: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.session_factory = session_factory
        self.enable_live_trading = enable_live_trading

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Key {self.api_key}"
        return headers

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.get(url, params=params, headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            response = await client.post(url, json=payload, headers=self._headers())
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if value > 10_000_000_000:
                return datetime.fromtimestamp(value / 1000, tz=UTC)
            return datetime.fromtimestamp(value, tz=UTC)
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
        val = str(value).strip().lower()
        if val in {"yes", "true", "1"}:
            return 1
        if val in {"no", "false", "0"}:
            return 0
        return None

    @staticmethod
    def _market_probability(raw: dict[str, Any]) -> float:
        for key in ("probability", "p", "lastProb"):
            value = raw.get(key)
            if value is None:
                continue
            try:
                prob = float(value)
            except (TypeError, ValueError):
                continue
            return max(0.0, min(1.0, prob))
        return 0.5

    @classmethod
    def _to_market(cls, raw: dict[str, Any]) -> ManifoldMarket:
        market_id = str(raw.get("id") or raw.get("marketId") or "")
        slug = str(raw.get("slug") or market_id)
        question = str(raw.get("question") or raw.get("text") or slug)
        probability = cls._market_probability(raw)
        close_time = cls._parse_ts(raw.get("closeTime") or raw.get("close_time") or raw.get("resolutionTime"))
        is_resolved = bool(raw.get("isResolved") or raw.get("resolved") or raw.get("resolution") is not None)
        outcome = cls._normalize_outcome(raw.get("resolution") or raw.get("outcome"))
        volume = float(raw.get("volume") or raw.get("totalLiquidity") or 0.0)

        return ManifoldMarket(
            market_id=market_id,
            slug=slug,
            question=question,
            probability=probability,
            close_time=close_time,
            is_resolved=is_resolved,
            outcome=outcome,
            volume=volume,
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
            ("crypto", ["bitcoin", "crypto", "eth", "solana", "token"]),
            ("macro", ["fed", "inflation", "rates", "cpi", "gdp"]),
            ("sports", ["nba", "nfl", "mlb", "soccer", "game", "tournament"]),
            ("tech", ["ai", "openai", "apple", "google", "microsoft", "nvidia"]),
            ("geo", ["war", "ukraine", "china", "taiwan", "israel", "gaza"]),
            ("entertainment", ["movie", "oscar", "music", "netflix", "series"]),
            ("science", ["space", "nasa", "vaccine", "trial", "climate", "research"]),
        ]
        for domain, keywords in mapping:
            if any(word in text for word in keywords):
                return domain
        return "other"

    async def fetch_markets(self, limit: int = 200, filter_type: str = "active") -> list[dict[str, Any]]:
        rows = await self._get("/markets", params={"limit": limit})
        if not isinstance(rows, list):
            return []

        normalized: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            market = self._to_market(raw)
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

    async def fetch_market(self, market_id: str) -> dict[str, Any]:
        raw = await self._get(f"/market/{market_id}")
        if not isinstance(raw, dict):
            raise ValueError(f"Unexpected response for market_id={market_id}")
        market = self._to_market(raw)
        return {
            "market_id": market.market_id,
            "slug": market.slug,
            "question": market.question,
            "domain": self._infer_domain(raw),
            "yes_token_id": market.market_id,
            "resolution_time": market.close_time,
            "status": "resolved" if market.is_resolved else "active",
            "probability": market.probability,
            "outcome": market.outcome,
            "raw": raw,
        }

    async def get_current_prob(self, market_id: str) -> float:
        market = await self.fetch_market(market_id)
        return float(market["probability"])

    async def get_price_history(self, market_id: str) -> list[dict[str, Any]]:
        # Manifold's endpoint naming can vary by API version.
        # Try contract metrics endpoint first, fallback to comments/probability stream if unavailable.
        try:
            raw = await self._get("/market-probability", params={"id": market_id})
        except Exception:
            raw = await self._get(f"/market/{market_id}")

        if isinstance(raw, list):
            history = raw
        else:
            history = raw.get("probabilityHistory") or raw.get("history") or []

        out: list[dict[str, Any]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            ts = self._parse_ts(item.get("time") or item.get("timestamp") or item.get("createdTime"))
            prob = item.get("probability") or item.get("p")
            if ts is None or prob is None:
                continue
            out.append({"timestamp": ts.isoformat(), "probability": float(prob)})
        return out

    async def place_paper_bet(
        self,
        market_id: str,
        outcome: str,
        amount_mana: float,
        is_paper: bool = True,
    ) -> dict[str, Any]:
        market = await self.fetch_market(market_id)
        prob = float(market["probability"])
        entry_price = prob if outcome.upper() == "YES" else (1 - prob)

        live_response: dict[str, Any] | None = None
        if self.enable_live_trading and self.api_key and not is_paper:
            payload = {
                "contractId": market_id,
                "outcome": outcome.upper(),
                "amount": amount_mana,
            }
            try:
                data = await self._post("/bet", payload)
                live_response = data if isinstance(data, dict) else {"raw": data}
            except Exception as exc:
                live_response = {"error": str(exc)}

        trade_record = {
            "market_id": market_id,
            "slug": market["slug"],
            "outcome": outcome.upper(),
            "amount_mana": amount_mana,
            "entry_price": entry_price,
            "is_paper": True,
            "live_response": live_response,
        }

        if self.session_factory is not None:
            with self.session_factory() as session:
                market_row = session.execute(select(Market).where(Market.slug == market["slug"])).scalar_one_or_none()
                if market_row is None:
                    market_row = Market(
                        slug=market["slug"],
                        question=market["question"],
                        domain=market["domain"],
                        resolution_time=market["resolution_time"],
                        status=market["status"],
                    )
                    session.add(market_row)
                    session.flush()

                trade = Trade(
                    market_id=market_row.id,
                    side=outcome.upper(),
                    size=amount_mana,
                    entry_price=entry_price,
                    kelly_fraction=0.0,
                    order_id=f"manifold-paper-{market_id}",
                    is_paper=True,
                    opened_at=datetime.now(UTC),
                    status="open",
                )
                session.add(trade)
                session.commit()
                trade_record["trade_id"] = trade.id

        return trade_record

    async def get_resolved_markets(self, limit: int = 200) -> list[dict[str, Any]]:
        return await self.fetch_markets(limit=limit, filter_type="resolved")

    # Compatibility wrappers used by scripts/service code.
    async def get_markets(self, limit: int = 200, before: str | None = None) -> list[ManifoldMarket]:
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before
        raw = await self._get("/markets", params=params)
        if not isinstance(raw, list):
            return []
        return [self._to_market(item) for item in raw if isinstance(item, dict)]

    async def get_active_markets(self, limit: int = 200) -> list[ManifoldMarket]:
        markets = await self.get_markets(limit=limit)
        return [m for m in markets if not m.is_resolved]

    async def get_market(self, market_id: str) -> ManifoldMarket:
        data = await self._get(f"/market/{market_id}")
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Manifold market response for {market_id}")
        return self._to_market(data)

    async def get_market_by_slug(self, slug: str) -> ManifoldMarket:
        data = await self._get(f"/slug/{slug}")
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Manifold slug response for {slug}")
        return self._to_market(data)
