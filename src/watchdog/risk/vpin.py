from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass(slots=True)
class TradeFlow:
    side: str
    volume: float


class VPINCalculator:
    def __init__(self, bucket_volume: float = 5000.0, rolling_buckets: int = 50) -> None:
        self.bucket_volume = bucket_volume
        self.rolling_buckets = rolling_buckets

    def compute(self, flows: list[TradeFlow]) -> float:
        if not flows:
            return 0.0

        imbalances: list[float] = []
        bucket_buy = 0.0
        bucket_sell = 0.0
        bucket_fill = 0.0

        for flow in flows:
            remaining = max(flow.volume, 0.0)
            while remaining > 0:
                room = self.bucket_volume - bucket_fill
                take = min(remaining, room)
                if flow.side.upper() == "BUY":
                    bucket_buy += take
                else:
                    bucket_sell += take

                bucket_fill += take
                remaining -= take

                if bucket_fill >= self.bucket_volume:
                    imbalance = abs(bucket_buy - bucket_sell) / max(self.bucket_volume, 1e-9)
                    imbalances.append(imbalance)
                    bucket_buy = 0.0
                    bucket_sell = 0.0
                    bucket_fill = 0.0

        if not imbalances:
            return 0.0

        recent = imbalances[-self.rolling_buckets :]
        return float(np.mean(recent))


def should_halt_maker(
    vpin: float,
    vpin_kill_threshold: float,
    resolution_time: datetime | None,
    near_resolution_hours: int,
    near_resolution_fraction: float,
    market_opened_at: datetime | None = None,
) -> bool:
    if vpin >= vpin_kill_threshold:
        return True

    if resolution_time is None:
        return False

    now = datetime.now(UTC)
    remaining_hours = (resolution_time - now).total_seconds() / 3600
    if remaining_hours <= near_resolution_hours:
        return True

    if market_opened_at is None:
        return False

    total_life = (resolution_time - market_opened_at).total_seconds()
    if total_life <= 0:
        return True

    remaining_fraction = max((resolution_time - now).total_seconds(), 0.0) / total_life
    return remaining_fraction <= near_resolution_fraction
