from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

import duckdb
import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from watchdog.db.models import CalibrationSurface

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CalibrationResult:
    model_probability: float
    adjustment: float
    price_bucket: int
    time_bucket_hours: int
    domain: str


class CalibrationSurfaceService:
    TIME_BUCKETS: ClassVar[tuple[int, ...]] = (1, 6, 24, 72, 168, 336, 720, 2160)

    def __init__(self) -> None:
        self._surface: dict[tuple[int, int, str], float] = {}

    @staticmethod
    def _bucket_time(hours_to_resolution: float) -> int:
        for bucket in CalibrationSurfaceService.TIME_BUCKETS:
            if hours_to_resolution <= bucket:
                return bucket
        return CalibrationSurfaceService.TIME_BUCKETS[-1]

    @staticmethod
    def _bucket_price(probability: float) -> int:
        return int(np.clip(round(probability * 100), 1, 99))

    def load_from_db(self, session: Session, dataset_source: str | None = None) -> int:
        stmt = select(CalibrationSurface)
        if dataset_source:
            stmt = stmt.where(CalibrationSurface.dataset_source == dataset_source)

        rows = session.execute(stmt).scalars().all()
        self._surface.clear()
        for row in rows:
            key = (row.price_bucket, row.time_bucket_hours, row.domain)
            self._surface[key] = row.model_adjustment
        return len(rows)

    def _lookup_adjustment(
        self,
        price_bucket: int,
        time_bucket: int,
        mapped_domain: str,
    ) -> float | None:
        """Look up calibration adjustment for exact (price, time, domain) combo.

        Tries mapped domain first, then 'other', then 'unknown', then any domain
        at the same (price, time) bucket.  Returns *None* when nothing matches.
        """
        for domain_try in (mapped_domain, "other", "unknown"):
            val = self._surface.get((price_bucket, time_bucket, domain_try))
            if val is not None:
                return val

        # Any domain at this (price, time)
        for k, v in self._surface.items():
            if k[0] == price_bucket and k[1] == time_bucket:
                return v

        return None

    def calibrate(
        self,
        market_probability: float,
        hours_to_resolution: float,
        domain: str,
        sentiment_score: float = 0.0,
    ) -> CalibrationResult:
        price_bucket = self._bucket_price(market_probability)
        time_bucket = self._bucket_time(hours_to_resolution)
        domain_norm = domain.lower().strip() or "unknown"

        # Domain alias mapping for Manifold/Polymarket categories
        domain_aliases: dict[str, str] = {
            "us-current-affairs": "politics",
            "coronavirus": "macro",
            "geo": "politics",
            "pop-culture": "other",
            "entertainment": "other",
            "business": "macro",
            "nfts": "crypto",
            "olympics": "sports",
            "science": "tech",
        }
        mapped_domain = domain_aliases.get(domain_norm, domain_norm)

        # --- Primary lookup at exact time bucket ---
        adjustment = self._lookup_adjustment(price_bucket, time_bucket, mapped_domain)

        # --- Time-bucket cascade: try progressively larger buckets ---
        # The Becker dataset is heavily concentrated in the 2160h bucket.
        # If the exact time bucket has no data for this price, cascade up
        # to larger buckets so we still get a calibration signal.
        if adjustment is None:
            tb_index = self.TIME_BUCKETS.index(time_bucket) if time_bucket in self.TIME_BUCKETS else -1
            for i in range(tb_index + 1, len(self.TIME_BUCKETS)):
                adjustment = self._lookup_adjustment(price_bucket, self.TIME_BUCKETS[i], mapped_domain)
                if adjustment is not None:
                    LOGGER.debug(
                        "Calibration cascade: no data at %dh, using %dh bucket (price=%d, domain=%s)",
                        time_bucket, self.TIME_BUCKETS[i], price_bucket, mapped_domain,
                    )
                    break

        if adjustment is None:
            adjustment = 0.0

        # Sentiment contributes a bounded secondary adjustment.
        sentiment_adj = float(np.clip(sentiment_score * 0.03, -0.05, 0.05))
        total_adjustment = adjustment + sentiment_adj
        model_probability = float(np.clip(market_probability + total_adjustment, 0.01, 0.99))

        return CalibrationResult(
            model_probability=model_probability,
            adjustment=total_adjustment,
            price_bucket=price_bucket,
            time_bucket_hours=time_bucket,
            domain=domain_norm,
        )

    def build_from_becker_parquet(
        self,
        session: Session,
        dataset_path: str,
        dataset_source: str,
        price_column: str = "price",
        outcome_column: str = "outcome",
        hours_to_resolution_column: str = "hours_to_resolution",
        domain_column: str = "domain",
    ) -> int:
        time_case = (
            f"CASE "
            f"WHEN {hours_to_resolution_column} * 24.0 <= 1 THEN 1 "
            f"WHEN {hours_to_resolution_column} * 24.0 <= 6 THEN 6 "
            f"WHEN {hours_to_resolution_column} * 24.0 <= 24 THEN 24 "
            f"WHEN {hours_to_resolution_column} * 24.0 <= 72 THEN 72 "
            f"WHEN {hours_to_resolution_column} * 24.0 <= 168 THEN 168 "
            f"WHEN {hours_to_resolution_column} * 24.0 <= 336 THEN 336 "
            f"WHEN {hours_to_resolution_column} * 24.0 <= 720 THEN 720 "
            f"ELSE 2160 END"
        )

        query = f"""
            WITH raw AS (
                SELECT
                    CAST(ROUND({price_column} * 100) AS INTEGER) AS price_bucket,
                    CAST({time_case} AS INTEGER) AS time_bucket_hours,
                    LOWER(COALESCE({domain_column}, 'unknown')) AS domain,
                    CAST({outcome_column} AS DOUBLE) AS outcome
                FROM read_parquet(?)
                WHERE {price_column} BETWEEN 0.01 AND 0.99
            )
            SELECT
                price_bucket,
                time_bucket_hours,
                domain,
                COUNT(*) AS sample_size,
                AVG(outcome) AS empirical_outcome_rate,
                AVG(outcome) - AVG(price_bucket) / 100.0 AS model_adjustment
            FROM raw
            GROUP BY 1,2,3
            HAVING COUNT(*) >= 1
        """

        con = duckdb.connect(database=":memory:")
        try:
            rows = con.execute(query, [dataset_path]).fetchall()
        finally:
            con.close()

        if not rows:
            raise ValueError("Calibration build returned 0 rows. Verify column names and dataset path.")

        session.execute(delete(CalibrationSurface).where(CalibrationSurface.dataset_source == dataset_source))

        inserts: list[CalibrationSurface] = []
        for row in rows:
            inserts.append(
                CalibrationSurface(
                    price_bucket=int(row[0]),
                    time_bucket_hours=int(row[1]),
                    domain=str(row[2]),
                    dataset_source=dataset_source,
                    sample_size=int(row[3]),
                    empirical_outcome_rate=float(row[4]),
                    model_adjustment=float(row[5]),
                )
            )
        session.add_all(inserts)
        session.commit()

        LOGGER.info("Built calibration surface with %d cells from %s", len(inserts), dataset_source)
        return len(inserts)
