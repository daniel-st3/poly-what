from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from watchdog.backtest.historical_loader import BeckerHistoricalLoader
from watchdog.core.config import Settings
from watchdog.db.init import init_db
from watchdog.db.models import CalibrationSurface
from watchdog.db.session import build_engine, build_session_factory
from watchdog.signals.calibration import CalibrationSurfaceService


def _write_parquet(path: Path) -> None:
    rows = []
    resolution = datetime(2024, 7, 1, tzinfo=UTC)
    for i in range(120):
        price = 0.02 if i < 60 else 0.40
        outcome = 0 if i < 60 else 1
        rows.append(
            {
                "market_id": f"m{i // 20}",
                "question": "Synthetic politics market",
                "domain": "politics",
                "resolution_time": resolution,
                "outcome": outcome,
                "price": price,
                "timestamp": resolution - timedelta(days=30 - (i % 20)),
                "volume": 100,
                "taker_direction": "buy",
                "platform": "polymarket",
            }
        )
    pd.DataFrame(rows).to_parquet(path / "calibration.parquet", index=False)


def test_build_calibration_surface_from_data(tmp_path: Path) -> None:
    _write_parquet(tmp_path)
    loader = BeckerHistoricalLoader(tmp_path)
    surface = loader.build_calibration_surface_from_data("polymarket", min_trades_per_bucket=1)
    assert not surface.empty
    assert {"p_bucket", "t_bucket", "domain", "empirical_win_rate", "n_trades"}.issubset(surface.columns)


def test_calibrate_probability_domain_adjustment() -> None:
    settings = Settings(database_url="sqlite:///:memory:")
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        session.add(
            CalibrationSurface(
                price_bucket=40,
                time_bucket_hours=720,
                domain="politics",
                dataset_source="test",
                sample_size=100,
                empirical_outcome_rate=0.60,
                model_adjustment=0.20,
            )
        )
        session.commit()

        service = CalibrationSurfaceService()
        service.load_from_db(session)
        result = service.calibrate(market_probability=0.40, hours_to_resolution=500, domain="politics", sentiment_score=0.0)

    assert result.model_probability != 0.40


def test_longshot_bias_correction_below_market_prob() -> None:
    settings = Settings(database_url="sqlite:///:memory:")
    engine = build_engine(settings)
    init_db(engine)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        session.add(
            CalibrationSurface(
                price_bucket=2,
                time_bucket_hours=720,
                domain="politics",
                dataset_source="test",
                sample_size=120,
                empirical_outcome_rate=0.01,
                model_adjustment=-0.01,
            )
        )
        session.commit()

        service = CalibrationSurfaceService()
        service.load_from_db(session)
        result = service.calibrate(market_probability=0.02, hours_to_resolution=500, domain="politics", sentiment_score=0.0)

    assert result.model_probability < 0.02
