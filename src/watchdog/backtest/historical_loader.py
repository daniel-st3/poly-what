from __future__ import annotations

# Dataset setup (one-time):
#   git clone https://github.com/Jon-Becker/prediction-market-analysis
#   cd prediction-market-analysis && uv sync && make setup
#   Set BECKER_DATASET_PATH in .env to the extracted data/ directory.
#   Extraction: 36GB compressed, 5-30 minutes.
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import duckdb
import pandas as pd


@dataclass(slots=True)
class ColumnBinding:
    logical_name: str
    column_name: str | None


class BeckerHistoricalLoader:
    CANDIDATES: ClassVar[dict[str, tuple[str, ...]]] = {
        "market_id": ("market_id", "condition_id", "market_slug", "slug", "market", "marketid"),
        "question": ("question", "title", "market_question"),
        "domain": ("domain", "category", "topic", "tag"),
        "resolution_time": ("resolution_time", "resolved_at", "end_time", "close_time", "end_date"),
        "outcome": ("outcome", "resolved_outcome", "result", "winner", "resolution", "is_yes"),
        "price": ("price", "trade_price", "fill_price", "probability"),
        "timestamp": ("timestamp", "ts", "created_at", "time", "trade_time"),
        "volume": ("size", "volume", "amount", "qty"),
        "taker_direction": ("taker_direction", "aggressor_side", "side", "direction", "taker_side"),
        "platform": ("platform", "venue", "exchange"),
        "is_maker": ("is_maker", "maker", "maker_flag"),
        "liquidity_role": ("liquidity_role", "role"),
    }

    def __init__(self, dataset_path: str | Path) -> None:
        self.dataset_path = Path(dataset_path)
        self.parquet_glob = str(self.dataset_path / "**/*.parquet")

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Becker dataset path not found: {self.dataset_path}. "
                "Set BECKER_DATASET_PATH to the extracted Parquet data directory."
            )

        self._available_columns = self._read_available_columns()
        self._bindings = self._build_bindings()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(database=":memory:")

    def _read_available_columns(self) -> dict[str, str]:
        con = self._connect()
        try:
            con.execute(
                "SELECT * FROM read_parquet(?, union_by_name=true, hive_partitioning=true) LIMIT 0",
                [self.parquet_glob],
            )
            description = con.description or []
            return {str(col[0]).lower(): str(col[0]) for col in description}
        finally:
            con.close()

    def _build_bindings(self) -> dict[str, ColumnBinding]:
        bindings: dict[str, ColumnBinding] = {}
        for logical, candidates in self.CANDIDATES.items():
            selected = None
            for candidate in candidates:
                if candidate.lower() in self._available_columns:
                    selected = self._available_columns[candidate.lower()]
                    break
            bindings[logical] = ColumnBinding(logical_name=logical, column_name=selected)

        required = ("market_id", "price", "timestamp", "outcome")
        missing = [name for name in required if bindings[name].column_name is None]
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {', '.join(missing)}. "
                f"Available columns: {', '.join(sorted(self._available_columns.values()))}"
            )

        return bindings

    @staticmethod
    def _q(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def _col(self, logical: str, default: str | None = None) -> str:
        bound = self._bindings[logical].column_name
        if bound is None:
            if default is None:
                raise ValueError(f"Missing required column for logical field: {logical}")
            return default
        return self._q(bound)

    def _outcome_expr(self) -> str:
        col = self._col("outcome")
        return (
            "CASE "
            f"WHEN LOWER(CAST({col} AS VARCHAR)) IN ('1','true','yes','y') THEN 1.0 "
            f"WHEN LOWER(CAST({col} AS VARCHAR)) IN ('0','false','no','n') THEN 0.0 "
            f"ELSE TRY_CAST({col} AS DOUBLE) END"
        )

    def _role_expr(self) -> str:
        is_maker = self._bindings["is_maker"].column_name
        role_col = self._bindings["liquidity_role"].column_name

        if is_maker is not None:
            col = self._q(is_maker)
            return (
                "CASE "
                f"WHEN LOWER(CAST({col} AS VARCHAR)) IN ('1','true','maker') THEN 'maker' "
                "ELSE 'taker' END"
            )

        if role_col is not None:
            col = self._q(role_col)
            return (
                "CASE "
                f"WHEN LOWER(CAST({col} AS VARCHAR)) LIKE '%maker%' THEN 'maker' "
                f"WHEN LOWER(CAST({col} AS VARCHAR)) LIKE '%taker%' THEN 'taker' "
                "ELSE 'unknown' END"
            )

        return "'unknown'"

    def _platform_filter_clause(self, platform: str) -> tuple[str, list[Any]]:
        platform_col = self._bindings["platform"].column_name
        if platform_col is None:
            return "", []
        return f"AND LOWER(CAST({self._q(platform_col)} AS VARCHAR)) = ?", [platform.lower()]

    def _base_cte(self, platform: str) -> tuple[str, list[Any]]:
        market_id_col = self._col("market_id")
        question_col = self._col("question", default=f"CAST({market_id_col} AS VARCHAR)")
        domain_col = self._col("domain", default="'unknown'")
        resolution_col = self._col("resolution_time", default=self._col("timestamp"))
        timestamp_col = self._col("timestamp")
        price_col = self._col("price")
        volume_col = self._col("volume", default="1.0")
        taker_col = self._col("taker_direction", default="'unknown'")

        platform_clause, platform_params = self._platform_filter_clause(platform)

        sql = f"""
            WITH raw AS (
                SELECT
                    CAST({market_id_col} AS VARCHAR) AS market_id,
                    CAST(COALESCE({question_col}, CAST({market_id_col} AS VARCHAR)) AS VARCHAR) AS question,
                    LOWER(CAST(COALESCE({domain_col}, 'unknown') AS VARCHAR)) AS domain,
                    CAST({timestamp_col} AS TIMESTAMP) AS ts,
                    CAST(COALESCE({resolution_col}, {timestamp_col}) AS TIMESTAMP) AS resolution_time,
                    TRY_CAST({price_col} AS DOUBLE) AS price,
                    TRY_CAST(COALESCE({volume_col}, 1.0) AS DOUBLE) AS volume,
                    {self._outcome_expr()} AS outcome,
                    LOWER(CAST(COALESCE({taker_col}, 'unknown') AS VARCHAR)) AS taker_direction,
                    {self._role_expr()} AS liquidity_role
                FROM read_parquet(?, union_by_name=true, hive_partitioning=true)
                WHERE 1=1 {platform_clause}
            ),
            base AS (
                SELECT *
                FROM raw
                WHERE market_id IS NOT NULL
                  AND price BETWEEN 0.0 AND 1.0
                  AND outcome IN (0.0, 1.0)
                  AND ts IS NOT NULL
                  AND resolution_time IS NOT NULL
            )
        """
        params: list[Any] = [self.parquet_glob, *platform_params]
        return sql, params

    def get_resolved_markets(
        self,
        platform: str,
        domain_filter: str | None,
        start_year: int,
        end_year: int,
    ) -> pd.DataFrame:
        base_sql, params = self._base_cte(platform)

        domain_clause = ""
        if domain_filter:
            domain_clause = "AND domain = ?"

        query = f"""
            {base_sql},
            market_base AS (
                SELECT
                    market_id,
                    ANY_VALUE(question) AS question,
                    ANY_VALUE(domain) AS domain,
                    MAX(resolution_time) AS resolution_time,
                    CAST(ROUND(AVG(outcome)) AS INTEGER) AS outcome,
                    SUM(volume) AS volume_total,
                    MIN(ts) AS first_seen
                FROM base
                WHERE EXTRACT(YEAR FROM resolution_time) BETWEEN ? AND ?
                {domain_clause}
                GROUP BY market_id
            )
            SELECT
                m.market_id,
                m.question,
                m.domain,
                m.resolution_time,
                m.outcome,
                (
                    SELECT b.price FROM base b
                    WHERE b.market_id = m.market_id AND b.ts <= m.resolution_time
                    ORDER BY ABS(DATE_DIFF('second', b.ts, m.resolution_time - INTERVAL 30 DAY)) ASC
                    LIMIT 1
                ) AS price_30d_before,
                (
                    SELECT b.price FROM base b
                    WHERE b.market_id = m.market_id AND b.ts <= m.resolution_time
                    ORDER BY ABS(DATE_DIFF('second', b.ts, m.resolution_time - INTERVAL 7 DAY)) ASC
                    LIMIT 1
                ) AS price_7d_before,
                (
                    SELECT b.price FROM base b
                    WHERE b.market_id = m.market_id AND b.ts <= m.resolution_time
                    ORDER BY ABS(DATE_DIFF('second', b.ts, m.resolution_time - INTERVAL 1 DAY)) ASC
                    LIMIT 1
                ) AS price_1d_before,
                DATE_DIFF('day', m.first_seen, m.resolution_time) AS days_to_resolution,
                m.volume_total
            FROM market_base m
            ORDER BY m.resolution_time
        """

        query_params = [*params, start_year, end_year]
        if domain_filter:
            query_params.append(domain_filter.lower())

        con = self._connect()
        try:
            return con.execute(query, query_params).df()
        finally:
            con.close()

    def get_price_snapshots(self, market_id: str, platform: str, n_snapshots: int = 30) -> pd.DataFrame:
        base_sql, params = self._base_cte(platform)

        query = f"""
            {base_sql},
            market_trades AS (
                SELECT
                    *,
                    MAX(resolution_time) OVER () AS market_resolution
                FROM base
                WHERE market_id = ?
                ORDER BY ts
            ),
            bucketed AS (
                SELECT
                    *,
                    NTILE(?) OVER (ORDER BY ts) AS bucket_id
                FROM market_trades
            )
            SELECT
                MIN(ts) AS timestamp,
                AVG(price) AS price,
                SUM(volume) AS volume_bucket,
                ANY_VALUE(taker_direction) AS taker_direction,
                DATE_DIFF('second', MIN(ts), MAX(market_resolution)) / 86400.0 AS days_remaining
            FROM bucketed
            GROUP BY bucket_id
            ORDER BY timestamp
        """

        con = self._connect()
        try:
            return con.execute(query, [*params, market_id, n_snapshots]).df()
        finally:
            con.close()

    def get_strategy_analogs(
        self,
        entry_price_max: float,
        model_prob_min: float,
        platform: str,
    ) -> pd.DataFrame:
        base_sql, params = self._base_cte(platform)

        query = f"""
            {base_sql},
            market_outcomes AS (
                SELECT
                    market_id,
                    AVG(outcome) AS market_resolution_rate
                FROM base
                GROUP BY market_id
            )
            SELECT
                b.market_id,
                b.price AS entry_price,
                mo.market_resolution_rate AS outcome,
                ((mo.market_resolution_rate - b.price) / NULLIF(b.price, 0)) AS return_pct
            FROM base b
            JOIN market_outcomes mo USING (market_id)
            WHERE b.price < ?
              AND mo.market_resolution_rate > ?
        """

        con = self._connect()
        try:
            return con.execute(query, [*params, entry_price_max, model_prob_min]).df()
        finally:
            con.close()

    def build_calibration_surface_from_data(
        self,
        platform: str,
        min_trades_per_bucket: int = 100,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        base_sql, params = self._base_cte(platform)

        year_clause = ""
        year_params: list[Any] = []
        if start_year is not None and end_year is not None:
            year_clause = "WHERE EXTRACT(YEAR FROM resolution_time) BETWEEN ? AND ?"
            year_params = [start_year, end_year]

        query = f"""
            {base_sql},
            bucketed AS (
                SELECT
                    FLOOR(price * 20) / 20 AS p_bucket,
                    CASE
                        WHEN DATE_DIFF('day', ts, resolution_time) < 7 THEN 'short'
                        WHEN DATE_DIFF('day', ts, resolution_time) <= 30 THEN 'mid'
                        ELSE 'long'
                    END AS t_bucket,
                    domain,
                    outcome
                FROM base
                {year_clause}
            )
            SELECT
                p_bucket,
                t_bucket,
                domain,
                AVG(outcome) AS empirical_win_rate,
                COUNT(*) AS n_trades
            FROM bucketed
            GROUP BY 1,2,3
            HAVING COUNT(*) >= ?
            ORDER BY 1,2,3
        """

        con = self._connect()
        try:
            return con.execute(query, [*params, *year_params, min_trades_per_bucket]).df()
        finally:
            con.close()

    def get_longshot_bias_profile(self, platform: str) -> pd.Series:
        base_sql, params = self._base_cte(platform)
        query = f"""
            {base_sql}
            SELECT
                ROUND(price, 2) AS implied_probability,
                AVG(outcome) AS empirical_win_rate
            FROM base
            WHERE price > 0.0 AND price < 0.15
            GROUP BY 1
            ORDER BY 1
        """

        con = self._connect()
        try:
            df = con.execute(query, params).df()
        finally:
            con.close()

        if df.empty:
            return pd.Series(dtype=float, name="empirical_win_rate")

        return df.set_index("implied_probability")["empirical_win_rate"]

    def get_maker_taker_stats(self, platform: str) -> pd.DataFrame:
        base_sql, params = self._base_cte(platform)
        query = f"""
            {base_sql},
            priced AS (
                SELECT
                    CAST(ROUND(price * 100) AS INTEGER) AS price_level,
                    liquidity_role,
                    CASE
                        WHEN taker_direction IN ('buy', 'yes', 'long', 'bid') THEN outcome - price
                        WHEN taker_direction IN ('sell', 'no', 'short', 'ask') THEN (1 - outcome) - (1 - price)
                        ELSE outcome - price
                    END AS trade_return,
                    outcome,
                    price
                FROM base
                WHERE price BETWEEN 0.01 AND 0.99
            )
            SELECT
                price_level,
                AVG(CASE WHEN liquidity_role = 'maker' THEN trade_return END) AS maker_excess_return,
                AVG(CASE WHEN liquidity_role = 'taker' THEN trade_return END) AS taker_excess_return,
                COUNT(CASE WHEN liquidity_role = 'maker' THEN 1 END) AS maker_n,
                COUNT(CASE WHEN liquidity_role = 'taker' THEN 1 END) AS taker_n,
                AVG(outcome) - AVG(price) AS market_excess_return
            FROM priced
            GROUP BY 1
            ORDER BY 1
        """

        con = self._connect()
        try:
            df = con.execute(query, params).df()
        finally:
            con.close()

        if df.empty:
            return df

        # If maker/taker role columns are missing in raw data, infer maker as opposite of taker.
        no_maker = df["maker_n"].fillna(0) == 0
        df.loc[no_maker, "maker_excess_return"] = -df.loc[no_maker, "taker_excess_return"]
        return df
