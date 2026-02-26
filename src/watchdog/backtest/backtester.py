from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from watchdog.backtest.metrics import brier_score, compute_max_drawdown, compute_sharpe_ratio
from watchdog.risk.kelly import EmpiricalKellySizer

StrategyMode = Literal["taker", "maker"]


@dataclass(slots=True)
class BacktestResults:
    n_total_markets: int
    n_traded: int
    n_won: int
    n_lost: int
    win_rate: float
    brier_score: float
    max_drawdown: float
    sharpe_ratio: float
    total_pnl_usdc: float
    avg_pnl_per_trade: float
    avg_slippage: float
    equity_curve: np.ndarray
    all_trades: pd.DataFrame
    pass_go_live: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "n_total_markets": self.n_total_markets,
            "n_traded": self.n_traded,
            "n_won": self.n_won,
            "n_lost": self.n_lost,
            "win_rate": self.win_rate,
            "brier_score": self.brier_score,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_pnl_usdc": self.total_pnl_usdc,
            "avg_pnl_per_trade": self.avg_pnl_per_trade,
            "avg_slippage": self.avg_slippage,
            "equity_curve": self.equity_curve.tolist(),
            "pass_go_live": self.pass_go_live,
            "all_trades": self.all_trades.to_dict(orient="records"),
        }


class Backtester:
    def __init__(
        self,
        initial_bankroll_usdc: float = 1000.0,
        min_divergence_backtest: float = 0.15,
        fee_rate: float = 0.005,
        slippage_proxy: float = 0.005,
        spread_proxy: float = 0.01,
        maker_fill_rate: float = 0.70,
        kelly_fraction: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.initial_bankroll_usdc = initial_bankroll_usdc
        self.min_divergence_backtest = min_divergence_backtest
        self.fee_rate = fee_rate
        self.slippage_proxy = slippage_proxy
        self.spread_proxy = spread_proxy
        self.maker_fill_rate = maker_fill_rate
        self.kelly_fraction = kelly_fraction
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _time_bucket(days_remaining: float) -> str:
        if days_remaining < 7:
            return "short"
        if days_remaining <= 30:
            return "mid"
        return "long"

    @staticmethod
    def _p_bucket(probability: float) -> float:
        return float(np.floor(np.clip(probability, 0.0, 1.0) * 20) / 20)

    @staticmethod
    def _market_years(markets_df: pd.DataFrame) -> set[int]:
        years = pd.to_datetime(markets_df["resolution_time"], errors="coerce", utc=True).dt.year.dropna()
        return set(years.astype(int).tolist())

    def _lookup_model_probability(
        self,
        calibration_index: dict[tuple[float, str, str], float],
        p_market: float,
        days_remaining: float,
        domain: str,
    ) -> float:
        p_bucket = self._p_bucket(p_market)
        t_bucket = self._time_bucket(days_remaining)
        domain_norm = (domain or "unknown").lower().strip() or "unknown"

        key_order = [
            (p_bucket, t_bucket, domain_norm),
            (p_bucket, t_bucket, "unknown"),
            (p_bucket, t_bucket, "all"),
        ]
        for key in key_order:
            if key in calibration_index:
                return float(np.clip(calibration_index[key], 0.01, 0.99))
        return float(np.clip(p_market, 0.01, 0.99))

    def _precompute_calibration_index(self, calibration_surface_df: pd.DataFrame) -> dict[tuple[float, str, str], float]:
        required = {"p_bucket", "t_bucket", "domain", "empirical_win_rate"}
        missing = required.difference(calibration_surface_df.columns)
        if missing:
            raise ValueError(f"Calibration surface is missing columns: {', '.join(sorted(missing))}")

        index: dict[tuple[float, str, str], float] = {}
        for _, row in calibration_surface_df.iterrows():
            p_bucket = float(row["p_bucket"])
            t_bucket = str(row["t_bucket"]).lower()
            domain = str(row["domain"]).lower()
            empirical = float(row["empirical_win_rate"])
            index[(p_bucket, t_bucket, domain)] = empirical
        return index

    def run_backtest(
        self,
        markets_df: pd.DataFrame,
        calibration_surface_df: pd.DataFrame,
        strategy_mode: StrategyMode = "taker",
        verbose: bool = False,
    ) -> BacktestResults:
        if strategy_mode not in {"taker", "maker"}:
            raise ValueError("strategy_mode must be 'taker' or 'maker'")

        if markets_df.empty:
            raise ValueError("markets_df is empty")

        required_market_cols = {
            "market_id",
            "domain",
            "outcome",
            "resolution_time",
            "price_30d_before",
            "price_7d_before",
            "price_1d_before",
            "pre_registered",
        }
        missing_cols = required_market_cols.difference(markets_df.columns)
        if missing_cols:
            raise ValueError(f"markets_df missing columns: {', '.join(sorted(missing_cols))}")

        if not markets_df["pre_registered"].fillna(False).all():
            raise ValueError("Backtest requires a pre-registered market subset (`pre_registered=True` for all rows)")

        years = self._market_years(markets_df)
        if 2024 in years and 2025 in years:
            raise ValueError("Training and out-of-sample years are mixed. Run OOS test on 2025 only.")
        if years != {2025}:
            raise ValueError(f"run_backtest expects 2025 OOS data only, got years={sorted(years)}")

        calibration_index = self._precompute_calibration_index(calibration_surface_df)

        bankroll = self.initial_bankroll_usdc
        equity_curve = [bankroll]
        trade_returns: list[float] = []
        brier_true: list[float] = []
        brier_pred: list[float] = []
        decisions: list[dict[str, Any]] = []

        decision_schedule = [(30, "price_30d_before"), (7, "price_7d_before"), (1, "price_1d_before")]

        for _, market in markets_df.iterrows():
            market_id = str(market["market_id"])
            domain = str(market["domain"] or "unknown")
            outcome = float(market["outcome"])

            for days_remaining, column in decision_schedule:
                p_market_raw = market[column]
                if p_market_raw is None or pd.isna(p_market_raw):
                    decisions.append(
                        {
                            "market_id": market_id,
                            "decision_point_days": days_remaining,
                            "did_trade": False,
                            "reason": "missing_snapshot_price",
                        }
                    )
                    continue

                p_market = float(np.clip(p_market_raw, 0.01, 0.99))
                p_model = self._lookup_model_probability(calibration_index, p_market, days_remaining, domain)
                divergence = abs(p_model - p_market)
                side = "YES" if p_model > p_market else "NO"

                decision_record: dict[str, Any] = {
                    "market_id": market_id,
                    "decision_point_days": days_remaining,
                    "strategy_mode": strategy_mode,
                    "domain": domain,
                    "p_market": p_market,
                    "p_model": p_model,
                    "divergence": divergence,
                    "side": side,
                    "did_trade": False,
                    "reason": "",
                }

                if divergence <= self.min_divergence_backtest:
                    decision_record["reason"] = "divergence_below_threshold"
                    decisions.append(decision_record)
                    continue

                net_edge = divergence - (self.spread_proxy + self.fee_rate + self.slippage_proxy)
                decision_record["net_edge"] = net_edge
                if net_edge <= 0:
                    decision_record["reason"] = "non_positive_net_edge_after_costs"
                    decisions.append(decision_record)
                    continue

                full_kelly = EmpiricalKellySizer.full_kelly_fraction(p_model=p_model, p_market=p_market, side=side)
                size_fraction = full_kelly * self.kelly_fraction
                if size_fraction <= 0:
                    decision_record["reason"] = "non_positive_kelly"
                    decisions.append(decision_record)
                    continue

                mid_yes = p_market
                bid_yes = float(np.clip(mid_yes - self.spread_proxy / 2, 0.01, 0.99))
                ask_yes = float(np.clip(mid_yes + self.spread_proxy / 2, 0.01, 0.99))

                if side == "YES":
                    intended_mid_price = mid_yes
                    taker_entry = ask_yes + self.slippage_proxy
                    maker_entry = bid_yes
                else:
                    mid_no = 1 - mid_yes
                    bid_no = float(np.clip(mid_no - self.spread_proxy / 2, 0.01, 0.99))
                    ask_no = float(np.clip(mid_no + self.spread_proxy / 2, 0.01, 0.99))
                    intended_mid_price = mid_no
                    taker_entry = ask_no + self.slippage_proxy
                    maker_entry = bid_no

                entry_price = taker_entry if strategy_mode == "taker" else maker_entry
                was_filled = True
                if strategy_mode == "maker":
                    was_filled = bool(self._rng.random() <= self.maker_fill_rate)

                if not was_filled:
                    decision_record["reason"] = "maker_quote_not_filled"
                    decisions.append(decision_record)
                    continue

                entry_price = float(np.clip(entry_price, 0.01, 0.99))
                notional = bankroll * min(size_fraction, 1.0)
                contracts = notional / entry_price
                payoff = outcome if side == "YES" else (1 - outcome)

                gross_pnl = contracts * (payoff - entry_price)
                fee_paid = notional * self.fee_rate
                pnl = gross_pnl - fee_paid

                pre_trade_bankroll = bankroll
                bankroll += pnl
                equity_curve.append(bankroll)

                trade_return = pnl / max(pre_trade_bankroll, 1e-12)
                trade_returns.append(trade_return)
                brier_true.append(outcome)
                brier_pred.append(p_model)

                slippage_paid = abs(entry_price - intended_mid_price)

                decision_record.update(
                    {
                        "did_trade": True,
                        "reason": "trade_executed",
                        "entry_price": entry_price,
                        "intended_mid_price": intended_mid_price,
                        "slippage_paid": slippage_paid,
                        "size_fraction": size_fraction,
                        "notional": notional,
                        "contracts": contracts,
                        "pnl": pnl,
                        "fee_paid": fee_paid,
                        "bankroll_before": pre_trade_bankroll,
                        "bankroll_after": bankroll,
                    }
                )
                decisions.append(decision_record)

                if verbose:
                    print(
                        f"market={market_id} d={days_remaining} side={side} "
                        f"p_mkt={p_market:.3f} p_model={p_model:.3f} pnl={pnl:.4f}"
                    )

        decisions_df = pd.DataFrame(decisions)
        traded_df = decisions_df[decisions_df["did_trade"] == True].copy()  # noqa: E712

        n_total_markets = int(markets_df.shape[0])
        n_traded = int(traded_df.shape[0])
        n_won = int((traded_df["pnl"] > 0).sum()) if n_traded > 0 else 0
        n_lost = int((traded_df["pnl"] <= 0).sum()) if n_traded > 0 else 0

        win_rate = float(n_won / n_traded) if n_traded > 0 else 0.0
        score = brier_score(brier_true, brier_pred) if brier_true else float("nan")
        max_dd = compute_max_drawdown(equity_curve)
        sharpe = compute_sharpe_ratio(trade_returns, periods_per_year=252, risk_free=0.0)

        total_pnl = float(bankroll - self.initial_bankroll_usdc)
        avg_pnl = float(traded_df["pnl"].mean()) if n_traded > 0 else 0.0
        avg_slippage = float(traded_df["slippage_paid"].mean()) if n_traded > 0 else 0.0

        pass_go_live = bool((win_rate >= 0.55) and (score < 0.22) and (n_traded >= 50))

        return BacktestResults(
            n_total_markets=n_total_markets,
            n_traded=n_traded,
            n_won=n_won,
            n_lost=n_lost,
            win_rate=win_rate,
            brier_score=score,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            total_pnl_usdc=total_pnl,
            avg_pnl_per_trade=avg_pnl,
            avg_slippage=avg_slippage,
            equity_curve=np.asarray(equity_curve, dtype=np.float64),
            all_trades=decisions_df,
            pass_go_live=pass_go_live,
        )
