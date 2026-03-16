PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

-- trades: INSERT OR IGNORE preserves newer data if rows already exist

-- portfolio_snapshots: INSERT OR REPLACE so latest balances always win
INSERT OR REPLACE INTO portfolio_snapshots (id, timestamp, starting_capital, current_balance, peak_balance, run_pnl, total_return_pct, drawdown_pct, trades_today) VALUES (1, '2026-03-15 00:00:00.000000', 100.0, 100.0, 100.0, 0.0, 0.0, 0.0, 0);
INSERT OR REPLACE INTO portfolio_snapshots (id, timestamp, starting_capital, current_balance, peak_balance, run_pnl, total_return_pct, drawdown_pct, trades_today) VALUES (2, '2026-03-15 00:00:00.000000', 500.0, 500.0, 500.0, 0.0, 0.0, 0.0, 0);

COMMIT;
PRAGMA foreign_keys = ON;
