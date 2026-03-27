PRAGMA foreign_keys = OFF;
BEGIN TRANSACTION;

-- trades: INSERT OR IGNORE preserves newer data if rows already exist

COMMIT;
PRAGMA foreign_keys = ON;
