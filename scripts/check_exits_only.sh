#!/bin/bash
# Exit-only checker — runs every 15 minutes via launchd (com.poly-agent.exits)
# Checks trailing stops, TP/SL, and time exits on all open paper positions.
# Does NOT place new trades.
# Log: /Users/danielstevenrodriguezsandoval/poly-agent/logs/exits.log

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

echo "=== $(date) Exit Check ==="

python scripts/check_exits_only.py --platform manifold
python scripts/check_exits_only.py --platform polymarket

echo "=== Done ==="
