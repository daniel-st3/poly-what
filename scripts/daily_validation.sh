#!/bin/bash
# Daily paper trading accumulation script
# Run via cron: 0 9 * * * /Users/danielstevenrodriguezsandoval/Desktop/trabajo\ bogota\ 2026/poly-agent/scripts/daily_validation.sh >> /Users/danielstevenrodriguezsandoval/Desktop/trabajo\ bogota\ 2026/poly-agent/logs/daily.log 2>&1
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

echo "=== $(date) Daily Validation Run ==="

# Run 5 Manifold iterations (5% threshold)
echo "Running Manifold paper trading..."
watchdog run-paper-trading --platform manifold --iterations 5

# Run 5 Polymarket iterations (2% threshold via REST API)
echo "Running Polymarket paper trading..."
watchdog run-paper-trading --platform polymarket --iterations 5

# Analyze results
echo ""
echo "Current stats:"
watchdog analyze-paper-trades

# Alert if win rate drops below 50%
WIN_RATE=$(watchdog analyze-paper-trades 2>/dev/null | grep "calibration" | awk '{print $6}' | tr -d '%')
if [ -n "$WIN_RATE" ]; then
    BELOW=$(echo "$WIN_RATE < 50" | bc -l 2>/dev/null || echo "0")
    if [ "$BELOW" = "1" ]; then
        echo "⚠️  WARNING: Win rate dropped to ${WIN_RATE}% (below 50% threshold)"
    fi
fi

echo "=== Run complete ==="
