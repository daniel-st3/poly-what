#!/bin/bash
# Daily paper trading accumulation script
# Run via launchd: com.poly-agent.daily (3x daily: 1am, 9am, 5pm)
# Script: /Users/danielstevenrodriguezsandoval/poly-agent/scripts/daily_validation.sh
# Log:    /Users/danielstevenrodriguezsandoval/poly-agent/logs/daily.log

# Do NOT use set -e — we want both platforms to run even if one fails.

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate

echo "=== $(date) Daily Validation Run ==="

# --- Network pre-check: wait up to 60s for connectivity ---
echo "Checking network connectivity..."
for i in $(seq 1 12); do
    if curl -s --max-time 5 -o /dev/null -w '' 'https://api.manifold.markets/v0/markets?limit=1' 2>/dev/null; then
        echo "Network OK (attempt $i)"
        break
    fi
    if [ "$i" -eq 12 ]; then
        echo "⚠️  WARNING: No network after 60s. Proceeding anyway (may fail)."
    else
        echo "  Waiting for network... (attempt $i/12)"
        sleep 5
    fi
done

# --- Manifold ---
echo ""
echo "Running Manifold paper trading..."
if watchdog run-paper-trading --platform manifold --iterations 5; then
    echo "✅ Manifold run complete"
else
    echo "⚠️  Manifold run failed (exit code $?). Continuing to Polymarket..."
fi

# --- Polymarket ---
echo ""
echo "Running Polymarket paper trading..."
if watchdog run-paper-trading --platform polymarket --iterations 5 --max-markets 300; then
    echo "✅ Polymarket run complete"
else
    echo "⚠️  Polymarket run failed (exit code $?). Continuing to analysis..."
fi

# --- Module 1: Intra-Event ARB Scanner ---
echo ""
echo "📊 ARB SCAN"
watchdog run-intra-event-arb || echo "⚠️  ARB scan failed"

# --- Module 2: Pair Cost Scanner ---
echo ""
echo "🔄 PAIR COST"
watchdog run-pair-cost-scan || echo "⚠️  Pair cost scan failed"

# --- Module 3: Resolution Proximity Filter ---
echo ""
echo "⏰ RESOLUTION CHECK"
watchdog run-resolution-check || echo "⚠️  Resolution check failed"

# --- Module 4: Whale Flow Detector ---
echo ""
echo "🐋 WHALE WATCH"
watchdog run-whale-watch || echo "⚠️  Whale watch failed"

# --- Module 5: OFI Signal ---
echo ""
echo "📈 OFI SCAN"
watchdog run-ofi-scan || echo "⚠️  OFI scan failed"

# --- Module 6: Ensemble Signal ---
echo ""
echo "🧠 ENSEMBLE SCAN"
watchdog run-ensemble-scan || echo "⚠️  Ensemble scan failed"

# --- Module Summary ---
echo ""
watchdog run-daily-summary || echo "⚠️ Daily summary failed"

# --- Analysis ---
echo ""
echo "Current stats:"
watchdog analyze-paper-trades || echo "⚠️  Analysis failed"

# Alert if win rate drops below 50%
WIN_RATE=$(watchdog analyze-paper-trades 2>/dev/null | grep "calibration" | awk '{print $6}' | tr -d '%')
if [ -n "$WIN_RATE" ]; then
    BELOW=$(echo "$WIN_RATE < 50" | bc -l 2>/dev/null || echo "0")
    if [ "$BELOW" = "1" ]; then
        echo "⚠️  WARNING: Win rate dropped to ${WIN_RATE}% (below 50% threshold)"
    fi
fi

# --- Telegram summary ---
echo ""
watchdog send-daily-telegram || echo "⚠️  Telegram send failed (non-fatal)"

echo "=== Run complete ==="
