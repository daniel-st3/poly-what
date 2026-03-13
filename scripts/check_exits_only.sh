#!/bin/bash
# Exit-only checker — runs every 15 minutes via launchd (com.poly-agent.exits)
# Checks trailing stops, TP/SL, and time exits on all open paper positions.
# Does NOT place new trades.
# Log: /Users/danielstevenrodriguezsandoval/poly-agent/logs/exits.log

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
source .venv/bin/activate
WATCHDOG_PYTHON="$SCRIPT_DIR/.venv/bin/python"
MAX_RUNTIME_SECONDS=300

echo "=== $(date) Exit Check ==="

run_exit_check() {
  local platform="$1"

  PLATFORM="$platform" WATCHDOG_PYTHON="$WATCHDOG_PYTHON" MAX_RUNTIME_SECONDS="$MAX_RUNTIME_SECONDS" python3 - <<'PY'
import os
import subprocess
import sys

platform = os.environ["PLATFORM"]
cmd = [os.environ["WATCHDOG_PYTHON"], "scripts/check_exits_only.py", "--platform", platform]
timeout = int(os.environ["MAX_RUNTIME_SECONDS"])

try:
    completed = subprocess.run(cmd, timeout=timeout, check=False)
except subprocess.TimeoutExpired:
    print(f"TIMEOUT: exit check exceeded {timeout}s | platform={platform}", flush=True)
    sys.exit(124)

sys.exit(completed.returncode)
PY

  local exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    echo "Exit check failed | platform=$platform | exit_code=$exit_code"
  fi
}

run_exit_check manifold
run_exit_check polymarket

echo "=== Done ==="
