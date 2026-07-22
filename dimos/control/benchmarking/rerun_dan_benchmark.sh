#!/usr/bin/env bash
# Re-run Dan's controller benchmark across all 5 speeds, safely.
# Run this from the repo root on the laptop connected to the robot.
set -euo pipefail

REPO_ROOT="$(pwd)"
RAW_ROOT="$REPO_ROOT/data/benchmark/go2_dan_raw"
FINAL_DIR="$REPO_ROOT/data/benchmark/go2_dan"

for SPEED in 0.3 0.5 0.7 0.9 1.0; do
  echo "=================================================="
  echo " Speed $SPEED m/s"
  echo "=================================================="

  # Kill any leftover dimos process so the new launch can't reuse stale state.
  pkill -f "bin/dimos" || true
  sleep 1

  SPEED_DIR="$RAW_ROOT/v$SPEED"
  mkdir -p "$SPEED_DIR"

  echo ">>> About to launch at DAN_SPEED_M_S=$SPEED"
  echo ">>> Watch for this line in the output BEFORE running the battery:"
  echo ">>>   Benchmarker: go2 battery=all speeds=[$SPEED] ..."
  echo ">>> If the printed speed does not match $SPEED, stop (Ctrl-C) and re-check the env var."
  read -p "Press Enter to launch this speed... " _

  DAN_SPEED_M_S="$SPEED" DAN_OUT_DIR="$SPEED_DIR" dimos run unitree-go2-dan-holonomic-benchmark

  # Sanity check: every file written for this launch must carry the intended speed.
  BAD=$(ls "$SPEED_DIR" 2>/dev/null | grep -v "_v${SPEED}_" || true)
  if [ -n "$BAD" ]; then
    echo "!!! Files in $SPEED_DIR do not match intended speed $SPEED -- env var did not take effect:"
    echo "$BAD"
    exit 1
  fi
  echo ">>> OK: $(ls "$SPEED_DIR" | wc -l) files recorded at v$SPEED, all correctly labeled."
done

echo "=================================================="
echo " Merging into flat directory for scoring: $FINAL_DIR"
echo "=================================================="
mkdir -p "$FINAL_DIR"
for SPEED_DIR in "$RAW_ROOT"/v*; do
  cp -n "$SPEED_DIR"/*.json "$FINAL_DIR"/
done

echo "Done. Score with:"
echo "  python -m dimos.control.benchmarking.score $FINAL_DIR"
