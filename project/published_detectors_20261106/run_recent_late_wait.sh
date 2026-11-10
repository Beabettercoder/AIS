#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
bridge=/home/ps/AIS_bridge_20261107
root=/home/ps/AIS_published_detectors_20261107/recent_original_late
mkdir -p "$root/logs"
echo 'Waiting for the complete TCN and TFT eval128 bridge audits.'
while true; do
  if [[ -f "$bridge/logs/tft128_finalize.log" ]] && grep -qE 'TRAIN FAILED|Traceback \(most recent call last\)' "$bridge/logs/tft128_finalize.log"; then
    echo 'TFT bridge failed; recent-model training not started.'; exit 1
  fi
  if [[ -s "$bridge/tcn/paired_test_summary.json" ]] && [[ -f "$bridge/logs/tft128_finalize.log" ]] && grep -q 'TFT BRIDGE COMPLETE' "$bridge/logs/tft128_finalize.log"; then break; fi
  sleep 30
done
if [[ -e "$root/bridge_gate_lock.json" || -d "$root/runs" ]]; then echo 'Existing run found; refuse duplicate launch.'; exit 1; fi
echo 'Both bridge summaries available; validate frozen protocol and launch recent models.'
bash run_recent_late_launch.sh
