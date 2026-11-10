#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY="${AIS_FORECAST_PYTHON:-/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python}"
# Accept both legacy GPU positional arguments and explicit v2 options.
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  GPU_IDS=()
  while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do GPU_IDS+=("$1"); shift; done
  exec "$PY" -u v2/pipeline.py --gpus "${GPU_IDS[@]}" "$@"
fi
exec "$PY" -u v2/pipeline.py "$@"
