#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY="${AIS_FORECAST_PYTHON:-/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python}"
if [ "$#" -eq 0 ]; then set -- 2 3 5 6; fi
"$PY" -u selftest.py
"$PY" -u pipeline.py --gpus "$@" --work work
