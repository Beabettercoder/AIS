#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${AIS_FORECAST_PYTHON:-/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python}"
DEVICE="${1:-cuda:0}"
DATA="${AIS_FORECAST_DATA:-data}"
RUNS="${AIS_FORECAST_RUNS:-runs}"
echo '[1/5] Environment check'
"$PYTHON" -u check_environment.py
echo '[2/5] Independent implementation checks'
"$PYTHON" -u independent_checks.py
echo '[3/5] Raw AIS -> historical inputs and observed future positions'
"$PYTHON" -u build_data.py --out "$DATA" --resume
echo '[4/5] Four encoders x four configurations x three seeds (48 runs)'
"$PYTHON" -u train.py --data "$DATA" --out "$RUNS" --device "$DEVICE" --resume
echo '[5/5] Validation tables and paired vessel-cluster intervals'
"$PYTHON" -u summarize.py --runs "$RUNS" --out reports --bootstrap 2000
echo 'Finished. See reports and per-run predictions/checkpoints.'
