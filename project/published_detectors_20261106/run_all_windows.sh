#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
method=${1:?method required}
gpu=${2:-0}
seed=${3:-0}
out=/home/ps/AIS_published_detectors_20261107/single_all_windows
for arm in raw gate ordinary reliable; do
    echo "开始 $method / $arm / seed=$seed"
    "$PY" -u train_all_windows.py --method "$method" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs 25 --out-root "$out"
    "$PY" -u train_all_windows.py --method "$method" --arm "$arm" --seed "$seed" --gpu "$gpu" --mode test --out-root "$out"
done
echo "$method 四组已完成"
