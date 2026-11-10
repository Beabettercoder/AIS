#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
gpu=${GPU:-0}
seed=${SEED:-0}
epochs=${EPOCHS:-25}
mkdir -p logs
for method in tft_gcn simad dmrad; do
  for arm in ordinary reliable; do
    free_kb=$(df -Pk . | awk 'NR==2 {print $4}')
    if (( free_kb < 2097152 )); then echo '磁盘可用空间不足 2 GiB，停止启动新训练'; exit 1; fi
    echo "开始 $method / $arm / seed=$seed"
    "$PY" -u train.py --method "$method" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs "$epochs"
    "$PY" -u train.py --method "$method" --arm "$arm" --seed "$seed" --gpu "$gpu" --mode test
  done
done
echo '三种方法的单种子普通场和可信场初步对照已完成；CARLA 两阶段单独执行。'
