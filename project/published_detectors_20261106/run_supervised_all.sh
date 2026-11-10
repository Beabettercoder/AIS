#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
method=${1:?method}; gpu=${2:?gpu}
PY=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
export PYTHONPATH="$PWD/deps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
for task in single relation; do
  if [[ "$task" == single ]]; then arms=(raw ordinary reliable); else arms=(C0 F0 F1); fi
  for seed in 0 1 2; do
    for arm in "${arms[@]}"; do
      echo "开始 supervised $task $method $arm seed=$seed"
      if [[ "$method" == carla ]]; then
        "$PY" -u train_carla_supervised.py --task "$task" --arm "$arm" --gpu "$gpu" --seed "$seed"
        "$PY" -u train_carla_supervised.py --task "$task" --arm "$arm" --gpu "$gpu" --seed "$seed" --mode test
      else
        "$PY" -u train_supervised_recent.py --method "$method" --task "$task" --arm "$arm" --gpu "$gpu" --seed "$seed"
        "$PY" -u train_supervised_recent.py --method "$method" --task "$task" --arm "$arm" --gpu "$gpu" --seed "$seed" --mode test
      fi
    done
  done
done
echo "完成 $method 监督双任务三种子"
