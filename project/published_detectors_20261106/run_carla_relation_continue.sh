#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106
arm=$1
gpu=$2
first_pid=$3
out=/home/ps/AIS_published_detectors_20261107
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
while kill -0 "$first_pid" 2>/dev/null; do sleep 10; done
for seed in 0 1 2; do
  if [ "$seed" -ne 0 ]; then
    echo "Starting $arm seed=$seed"
    "$py" -u train_carla_relation.py --arm "$arm" --seed "$seed" --device "cuda:$gpu"
  fi
  "$py" -c 'import sys, torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert x["epoch"]==50, "last epoch incomplete"' "$out/carla_relation_pilot/$arm/seed$seed/last.pt"
  "$py" -u train_carla_relation.py --arm "$arm" --seed "$seed" --device "cuda:$gpu" --mode test
done
echo "COMPLETED $arm seeds=0,1,2"
