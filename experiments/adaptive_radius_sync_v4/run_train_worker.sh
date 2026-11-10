#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/AIS_adaptive_sync_20261110
task=${1:?task};name=${2:?name};seed=${3:?seed};gpu=${4:?gpu}
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
arms=(target_only tight_plain wide_plain adaptive_plain_nofallback adaptive_plain_fallback adaptive_qualified_fallback adaptive_credible_fallback)
if [[ "$task" == single ]]; then script=train_single_v2.py; elif [[ "$task" == relation ]]; then script=train_relation_v2.py; else exit 2; fi
for arm in "${arms[@]}"; do
 echo "TRAIN_START $task $name $arm seed=$seed gpu=$gpu"
 "$py" -u "$script" --name "$name" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs 25
 echo "TRAIN_DONE $task $name $arm seed=$seed"
done
echo "WORKER_COMPLETE $task $name seed=$seed"
