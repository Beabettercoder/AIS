#!/usr/bin/env bash
set -euo pipefail
cd /home/ps/AIS_adaptive_radius_relation_v3_20261109
exec 9>build_launcher.lock
flock -n 9 || { echo 'Build launcher already active'; exit 1; }
if [[ -d relation ]]; then echo 'Existing relation output; refusing overwrite'; exit 1; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
mkdir -p logs
"$py" -u test_radius_core.py
regions=(houston ny la puget great_belt haugesund kattegat shanghai)
for split in train val test; do
 pids=()
 for region in "${regions[@]}"; do "$py" -u build_relation_unfiltered_v3.py --split "$split" --region "$region" > "logs/build_${split}_${region}.log" 2>&1 & pids+=("$!"); done
 failed=0
 for i in "${!pids[@]}"; do if wait "${pids[$i]}"; then echo "BUILD_OK $split ${regions[$i]}"; else echo "BUILD_FAILED $split ${regions[$i]}";failed=1;fi;done
 if ((failed)); then exit 1;fi
done
echo 'UNFILTERED_BUILD_COMPLETE'
