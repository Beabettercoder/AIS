#!/usr/bin/env bash
set -uo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106 || exit 1
gpu=$1
queue_pid=$2
shift 2
models=("$@")
root=/home/ps/AIS_published_detectors_20261107
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
status="$root/logs/eight_accel_gpu${gpu}_status.tsv"
printf 'model\tarm\tseed\ttrain_rc\ttest_rc\tpublication\n' > "$status"
mkdir -p "$root/logs/eight_accel_jobs"
for model in "${models[@]}"; do
  for arm in F1 F0 C0; do
    for seed in 2 1 0; do
      marker="START relation $model $arm seed=$seed"
      if grep -Fxq "$marker" "$root/logs/eight_matched_queue.log"; then
        printf '%s\t%s\t%s\t-\t-\tSERIAL_ALREADY_STARTED\n' "$model" "$arm" "$seed" >> "$status";continue
      fi
      log="$root/logs/eight_accel_jobs/relation_${model}_${arm}_${seed}.log"
      echo "START relation $model $arm seed=$seed GPU=$gpu"
      trc=0;erc=0;pub=NOT_ATTEMPTED
      "$py" -u train_eight_matched.py --out "$root/eight_matched_accel" --task relation --name "$model" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs 25 > "$log" 2>&1 || trc=$?
      if [ "$trc" -eq 0 ]; then
        "$py" -u train_eight_matched.py --out "$root/eight_matched_accel" --task relation --name "$model" --arm "$arm" --seed "$seed" --gpu "$gpu" --mode test >> "$log" 2>&1 || erc=$?
        if [ "$erc" -eq 0 ]; then
          pub=$("$py" publish_matched_accel.py "$queue_pid" "$model" "$arm" "$seed" 2>&1) || pub="PUBLICATION_ERROR $pub"
        fi
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model" "$arm" "$seed" "$trc" "$erc" "$pub" >> "$status"
      echo "END relation $model $arm seed=$seed train=$trc test=$erc $pub"
    done
  done
done
echo 'ACCEL_WORKER_FINISHED; inspect per-job status.'
