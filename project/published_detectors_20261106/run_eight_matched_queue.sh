#!/usr/bin/env bash
# Eight backbones x three arms x three seeds x two tasks = 144.  Independent
# per-job status; training failure is never silently marked as complete.
set -uo pipefail
cd /mnt/sda/WangZijia/AIS/published_detectors_20261106 || exit 1
root=/home/ps/AIS_published_detectors_20261107
py=/mnt/sda/WangZijia/AIS/forecast_pilot_20261105/.runtime/bin/python
gpu=${GPU:-7}
mkdir -p "$root/logs/eight_matched_jobs"
status="$root/logs/eight_matched_status.tsv"
printf 'task\tmodel\tarm\tseed\ttrain_rc\ttest_rc\tstatus\n' > "$status"
echo 'Waiting for a recent supervised model to finish training and testing.'
while ! find "$root/supervised" -path '*/single/*/test_metrics.json' -print -quit 2>/dev/null | grep -q .; do sleep 20; done
for task in single relation; do
  if [ "$task" = single ]; then arms=(raw ordinary reliable); else arms=(C0 F0 F1); fi
  for model in tcn gru transformer itransformer patchtst mamba neuralcde graph; do
    for arm in "${arms[@]}"; do
      for seed in 0 1 2; do
        d="$root/eight_matched/$task/$model/$arm/seed$seed"
        log="$root/logs/eight_matched_jobs/${task}_${model}_${arm}_${seed}.log"
        echo "START $task $model $arm seed=$seed"
        trc=0; erc=0; outcome=OK
        if [ -e "$d/last.pt" ] || [ -e "$d/best_val.pt" ]; then
          "$py" -c 'import torch,sys; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); assert x["epoch"]==25' "$d/last.pt" >> "$log" 2>&1 || trc=$?
        else
          "$py" -u train_eight_matched.py --task "$task" --name "$model" --arm "$arm" --seed "$seed" --gpu "$gpu" --epochs 25 >> "$log" 2>&1 || trc=$?
        fi
        if [ "$trc" -eq 0 ]; then
          if [ ! -e "$d/test_metrics.json" ] || [ ! -e "$d/test_scores.npz" ]; then
            "$py" -u train_eight_matched.py --task "$task" --name "$model" --arm "$arm" --seed "$seed" --gpu "$gpu" --mode test >> "$log" 2>&1 || erc=$?
          fi
          if [ "$erc" -eq 0 ]; then
            "$py" -c 'import sys,numpy as np; z=np.load(sys.argv[1]); r=("y","mmsi","region","base"); assert all(k in z.files for k in r); assert len({len(z[k]) for k in r})==1; assert ("window_id" in z.files) or (("source" in z.files) and ("case" in z.files))' "$d/test_scores.npz" >> "$log" 2>&1 || erc=$?
          fi
        fi
        if [ "$trc" -ne 0 ]; then outcome=TRAIN_FAILED; elif [ "$erc" -ne 0 ]; then outcome=TEST_FAILED; fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$task" "$model" "$arm" "$seed" "$trc" "$erc" "$outcome" >> "$status"
        echo "END $task $model $arm seed=$seed $outcome"
      done
    done
  done
done
echo 'QUEUE_FINISHED: inspect per-task statuses before reporting completion.'
