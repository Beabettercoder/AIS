#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/mnt/sda/HaiShangTS
DEPLOY_DIR=/mnt/sda/WangZijia/AIS/system3
MODEL_DIR="$BASE/models/Qwen3.8-27B"
ENV_DIR="$BASE/conda-envs/qwen38"
LOG_DIR="$DEPLOY_DIR/logs"
PORT=8002

mkdir -p "$LOG_DIR"
if ss -ltn | awk -v port=":$PORT" '$4 ~ port"$" { found=1 } END { exit !found }'; then
  echo "Port $PORT is already in use; refusing to start."
  exit 1
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$ENV_DIR/bin:$PATH"
exec > >(tee -a "$LOG_DIR/vllm_server_gpu4.log") 2>&1
printf '\nUTC: %s\nPhysical GPU: 4\nModel: %s\n' "$(date -u +%FT%TZ)" "$MODEL_DIR"
exec "$ENV_DIR/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.88 \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --host 127.0.0.1 \
  --port "$PORT"
