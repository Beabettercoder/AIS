#!/usr/bin/env bash
cd /mnt/sda/WangZijia/AIS
PY=/mnt/sda/HaiShangTS/conda-envs/qwen38/bin/python
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY scripts/109_llm_fix.py "Qwen2.5-0.5B-Instruct,Qwen2.5-1.5B-Instruct" > logs/110_w1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY scripts/109_llm_fix.py "Qwen2.5-3B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B,Mistral-7B-Instruct-v0.2" > logs/111_w2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 setsid nohup $PY scripts/109_llm_fix.py "Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct" > logs/112_w3.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 setsid nohup $PY scripts/109_llm_fix.py "gemma-2-9b-it,Qwen2.5-14B-Instruct,Qwen3.8-27B" > logs/113_w4.log 2>&1 &
sleep 3
echo "4 workers launched"
pgrep -af "109_llm_fix" | wc -l
