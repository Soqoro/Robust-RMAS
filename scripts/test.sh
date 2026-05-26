#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH -p NA100q
#SBATCH -w node01
#SBATCH --output=logs/test%j.out
#SBATCH --error=logs/test%j.err

# GPU visibility / status
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "===== nvidia-smi -L ====="
nvidia-smi -L || true
echo "===== initial nvidia-smi ====="
nvidia-smi || true
echo "===== detailed GPU memory/status ====="
nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
           --format=csv || true

python - <<'PY'
import os, torch
print("torch.cuda.device_count() =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} ->", torch.cuda.get_device_name(i))
PY