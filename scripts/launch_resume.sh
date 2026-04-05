#!/bin/bash
# Resume DynaCLIP pre-training from step 15000 on GPU 6
# Memory reduced: batch_size=128, grad_accum=8, gradient_checkpointing=true
# Effective batch size unchanged: 128 * 8 = 1024

set -e

cd /home/kztrgg/DynaCLIP

# Only use GPU 6 (the only free one)
export CUDA_VISIBLE_DEVICES=6

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynaclip

echo "=== Resuming DynaCLIP training from step 15000 ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
nvidia-smi --id=6 --query-gpu=memory.used,memory.total --format=csv

python scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --resume checkpoints/pretrain/dynaclip_step_15000.pt \
    --seed 42

echo "=== Training completed at $(date) ==="
