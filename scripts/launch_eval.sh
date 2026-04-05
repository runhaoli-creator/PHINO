#!/bin/bash
# Run DynaCLIP evaluation after training completes.
# This script evaluates DynaCLIP and baseline backbones on all experiments.
#
# Usage: bash scripts/launch_eval.sh [checkpoint_path]
#   Default: uses latest checkpoint found in checkpoints/pretrain/

set -e

cd /home/kztrgg/DynaCLIP

export CUDA_VISIBLE_DEVICES=6

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dynaclip

# Find checkpoint
CKPT="${1:-}"
if [ -z "$CKPT" ]; then
    # Find latest checkpoint
    CKPT=$(ls -t checkpoints/pretrain/dynaclip_step_*.pt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "ERROR: No checkpoint found!"
        exit 1
    fi
fi

echo "=== DynaCLIP Evaluation ==="
echo "Checkpoint: $CKPT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"

python scripts/evaluate.py \
    --checkpoint "$CKPT" \
    --data_dir data_cache/dynaclip_data \
    --output_dir results/eval \
    --backbones dynaclip dinov2_vitb14 siglip clip_vitl14 \
    --experiments linear_probing invisible_physics zero_shot \
    --num_seeds 5 \
    --num_epochs 100 \
    --n_lib 2000 \
    --n_query 500

echo "=== Evaluation complete at $(date) ==="
