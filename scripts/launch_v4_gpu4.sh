#!/bin/bash
# launch_v4_gpu4.sh — Run backbones on GPU 4
set -e
cd /home/kztrgg/DynaCLIP

export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

# Source conda
source /home/kztrgg/miniconda3/etc/profile.d/conda.sh
conda activate dynaclip

LOG_DIR="logs/libero_v4"
mkdir -p "$LOG_DIR"

for BACKBONE in DynaCLIP DINOv2 CLIP R3M SigLIP; do
    echo "[$(date)] Starting $BACKBONE on GPU 4"
    python -u scripts/evaluate_libero_v4.py \
        --gpu 0 \
        --backbone "$BACKBONE" \
        --n_seeds 3 \
        --n_epochs 200 \
        --n_episodes 50 \
        --tasks all \
        > "${LOG_DIR}/${BACKBONE}_gpu4.log" 2>&1
    echo "[$(date)] $BACKBONE DONE"
done
echo "[$(date)] GPU 4 ALL DONE"
