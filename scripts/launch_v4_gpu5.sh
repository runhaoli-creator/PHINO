#!/bin/bash
# launch_v4_gpu5.sh — Run backbones on GPU 5
set -e
cd /home/kztrgg/DynaCLIP

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

# Source conda
source /home/kztrgg/miniconda3/etc/profile.d/conda.sh
conda activate dynaclip

LOG_DIR="logs/libero_v4"
mkdir -p "$LOG_DIR"

for BACKBONE in VC-1 MCR MVP; do
    echo "[$(date)] Starting $BACKBONE on GPU 5"
    python -u scripts/evaluate_libero_v4.py \
        --gpu 0 \
        --backbone "$BACKBONE" \
        --n_seeds 3 \
        --n_epochs 200 \
        --n_episodes 50 \
        --tasks all \
        > "${LOG_DIR}/${BACKBONE}_gpu5.log" 2>&1
    echo "[$(date)] $BACKBONE DONE"
done
echo "[$(date)] GPU 5 ALL DONE"
