#!/bin/bash
# DynaCLIP Full Pre-training Launch Script
# Uses GPU 5 (single GPU), nohup for persistence

set -e

cd /home/kztrgg/DynaCLIP

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=/home/kztrgg/DynaCLIP:$PYTHONPATH

# Create output dirs
mkdir -p checkpoints/pretrain logs/pretrain

echo "=== DynaCLIP Pre-training ==="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: configs/pretrain.yaml"
echo "Start time: $(date)"

# Run training
conda run -n dynaclip python scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --seed 42

echo "Training complete at $(date)"
