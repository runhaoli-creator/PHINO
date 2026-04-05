#!/bin/bash
# DynaCLIP Multi-GPU Training Launch Script
# Uses GPUs 1-5 for DDP training (GPU 0 = fisher_steer, GPU 6 = trajcache, GPU 7 = reserved for eval/ablations)
set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4,5
NUM_GPUS=5

cd /home/kztrgg/DynaCLIP
source ~/miniconda3/bin/activate dynaclip

echo "=== DynaCLIP Training (v2: category-grounded physics) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES ($NUM_GPUS GPUs)"
echo "Config: configs/pretrain.yaml"
echo "Start time: $(date)"

# Launch distributed training (use python -m to ensure correct conda env Python)
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --seed 42

echo "Training completed at $(date)"
