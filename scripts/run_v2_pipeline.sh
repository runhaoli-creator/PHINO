#!/bin/bash
# DynaCLIP v2: Full pipeline — regenerate data + retrain with all 5 fixes
# Uses GPUs 4,5 (free) — does NOT touch GPUs 0-3,6-7 (TrajCache running)

set -e

export CUDA_VISIBLE_DEVICES=4,5

cd /home/kztrgg/DynaCLIP

echo "=============================================="
echo " DynaCLIP v2: All 5 Fixes Applied"
echo "  Fix 1: No hard mining (hard_neg=0, hard_pos=0)"
echo "  Fix 2: Actual pairwise physics distances"
echo "  Fix 3: WiSE-FT regularization (alpha=0.1)"
echo "  Fix 4: Multi-material categories"
echo "  Fix 5: PairwisePhysicsRnCLoss"
echo "=============================================="

# Step 1: Regenerate data with multi-material mapping
echo ""
echo "[Step 1/2] Regenerating data with multi-material categories..."
python scripts/generate_v2_data.py 2>&1 | tee logs/generate_v2.log

# Step 2: Train with all fixes on 2 GPUs
echo ""
echo "[Step 2/2] Training DynaCLIP v2 on GPUs 4,5..."
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/pretrain.py \
    --config configs/pretrain_v2.yaml \
    --seed 42 \
    2>&1 | tee logs/pretrain_v2.log

echo ""
echo "DynaCLIP v2 training complete!"
echo "Checkpoints saved to: checkpoints/pretrain_v2/"
