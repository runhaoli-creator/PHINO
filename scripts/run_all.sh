#!/bin/bash
# DynaCLIP: Full Pipeline Run Script
# Runs data generation, pre-training, evaluation, and ablations.
set -e

echo "================================================="
echo "  DynaCLIP: Physics-Grounded Visual Representations"
echo "  via Dynamics Contrastive Learning"
echo "================================================="

# Configuration
CONDA_ENV="dynaclip"
NUM_GPUS=${NUM_GPUS:-8}
CONFIG_DIR="configs"
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# -------------------------------------------------------------------
# Step 1: Data Generation
# -------------------------------------------------------------------
echo ""
echo "=== Step 1: Data Generation ==="
python scripts/generate_data.py \
    --config ${CONFIG_DIR}/data_generation.yaml \
    --seed 42

# -------------------------------------------------------------------
# Step 2: Pre-compute DINOv2 Embeddings (for pair mining)
# -------------------------------------------------------------------
echo ""
echo "=== Step 2: Pre-compute DINOv2 Embeddings ==="
python -c "
from dynaclip.data.precompute import precompute_dino_embeddings
import json
with open('data_cache/dynaclip_data/metadata.json') as f:
    meta = json.load(f)
paths = [m['image_path'] for m in meta]
precompute_dino_embeddings(paths, 'data_cache/dynaclip_data/dino_embeddings.npz')
"

# -------------------------------------------------------------------
# Step 3: DynaCLIP Pre-training
# -------------------------------------------------------------------
echo ""
echo "=== Step 3: DynaCLIP Pre-training ==="
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS \
        scripts/pretrain.py \
        --config ${CONFIG_DIR}/pretrain.yaml \
        --seed 42
else
    python scripts/pretrain.py \
        --config ${CONFIG_DIR}/pretrain.yaml \
        --seed 42
fi

# -------------------------------------------------------------------
# Step 4: Evaluation (Experiments 1-5)
# -------------------------------------------------------------------
echo ""
echo "=== Step 4: Evaluation ==="
python scripts/evaluate.py \
    --config ${CONFIG_DIR}/evaluation.yaml \
    --checkpoint ${CHECKPOINT_DIR}/pretrain/dynaclip_best.pt \
    --experiments 1 2 4 5 \
    --seed 42

# -------------------------------------------------------------------
# Step 5: Ablation Studies
# -------------------------------------------------------------------
echo ""
echo "=== Step 5: Ablation Studies ==="
python scripts/run_ablations.py \
    --config ${CONFIG_DIR}/pretrain.yaml \
    --output_dir ${RESULTS_DIR}/ablations \
    --seed 42

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "================================================="
echo "  DynaCLIP Pipeline Complete!"
echo ""
echo "  Checkpoints:  ${CHECKPOINT_DIR}/"
echo "  Results:       ${RESULTS_DIR}/"
echo "  Figures:       ${RESULTS_DIR}/figures/"
echo "  Ablations:     ${RESULTS_DIR}/ablations/"
echo "================================================="
