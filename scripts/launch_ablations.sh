#!/bin/bash
# Launch all DynaCLIP ablation training runs sequentially on GPU 7
# Each ablation trains for 10K steps (~2h each on 1 GPU)

set -e

export CUDA_VISIBLE_DEVICES=7
CONDA_ENV="dynaclip"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "DynaCLIP Ablation Training Suite"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"
echo "========================================"

# Ablation 1: Frozen backbone (projection head only)
echo ""
echo "[1/4] Ablation: Frozen Backbone"
echo "Config: configs/ablation_frozen.yaml"
python scripts/pretrain.py --config configs/ablation_frozen.yaml \
    2>&1 | tee "$LOG_DIR/ablation_frozen.log"
echo "[1/4] Done at $(date)"

# Ablation 2: No hard negative mining
echo ""
echo "[2/4] Ablation: No Hard Negative Mining"
echo "Config: configs/ablation_nohardneg.yaml"
python scripts/pretrain.py --config configs/ablation_nohardneg.yaml \
    2>&1 | tee "$LOG_DIR/ablation_nohardneg.log"
echo "[2/4] Done at $(date)"

# Ablation 3: Random physics assignment (old data)
echo ""
echo "[3/4] Ablation: Random Physics (old data v1)"
echo "Config: configs/ablation_random_physics.yaml"
python scripts/pretrain.py --config configs/ablation_random_physics.yaml \
    2>&1 | tee "$LOG_DIR/ablation_random_physics.log"
echo "[3/4] Done at $(date)"

# Ablation 4: Standard InfoNCE (no soft labels)
echo ""
echo "[4/4] Ablation: Standard InfoNCE Loss"
echo "Config: configs/ablation_infonce.yaml"
python scripts/pretrain.py --config configs/ablation_infonce.yaml \
    2>&1 | tee "$LOG_DIR/ablation_infonce.log"
echo "[4/4] Done at $(date)"

echo ""
echo "========================================"
echo "All ablation training complete!"
echo "End: $(date)"
echo "========================================"
echo ""
echo "Checkpoints saved to:"
echo "  checkpoints/ablation_frozen/"
echo "  checkpoints/ablation_nohardneg/"
echo "  checkpoints/ablation_random_physics/"
echo "  checkpoints/ablation_infonce/"
echo ""
echo "Next: run evaluation with scripts/evaluate_ablations.py"
