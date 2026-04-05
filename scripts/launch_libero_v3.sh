#!/bin/bash
# Launch LIBERO-10 v3 evaluation for all 6 backbones across 6 GPUs.
# GPUs 5,6 are occupied by other projects — use GPUs 0,1,2,3,4,7.
#
# Usage: bash scripts/launch_libero_v3.sh

set -e

CONDA_RUN="/home/kztrgg/miniconda3/bin/conda run -n dynaclip --no-capture-output"
SCRIPT="scripts/evaluate_libero_v3.py"
LOG_DIR="logs/libero_v3"
mkdir -p "$LOG_DIR"

echo "=== Launching LIBERO-10 v3 evaluation ==="
echo "  6 backbones × 10 tasks × 3 seeds × 50 episodes"
echo "  GPUs: 0,1,2,3,4,7"
echo ""

# GPU assignments: one backbone per GPU
declare -A GPU_MAP
GPU_MAP[DynaCLIP]=0
GPU_MAP[DINOv2]=1
GPU_MAP[CLIP]=2
GPU_MAP[R3M]=3
GPU_MAP[MCR]=4
GPU_MAP[SigLIP]=7

for BACKBONE in DynaCLIP DINOv2 CLIP R3M MCR SigLIP; do
    GPU=${GPU_MAP[$BACKBONE]}
    LOG="$LOG_DIR/${BACKBONE}_gpu${GPU}.log"
    
    echo "  Launching $BACKBONE on GPU $GPU → $LOG"
    
    nohup env MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$GPU \
        OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        $CONDA_RUN \
        python $SCRIPT \
        --gpu 0 \
        --backbone $BACKBONE \
        --n_epochs 100 \
        --n_episodes 50 \
        --n_seeds 3 \
        > "$LOG" 2>&1 &
    
    echo "    PID: $!"
    sleep 2  # Stagger launches to avoid model download races
done

echo ""
echo "All 6 backbones launched. Monitor with:"
echo "  tail -f $LOG_DIR/*.log"
echo "  grep 'success rate\|Average\|Training done' $LOG_DIR/*.log"
echo ""
echo "Results will be saved to results/libero10_v3/"
