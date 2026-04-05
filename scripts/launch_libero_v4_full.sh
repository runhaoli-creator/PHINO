#!/bin/bash
# launch_libero_v4_full.sh — Run all LIBERO-10 v4 evaluations
# Uses GPUs 4 and 5 (free), serializes backbones per GPU
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

LOG_DIR="logs/libero_v4"
mkdir -p "$LOG_DIR"

echo "=== LIBERO-10 v4 Full Evaluation ==="
echo "Start time: $(date)"
echo ""

# GPU 4: DynaCLIP, DINOv2, CLIP, R3M, SigLIP (5 backbones)
# GPU 5: VC-1, MCR, MVP (3 backbones — larger models need more time)

run_backbone() {
    local GPU=$1
    local BACKBONE=$2
    local LOGFILE="${LOG_DIR}/${BACKBONE}_gpu${GPU}.log"
    
    echo "[$(date +%H:%M:%S)] Starting ${BACKBONE} on GPU ${GPU}..."
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 conda run -n dynaclip python -u scripts/evaluate_libero_v4.py \
        --gpu 0 \
        --backbone "$BACKBONE" \
        --n_seeds 3 \
        --n_epochs 200 \
        --n_episodes 50 \
        --tasks all \
        > "$LOGFILE" 2>&1
    echo "[$(date +%H:%M:%S)] ${BACKBONE} DONE."
    echo ""
}

# Run GPU 4 backbones sequentially
(
    run_backbone 4 DynaCLIP
    run_backbone 4 DINOv2
    run_backbone 4 CLIP
    run_backbone 4 R3M
    run_backbone 4 SigLIP
) &
PID_GPU4=$!

# Run GPU 5 backbones sequentially  
(
    run_backbone 5 VC-1
    run_backbone 5 MCR
    run_backbone 5 MVP
) &
PID_GPU5=$!

echo "GPU 4 PID: $PID_GPU4"
echo "GPU 5 PID: $PID_GPU5"
echo "Waiting for all backbones to finish..."

wait $PID_GPU4
wait $PID_GPU5

echo ""
echo "=== ALL EVALUATIONS COMPLETE ==="
echo "End time: $(date)"
echo ""
echo "Results in: results/libero10_v4/"
ls -la results/libero10_v4/
