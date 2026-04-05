#!/bin/bash
# Run all evaluation after training completes.
# Usage: bash scripts/run_all_eval.sh [CHECKPOINT_PATH]
# If no checkpoint specified, uses the latest from checkpoints/pretrain/

set -e
cd "$(dirname "$0")/.."

CHECKPOINT="${1:-$(ls -1t checkpoints/pretrain/dynaclip_step_*.pt 2>/dev/null | head -1)}"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found. Provide path or wait for training to finish."
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
STEP=$(echo "$CHECKPOINT" | grep -oP 'step_\K[0-9]+' || echo "final")
OUTPUT_DIR="results/step_${STEP}"

export CUDA_VISIBLE_DEVICES=7
mkdir -p "$OUTPUT_DIR" results/ablations logs

echo ""
echo "========================================"
echo "Stage 1: Full Backbone Evaluation"
echo "========================================"
python scripts/evaluate_full.py \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --batch_size 256 \
    --num_seeds 5 \
    2>&1 | tee logs/eval_full.log

echo ""
echo "========================================"
echo "Stage 2: Ablation Evaluation"
echo "========================================"
python scripts/evaluate_ablations.py \
    --data_dir data_cache/dynaclip_data \
    --output_dir results/ablations \
    --device cuda:0 \
    --batch_size 256 \
    2>&1 | tee logs/eval_ablations.log

echo ""
echo "========================================"
echo "All Evaluations Complete!"
echo "========================================"
echo "Results at:"
echo "  Full eval:    $OUTPUT_DIR/"
echo "  Ablation:     results/ablations/"
echo "  t-SNE plots:  $OUTPUT_DIR/tsne/"
echo ""
echo "To update paper with results, run:"
echo "  python scripts/update_paper_results.py"
