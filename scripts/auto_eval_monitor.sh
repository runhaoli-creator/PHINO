#!/bin/bash
# Auto-evaluation monitor: watches for new checkpoints and runs evaluation
# Usage: nohup bash scripts/auto_eval_monitor.sh &

set -e
cd "$(dirname "$0")/.."

CKPT_DIR="checkpoints/pretrain"
EVAL_SCRIPT="scripts/evaluate_full.py"
EVAL_DEVICE="cuda:7"  # Use GPU 7 if free, else wait
RESULTS_DIR="results"
LOG_FILE="logs/auto_eval.log"

mkdir -p "$RESULTS_DIR" logs

echo "$(date): Auto-eval monitor started" >> "$LOG_FILE"
echo "Watching: $CKPT_DIR for checkpoints" >> "$LOG_FILE"

LAST_EVALUATED=""

while true; do
    # Find the latest checkpoint
    LATEST=$(ls -1t "$CKPT_DIR"/dynaclip_step_*.pt 2>/dev/null | head -1)

    if [[ -n "$LATEST" && "$LATEST" != "$LAST_EVALUATED" ]]; then
        STEP=$(echo "$LATEST" | grep -oP 'step_\K[0-9]+')
        echo "$(date): Found new checkpoint: $LATEST (step $STEP)" >> "$LOG_FILE"

        # Check if GPU 7 is free (memory < 1000 MiB)
        GPU7_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo "99999")
        if [[ "$GPU7_USED" -lt 1000 ]]; then
            echo "$(date): GPU 7 free ($GPU7_USED MiB used). Running eval..." >> "$LOG_FILE"

            CUDA_VISIBLE_DEVICES=7 python "$EVAL_SCRIPT" \
                --checkpoint "$LATEST" \
                --output_dir "$RESULTS_DIR/step_$STEP" \
                --device "cuda:0" \
                --batch_size 256 \
                2>&1 | tee -a "$LOG_FILE"

            LAST_EVALUATED="$LATEST"
            echo "$(date): Evaluation complete for step $STEP" >> "$LOG_FILE"
        else
            echo "$(date): GPU 7 busy ($GPU7_USED MiB). Will retry in 5 min." >> "$LOG_FILE"
        fi
    fi

    # Check if main training is still running
    if ! pgrep -f "pretrain.py --config configs/pretrain.yaml" > /dev/null 2>&1; then
        echo "$(date): Main training no longer running." >> "$LOG_FILE"

        # One final evaluation on the last checkpoint
        LATEST=$(ls -1t "$CKPT_DIR"/dynaclip_step_*.pt 2>/dev/null | head -1)
        if [[ -n "$LATEST" && "$LATEST" != "$LAST_EVALUATED" ]]; then
            echo "$(date): Final evaluation on $LATEST" >> "$LOG_FILE"
            GPU7_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 2>/dev/null || echo "99999")
            if [[ "$GPU7_USED" -lt 1000 ]]; then
                STEP=$(echo "$LATEST" | grep -oP 'step_\K[0-9]+')
                CUDA_VISIBLE_DEVICES=7 python "$EVAL_SCRIPT" \
                    --checkpoint "$LATEST" \
                    --output_dir "$RESULTS_DIR/step_$STEP" \
                    --device "cuda:0" \
                    --batch_size 256 \
                    2>&1 | tee -a "$LOG_FILE"
            fi
        fi
        echo "$(date): Monitor exiting." >> "$LOG_FILE"
        break
    fi

    sleep 300  # Check every 5 minutes
done
