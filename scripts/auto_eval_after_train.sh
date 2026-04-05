#!/bin/bash
# Monitor training and auto-launch evaluation when complete.
# Polls the training log every 5 minutes. When training finishes, launches eval.
#
# Usage: nohup bash scripts/auto_eval_after_train.sh &

set -e

cd /home/kztrgg/DynaCLIP

LOG="logs/resume_training.log"
EVAL_LOG="logs/auto_eval.log"

echo "=== Auto-eval monitor started at $(date) ===" | tee "$EVAL_LOG"
echo "Monitoring: $LOG" | tee -a "$EVAL_LOG"

while true; do
    # Check if training process is still running
    if ! pgrep -f "scripts/pretrain.py.*resume" > /dev/null 2>&1; then
        echo "[$(date)] Training process not found. Checking log..." | tee -a "$EVAL_LOG"

        # Check if training completed or crashed
        if grep -q "Pre-training complete" "$LOG" 2>/dev/null; then
            echo "[$(date)] Training completed! Launching evaluation..." | tee -a "$EVAL_LOG"
            break
        fi

        # Check for the final checkpoint save as backup signal
        if grep -q "Step 50000" "$LOG" 2>/dev/null; then
            echo "[$(date)] Reached step 50000. Launching evaluation..." | tee -a "$EVAL_LOG"
            break
        fi

        echo "[$(date)] Training seems to have stopped unexpectedly." | tee -a "$EVAL_LOG"
        echo "[$(date)] Last log line:" | tee -a "$EVAL_LOG"
        tail -1 "$LOG" | tee -a "$EVAL_LOG"
        echo "[$(date)] Will launch eval on latest checkpoint anyway." | tee -a "$EVAL_LOG"
        break
    fi

    # Still running — report status
    LAST_STEP=$(grep "^[0-9].*Step " "$LOG" 2>/dev/null | tail -1 | grep -oP 'Step \K[0-9]+' || echo "?")
    echo "[$(date)] Training still running. Step: $LAST_STEP/50000" | tee -a "$EVAL_LOG"
    sleep 300  # 5 minutes
done

# Launch evaluation
echo "[$(date)] Starting evaluation pipeline..." | tee -a "$EVAL_LOG"
bash scripts/launch_eval.sh >> "$EVAL_LOG" 2>&1
echo "[$(date)] Evaluation complete!" | tee -a "$EVAL_LOG"
