#!/bin/bash
# Launch all remaining ablations in parallel across 7 GPUs.
# GPU 7 is NOT USED (reserved for others).
set -e

cd /home/kztrgg/DynaCLIP

export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
export PYTHONPATH="."

CONDA_PREFIX="conda run -n dynaclip --no-capture-output"
SCRIPT="scripts/run_remaining_ablations.py"

echo "================================================================"
echo " DynaCLIP Remaining Ablations — Parallel Launch"
echo " $(date)"
echo "================================================================"
echo ""
echo "Launching 7 ablation processes across GPUs 0-6..."
echo "GPU 7 is reserved and will NOT be used."
echo ""

mkdir -p logs

# GPU 0: Hard negative ratio 15%
echo "[GPU 0] hardneg_15pct — starting..."
CUDA_VISIBLE_DEVICES=0 $CONDA_PREFIX python $SCRIPT --mode single --ablation hardneg_15pct --gpu 0 \
    > logs/ablation_hardneg_15pct.log 2>&1 &
PID_0=$!
echo "  PID=$PID_0"

# GPU 1: Hard negative ratio 50%
echo "[GPU 1] hardneg_50pct — starting..."
CUDA_VISIBLE_DEVICES=1 $CONDA_PREFIX python $SCRIPT --mode single --ablation hardneg_50pct --gpu 0 \
    > logs/ablation_hardneg_50pct.log 2>&1 &
PID_1=$!
echo "  PID=$PID_1"

# GPU 2: ImageNet init
echo "[GPU 2] init_imagenet — starting..."
CUDA_VISIBLE_DEVICES=2 $CONDA_PREFIX python $SCRIPT --mode single --ablation init_imagenet --gpu 0 \
    > logs/ablation_init_imagenet.log 2>&1 &
PID_2=$!
echo "  PID=$PID_2"

# GPU 3: Random init
echo "[GPU 3] init_random — starting..."
CUDA_VISIBLE_DEVICES=3 $CONDA_PREFIX python $SCRIPT --mode single --ablation init_random --gpu 0 \
    > logs/ablation_init_random.log 2>&1 &
PID_3=$!
echo "  PID=$PID_3"

# GPU 4: Data scale (sequential: 10K, 25K, 50K, 100K)
echo "[GPU 4] data_scale (10K→100K sequential) — starting..."
CUDA_VISIBLE_DEVICES=4 $CONDA_PREFIX python $SCRIPT --mode scale --gpu 0 \
    > logs/ablation_data_scale.log 2>&1 &
PID_4=$!
echo "  PID=$PID_4"

# GPU 5: Property diversity — mass only
echo "[GPU 5] mass_only — starting..."
CUDA_VISIBLE_DEVICES=5 $CONDA_PREFIX python $SCRIPT --mode single --ablation mass_only --gpu 0 \
    > logs/ablation_mass_only.log 2>&1 &
PID_5=$!
echo "  PID=$PID_5"

# GPU 6: Property diversity — friction only
echo "[GPU 6] friction_only — starting..."
CUDA_VISIBLE_DEVICES=6 $CONDA_PREFIX python $SCRIPT --mode single --ablation friction_only --gpu 0 \
    > logs/ablation_friction_only.log 2>&1 &
PID_6=$!
echo "  PID=$PID_6"

echo ""
echo "All 7 ablation processes launched."
echo "PIDs: $PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5 $PID_6"
echo ""
echo "Monitoring logs:"
echo "  tail -f logs/ablation_hardneg_15pct.log"
echo "  tail -f logs/ablation_hardneg_50pct.log"
echo "  tail -f logs/ablation_init_imagenet.log"
echo "  tail -f logs/ablation_init_random.log"
echo "  tail -f logs/ablation_data_scale.log"
echo "  tail -f logs/ablation_mass_only.log"
echo "  tail -f logs/ablation_friction_only.log"
echo ""

# Wait for all processes
echo "Waiting for all ablation processes to complete..."
wait $PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5 $PID_6

echo ""
echo "================================================================"
echo " All ablations completed! $(date)"
echo "================================================================"
