#!/bin/bash
# Master launcher: Resume all ablation trainings + ManiSkill3 data gen across 8 GPUs
# GPU 0: hardneg_15pct (resume from step 5000)
# GPU 1: hardneg_50pct (resume from step 5000)
# GPU 2: init_imagenet  (resume from step 5000)
# GPU 3: init_random    (resume from step 5000)
# GPU 4: data_scale     (re-run from scratch — 4 sequential jobs: 10K,25K,50K,100K)
# GPU 5: mass_only      (resume from step 5000)
# GPU 6: friction_only  (resume from step 5000)
# GPU 7: ManiSkill3 data generation (resume from ~8424 configs)

set -e
cd /home/kztrgg/DynaCLIP

export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

CONDA_RUN="/home/kztrgg/miniconda3/bin/conda run -n dynaclip --no-capture-output"

echo "[$(date)] Launching all jobs..."

# GPU 0-3, 5-6: Resume 6 ablation trainings
for abl in hardneg_15pct hardneg_50pct init_imagenet init_random mass_only friction_only; do
    case $abl in
        hardneg_15pct) GPU=0 ;;
        hardneg_50pct) GPU=1 ;;
        init_imagenet) GPU=2 ;;
        init_random)   GPU=3 ;;
        mass_only)     GPU=5 ;;
        friction_only) GPU=6 ;;
    esac
    echo "[$(date)] Starting $abl on GPU $GPU (resume from step 5000)"
    CUDA_VISIBLE_DEVICES=$GPU nohup $CONDA_RUN python scripts/run_remaining_ablations.py \
        --mode single --ablation $abl --gpu 0 \
        > logs/resume_${abl}.log 2>&1 &
    echo "  PID=$!"
done

# GPU 4: Data scale ablation (from scratch)
echo "[$(date)] Starting data_scale on GPU 4 (from scratch)"
CUDA_VISIBLE_DEVICES=4 nohup $CONDA_RUN python scripts/run_remaining_ablations.py \
    --mode scale --gpu 0 \
    > logs/resume_data_scale.log 2>&1 &
echo "  PID=$!"

# GPU 7: ManiSkill3 data generation (resume)
echo "[$(date)] Starting ManiSkill3 data gen on GPU 7 (resume from ~8424)"
CUDA_VISIBLE_DEVICES=7 nohup $CONDA_RUN python dynaclip/data/generation_maniskill3.py \
    --output_dir data_cache/maniskill3_data \
    --num_configs 20000 --num_physics 5 --gpu_id 0 --seed 42 \
    > logs/resume_maniskill3.log 2>&1 &
echo "  PID=$!"

echo "[$(date)] All 8 jobs launched. Monitor with: tail -f logs/resume_*.log"
echo "Check GPU usage with: nvidia-smi"
