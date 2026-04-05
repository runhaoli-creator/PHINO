# DynaCLIP V3: Experiment Instructions for Agent

## What Changed and Why

We switched from **parameter-distance** soft labels (V2) back to **trajectory-based** soft labels (V3).

**V2 problem**: The `PairwisePhysicsRnCLoss` computed `L2([log(mass), friction, restitution])` between image pairs as the contrastive soft label. This is a linear distance that misses nonlinear physics interactions. V2 showed worse physics probing than V1 on friction (0.378 vs 0.416) and restitution (0.652 vs 0.659).

**V3 fix**: We use the `SoftInfoNCELoss` with dynamics similarity computed from actual physics trajectories. The analytical physics engine simulates 5 diagnostic actions (push, lift, flick, press) under Newtonian mechanics. The trajectory similarity between two objects captures how mass, friction, and restitution *jointly* determine physical behavior — information that raw parameter distance loses.

**Code changes already made** (in this commit):

1. `dynaclip/data/dataset.py` — `_compute_proxy_similarity()` now computes **actual trajectory similarity** using the analytical physics engine, instead of falling back to parameter L2 distance. A new `_get_trajectory()` method handles trajectory computation with three-level lookup: (1) in-memory cache, (2) precomputed `.npz` fingerprint on disk, (3) on-the-fly computation via the physics engine. This ensures all training pairs use real trajectory similarity while keeping dataset initialization fast.

2. `configs/pretrain_v3.yaml` — New config using `soft_infonce` loss with `use_physics_vectors: false` and `wiseft_alpha: 0.1`.

3. `scripts/generate_v3_ablation_data.py` — Data generation script for all ablation variants.

---

## Step-by-Step Experiment Plan

### Phase 1: Generate Data (~1-2 hours, CPU only)

Generate training data for V3 full and all ablation variants. Each variant needs its own data directory because the physics parameters (and thus trajectories and similarities) differ.

```bash
# V3 full (all 3 physics dimensions vary, category-grounded)
python scripts/generate_v3_ablation_data.py --variant full \
    --output_dir data_cache/dynaclip_v3_data \
    --dataset_root <YOUR_DATASET_ROOT> \
    --max_images 20000 --num_physics 5

# Ablation: random physics (no category correlation)
python scripts/generate_v3_ablation_data.py --variant random \
    --output_dir data_cache/dynaclip_v3_random \
    --dataset_root <YOUR_DATASET_ROOT> \
    --max_images 20000 --num_physics 5

# Ablation: mass only (friction & restitution fixed to material midpoint)
python scripts/generate_v3_ablation_data.py --variant mass_only \
    --output_dir data_cache/dynaclip_v3_mass_only \
    --dataset_root <YOUR_DATASET_ROOT> \
    --max_images 20000 --num_physics 5

# Ablation: friction only
python scripts/generate_v3_ablation_data.py --variant fric_only \
    --output_dir data_cache/dynaclip_v3_fric_only \
    --dataset_root <YOUR_DATASET_ROOT> \
    --max_images 20000 --num_physics 5

# Ablation: restitution only
python scripts/generate_v3_ablation_data.py --variant rest_only \
    --output_dir data_cache/dynaclip_v3_rest_only \
    --dataset_root <YOUR_DATASET_ROOT> \
    --max_images 20000 --num_physics 5
```

Replace `<YOUR_DATASET_ROOT>` with the path to the directory containing `domainnet/real/` and optionally `coco/raw/train2017/`.

These can all run in parallel on different CPU cores. Each takes ~15-30 minutes.

### Phase 2: Train V3 Full Model (~2 hours, multi-GPU)

```bash
torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42
```

Adjust `training.batch_size` in the config if needed based on GPU count. The effective batch size should be ~1024. For example, with 4 GPUs: set `batch_size: 1024` (256 per GPU).

The checkpoint will be saved to `checkpoints/pretrain_v3/dynaclip_final.pt`.

### Phase 3: Evaluate V3 — Physics Probing & Clustering (~1 hour)

Run linear probing and material clustering on V3 checkpoint + all baselines.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/pretrain_v3/dynaclip_final.pt \
    --data_dir data_cache/dynaclip_v3_data \
    --experiments linear_probing clustering
```

**Checkpoint**: Compare V3 probing numbers against V2 (mass R²=0.553, friction R²=0.378, restitution R²=0.652). V3 should match or exceed V2 on all metrics, especially friction. If friction R² < 0.40 or mass R² < 0.53, something is wrong — check data generation and similarity computation.

Expected V3 probing (based on V1 "no hard mining" ablation extrapolation):
- Mass R²: ~0.60-0.70
- Friction R²: ~0.50-0.60
- Restitution R²: ~0.70-0.75

### Phase 4: Evaluate V3 — LIBERO-10 (~5-6 hours)

This is the most important evaluation. Run LIBERO-10 with V3 checkpoint using the standard frozen-encoder LSTM BC protocol.

```bash
python scripts/evaluate_libero_v4.py \
    --checkpoint checkpoints/pretrain_v3/dynaclip_final.pt \
    --seeds 3 --epochs 200
```

(Adapt the script and arguments to your LIBERO evaluation setup.)

**Checkpoint**: V3 LIBERO-10 avg must be ≥ 59% to proceed. If it's significantly below V2's 60.4%, we may need to fall back to V2 and include V3 as an ablation row.

**Decision point**:
- V3 LIBERO ≥ 60%: Use V3 as the final model. Proceed to Phase 5.
- V3 LIBERO 58-60%: Still usable. The probing improvement + competitive LIBERO makes a good story.
- V3 LIBERO < 58%: Fall back to V2. Report V3 as an ablation showing trajectory > param distance on probing.

### Phase 5: Train Ablation Variants (~1 hour each, parallelize)

Train 4 ablation variants at 10K steps each. Use the same config as V3 but with:
- `training.total_steps: 10000`
- `data.data_dir:` pointing to the variant's data directory

```bash
# Random physics
torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    training.total_steps=10000 \
    data.data_dir=data_cache/dynaclip_v3_random \
    output.checkpoint_dir=checkpoints/ablation_random

# Mass only
torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    training.total_steps=10000 \
    data.data_dir=data_cache/dynaclip_v3_mass_only \
    output.checkpoint_dir=checkpoints/ablation_mass_only

# Friction only
torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    training.total_steps=10000 \
    data.data_dir=data_cache/dynaclip_v3_fric_only \
    output.checkpoint_dir=checkpoints/ablation_fric_only

# Restitution only
torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    training.total_steps=10000 \
    data.data_dir=data_cache/dynaclip_v3_rest_only \
    output.checkpoint_dir=checkpoints/ablation_rest_only
```

These 4 can run in parallel if you have enough GPUs (each needs ≥2 GPUs for DDP, or 1 GPU with smaller batch).

### Phase 6: Evaluate Ablation Variants — Probing (~30 min)

Run linear probing on all 4 ablation checkpoints:

```bash
for variant in random mass_only fric_only rest_only; do
    python scripts/evaluate.py \
        --checkpoint checkpoints/ablation_${variant}/dynaclip_final.pt \
        --data_dir data_cache/dynaclip_v3_${variant} \
        --experiments linear_probing
done
```

Record mass R², friction R², restitution R² for each variant.

### Phase 7: Evaluate Ablation Variants — LIBERO-10 (~6 hours)

Run LIBERO-10 on **random physics** variant (most critical) and optionally 1-2 other interesting variants.

**Must run**: `random` — this is the control that proves physics signal matters.
**Optional**: the single-dimension variant with the best probing (likely `mass_only`) and the worst (for contrast).

```bash
# Random physics on LIBERO-10
python scripts/evaluate_libero_v4.py \
    --checkpoint checkpoints/ablation_random/dynaclip_final.pt \
    --seeds 3 --epochs 200

# Best single-dimension variant on LIBERO-10
python scripts/evaluate_libero_v4.py \
    --checkpoint checkpoints/ablation_mass_only/dynaclip_final.pt \
    --seeds 3 --epochs 200
```

### Phase 8: Also Rerun with V3 Checkpoint

Run these existing evaluations with the V3 checkpoint to ensure all reported numbers are from the same model:

```bash
# DROID-100 offline evaluation
python scripts/evaluate_droid.py \
    --checkpoint checkpoints/pretrain_v3/dynaclip_final.pt

# CALVIN offline BC
python scripts/evaluate_calvin_offline.py \
    --checkpoint checkpoints/pretrain_v3/dynaclip_final.pt
```

---

## Expected Final Results Table

### Main Results (all V3 checkpoint)

| Experiment | DynaCLIP V3 | DINOv2-B/14 (base) | Best Other Baseline |
|---|---|---|---|
| Linear Probing Mass R² | ~0.65 | 0.370 | 0.439 (DINOv2-L) |
| Linear Probing Fric R² | ~0.55 | 0.132 | 0.295 (CLIP) |
| Linear Probing Rest R² | ~0.72 | 0.544 | 0.605 (DINOv2-L) |
| Material Clustering NMI | ~0.45 | 0.331 | 0.350 (CLIP) |
| LIBERO-10 Avg Success | ≥59% | 51.5% | 54.5% (SigLIP) |
| DROID-100 CosSim | ~0.79 | 0.774 | 0.887 (R3M) |

### Ablation Table (the key table for the story)

| Variant | Physics Signal | Mass R² | Fric R² | Rest R² | LIBERO-10 |
|---|---|---|---|---|---|
| DINOv2 frozen | None | 0.370 | 0.132 | 0.544 | 51.5% |
| Random physics | Random trajectories | ? | ? | ? | ? |
| Mass only | Mass-driven trajectories | ? | ? | ? | ? |
| Friction only | Friction-driven trajectories | ? | ? | ? | ? |
| Restitution only | Restitution-driven trajectories | ? | ? | ? | ? |
| **DynaCLIP V3** | **Full trajectories** | **?** | **?** | **?** | **?** |

---

## Important Notes

1. **Dataset root**: Make sure `<YOUR_DATASET_ROOT>` contains `domainnet/real/` with the 345 category subdirectories. The generation script expects this structure.

2. **Trajectory computation**: The updated `dataset.py` uses a three-level trajectory lookup: (1) in-memory cache (instant), (2) precomputed `.npz` fingerprint files on disk (fast), (3) on-the-fly computation via the physics engine (~10ms per entry). After the first pass through all entries, trajectories are fully cached in memory. The `similarity_matrix.npz` provides precomputed pairwise similarities for 500K pairs; uncovered pairs compute trajectory similarity on the fly. If dataset initialization is too slow, increase `max_sim_pairs` in `DynaCLIPDataGenerator.generate_all()` to 1M or 2M.

3. **V3 vs V2 comparison**: If V3 probing is better but LIBERO is slightly worse (e.g., 58% vs 60.4%), this is actually fine. We can report both and discuss trajectory vs parameter distance as an ablation. The key requirement is that V3 LIBERO must be clearly above random physics baseline.

4. **Baseline numbers**: Do NOT rerun baselines (DINOv2, CLIP, SigLIP, etc.) — their numbers from V2 evaluation are still valid since those are frozen encoders evaluated on the same benchmarks. Only DynaCLIP numbers need updating with the V3 checkpoint.

5. **Report per-seed results**: For LIBERO-10, save per-seed breakdowns (Seed 0, 1, 2) for the V3 model AND the ablation variants. We need these for error bars and statistical significance.
