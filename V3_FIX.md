# V3 Fix: Trajectory Similarity Bug + Correct Probing Evaluation

## Bug: `compute_dynamics_similarity_l2` only uses last frame

**File**: `dynaclip/data/generation.py`, line 474-476

**Current code** (broken):
```python
def compute_dynamics_similarity_l2(traj1: np.ndarray, traj2: np.ndarray) -> float:
    diff = np.linalg.norm(traj1[-1] - traj2[-1])   # ONLY last frame!
    return float(np.exp(-diff / 5.0))
```

**Problem**: Trajectories are 250×13 (5 actions × 50 timesteps × 13-dim state), but
this function compares only the final frame (frame 250). All mid-trajectory information
is discarded:
- Friction signal (sliding deceleration) — happens in frames 1-30, gone by last frame
- Restitution signal (bounce pattern) — happens in frames 10-40, gone by last frame
- Only mass signal partially survives (final resting position correlates with mass)

This explains why V3 friction R²≈0 and restitution R²≈0.2 while mass R²≈0.18.

**Fix**: Change to use full trajectory MSE:
```python
def compute_dynamics_similarity_l2(traj1: np.ndarray, traj2: np.ndarray) -> float:
    mse = np.mean((traj1 - traj2) ** 2)
    return float(np.exp(-mse / 5.0))
```

Note: `compute_dynamics_similarity_mse` (line 478-480) already does exactly this.
You can either fix `_l2` or switch all callers to use `_mse`.

This function is used in two places:
1. `DynaCLIPDataGenerator._compute_similarity_matrix()` — precomputes `similarity_matrix.npz`
2. `DynaCLIPContrastiveDataset._compute_proxy_similarity()` — fallback when pair not in precomputed matrix

Both need the fix.

## After fixing: regenerate data + retrain V3

The fix changes the similarity values, so existing data must be regenerated.

### Step 1: Regenerate V3 training data
```bash
python scripts/generate_v3_ablation_data.py --variant full \
    --output_dir data_cache/dynaclip_v3_fixed \
    --dataset_root <YOUR_DATASET_ROOT> \
    --max_images 20000 --num_physics 5
```

### Step 2: Retrain V3
```bash
torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    data.data_dir=data_cache/dynaclip_v3_fixed \
    output.checkpoint_dir=checkpoints/pretrain_v3_fixed
```

### Step 3: Run linear probing correctly

**Critical**: Evaluate on the SAME data the model was trained on (V3's own data),
not V2's data. The previous V3 probing (mass 0.176) was evaluated on V2's evaluation
data, which uses different physics distributions (multi-material mapping with different
parameter ranges). This makes the comparison unfair.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/pretrain_v3_fixed/dynaclip_final.pt \
    --data_dir data_cache/dynaclip_v3_fixed \
    --experiments linear_probing clustering
```

For fair comparison, also evaluate ALL baselines (DINOv2, CLIP, etc.) on the same
V3 evaluation data:

```bash
python scripts/evaluate.py \
    --data_dir data_cache/dynaclip_v3_fixed \
    --experiments linear_probing clustering
    # This should run all registered baselines on V3's data
```

### Step 4: Verify improvement

Expected results after fix:
- Mass R²: should be >0.5 (was 0.176 with last-frame-only bug)
- Friction R²: should be >0.3 (was 0.007 — friction signal was completely lost)
- Restitution R²: should be >0.5 (was 0.227 — bounce signal was mostly lost)

If the numbers are still low after the fix, check:
1. Is the fixed `compute_dynamics_similarity_l2` actually being called? Add a print
   to confirm.
2. Are precomputed fingerprints (`similarity_matrix.npz`) regenerated? Old cached
   files use the broken similarity.
3. Is the evaluation data using the regenerated data, not the old data?

### Step 5: Rerun IntPhys + LIBERO with fixed V3

After confirming probing improvement:
```bash
# IntPhys
python scripts/evaluate_intphys.py \
    --data_dir /path/to/intphys2 \
    --checkpoints checkpoints/pretrain_v3_fixed/dynaclip_final.pt \
    --checkpoint_names DynaCLIP-V3-fixed \
    --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16 \
    --output_dir results/intphys_v3_fixed

# LIBERO-10
python scripts/evaluate_libero_v4.py \
    --checkpoint checkpoints/pretrain_v3_fixed/dynaclip_final.pt \
    --seeds 3 --epochs 200
```
