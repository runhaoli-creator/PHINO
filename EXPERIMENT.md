# IntPhys Benchmark Evaluation — Experiment Instructions

## Goal

Evaluate whether physics-aligned visual encoders (DynaCLIP V2 and V3) can
detect violations of intuitive physics better than frozen baselines (DINOv2, CLIP).
This uses the IntPhys benchmark (Riochet et al., 2022), the same benchmark used by
V-JEPA (Garrido et al., 2025) to demonstrate emergent physics understanding.

We test **both** V2 (parameter-distance fine-tuning) and V3 (trajectory-similarity
fine-tuning) to compare which type of physics alignment better transfers to
intuitive physics reasoning.

---

## Background: How IntPhys Evaluation Works

IntPhys presents pairs of videos: one physically **plausible** (ball rolls normally),
one **implausible** (ball vanishes behind occluder, teleports, or changes shape).

For V-JEPA (a video prediction model), "surprise" is measured as prediction error
in representation space — higher surprise on implausible videos = physics understanding.

For **static image encoders** like DynaCLIP, we adapt this by encoding each frame
independently and measuring **temporal embedding discontinuity**:
- Plausible videos → smooth embedding trajectories
- Implausible videos → sudden jumps in embedding space (object vanishing = embedding discontinuity)

If a physics-aligned encoder has learned to encode physical properties, it should
produce more distinct embeddings for physically different states, making violations
(object disappearing, changing shape) more detectable as embedding jumps.

Three surprise metrics are computed:
1. **embedding_diff**: Total squared L2 distance between consecutive frame embeddings
2. **max_jump**: Largest single frame-to-frame embedding discontinuity
3. **prediction_error**: Fit linear predictor (emb[t] → emb[t+1]) on plausible videos, measure error on all videos

Pairwise accuracy: for each (plausible, implausible) pair, check if the implausible
video has higher surprise. Random chance = 50%.

---

## Step 0: Prerequisites

### Install dependencies
```bash
pip install decord  # For video decoding (if videos are .mp4)
# Other deps (torch, torchvision, sklearn, numpy) should already be installed
```

### Get IntPhys data

**Option A — IntPhys dev set (recommended):**

The IntPhys dev set is publicly available. Download from the IntPhys challenge:
```bash
# Check if available at:
# https://intphys.cognitive-ml.fr
# or via the jepa-intuitive-physics repo:
git clone https://github.com/facebookresearch/jepa-intuitive-physics.git
cd jepa-intuitive-physics
tar -xzvf data_intphys.tar.gz
```

Note: The `jepa-intuitive-physics` repo contains pre-computed surprise values from
V-JEPA, but we need the raw video frames. If the dev set videos are not included,
download them separately from the IntPhys challenge website.

**Option B — IntPhys 2 (newer version):**

IntPhys 2 (Bordes et al., 2025, arXiv:2506.09849) may have updated data.
Check: https://github.com/facebookresearch/intphys2

**Option C — Generate IntPhys-style test data:**

If the original IntPhys data is hard to obtain, we can generate similar
violation-of-expectation videos using our analytical physics engine. See the
"Fallback: Synthetic VoE Test" section at the bottom.

### Required checkpoints

You need these checkpoint files:
- `checkpoints/pretrain_v2/dynaclip_final.pt` — V2 model (parameter distance)
- `checkpoints/pretrain_v3/dynaclip_final.pt` — V3 model (trajectory similarity)

If V3 checkpoint doesn't exist yet, train it first following `AGENT_INSTRUCTIONS.md`.
The V3 ablation model with WiSE-FT (`checkpoints/ablation_v3_wiseft/dynaclip_final.pt`)
can also be used.

---

## Step 1: Run Evaluation

### Full evaluation (all backbones)

```bash
python scripts/evaluate_intphys.py \
    --data_dir /path/to/intphys/dev \
    --checkpoints \
        checkpoints/pretrain_v2/dynaclip_final.pt \
        checkpoints/pretrain_v3/dynaclip_final.pt \
    --checkpoint_names DynaCLIP-V2 DynaCLIP-V3 \
    --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16 \
    --output_dir results/intphys \
    --frame_stride 2 \
    --max_frames 100 \
    --batch_size 32
```

### Quick test (just V2 + DINOv2)

```bash
python scripts/evaluate_intphys.py \
    --data_dir /path/to/intphys/dev \
    --checkpoints checkpoints/pretrain_v2/dynaclip_final.pt \
    --checkpoint_names DynaCLIP-V2 \
    --baselines dinov2_vitb14 \
    --output_dir results/intphys_quick \
    --frame_stride 4 \
    --max_frames 50
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--data_dir` | Path to IntPhys video data | Required |
| `--checkpoints` | DynaCLIP checkpoint paths | [] |
| `--checkpoint_names` | Display names for checkpoints | auto |
| `--baselines` | Frozen baseline names | dinov2_vitb14, dinov2_vitl14 |
| `--frame_stride` | Sample every N-th frame | 2 |
| `--max_frames` | Max frames per video | 100 |
| `--batch_size` | Frames per GPU batch | 32 |
| `--no_predictor` | Skip linear predictor training | false |

---

## Step 2: Interpret Results

### Expected output

The script prints a summary table:

```
================================================================================
IntPhys Pairwise Classification Accuracy
================================================================================
Backbone                  Best Metric          Overall Acc
------------------------------------------------------------
DynaCLIP-V2               prediction_error          ??%
DynaCLIP-V3               prediction_error          ??%
dinov2_vitb14             embedding_diff             ??%
dinov2_vitl14             embedding_diff             ??%
================================================================================
```

### What we hope to see

1. **DynaCLIP-V2 > DINOv2-B/14**: Physics alignment improves violation detection
2. **DynaCLIP-V2 and/or V3 > 50%**: Above chance = some physics understanding
3. **V2 vs V3 comparison**: Which fine-tuning strategy better transfers to intuitive physics

### What would be a strong result

- V-JEPA gets 98% on IntPhys (but it's a video prediction model trained on massive video data)
- If DynaCLIP (static image encoder, trained on DomainNet) gets **60-70%**, that's noteworthy
- Even **55%** with clear improvement over frozen DINOv2 is publishable

### Results are saved to

```
results/intphys/
├── intphys_results.json    # Full results with per-property breakdown
```

---

## Step 3: Additional Analyses (Optional)

### Per-property breakdown

The results JSON contains accuracy per physics property (object_permanence,
continuity, shape_constancy). Report these in the paper:

```python
import json
with open("results/intphys/intphys_results.json") as f:
    results = json.load(f)

for backbone, data in results.items():
    print(f"\n{backbone} (best metric: {data['best_metric']})")
    best = data["all_metrics"][data["best_metric"]]
    for prop, prop_data in best.items():
        if prop != "overall" and isinstance(prop_data, dict):
            print(f"  {prop}: {prop_data['accuracy']:.1%}")
```

### Sensitivity to frame_stride

Run with stride 1, 2, 4, 8 to check robustness:

```bash
for stride in 1 2 4 8; do
    python scripts/evaluate_intphys.py \
        --data_dir /path/to/intphys/dev \
        --checkpoints checkpoints/pretrain_v2/dynaclip_final.pt \
        --checkpoint_names DynaCLIP-V2 \
        --baselines dinov2_vitb14 \
        --output_dir results/intphys_stride${stride} \
        --frame_stride ${stride}
done
```

---

## Fallback: Synthetic VoE Test (if IntPhys data unavailable)

If the original IntPhys dev set is difficult to obtain, we can create a synthetic
violation-of-expectation test using our analytical physics engine + a renderer:

1. **Plausible videos**: Render a ball with correct physics (metal ball bounces high,
   fabric ball doesn't bounce)
2. **Implausible videos**: Same visual appearance but wrong physics (metal ball
   doesn't bounce, fabric ball bounces like rubber)

This is simpler than IntPhys but tests the same principle: does the encoder detect
that something is physically wrong?

To implement this, generate trajectory pairs using `dynaclip/data/generation.py`:
```python
from dynaclip.data.generation import AnalyticalPhysicsEngine, PhysicsConfig, DIAGNOSTIC_ACTIONS

engine = AnalyticalPhysicsEngine()

# Plausible: metal ball with metal physics
metal = PhysicsConfig(mass=5.0, static_friction=0.2, restitution=0.8)
traj_plausible = engine.execute_diagnostic_action(DIAGNOSTIC_ACTIONS[2], metal)  # grasp-lift-release

# Implausible: metal ball with fabric physics (shouldn't bounce)
fabric = PhysicsConfig(mass=0.1, static_friction=0.8, restitution=0.05)
traj_implausible = engine.execute_diagnostic_action(DIAGNOSTIC_ACTIONS[2], fabric)
```

Then render frames using a simple renderer (matplotlib 3D plot or PyBullet visual)
and run the same evaluation pipeline.

---

## Notes

1. **GPU memory**: Each backbone is loaded one at a time and freed after evaluation.
   A single GPU with 24GB VRAM should suffice (batch_size=32 with ViT-B/14).

2. **Runtime**: Encoding ~300 videos × 50 frames = ~15K frames per backbone.
   At ~1K frames/sec on a single GPU, each backbone takes ~15 seconds.
   Total for 6 backbones: ~2 minutes. The predictor training adds ~30 seconds.

3. **DINOv2 feature extraction**: The script extracts CLS + mean-pooled patch
   tokens (1536-d) for DINOv2 models, matching DynaCLIP's feature format.
   For CLIP and SigLIP, it uses the model's native output format.

4. **V2 vs V3 hypothesis**: V2 (parameter distance) may excel on IntPhys because
   its embedding space preserves per-dimension physics information that helps
   detect specific violations. V3 (trajectory similarity) might excel if the
   evaluation depends on dynamic trajectory prediction (which V3's training
   signal captures better). We test both to see which transfers better.
