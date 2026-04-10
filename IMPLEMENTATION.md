# PHINO: Implementation Guide

> **PHINO** = **PH**ysical D**INO**v2
>
> Teaching visual encoders physics through simple simulated interactions.

---

## Core Idea

Human infants learn about object physics not by reading labels, but by pushing, lifting, and dropping things. PHINO does the same: we simulate simple physical interactions (push, lift, drop, flick, press) for each object using Newtonian mechanics, compute how similarly two objects behave under the same forces, and use that behavioral similarity to fine-tune DINOv2 via contrastive learning. The result is a visual encoder that knows not just *what* objects are, but *how they would physically behave*.

---

## Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PHINO TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

 ┌──────────────┐
 │  DomainNet   │  20,000 real images, 345 object categories
 │  Images      │  (hammer, pillow, basketball, cup, ...)
 └──────┬───────┘
        │
        ▼
 ┌──────────────┐     ┌──────────────────────────────────┐
 │  Category    │────▶│  Material Prior Lookup            │
 │  "hammer"    │     │                                    │
 └──────────────┘     │  hammer → metal (50%)              │
                      │           rubber_plastic (25%)      │
                      │           glass_ceramic (12.5%)     │
                      │           wood (12.5%)              │
                      └──────────────┬───────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────┐
                      │  Sample Physics Parameters        │
                      │  × 5 per image                    │
                      │                                    │
                      │  If metal:                         │
                      │    mass ∈ [2.0, 8.0] kg            │
                      │    friction ∈ [0.1, 0.4]           │
                      │    restitution ∈ [0.5, 0.9]        │
                      └──────────────┬───────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────┐
                      │  Analytical Physics Engine        │
                      │  (Pure Newtonian Mechanics)        │
                      │                                    │
                      │  For each (mass, μ, e):            │
                      │                                    │
                      │  ┌─────────────────────────────┐   │
                      │  │ Action 1: Horizontal Push    │   │
                      │  │   F=ma, friction=μmg         │   │
                      │  │   → 50 frames of sliding     │   │
                      │  ├─────────────────────────────┤   │
                      │  │ Action 2: Vertical Push      │   │
                      │  │   Ballistic trajectory       │   │
                      │  │   → 50 frames of arc         │   │
                      │  ├─────────────────────────────┤   │
                      │  │ Action 3: Lift & Release     │   │
                      │  │   Drop + bounce (h₁=e²h)    │   │
                      │  │   → 50 frames of bouncing    │   │
                      │  ├─────────────────────────────┤   │
                      │  │ Action 4: Lateral Flick      │   │
                      │  │   High impulse sliding       │   │
                      │  │   → 50 frames of deceleration│   │
                      │  ├─────────────────────────────┤   │
                      │  │ Action 5: Slow Press Down    │   │
                      │  │   Static equilibrium test    │   │
                      │  │   → 50 frames               │   │
                      │  └─────────────────────────────┘   │
                      │                                    │
                      │  Output: 250 × 13 trajectory       │
                      │  (position, rotation, velocity,     │
                      │   angular velocity per frame)       │
                      └──────────────┬───────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────┐
                      │  Trajectory Similarity            │
                      │                                    │
                      │  For pair (i, j):                   │
                      │                                    │
                      │  MSE = mean((Tᵢ - Tⱼ)²)           │
                      │         over all 250×13 elements   │
                      │                                    │
                      │  similarity = exp(-MSE / σ)        │
                      │            ∈ [0, 1]                │
                      │                                    │
                      │  Similar physics → similar traj    │
                      │  → high similarity score           │
                      └──────────────┬───────────────────┘
                                     │
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                      CONTRASTIVE FINE-TUNING                         │
 │                                                                       │
 │   Batch of 1024 image pairs, each with trajectory similarity score   │
 │                                                                       │
 │   ┌─────────┐          ┌──────────────┐         ┌──────────┐        │
 │   │ Image A │───────▶ │  DINOv2      │───────▶ │ Proj.    │──▶ zₐ  │
 │   │ (224²)  │         │  ViT-B/14    │  1536d  │ Head     │  512d  │
 │   └─────────┘         │  (UNFROZEN)  │         │ (discard │        │
 │                        │              │         │  after)  │        │
 │   ┌─────────┐         │              │         │          │        │
 │   │ Image B │───────▶ │              │───────▶ │          │──▶ z_b │
 │   │ (224²)  │         └──────────────┘         └──────────┘        │
 │                                                                       │
 │   Loss = SoftInfoNCE(zₐ, z_b, trajectory_similarity)                │
 │     → Pull together images whose objects would behave similarly      │
 │     → Push apart images whose objects would behave differently       │
 │                                                                       │
 │   + WiSE-FT regularization:                                          │
 │     L_reg = α · ||features - frozen_DINOv2_features||²               │
 │     (prevents backbone from drifting too far from pretrained)        │
 │                                                                       │
 │   Training: AdamW, backbone LR=1e-5, head LR=1e-3                   │
 │             20K steps, cosine warmup, bf16, gradient checkpointing   │
 └───────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │                         DEPLOYMENT                                    │
 │                                                                       │
 │   Discard projection head. Keep fine-tuned DINOv2 backbone.          │
 │                                                                       │
 │   Image → DINOv2 (fine-tuned) → [CLS ∥ mean_patch] → 1536d features │
 │                                                                       │
 │   Same latency as original DINOv2 (5.6ms, 873 img/s)                │
 │   But features now carry physics information                         │
 │                                                                       │
 │   Plug into any downstream task:                                     │
 │     • BC policy for manipulation (LIBERO: 62%)                       │
 │     • Linear probe for physics prediction                            │
 │     • Violation-of-expectation detection (IntPhys2: 63%)             │
 └───────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Material Priors (`dynaclip/data/generation.py`)

345 DomainNet categories → 10 material archetypes → physics parameter ranges.

```
Material        Mass (kg)      Friction (μ)    Restitution (e)
─────────────────────────────────────────────────────────────
Metal           [2.0, 8.0]     [0.10, 0.40]    [0.50, 0.90]
Wood            [0.5, 3.0]     [0.30, 0.60]    [0.30, 0.60]
Fabric          [0.05, 0.50]   [0.60, 0.95]    [0.10, 0.30]
Glass/Ceramic   [0.3, 3.0]     [0.20, 0.50]    [0.50, 0.80]
Rubber/Plastic  [0.1, 2.0]     [0.40, 0.80]    [0.40, 0.80]
Food/Organic    [0.1, 2.0]     [0.30, 0.70]    [0.10, 0.40]
Paper/Light     [0.01, 0.30]   [0.40, 0.70]    [0.10, 0.30]
Animal          [0.1, 5.0]     [0.30, 0.60]    [0.20, 0.50]
Stone/Heavy     [2.0, 15.0]    [0.30, 0.70]    [0.10, 0.30]
Default         [0.1, 5.0]     [0.20, 0.80]    [0.10, 0.90]
```

Probabilistic multi-material mapping prevents trivial category→physics shortcut:
each category maps to primary material (50%) + 2-3 alternatives (50%).

### 2. Analytical Physics Engine (`dynaclip/data/generation.py`)

Three equations from classical mechanics:

```
Newton's 2nd law:    a = F_net / m
Coulomb friction:    |F_friction| ≤ μ_s · mg    (static)
                     F_kinetic = -μ_k · mg · v̂  (kinetic, μ_k = 0.8μ_s)
Restitution:         v_after = -e · v_before     (ground collision)
```

Five diagnostic actions probe complementary physics:

| Action | What it tests | Key signal |
|---|---|---|
| Horizontal push | Mass + friction | Sliding distance & deceleration |
| Vertical push | Mass | Ballistic arc height |
| Lift & release | Mass + restitution | Bounce pattern |
| Lateral flick | Friction | High-speed sliding deceleration |
| Slow press | Mass | Static equilibrium response |

Each action: 50 timesteps × 13 dimensions = 650 values.
Five actions concatenated: 250 × 13 = 3,250 dimensional trajectory fingerprint.

### 3. Trajectory Similarity (`dynaclip/data/generation.py`)

```python
def compute_dynamics_similarity_l2(traj1, traj2):
    mse = np.mean((traj1 - traj2) ** 2)   # Full 250×13 trajectory
    return exp(-mse / 5.0)                  # → [0, 1]
```

Uses full trajectory MSE (not just last frame). This preserves:
- Friction signal: sliding deceleration in push/flick (frames 1-30)
- Restitution signal: bounce patterns in lift-release (frames 10-40)
- Mass signal: acceleration magnitude across all actions

### 4. Contrastive Loss (`dynaclip/losses/contrastive.py`)

```
SoftInfoNCE:
  logits[i,j] = (z_a[i] · z_b[j]) / τ        # B×B cosine similarity
  soft_labels[i,i] = trajectory_similarity[i]   # diagonal = pair similarity
  soft_labels[i,j] = 0  for i≠j                 # off-diagonal = 0
  
  loss = -Σ softmax(labels) · log_softmax(logits)
  
  τ = learnable temperature (init 0.07)
```

High-similarity pairs contribute more to the loss. Low-similarity pairs act as negatives.

### 5. WiSE-FT Regularization (`dynaclip/trainers/pretrain.py`)

```python
frozen_features = frozen_dinov2(images)   # Original DINOv2 (frozen copy)
live_features = model(images)              # Fine-tuned DINOv2
reg_loss = α * MSE(live_features, frozen_features)   # α = 0.1
total_loss = contrastive_loss + reg_loss
```

Prevents backbone from drifting too far, preserving semantic structure while adding physics.

### 6. Model Architecture (`dynaclip/models/dynaclip.py`)

```
Input Image (224×224)
        │
        ▼
┌──────────────────────┐
│  DINOv2-ViT-B/14     │   86M params, UNFROZEN during training
│  (Backbone)           │
└──────────┬───────────┘
           │
     ┌─────┴─────┐
     │           │
  CLS Token   Mean-Pool
  (768-d)    (768-d)
     │           │
     └─────┬─────┘
           │
    Concatenate → 1536-d     ← This is the output feature for downstream
           │
           ▼
┌──────────────────────┐
│  Projection Head      │   DISCARDED after training
│  Linear(1536, 768)    │
│  LayerNorm → GELU     │
│  Linear(768, 512)     │
│  L2 Normalize         │
└──────────┬───────────┘
           │
           ▼
    512-d Unit-Norm Embedding  ← Used only during contrastive training
```

---

## How to Run

### Step 1: Generate Data
```bash
python scripts/generate_v3_ablation_data.py \
    --variant full \
    --output_dir data_cache/phino_data \
    --dataset_root datasets \
    --max_images 20000 --num_physics 5
```

### Step 2: Train
```bash
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    data.data_dir=data_cache/phino_data \
    output.checkpoint_dir=checkpoints/phino
```

### Step 3: Evaluate
```bash
# Linear probing
python scripts/evaluate.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --data_dir data_cache/phino_data \
    --experiments linear_probing

# LIBERO-10
python scripts/evaluate_libero_v4.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --seeds 3 --epochs 200

# IntPhys2
python scripts/evaluate_intphys.py \
    --data_dir datasets/intphys2 \
    --checkpoints checkpoints/phino/dynaclip_final.pt \
    --checkpoint_names PHINO \
    --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16
```

---

## What Makes PHINO Different

| Aspect | Other methods | PHINO |
|---|---|---|
| Physics source | Video data, simulators, human annotation | Analytical equations (F=ma) |
| Training data | Videos, robot demonstrations | Static images only |
| Compute cost | Hours/days of simulation | Minutes (numpy) |
| Deployment overhead | Additional heads/modules | Zero (same as DINOv2) |
| Physics signal | Implicit (learned from data) | Explicit (trajectory similarity) |

---

## Bug Fixes Applied (vs original DynaCLIP)

1. **Trajectory similarity**: Changed from last-frame-only to full-trajectory MSE
2. **Soft labels**: Fixed off-diagonal from meaningless geometric mean to zero
3. **Fingerprint paths**: Added data_dir-relative fallback for portability
4. **WiSE-FT dtype**: Fixed f32/bf16 mismatch in regularization loss
5. **Pair mining**: Added fast path using precomputed similarity when no hard mining
