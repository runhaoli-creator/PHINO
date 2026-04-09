# PHINO: Complete Setup, Training & Evaluation Guide

This repo is a bug-fixed version of DynaCLIP V3 (trajectory-based physics contrastive
fine-tuning of DINOv2). This document covers everything needed to run the full pipeline
on a fresh machine.

---

## 0. Environment Setup

```bash
git clone https://github.com/runhaoli-creator/PHINO.git
cd PHINO

# Create environment
conda create -n phino python=3.10 -y
conda activate phino

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project
pip install -e .

# Additional dependencies
pip install decord          # For IntPhys video evaluation
pip install tslearn         # Optional: DTW trajectory similarity
```

---

## 1. Prepare Source Dataset

The training pipeline needs DomainNet real-domain images (345 object categories).

### Download DomainNet
```bash
mkdir -p datasets/domainnet/real
cd datasets/domainnet
# Download the "real" split from DomainNet:
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
unzip real.zip
cd ../..
```

Verify the structure:
```
datasets/
└── domainnet/
    └── real/
        ├── aircraft_carrier/
        │   ├── real_000001.jpg
        │   └── ...
        ├── airplane/
        ├── alarm_clock/
        └── ... (345 categories)
```

---

## 2. Generate Training Data

Generate physics-grounded training data with trajectory similarities.
Each image gets 5 physics configurations sampled from category-grounded material
priors. The analytical physics engine computes trajectories for 5 diagnostic
actions (push, lift, flick, press), and pairwise trajectory MSE is used as
the contrastive similarity label.

### Main training data (V3 full)
```bash
python scripts/generate_v3_ablation_data.py \
    --variant full \
    --output_dir data_cache/phino_data \
    --dataset_root datasets \
    --max_images 20000 \
    --num_physics 5 \
    --seed 42
```

This produces:
```
data_cache/phino_data/
├── metadata.json              # 100K entries (20K images × 5 physics configs)
├── fingerprints/              # 100K .npz files with trajectory data
├── similarity_matrix.npz      # 500K precomputed pairwise similarities
├── invisible_physics_test.json
└── cross_material_test.json
```

**Time**: ~30-60 minutes on CPU. Can run in parallel with other setup tasks.

### Ablation data (for ablation experiments)
```bash
# Random physics (no category correlation) — control group
python scripts/generate_v3_ablation_data.py \
    --variant random \
    --output_dir data_cache/phino_random \
    --dataset_root datasets \
    --max_images 20000

# Mass only (friction & restitution fixed to material midpoint)
python scripts/generate_v3_ablation_data.py \
    --variant mass_only \
    --output_dir data_cache/phino_mass_only \
    --dataset_root datasets \
    --max_images 20000

# Friction only
python scripts/generate_v3_ablation_data.py \
    --variant fric_only \
    --output_dir data_cache/phino_fric_only \
    --dataset_root datasets \
    --max_images 20000

# Restitution only
python scripts/generate_v3_ablation_data.py \
    --variant rest_only \
    --output_dir data_cache/phino_rest_only \
    --dataset_root datasets \
    --max_images 20000
```

---

## 3. Train PHINO (V3 Fixed)

### Single GPU
```bash
python scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    data.data_dir=data_cache/phino_data \
    output.checkpoint_dir=checkpoints/phino
```

### Multi-GPU (recommended)
```bash
torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    data.data_dir=data_cache/phino_data \
    output.checkpoint_dir=checkpoints/phino
```

Adjust `training.batch_size` if needed. Default is 1024 (total across all GPUs).
With 4 GPUs: 256 per GPU. With 8 GPUs: 128 per GPU.

**Training takes ~2-3 hours on 4-8 modern GPUs (H100/A100/RTX 4090).**

Checkpoint saved to: `checkpoints/phino/dynaclip_final.pt`

### Train ablation variants (10K steps each)
```bash
for variant in random mass_only fric_only rest_only; do
    torchrun --nproc_per_node=<NUM_GPUS> scripts/pretrain.py \
        --config configs/pretrain_v3.yaml \
        --seed 42 \
        training.total_steps=10000 \
        data.data_dir=data_cache/phino_${variant} \
        output.checkpoint_dir=checkpoints/ablation_${variant}
done
```

These can run in parallel if you have enough GPUs.

---

## 4. Evaluate — Linear Probing

Test whether the fine-tuned encoder can predict physics properties from
frozen features.

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --data_dir data_cache/phino_data \
    --experiments linear_probing clustering
```

**IMPORTANT**: Always evaluate on the SAME data the model was trained on
(or data generated with the same pipeline). Evaluating on data from a
different generation run (different material mappings, different similarity
computation) will produce misleadingly low numbers.

For fair baseline comparison, run ALL baselines on the same evaluation data:
```bash
python scripts/evaluate.py \
    --data_dir data_cache/phino_data \
    --experiments linear_probing clustering
```

This evaluates DINOv2-B/14, DINOv2-L/14, CLIP-L/14, SigLIP, R3M, VIP, etc.
on the same data split, ensuring apples-to-apples comparison.

### Ablation probing
```bash
for variant in random mass_only fric_only rest_only; do
    python scripts/evaluate.py \
        --checkpoint checkpoints/ablation_${variant}/dynaclip_final.pt \
        --data_dir data_cache/phino_${variant} \
        --experiments linear_probing
done
```

### Expected results

| Model | Mass R² | Fric R² | Rest R² |
|---|---|---|---|
| DINOv2-B/14 (frozen) | ~0.37 | ~0.13 | ~0.54 |
| PHINO (V3 fixed) | **>0.50** | **>0.30** | **>0.60** |
| Random physics | <0 | <0 | <0 |
| Mass only | ~0.5 | ~0 | ~0.2 |
| Friction only | ~0.1 | ~0.3 | ~0.2 |
| Restitution only | ~0.1 | ~0 | ~0.5 |

If PHINO mass R² is below 0.40, something is wrong — check that:
1. Data was generated with the fixed `compute_dynamics_similarity_l2` (full MSE, not last frame)
2. Evaluation data matches training data
3. The `similarity_matrix.npz` was regenerated (not cached from old run)

---

## 5. Evaluate — IntPhys Benchmark

Violation-of-expectation test: can the encoder detect when physics is violated
in a video (object vanishing, teleporting, passing through walls)?

### Get IntPhys2 data
```bash
wget https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip
unzip IntPhys2.zip -d datasets/intphys2
```

### Run evaluation
```bash
python scripts/evaluate_intphys.py \
    --data_dir datasets/intphys2 \
    --split Main \
    --checkpoints checkpoints/phino/dynaclip_final.pt \
    --checkpoint_names PHINO \
    --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16 \
    --output_dir results/intphys \
    --frame_stride 2 \
    --max_frames 100 \
    --batch_size 32
```

Results saved to `results/intphys/intphys_results.json`.

---

## 6. Evaluate — LIBERO-10

Downstream robotic manipulation with frozen encoder + LSTM BC policy.

### Setup LIBERO
```bash
pip install libero
# Or follow: https://github.com/Lifelong-Robot-Learning/LIBERO
```

### Run evaluation
```bash
python scripts/evaluate_libero_v4.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --seeds 3 \
    --epochs 200
```

This trains a BC policy on each of 10 LIBERO tasks with the frozen PHINO encoder,
3 seeds × 50 evaluation episodes. Takes ~5-6 hours on GPU.

Run baselines with the same script (omit --checkpoint to use frozen DINOv2/CLIP/etc).

---

## 7. Evaluate — DROID-100 (Optional)

Offline action prediction on real-world manipulation data.

```bash
python scripts/evaluate_droid.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt
```

---

## Summary: What to Run and In What Order

```
Step 1: Environment setup                    (~10 min)
Step 2: Download DomainNet                   (~10 min)
Step 3: Generate training data               (~30-60 min, CPU)
Step 4: Train PHINO                          (~2-3 hours, GPU)
Step 5: Linear probing + clustering          (~1 hour)
        → Check: mass R² > 0.5?
Step 6: Download IntPhys2 + evaluate         (~30 min)
Step 7: LIBERO-10 evaluation                 (~5-6 hours, GPU)
Step 8: (Optional) Ablation variants         (~4 hours total)
Step 9: (Optional) DROID-100                 (~2 hours)
```

Total: ~1 day for full pipeline on a multi-GPU machine.

---

## Key Files

| File | Purpose |
|---|---|
| `configs/pretrain_v3.yaml` | Training config (SoftInfoNCE + WiSE-FT + no hard mining) |
| `dynaclip/data/generation.py` | Physics engine + data generation |
| `dynaclip/data/dataset.py` | Contrastive dataset with trajectory similarity |
| `dynaclip/losses/contrastive.py` | SoftInfoNCE loss (fixed soft labels) |
| `dynaclip/trainers/pretrain.py` | Training loop (fixed WiSE-FT dtype) |
| `dynaclip/models/dynaclip.py` | DynaCLIP model (DINOv2 + projection head) |
| `scripts/pretrain.py` | Training entry point |
| `scripts/generate_v3_ablation_data.py` | Data generation for all variants |
| `scripts/evaluate.py` | Probing + clustering evaluation |
| `scripts/evaluate_intphys.py` | IntPhys benchmark evaluation |
| `scripts/evaluate_libero_v4.py` | LIBERO-10 evaluation |

## Bug Fixes Applied (vs DynaCLIP repo)

1. `generation.py`: `compute_dynamics_similarity_l2` uses full trajectory MSE (was: last frame only)
2. `contrastive.py`: Off-diagonal soft labels set to 0 (was: meaningless geometric mean)
3. `dataset.py`: fingerprint_path falls back to data_dir-relative (was: absolute path only)
4. `pretrain.py`: WiSE-FT frozen features cast to AMP dtype (was: f32/bf16 mismatch)
