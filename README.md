# DynaCLIP: Physics-Grounded Visual Representations for Robotic Manipulation via Dynamics Contrastive Learning

## Overview

DynaCLIP learns visual representations where embedding similarity reflects **physical dynamics similarity** rather than mere visual similarity. Two objects that look identical but have different masses, friction coefficients, or restitution values will receive distinct embeddings—because they *behave* differently when a robot interacts with them.

### Key Idea

```
Traditional CLIP:  "looks similar" → close embeddings
DynaCLIP:          "behaves similarly" → close embeddings
```

We fine-tune a DINOv2-ViT-B/14 backbone with a **Soft InfoNCE** contrastive loss where the soft targets come from **dynamics fingerprint similarity** (computed via DTW over diagnostic manipulation trajectories in ManiSkill3).

---

## Project Structure

```
DynaCLIP/
├── dynaclip/                    # Main Python package
│   ├── data/
│   │   ├── generation.py        # ManiSkill3 data generation, physics configs, dynamics fingerprints
│   │   ├── dataset.py           # PyTorch datasets (contrastive, invisible physics, probe)
│   │   └── precompute.py        # DINOv2 embedding pre-computation & hard pair mining
│   ├── models/
│   │   ├── dynaclip.py          # DynaCLIP model (DINOv2 + projection head)
│   │   └── backbones.py         # 8 backbone registry (DINOv2-B/L, SigLIP, CLIP, R3M, VIP, MCR)
│   ├── losses/
│   │   └── contrastive.py       # Soft InfoNCE, Standard InfoNCE, Triplet, BYOL losses
│   ├── trainers/
│   │   ├── pretrain.py          # DynaCLIP pre-training (DDP, bf16, cosine warmup)
│   │   └── diffusion_policy.py  # Diffusion Policy for downstream tasks
│   ├── eval/
│   │   ├── linear_probing.py    # Exp 1: Physics property linear probing
│   │   ├── invisible_physics.py # Exp 2: Visually-identical pair discrimination
│   │   ├── world_model.py       # Exp 3: RSSM world model prediction
│   │   ├── downstream.py        # Exp 4: 6 manipulation benchmarks
│   │   ├── zero_shot.py         # Exp 5: k-NN physics inference
│   │   └── ablations.py         # 8 ablation studies
│   ├── baselines/
│   │   └── policies.py          # ACT, OpenVLA-OFT, Octo-Base, Dreamer-v3, TD-MPC2
│   ├── analysis/
│   │   └── visualize.py         # t-SNE/UMAP, Jacobian analysis, publication figures
│   └── utils/
│       └── helpers.py           # Logging, seeding, DDP utilities
├── configs/                     # Hydra/YAML configuration files
│   ├── pretrain.yaml
│   ├── data_generation.yaml
│   ├── diffusion_policy.yaml
│   └── evaluation.yaml
├── scripts/                     # Entry-point scripts
│   ├── generate_data.py
│   ├── pretrain.py
│   ├── evaluate.py
│   ├── run_ablations.py
│   └── run_all.sh               # Master pipeline
├── tests/
│   └── test_dynaclip.py         # Unit tests
├── requirements.txt
├── setup.py
├── setup_env.sh
└── README.md
```

---

## Installation

### Prerequisites
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA 12.1+
- Conda (Miniconda or Anaconda)

### Setup

```bash
# 1. Create conda environment
conda create -n dynaclip python=3.10 -y
conda activate dynaclip

# 2. Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install all dependencies
cd DynaCLIP
pip install -r requirements.txt

# 4. Install DynaCLIP in editable mode
pip install -e .

# Or simply run:
bash setup_env.sh
```

---

## Architecture

### DynaCLIP Model

| Component | Details |
|-----------|---------|
| **Backbone** | DINOv2-ViT-B/14 (86M params) |
| **Feature extraction** | CLS token ⊕ mean-pooled patch tokens → 1536d |
| **Projection head** | Linear(1536→768) → LayerNorm → GELU → Linear(768→512) → L2-norm |
| **Temperature** | Learnable, init=0.07 |
| **Training** | Soft InfoNCE with dynamics similarity as soft targets |

### Dynamics Fingerprint

Each object configuration (geometry × texture × physics) receives a **dynamics fingerprint**: the concatenation of state trajectories from 5 standardized diagnostic actions:

1. `push_x` — Push along X axis
2. `push_y` — Push along Y axis  
3. `grasp_lift_release` — Pick up and drop
4. `lateral_flick` — Quick sideways flick
5. `slow_press_down` — Gradual downward press

Each trajectory: 50 timesteps × 13 dimensions (3 pos + 4 quat + 3 lin_vel + 3 ang_vel) at 20 Hz.

### Similarity Metrics

- **DTW** (default): Dynamic Time Warping via tslearn
- **L2**: Normalized Euclidean distance
- **MSE**: Mean squared error
- **Velocity-DTW**: DTW on velocity channels only

---

## Visual Backbones (8 total)

| Backbone | Source | Embedding Dim |
|----------|--------|--------------|
| **DynaCLIP** | Ours | 512 |
| DINOv2-ViT-B/14 | `facebookresearch/dinov2` | 768 |
| DINOv2-ViT-L/14 | `facebookresearch/dinov2` | 1024 |
| SigLIP-ViT-B/16 | `google/siglip-base-patch16-224` | 768 |
| CLIP-ViT-L/14 | `openai/clip-vit-large-patch14` | 768 |
| R3M | `r3m` | 2048 |
| VIP | `vip` | 1024 |
| MCR | `mcr` | 512 |

---

## Policy Baselines (7 total)

| Policy | Type | Details |
|--------|------|---------|
| **Diffusion Policy** (DynaCLIP) | Ours | DDPM train / DDIM-10 inference, 16-step action chunks |
| Diffusion Policy (DINOv2) | Baseline | Same architecture, frozen DINOv2 encoder |
| Diffusion Policy (R3M) | Baseline | Same architecture, frozen R3M encoder |
| **ACT** | Baseline | CVAE + Transformer decoder |
| **OpenVLA-OFT** | Baseline | Vision-Language-Action with orthogonal fine-tuning |
| **Octo-Base** | Baseline | Transformer-based generalist policy |
| **Dreamer-v3** | Baseline | Model-based RL with RSSM world model |
| **TD-MPC2** | Baseline | Latent dynamics + Model Predictive Control |

---

## Experiments

### Experiment 1: Physics Property Linear Probing

Freeze each backbone → train linear heads to predict mass, friction, restitution.

**Metrics:** R² per property, material classification accuracy  
**Run:** `python scripts/evaluate.py --experiment linear_probing`

### Experiment 2: Invisible Physics Discrimination

500 visually-identical object pairs with different physics → test if embeddings can distinguish them.

**Metrics:** Cosine similarity distributions, "heavier" classification accuracy, sensitivity  
**Run:** `python scripts/evaluate.py --experiment invisible_physics`

### Experiment 3: World Model Prediction

Train RSSM world model (Dreamer-v3 style) on each backbone → test latent prediction quality.

**Metrics:** Latent MSE at horizons t+1, t+5, t+10, t+20  
**Run:** `python scripts/evaluate.py --experiment world_model`

### Experiment 4: Downstream Policy Learning

Train Diffusion Policy (and all baselines) on 6 manipulation benchmarks.

**Benchmarks:**
- LIBERO-10 (10 tasks, success rate)
- LIBERO-Long (10 long-horizon tasks)
- CALVIN (ABC→D, chain length metric)
- ManiSkill3 (8 tasks, success rate)
- Physics-Varying (OOD mass/friction, 30 episodes each)
- RLBench-18 (18 tasks, multi-variation)

**Run:** `python scripts/evaluate.py --experiment downstream`

### Experiment 5: Zero-Shot Physics Inference

Encode a library of known objects → use k-NN on new images to predict physics properties.

**Metrics:** Mass/friction/restitution MAE, top-5 retrieval P@5  
**Run:** `python scripts/evaluate.py --experiment zero_shot`

---

## Ablation Studies (8 total)

| # | Ablation | What varies |
|---|----------|-------------|
| A1 | Similarity metric | DTW vs L2 vs MSE vs velocity-DTW |
| A2 | Number of diagnostic actions | 1 → 5 actions |
| A3 | Loss formulation | Soft InfoNCE vs InfoNCE vs Triplet vs BYOL |
| A4 | Hard negative ratio | 0%, 10%, 20%, 30% (default), 50%, 70% |
| A5 | Backbone initialization | DINOv2, SigLIP, CLIP, random |
| A6 | Data scale | 10%, 25%, 50%, 75%, 100% of training data |
| A7 | Property diversity | Mass-only, friction-only, restitution-only, all three |
| A8 | Fine-tuning depth | Frozen, last 2 layers, last 4 layers, full fine-tune |

**Run:** `python scripts/run_ablations.py`

---

## Quick Start

### Full Pipeline

```bash
conda activate dynaclip
cd DynaCLIP

# Generate data (or use synthetic fallback for testing)
python scripts/generate_data.py --num_geometries 50 --num_textures 5 --num_physics 100

# Pre-compute DINOv2 embeddings & mine hard pairs
python scripts/generate_data.py --precompute_only

# Pre-train DynaCLIP (multi-GPU)
torchrun --nproc_per_node=4 scripts/pretrain.py

# Run all evaluations
python scripts/evaluate.py --experiment all

# Run ablations
python scripts/run_ablations.py
```

### Or run everything at once:

```bash
bash scripts/run_all.sh
```

---

## Configuration

All configurations use YAML files in `configs/`. Key parameters:

### Pre-training (`configs/pretrain.yaml`)
```yaml
model:
  backbone: dinov2_vitb14
  projection_dim: 512
  unfreeze_last_n: 4

training:
  epochs: 100
  batch_size: 256
  lr_backbone: 1.0e-5
  lr_head: 1.0e-3
  warmup_steps: 500
  loss: soft_infonce
  hard_negative_ratio: 0.3
```

### Data Generation (`configs/data_generation.yaml`)
```yaml
simulation:
  num_geometries: 50
  num_textures_per_geometry: 5
  num_physics_per_config: 100
  trajectory_steps: 50
  control_freq: 20
```

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Backbone LR | 1e-5 |
| Head LR | 1e-3 |
| Weight decay | 0.01 |
| Batch size | 256 |
| Epochs | 100 |
| Warmup steps | 500 |
| Scheduler | Cosine annealing |
| Precision | bf16 |
| GPUs | 4× (DDP) |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_dynaclip.py::TestLosses -v
```

---

## Citation

```bibtex
@article{dynaclip2025,
  title={DynaCLIP: Physics-Grounded Visual Representations for Robotic Manipulation via Dynamics Contrastive Learning},
  year={2025}
}
```

## License

MIT License
