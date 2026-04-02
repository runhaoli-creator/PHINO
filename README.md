# DynaCLIP: Physics-Grounded Visual Representations via Dynamics Contrastive Learning

Official implementation of **DynaCLIP**, a self-supervised visual representation learning framework that embeds *implicit physical dynamics* into visual encoders through contrastive pre-training with analytically computed physics priors.

## Overview

Standard visual encoders (CLIP, DINOv2, etc.) learn representations optimized for semantic similarity — objects that *look alike* are mapped nearby in embedding space. However, for embodied AI and robotics, we need representations that capture **how objects behave physically**: a metal sphere and a rubber ball may look similar, but their dynamics (bouncing, rolling, deformation) differ fundamentally.

DynaCLIP bridges this gap by:
1. **Category-Grounded Physics Priors**: Mapping 345 DomainNet visual categories to 10 material archetypes (metal, wood, fabric, glass/ceramic, rubber/plastic, etc.), each with analytical physical parameters (mass, friction, restitution, deformability).
2. **Analytical Physics Engine**: Computing dynamics trajectories (projectile motion, surface sliding, collision, pendulum, deformation) from first principles — no simulator needed.
3. **Soft InfoNCE Contrastive Loss**: Using continuous dynamics similarity as soft labels (not binary pos/neg), with a learnable temperature parameter.
4. **DINOv2 Fine-tuning**: Unfreezing a DINOv2-ViT-B/14 backbone during pre-training so gradients from the physics-aware loss reshape the entire feature space.

## Model Architecture

```
Input Image (224×224)
        │
        ▼
┌──────────────────────┐
│  DINOv2-ViT-B/14     │   86M params, UNFROZEN
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
    Concatenate → 1536-d
           │
           ▼
┌──────────────────────┐
│  Projection Head      │   (discarded after pre-training)
│  Linear(1536, 768)    │
│  LayerNorm → GELU     │
│  Linear(768, 512)     │
│  L2 Normalize         │
└──────────┬───────────┘
           │
           ▼
    512-d Unit-Norm Embedding
```

**Total parameters**: 88.2M (86M backbone + 2.2M projection head)

After pre-training, the projection head is discarded and the backbone serves as a general-purpose physics-aware visual encoder outputting 1536-d features.

## Project Structure

```
DynaCLIP/
├── dynaclip/
│   ├── models/
│   │   ├── dynaclip.py        # DynaCLIPModel, DynaCLIPEncoder, ProjectionHead
│   │   └── backbones.py       # BackboneWrapper + backbone registry
│   ├── losses/
│   │   └── contrastive.py     # SoftInfoNCELoss with learnable temperature
│   ├── trainers/
│   │   └── pretrain.py        # DynaCLIPTrainer + CosineWarmupScheduler
│   ├── data/
│   │   ├── generation.py      # AnalyticalPhysicsEngine + material priors
│   │   ├── dataset.py         # DynaCLIPContrastiveDataset + dataloaders
│   │   └── precompute.py      # DINOv2 embedding pre-computation
│   ├── eval/
│   │   ├── linear_probing.py  # Exp 1: Physics property linear probes
│   │   ├── invisible_physics.py  # Exp 2: Same-appearance discrimination
│   │   ├── world_model.py     # Exp 3: RSSM world model evaluation
│   │   └── zero_shot.py       # Exp 5: Zero-shot physics inference
│   └── utils/
│       └── helpers.py         # Logging, seeding, distributed utils
├── configs/
│   ├── pretrain.yaml          # Pre-training configuration
│   ├── data_generation.yaml   # Data generation configuration
│   └── evaluation.yaml        # Evaluation configuration
├── scripts/
│   ├── pretrain.py            # Training entry point
│   ├── generate_data.py       # Data generation entry point
│   └── evaluate.py            # Evaluation entry point
├── tests/
│   └── test_dynaclip.py       # Unit tests
├── paper/
│   ├── dynaclip_neurips2026.tex   # Full paper (NeurIPS 2026 format)
│   └── neurips_2026.sty           # NeurIPS style file
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

### Requirements
- Python ≥ 3.10
- PyTorch ≥ 2.2.0 with CUDA support
- 1+ GPU with ≥ 24GB VRAM (8× RTX PRO 6000 recommended for full training)

### Setup

```bash
# Clone the repository
git clone git@github.com:zhengtaoyao/DynaCLIP.git
cd DynaCLIP

# Create conda environment
conda create -n dynaclip python=3.10 -y
conda activate dynaclip

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install DynaCLIP
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Training Data

DynaCLIP uses an analytical physics engine — no simulator installation needed.

```bash
python scripts/generate_data.py \
    --config configs/data_generation.yaml \
    --output_dir data_cache/domainnet_physics \
    --num_workers 8
```

This generates contrastive pairs with dynamics similarity labels from DomainNet images.

### 2. Pre-train DynaCLIP

**Single GPU:**
```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

**Multi-GPU (recommended):**
```bash
torchrun --nproc_per_node=8 scripts/pretrain.py \
    --config configs/pretrain.yaml \
    training.total_steps=30000 \
    training.effective_batch_size=1280
```

Key training hyperparameters (from `configs/pretrain.yaml`):
| Parameter | Value |
|-----------|-------|
| Backbone LR | 1e-5 |
| Head LR | 1e-3 |
| Weight Decay | 0.05 |
| Warmup Steps | 500 |
| Total Steps | 30,000 |
| Effective Batch Size | 1,280 |
| Precision | bf16 |
| Temperature (init) | 0.07 (learnable) |

### 3. Evaluate

Run the full evaluation suite:

```bash
python scripts/evaluate.py \
    --config configs/evaluation.yaml \
    --checkpoint checkpoints/dynaclip_best.pt
```

Individual evaluations:

```python
from dynaclip.models import DynaCLIPModel, DynaCLIPEncoder
from dynaclip.eval.linear_probing import PhysicsLinearProbing

# Load pre-trained encoder (backbone only, no projection head)
encoder = DynaCLIPEncoder.from_pretrained("checkpoints/dynaclip_best.pt")
encoder.eval().cuda()

# Run linear probing evaluation
evaluator = PhysicsLinearProbing(encoder, feature_dim=1536)
results = evaluator.evaluate(data_dir="data_cache/physics_probes", num_seeds=5)
print(results)
```

## Evaluation Suite

| Experiment | Script | Description |
|-----------|--------|-------------|
| **Exp 1**: Linear Probing | `eval/linear_probing.py` | 5 physics properties (mass, friction, restitution, density, elasticity) via linear probes |
| **Exp 2**: Invisible Physics | `eval/invisible_physics.py` | Discriminate visually identical objects with different physical properties |
| **Exp 3**: World Model | `eval/world_model.py` | RSSM world model: predict future states from DynaCLIP features |
| **Exp 5**: Zero-Shot | `eval/zero_shot.py` | Zero-shot physics property inference on novel object categories |

## Key Results

**Physics Linear Probing** (5-property average R², 5 seeds):
| Model | Mass | Friction | Restitution | Density | Elasticity | **Mean R²** |
|-------|------|----------|-------------|---------|------------|-------------|
| CLIP ViT-B/16 | 0.328 | 0.156 | 0.218 | 0.289 | 0.142 | 0.227 |
| DINOv2 ViT-B/14 | 0.412 | 0.298 | 0.345 | 0.378 | 0.267 | 0.340 |
| **DynaCLIP (Ours)** | **0.687** | **0.623** | **0.712** | **0.654** | **0.598** | **0.655** |

**Downstream: LIBERO-10 Robotic Manipulation** (v4, 200 epochs, action chunking):
| Backbone | Success Rate |
|----------|-------------|
| CLIP ViT-B/16 | 46.9% |
| DINOv2 ViT-B/14 | 51.5% |
| **DynaCLIP (Ours)** | **59.0%** |

## Citation

```bibtex
@inproceedings{yao2026dynaclip,
  title={DynaCLIP: Physics-Grounded Visual Representations via Dynamics Contrastive Learning},
  author={Yao, Zhengtao},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is released under the MIT License.
