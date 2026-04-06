# IntPhys2 Benchmark Evaluation

## Overview

Evaluation of DynaCLIP (V2/V3) on the [IntPhys2](https://github.com/facebookresearch/IntPhys2) violation-of-expectation (VoE) benchmark (Bordes et al., 2025).

**Task:** Given paired videos (plausible vs. implausible physics), determine which video violates intuitive physics using embedding trajectory analysis.

## Results

### Main Evaluation (stride=2, 1012 videos, 253 scenes)

| Backbone | Best Metric | Overall Acc | embedding_diff | max_jump | prediction_error |
|----------|-------------|:-----------:|:--------------:|:--------:|:----------------:|
| **clip_vitl14** | prediction_error | **66.5%** | 51.2% | 51.0% | 66.5% |
| **siglip_vitb16** | prediction_error | **66.5%** | 50.5% | 50.5% | 66.5% |
| dinov2_vitl14 | prediction_error | 59.6% | 48.0% | 48.1% | 59.6% |
| dinov2_vitb14 | prediction_error | 58.9% | 50.5% | 49.9% | 58.9% |
| DynaCLIP-V3 | prediction_error | 58.0% | 52.2% | 50.2% | 58.0% |
| DynaCLIP-V2 | prediction_error | 57.0% | 49.7% | 48.9% | 57.0% |

*Random chance = 50%*

### Per-Condition Breakdown (prediction_error)

| Backbone | Continuity | Immutability | Permanence | Solidity |
|----------|:----------:|:------------:|:----------:|:--------:|
| clip_vitl14 | 64.2% | 63.3% | 69.6% | 68.5% |
| siglip_vitb16 | 63.7% | 64.2% | 66.7% | 70.5% |
| dinov2_vitl14 | 60.4% | 57.1% | 61.3% | 59.6% |
| dinov2_vitb14 | 55.0% | 58.3% | 60.8% | 61.0% |
| DynaCLIP-V3 | 60.0% | 57.9% | 60.0% | 54.8% |
| DynaCLIP-V2 | 52.1% | 60.0% | 55.0% | 60.3% |

### Per-Difficulty Breakdown

| Backbone | Easy | Medium | Hard | Unknown |
|----------|:----:|:------:|:----:|:-------:|
| clip_vitl14 | 68.3% | 66.5% | 70.5% | 57.6% |
| siglip_vitb16 | 73.1% | 68.5% | 67.6% | 55.8% |
| dinov2_vitl14 | 55.8% | 62.7% | 57.1% | 59.3% |
| dinov2_vitb14 | 61.5% | 59.0% | 56.2% | 62.2% |
| DynaCLIP-V3 | 61.5% | 62.3% | 51.8% | 58.1% |
| DynaCLIP-V2 | 56.7% | 56.0% | 59.8% | 54.1% |

### Stride Sensitivity Analysis

| Stride | DynaCLIP-V2 | dinov2_vitb14 |
|:------:|:-----------:|:-------------:|
| 1 | 59.0% | 59.7% |
| 2 | 57.0% | 58.9% |
| 4 | 56.4% | 59.7% |
| 8 | 59.1% | 61.5% |

### Training Loss Curves

| Version | Start Loss | End Loss | Steps |
|---------|:----------:|:--------:|:-----:|
| V2 | 6.2807 | 6.0780 | 20,000 |
| V3 | 5.6321 | 4.9210 | 20,000 |

## Methodology

### Surprise Metrics
1. **embedding_diff**: Sum of squared L2 distances between consecutive frame embeddings
2. **max_jump**: Largest single frame-to-frame L2 distance  
3. **prediction_error**: Train ridge regression (emb[t] → emb[t+1]) on plausible videos only; measure prediction residual on all videos

### Pairwise Classification
For each scene with both plausible and implausible variants, classify correctly if the implausible video has higher surprise score. Random chance = 50%.

### Video Processing
- Frame extraction: decord library
- Frame stride: 2 (default), sensitivity tested at 1, 4, 8
- Max frames per video: 100
- Resolution: 224×224 (center crop)
- Normalization: ImageNet mean/std

## Figures

- `paper/figures/intphys_accuracy_bar.pdf` — Overall accuracy bar chart
- `paper/figures/intphys_stride_sensitivity.pdf` — Stride sensitivity line plot  
- `paper/figures/intphys_condition_heatmap.pdf` — Per-condition accuracy heatmap
- `paper/figures/intphys_metric_comparison.pdf` — 3-metric comparison grouped bars
- `paper/figures/loss_curves_v2_v3.pdf` — Training loss/temperature/LR curves

## Scripts

```bash
# Main IntPhys2 evaluation
python scripts/evaluate_intphys.py \
    --data_dir /path/to/intphys2 \
    --split Main \
    --checkpoints checkpoints/pretrain_v2/dynaclip_final.pt \
                  checkpoints/pretrain_v3/dynaclip_final.pt \
    --checkpoint_names DynaCLIP-V2 DynaCLIP-V3 \
    --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16 \
    --output_dir results/intphys \
    --frame_stride 2 --max_frames 100 --batch_size 64 \
    --gpu 0 1 2 3 4 5

# Extract training loss curves
python scripts/extract_loss_curves.py

# Synthetic VoE test
python scripts/evaluate_synthetic_voe.py

# Compile all results and generate figures
python scripts/compile_intphys_results.py
```

## Data

IntPhys2 dataset: https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip (1.68 GB)

Reference: Bordes, F. et al. (2025). IntPhys 2.0: Benchmarking Intuitive Physics Understanding in Foundation Models.
