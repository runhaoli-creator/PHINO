# DynaCLIP: Comprehensive Analysis of 5 Critical Issues

## Executive Summary

After thorough code inspection and literature review, I assess each issue with severity, supporting evidence from the codebase and related work, and concrete fix proposals.

| Issue | Severity | Verdict | Key Finding |
|-------|----------|---------|-------------|
| P0-1: Deterministic Category→Physics | **Critical** | Valid — fundamental confound | Physics is a deterministic function of category; the "physics-aware" signal is redundant to category recognition |
| P0-2: Ablation Contradicts Core Claims | **Critical** | Valid — undermines paper narrative | "No hard mining" outperforms full model by +22% Mass R²; Soft InfoNCE has no theoretical basis |
| P1-1: Zero-Shot Retrieval Degradation | **High** | Valid — well-documented phenomenon | DynaCLIP ranks last in 5000-library retrieval; fine-tuning distorts DINOv2's embedding space |
| P1-2: ManiSkill3 Floor Effect | **Medium** | Valid — misleading metric | Gen Ratio 1.02 at 26.7% absolute is a floor effect, not physics resilience |
| P1-3: Table 4 vs App. Table Contradiction | **Medium** | Valid — protocol-dependent results | Small library makes all models equivalent via category matching; large library exposes distortion |

---

## P0-1: Category–Physics Deterministic Coupling (FUNDAMENTAL)

### The Problem

In `dynaclip/data/generation.py:81-235`, `CATEGORY_TO_MATERIAL` implements a **deterministic 1-to-1 mapping** of 345 DomainNet categories → 10 material types. Each material type has fixed physics ranges in `MATERIAL_PRIORS` (lines 28-80). This means:

```
"cup" → glass_ceramic → mass ∈ [0.30, 2.00], friction ∈ [0.20, 0.50], restitution ∈ [0.40, 0.70]
"basketball" → rubber_plastic → mass ∈ [0.20, 1.50], friction ∈ [0.40, 0.80], restitution ∈ [0.30, 0.70]
```

**Any model that can recognize categories can deterministically predict physics ranges.** Since Table 1 shows all strong baselines achieve near-perfect category accuracy (DINOv2-B: 0.987, CLIP: 0.941, DynaCLIP: 0.997), the physics prediction improvements could simply reflect better category boundaries, not genuine physics understanding.

### Evidence from Tables

**Table 4 (k-NN Physics, small library):** All models are nearly tied:
- DynaCLIP: 0.675 / 0.662 / 0.816
- DINOv2-L: 0.669 / 0.666 / 0.821
- CLIP: 0.674 / 0.661 / 0.813

This is expected: k-NN retrieves same-category images, which have the same material type, hence the same physics range. The "physics prediction" is actually just category matching.

**Table 1 (Linear Probing):** DynaCLIP does show superior R² (Mass: 0.550 vs DINOv2-L: 0.440). But this could be explained by the contrastive loss sharpening within-material boundaries, not learning actual physics. A control experiment with category-conditioned baselines would disentangle this.

### What the Literature Says

- **Rank-N-Contrast** (Zha et al., NeurIPS 2023 Spotlight, `arXiv:2210.01189`) addresses contrastive learning for continuous regression targets. It ranks samples by target values and contrasts based on ranking, guaranteeing order preservation in the representation space. DynaCLIP's approach of mapping category→material→physics is fundamentally different: it's discrete, not continuous.

- **Roboscape** (Shang et al., 2025, `arXiv:2506.23135`) and **Pin-WM** (Li et al., 2025, `arXiv:2504.16693`) demonstrate physics-informed world models that learn physical properties from *interaction data*, not from category labels. This is a more principled approach.

### How to Fix

1. **Multi-material categories:** Allow categories to map to multiple materials with a probability distribution:
   ```python
   CATEGORY_TO_MATERIALS = {
       "cup": {"glass_ceramic": 0.5, "rubber_plastic": 0.3, "metal": 0.2},
       "ball": {"rubber_plastic": 0.6, "fabric": 0.2, "metal": 0.2},
   }
   ```

2. **Cross-material physics overlap:** Increase overlap between material physics ranges so that category alone doesn't determine physics:
   ```python
   # Instead of: metal mass [2.0, 8.0], wood mass [0.5, 3.0]
   # Use overlapping ranges with noise
   "metal": {"mass": (1.0, 10.0, 0.3)},  # (min, max, noise_std)
   "wood":  {"mass": (0.3, 5.0, 0.3)},
   ```

3. **Category-controlled evaluation:** Add a Table showing physics R² *within the same category* (e.g., predict which of two cups is heavier). This would isolate genuine physics understanding from category recognition.

4. **Material-confused test set:** Create eval pairs where both objects are from the same material type but different categories, or different material types but visually similar.

---

## P0-2: Ablation Contradicts Core Contributions (FUNDAMENTAL)

### The Problem

Table 5 (`tab:ablations`) shows devastating results for DynaCLIP's two key contributions:

| Variant | Mass R² | Friction R² | Rest. R² | Cat. Acc |
|---------|---------|-------------|----------|----------|
| **DynaCLIP (full)** | 0.584 | 0.523 | 0.689 | 0.999 |
| **No hard mining** | **0.714** | **0.618** | **0.763** | 1.000 |
| Standard InfoNCE | 0.601 | 0.557 | 0.715 | 0.999 |

1. **Removing hard mining improves EVERY metric** — Mass +22%, Friction +18%, Rest +11%
2. **Standard InfoNCE outperforms Soft InfoNCE** — even without the "novel" soft label formulation
3. **Hard mining destroys category structure** — 15% hard neg → Cat Acc 0.437, 50% → 0.403

### Why Hard Mining Hurts

The 30% hard negatives are "same-category pairs where physics differ." But because of P0-1 (deterministic mapping), same-category pairs always have the *same* material type with physics drawn from the *same* range. So "hard negatives" are actually samples that are physics-similar but the model is forced to push apart. This is **self-contradictory training signal**.

From the code (`dataset.py`):
```python
# Hard negatives: SAME image group, DIFFERENT physics
# But same image group = same category = same material = same physics range!
```

### Why Soft InfoNCE is Unjustified

The geometric mean approximation in `contrastive.py:87-93`:
```python
sim_matrix = torch.sqrt(pair_sim_i * pair_sim_j + 1e-8) * 0.5
```

For off-diagonal entry $(i,j)$, this computes $S_{ij} = 0.5 \sqrt{s_i \cdot s_j}$ where $s_i$ is the dynamics similarity for pair $i$. This is **not** the pairwise similarity between samples $i$ and $j$ — it's a heuristic combining single-pair similarities. There is no theoretical justification for why the geometric mean of individual pair similarities should approximate cross-pair similarity.

### What the Literature Says

- **Rank-N-Contrast** (NeurIPS 2023) provides a principled approach: for each anchor, every other sample can be a "positive" — the loss uses ranking to weight the contribution. Negatives are only samples with **larger** label differences. The loss is:
  ```
  L_RnC = -Σ_k [log(exp(f_anchor·f_k/τ) / Σ_{j: |y_j-y_a|≥|y_k-y_a|} exp(f_anchor·f_j/τ))]
  ```
  This **preserves continuous ordering** in the representation, unlike DynaCLIP's ad-hoc geometric mean.

- **Similarity Contrastive Estimation (SCE)** (Denize et al., WACV 2023) derives soft contrastive targets from teacher-student momentum networks, not from ad-hoc similarity approximation.

- **Supervised Contrastive Learning** (Khosla et al., NeurIPS 2020) uses actual class membership for positive/negative assignment. Extending to continuous labels requires actual pairwise target distances (e.g., $|y_i - y_j|$), not proxy approximations.

### How to Fix

1. **Replace geometric mean with actual pairwise physics distance:**
   ```python
   # Instead of: sqrt(s_i * s_j) * 0.5
   # Compute actual pairwise similarity from physics vectors:
   physics_i = physics_vectors[i]  # (mass, friction, restitution)
   physics_j = physics_vectors[j]
   S_ij = exp(-||log(physics_i) - log(physics_j)||_1 / sigma)
   ```

2. **Adopt Rank-N-Contrast formulation** for physics regression:
   - Use actual physics values as continuous labels
   - Contrast based on relative rankings of physics distances
   - Guarantees order preservation (theoretical backing)

3. **Remove hard mining entirely.** The ablation already proves this improves results. Use random pair sampling with the contrastive loss providing the learning signal.

4. **Retrain with the better configuration:** No hard mining + Standard InfoNCE (or RnC) as the default. This would yield an honest improvement of +22% Mass R², +18% Friction R², +11% Restitution R² over the current "full" model.

---

## P1-1: Zero-Shot Retrieval Degradation

### The Problem

Appendix Table (`tab:zeroshot`, 5000-image library):

| Backbone | Mass R² | Fric R² | Rest R² | Mean R² |
|----------|---------|---------|---------|---------|
| DynaCLIP | 0.536 | 0.426 | 0.638 | **0.533** |
| DINOv2-B/14 | 0.582 | 0.567 | 0.741 | 0.630 |
| DINOv2-L/14 | 0.591 | 0.568 | 0.753 | **0.637** |

DynaCLIP ranks **last** (Mean R² = 0.533) among competitive models, even worse than its backbone DINOv2-B/14 (0.630). Fine-tuning destroyed the general-purpose embedding quality.

### What the Literature Says

This is a well-documented phenomenon:

1. **"Fine-tuning can cripple your foundation model"** (Mukhoti et al., TMLR 2024, `arXiv:2308.13320`): Fine-tuning causes "concept forgetting" — the model loses pre-trained representations while learning task-specific features. The paper proposes **LDIFS** (ℓ₂ distance in feature space regularization).

2. **WiSE-FT** (Wortsman et al., CVPR 2022): Weight-space ensembling between pre-trained and fine-tuned weights: $\theta_{WiSE} = \alpha \cdot \theta_{pretrained} + (1-\alpha) \cdot \theta_{finetuned}$. Simple but effective at preserving zero-shot capabilities.

3. **LP-FT** (Kumar et al., NeurIPS 2022): Linear Probing then Fine-Tuning. First trains a linear head on frozen features, then unfreezes for full fine-tuning. This preserves more of the pre-trained structure.

4. **FLYP** (Goyal et al., CVPR 2023): "Finetune Like You Pretrain" — maintains the contrastive pre-training objective during fine-tuning to prevent embedding distortion.

5. **"Preventing Zero-Shot Transfer Degradation"** (Zheng et al., ICCV 2023): Directly addresses preventing zero-shot capability loss during fine-tuning.

6. **Dual Risk Minimization** (Li et al., NeurIPS 2024): Formulates robust fine-tuning as a dual risk problem, mitigating embedding distortion.

### How to Fix

1. **Add feature-space regularization during fine-tuning:**
   ```python
   # In trainer, add L2 regularization toward frozen DINOv2 features
   with torch.no_grad():
       z_frozen = frozen_backbone(img)
   reg_loss = alpha * F.mse_loss(z_finetuned, z_frozen)
   total_loss = contrastive_loss + reg_loss
   ```

2. **Apply WiSE-FT post-training:** Interpolate between DINOv2 weights and DynaCLIP weights:
   ```python
   for (name, p_dyna), (_, p_dino) in zip(dynaclip.named_parameters(), dinov2.named_parameters()):
       p_dyna.data = alpha * p_dino.data + (1 - alpha) * p_dyna.data
   ```

3. **Use LP-FT protocol:** First train projection head only, then unfreeze backbone. The ablation Table 5 already shows "Frozen backbone" gets R² = 0.542/0.533/0.693 and "Last-2 blocks" gets 0.603/0.569/0.717 — suggesting partial fine-tuning may be competitive while preserving retrieval quality.

---

## P1-2: ManiSkill3 — Worst Absolute Performance

### The Problem

Table 6 (`tab:downstream_policy`, ManiSkill3 PickCube):

| Backbone | Standard | Heavy | Varied | Gen Ratio |
|----------|----------|-------|--------|-----------|
| **DynaCLIP** | 26.7 | 6.7 | 27.1 | **1.02** |
| CLIP | 56.7 | 13.3 | 42.9 | 0.76 |
| DINOv2 | 40.0 | 10.0 | 35.2 | 0.88 |

DynaCLIP's Gen Ratio of 1.02 is highlighted as "near-zero degradation under physics shift," but this is misleading:
- CLIP at 56.7% with 0.76 ratio → effective ~43% on varied → **still much better than DynaCLIP's 27.1%**
- DynaCLIP's low absolute performance means there's little room to degrade → floor effect

The generalization ratio is $\text{Gen} = \text{Varied} / \text{Standard}$. When Standard is already low, the ratio is artifactually stable. A model that succeeds 0% and fails 0% also has perfect Gen Ratio.

### How to Fix

1. **Report absolute performance as the primary metric.** Gen Ratio should be secondary.
2. **Investigate why DynaCLIP underperforms:** The contrastive fine-tuning may have hurt action-relevant features while improving physics features. This aligns with P1-1 (embedding distortion).
3. **Try CLS-only features vs. patch features:** DINOv2 typically works better with patch tokens for manipulation policies.
4. **Scale up demonstrations:** 10 demos may be insufficient; try 25, 50, 100 demos.

---

## P1-3: Table 4 vs Appendix Table Contradiction

### The Problem

**Table 4 (k-NN, small library):** All models tied (~0.67-0.68 Mass R²)
**Appendix (`tab:zeroshot`, 5000 library):** DynaCLIP worst (0.533 Mean R²)

These tables tell opposite stories about DynaCLIP's physics understanding.

### Root Cause

The small-library k-NN is confounded by P0-1 (category-physics coupling):
- With a small library, k-NN finds same-category images easily → same material → same physics range → all models perform equally
- With a 5000-image library, the retrieval problem becomes harder, requiring genuine embedding quality → DynaCLIP's distorted embeddings fail

### How to Fix

1. **Unify evaluation protocols:** Use a single library size (or report multiple sizes: 100, 500, 1000, 5000).
2. **Add category-controlled k-NN:** Retrieve only from *different*-category images to isolate genuine physics understanding.
3. **Report both consistently** in the main paper, not hiding the worse result in the appendix.

---

## Recommended Priority of Fixes

### Immediate (before any resubmission):

1. **Remove hard mining** — The ablation already proves this improves results. No new experiments needed, just retrain with the already-proven-better configuration.

2. **Replace geometric mean with actual pairwise physics distances** — Simple code change in `contrastive.py`.

3. **Add WiSE-FT or feature-space regularization** — Prevents embedding distortion. Likely fixes P1-1 and P1-3 simultaneously.

### Short-term:

4. **Multi-material categories** — Fundamental fix for P0-1. Requires re-generating the dataset.

5. **Category-controlled evaluation** — Add within-category and cross-category physics tests.

6. **Report absolute ManiSkill3 performance first**, Gen Ratio second.

### Longer-term:

7. **Adopt Rank-N-Contrast** (NeurIPS 2023 Spotlight) as the contrastive loss — theoretically grounded for continuous regression targets.

8. **Use interaction-based physics** (push/grasp trajectories) as the training signal instead of category-derived physics properties.

---

## Key References

1. **Rank-N-Contrast** — Zha et al., NeurIPS 2023 Spotlight. `arXiv:2210.01189`. Ranking-based contrastive loss for continuous regression. Theoretically grounded order preservation.

2. **Fine-tuning can cripple your foundation model** — Mukhoti et al., TMLR 2024. `arXiv:2308.13320`. Documents concept forgetting during fine-tuning; proposes LDIFS regularization.

3. **WiSE-FT** — Wortsman et al., CVPR 2022. Weight-space interpolation to preserve zero-shot capabilities during fine-tuning.

4. **LP-FT** — Kumar et al., NeurIPS 2022. Linear probing then fine-tuning to preserve pre-trained features.

5. **FLYP** — Goyal et al., CVPR 2023. Finetune Like You Pretrain. Contrastive fine-tuning that preserves embedding structure.

6. **Preventing Zero-Shot Transfer Degradation** — Zheng et al., ICCV 2023. Cited 196 times. Methods for maintaining zero-shot generalization during fine-tuning.

7. **Dual Risk Minimization** — Li et al., NeurIPS 2024. Robust fine-tuning of zero-shot models.

8. **Similarity Contrastive Estimation (SCE)** — Denize et al., WACV 2023. Cited 50 times. Soft contrastive learning using momentum-estimated soft targets.

9. **Supervised Contrastive Learning** — Khosla et al., NeurIPS 2020. Foundational SupCon paper.

10. **Roboscape** — Shang et al., 2025. `arXiv:2506.23135`. Physics-informed embodied world model.

---

## Code-Level Summary of Issues

| File | Line(s) | Issue |
|------|---------|-------|
| `generation.py` | 81–235 | `CATEGORY_TO_MATERIAL`: 1-to-1 deterministic mapping (345 cats → 10 materials) |
| `generation.py` | 28–80 | `MATERIAL_PRIORS`: non-overlapping enough physics ranges per material |
| `contrastive.py` | 87–93 | `_build_similarity_matrix`: geometric mean `sqrt(s_i * s_j) * 0.5` — theoretically unjustified |
| `contrastive.py` | 20–99 | `SoftInfoNCELoss`: underperforms `StandardInfoNCELoss` per ablations |
| `dataset.py` | 61–62 | `hard_neg_ratio=0.3, hard_pos_ratio=0.3` — hard mining hurts all metrics |
| Paper discrepancy | — | Code `MATERIAL_PRIORS` values differ from paper Table (e.g., wood restitution: code [0.20, 0.40] vs paper [0.30, 0.60]) |
