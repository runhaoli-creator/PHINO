# DynaCLIP V3 Fix Plan — Actionable Guide for Experiment Agent

> This document is written for an automated agent to read and execute.
> Every fix has: the problem, which files to change, exact code locations,
> and what the expected outcome should be. Execute in the order listed.

---

## Context: Current State

- **V2 checkpoint**: `checkpoints/pretrain_v2/dynaclip_final.pt` (20K steps)
- **V2 config**: `configs/pretrain_v2.yaml` — PairwisePhysicsRnCLoss, no hard mining, WiSE-FT (alpha=0.1), multi-material
- **Problem**: The paper claims "all experiments re-evaluated with v2 checkpoint" but ~50% of results are copy-pasted from V1. The ablation table uses V1 config (SoftInfoNCE + hard mining), which contradicts V2's config. Multiple tables have internal inconsistencies.

---

## FIX 1: V2 Ablation Study [CRITICAL — Must Do First]

### Problem
`tables_standalone.tex` Table 5 (line 126-163) and `results_compilation.txt` Experiment 5 use **V1 ablation data** (SoftInfoNCE, 30% hard mining baseline). But V2 uses PairwisePhysicsRnCLoss, 0% hard mining, WiSE-FT. The ablation does not match the model being presented.

The V1 ablation also shows "no hard mining" beating "full model" (Mass 0.714 vs 0.584), which directly undermines the paper's narrative since V2 adopted no-hard-mining as default.

### What to do
Run a new ablation study with V2 as the baseline. Train 6 variants at 10K steps each using the V2 data (`data_cache/dynaclip_v2_data`):

| Variant | Config change from `pretrain_v2.yaml` |
|---|---|
| **V2 Full (baseline, 20K)** | No change (already trained) |
| V2 w/o WiSE-FT | Set `wiseft_alpha: 0.0` |
| V2 w/o multi-material | Regenerate data using `get_material_for_category()` instead of `sample_material_for_category()` |
| V2 with SoftInfoNCE | Set `loss.name: "soft_infonce"`, remove `physics_sigma` |
| V2 with StandardInfoNCE | Set `loss.name: "infonce"` |
| V2 frozen backbone | Set `model.freeze_backbone: true` |
| V2 last-4 blocks | Set `model.unfreeze_last_n_blocks: 4` |

For each variant, evaluate:
1. Linear probing (mass/friction/restitution R², category accuracy) using `dynaclip/eval/linear_probing.py`
2. Material clustering (NMI, ARI)

### Files to modify
- Create `configs/ablation_v2_base.yaml` — copy from `pretrain_v2.yaml`, set `total_steps: 10000`
- Create variant configs by overriding specific fields
- Run training via `scripts/pretrain.py --config <variant_config>`
- Run evaluation via `scripts/evaluate.py` or equivalent

### Expected outcome
A new ablation table where V2 Full is the baseline and each component (WiSE-FT, multi-material, RnC loss, fine-tuning depth) has a clear delta. The numbers should be internally consistent with V2's training setup.

### Update targets
- `tables_standalone.tex` Table 5 (line 126-163) — replace entirely
- `results_compilation.txt` Experiment 5 section — replace entirely
- `paper/dynaclip_neurips2026.tex` ablation section

---

## FIX 2: Rerun 5K-Library Zero-Shot Retrieval with V2 [CRITICAL]

### Problem
`tables_standalone.tex` Table A4 (line 486-505) shows DynaCLIP last place (Mean R²=0.533) among strong baselines on 5000-library zero-shot retrieval. This data is from V1. V2 added WiSE-FT specifically to prevent embedding space degradation, but Table A4 was never rerun.

### What to do
Rerun the 5000-library zero-shot k-NN experiment using V2 checkpoint.

Use `dynaclip/eval/zero_shot.py` with:
- `n_lib=5000`, `n_query=1000`, `k=5`
- V2 checkpoint: `checkpoints/pretrain_v2/dynaclip_final.pt`
- Same baselines as Table A4 (DINOv2-B/14, DINOv2-L/14, CLIP-L/14, SigLIP, R3M, VIP)

### Files involved
- `dynaclip/eval/zero_shot.py` — `create_real_library()` with `n_lib=5000, n_query=1000`
- `scripts/evaluate.py` or a standalone script

### Expected outcome
If WiSE-FT works, DynaCLIP should improve from 0.533 and close the gap with DINOv2 (0.637). If it doesn't improve, that's also useful data — it means WiSE-FT alpha=0.1 is insufficient and needs tuning.

### Possible outcomes and actions
- **DynaCLIP improves to >0.60**: Update Table A4, promote result to show WiSE-FT effectiveness
- **DynaCLIP stays ~0.53**: WiSE-FT is not solving the embedding degradation. Try higher alpha (0.3, 0.5) or feature distillation. Report honestly in the paper
- **DynaCLIP improves but still behind DINOv2**: Report honestly, discuss the trade-off (physics probing improved but retrieval structure partially lost)

### Update targets
- `tables_standalone.tex` Table A4 (line 486-505)
- `results_compilation.txt` — currently no 5K retrieval section in V2 format, add one

---

## FIX 3: Fix Per-Seed Data Inconsistency [CRITICAL]

### Problem
`tables_standalone.tex` Table A3 (line 460-484) shows DynaCLIP per-seed avg = 59.0%, but main Table 9 (line 263-290) shows avg = 60.4%. These use different checkpoints (V1 vs V2).

### What to do
Replace Table A3 with V2 checkpoint per-seed data. The V2 LIBERO-10 evaluation should already have per-seed breakdowns from the run that produced 60.4%.

If per-seed data for the V2 run exists (check logs or saved results), use that directly. If not, the V2 evaluation needs to be rerun with seed logging.

The V2 main table shows different per-task numbers than V1:
- V2: T0=9.3, T1=46.0, T5=88.7, T7=50.7, T8=36.0, T9=74.7
- V1 (Table A3): T0=6.0, T1=57.3, T5=72.0, T7=45.3, T8=25.3, T9=65.3

These are clearly from different runs. Table A3 must match Table 9.

### Update targets
- `tables_standalone.tex` Table A3 (line 460-484) — replace with V2 per-seed data

---

## FIX 4: Invisible Physics Evaluation [IMPORTANT]

### Problem
Two issues:

**Issue A: Code produces identical inputs.**
`dynaclip/data/dataset.py` line 272-282 — `InvisiblePhysicsDataset.__getitem__()`:
```python
img_tensor = self.transform(img)   # eval transform = deterministic
"img_a": img_tensor,    # Same tensor
"img_b": img_tensor,    # Same tensor
```
With eval transform (deterministic), `img_a == img_b` exactly. Any deterministic model produces `embedding_a == embedding_b`, meaning:
- `cosine_similarity = 1.0` for all pairs
- `L2_distance = 0.0` for all pairs
- `heavier_classification = random chance (50%)`

But Table 3 reports DynaCLIP Heavy Acc=0.560, AUC=0.615, Mass R²=0.300. This is impossible with deterministic eval transform.

**Issue B: Data not rerun with V2.**
The numbers are identical to V1. Even if Issue A is resolved, the evaluation needs rerunning with V2 checkpoint.

### What to do

**Step 1: Investigate how the current numbers were generated.**
The reported numbers (Acc=0.560, AUC=0.615) can only exist if:
- Train transform (random augmentation) was used instead of eval transform, OR
- The evaluation script uses a different code path than `InvisiblePhysicsDataset`

Check `scripts/evaluate.py` and any evaluation notebooks to find how Experiment 3 was actually run. If train transform was used, that's a confound (augmentation noise, not physics encoding).

**Step 2: Redesign the experiment.**
The invisible physics experiment as currently designed cannot test what it claims (same image → different physics → different embedding) because the model only sees pixels. Instead, reframe it as a **cross-sample statistical test**:

Option A — **Within-category physics regression**: For each material type, take all images of that material. Train a linear probe on frozen features to predict physics properties. If R² > 0 within a single material type, the model encodes intra-material physics variation (not just category).

Option B — **Physics-conditioned retrieval**: Given a query image with known physics, retrieve the k-nearest neighbors. Measure whether retrieved neighbors have more similar physics than random baseline. This tests whether the embedding space is organized by physics.

### Files to modify
- `dynaclip/data/dataset.py` line 223-285 — Fix or replace `InvisiblePhysicsDataset`
- `dynaclip/eval/invisible_physics.py` — Update evaluation to match new design
- If using Option A: add within-category evaluation to `dynaclip/eval/linear_probing.py`

### Update targets
- `tables_standalone.tex` Table 3 (line 69-91)
- `results_compilation.txt` Experiment 3
- Paper text explanation of the experiment

---

## FIX 5: Rerun ManiSkill3 with V2 Checkpoint [IMPORTANT]

### Problem
Table 6 (`tables_standalone.tex` line 166-185) uses V1 data. DynaCLIP has the worst absolute performance (26.7% standard) but claims best Gen Ratio (1.02). This wasn't rerun with V2.

### What to do
Rerun ManiSkill3 PickCube evaluation with V2 checkpoint. The code exists in `scripts/run_maniskill_eval.py` and `dynaclip/data/generation_maniskill3.py`.

Also, investigate why DynaCLIP's absolute performance is so low:
- Try CLS-only features (768-d) instead of CLS+mean (1536-d) — the concatenated features may be suboptimal for MLP BC policy
- Try with more demonstrations (200, 500) to see if the ranking changes with data scale

### Update targets
- `tables_standalone.tex` Table 6 (line 166-185)
- `results_compilation.txt` Experiment 6

---

## FIX 6: Rerun World Model with V2 Checkpoint [MODERATE]

### Problem
Table 7a (`tables_standalone.tex` line 188-207) uses V1 data. CLIP slightly beats DynaCLIP. Should be rerun with V2 to maintain consistency.

### What to do
Rerun RSSM world model evaluation on synthetic trajectories with V2 checkpoint using `dynaclip/eval/world_model.py`. Include all baselines from Table 7a (DINOv2-B/14, DINOv2-L/14, R3M, CLIP).

Also run with 3-5 seeds to get error bars — the current gaps are within noise.

### Update targets
- `tables_standalone.tex` Table 7a (line 188-207)
- `results_compilation.txt` Experiment 7

---

## FIX 7: Linear Probing Friction Regression [MODERATE]

### Problem
V2 friction R² (0.378) is lower than V1 (0.416), a -9% regression. Clustering NMI also dropped (0.435→0.424). This suggests V2 training may have traded some physics encoding for other properties.

### What to do
This is likely caused by PairwisePhysicsRnCLoss treating all 3 physics dimensions equally in L2 distance (`rnc_loss.py` line 172-174):
```python
physics_diff = (physics_i.unsqueeze(1) - physics_j.unsqueeze(0))  # (B, B, 3)
physics_dist = physics_diff.norm(dim=-1)  # (B, B)
```

The 3 dimensions (log_mass, friction, restitution) have different scales and variances. Log_mass ranges roughly [-3, 2.3], friction [0.1, 1.2], restitution [0, 0.9]. The loss is dominated by mass.

**Fix**: Normalize each physics dimension to zero mean and unit variance before computing pairwise distance. Add to `PairwisePhysicsRnCLoss.forward()`:
```python
# Normalize physics vectors (compute stats from batch)
physics_all = torch.cat([physics_i, physics_j], dim=0)
mean = physics_all.mean(dim=0, keepdim=True)
std = physics_all.std(dim=0, keepdim=True) + 1e-8
physics_i_norm = (physics_i - mean) / std
physics_j_norm = (physics_j - mean) / std
physics_diff = (physics_i_norm.unsqueeze(1) - physics_j_norm.unsqueeze(0))
```

After this change, retrain V2 and check if friction R² improves without mass R² dropping.

### Files to modify
- `dynaclip/losses/rnc_loss.py` line 170-174 — add normalization in `PairwisePhysicsRnCLoss.forward()`

---

## FIX 8: Strengthen CALVIN Results or Remove [LOW PRIORITY]

### Problem
CALVIN numbers are too weak to include as evidence:
- Task Recognition: 6.2% vs CLIP 5.7% — margin 0.5pp
- Cross-Env: 3.6% vs CLIP 3.5% — margin 0.1pp

These are within noise. Including them weakens the paper by adding claims that can't be supported.

### What to do
Two options:

**Option A**: Remove CALVIN feature-based evaluation (Table 11) from the main paper. Keep CALVIN Offline BC (Table 13) which has better margins (CosSim 0.479 vs next best 0.475).

**Option B**: If keeping CALVIN, add error bars (3-5 seeds). If the margin is not statistically significant, acknowledge this in the text instead of claiming "#1".

---

## Execution Priority and Order

```
Priority 1 (blocks paper submission):
  FIX 1 — V2 Ablation [~2-3 days GPU time, 10K steps × 6 variants]
  FIX 2 — 5K Retrieval rerun [~2 hours]  
  FIX 3 — Per-seed data fix [~30 min if data exists, rerun if not]

Priority 2 (significantly strengthens paper):
  FIX 4 — Invisible physics redesign [~1-2 days]
  FIX 5 — ManiSkill3 rerun [~1 day]
  FIX 7 — Friction regression fix [requires retrain, ~1-2 days]

Priority 3 (cleanup):
  FIX 6 — World model rerun [~4 hours]
  FIX 8 — CALVIN decision [editorial only]
```

---

## Verification Checklist

After all fixes, verify:
- [ ] Every table in `tables_standalone.tex` uses V2 checkpoint data (or clearly states otherwise)
- [ ] Ablation table baseline config matches `configs/pretrain_v2.yaml`
- [ ] Per-seed data (Table A3) matches main table (Table 9) averages
- [ ] 5K retrieval (Table A4) uses V2 checkpoint
- [ ] Invisible physics experiment has a sound methodology explanation
- [ ] No "(tbd)" or "plausible projection" entries remain
- [ ] Paper text claims match actual table numbers (grep for specific numbers)

---

## Files Summary

| File | Fixes |
|---|---|
| `configs/ablation_v2_*.yaml` | FIX 1 — create 6 variant configs |
| `dynaclip/losses/rnc_loss.py:170-174` | FIX 7 — normalize physics dimensions |
| `dynaclip/data/dataset.py:272-282` | FIX 4 — invisible physics dataset |
| `dynaclip/eval/invisible_physics.py` | FIX 4 — evaluation redesign |
| `dynaclip/eval/linear_probing.py` | FIX 4 Option A — within-category probe |
| `paper/tables_standalone.tex` | FIX 1,2,3,4,5,6 — update all stale tables |
| `paper/results_compilation.txt` | FIX 1,2,3,4,5,6 — update all stale sections |
| `paper/dynaclip_neurips2026.tex` | All fixes — update text to match new numbers |
