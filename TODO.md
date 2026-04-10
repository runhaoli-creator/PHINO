# PHINO: What to Fix and What to Run

> 给接手的人看的完整指南。基于 DynaCLIP repo 的 benchmark 分支。
> 项目目标：NeurIPS 2026 投稿，方法名 PHINO (PHysical DINOv2)。

---

## 当前状态

- 代码在 `benchmark` 分支上，已修了 4 个 bug（轨迹相似度、soft label、fingerprint 路径、WiSE-FT dtype）
- 有部分实验结果（probing、LIBERO、IntPhys、ablation），但来自不同版本，需要统一
- 论文在 Overleaf 上，框架已写好，需要填数据

---

## Part 1: 代码修改（做实验之前先改）

### Fix 1: `evaluate.py` 缺少 baseline loader

`scripts/evaluate.py` 的 `load_backbones()` 函数只支持 7 个 backbone，缺少 r3m、vip、voltron、theia、mcr。加上这些：

```python
# 在 load_backbones() 的 if-elif 链里加：
elif name == "r3m":
    from dynaclip.models.backbones import load_r3m
    bb = load_r3m()
elif name == "vip":
    from dynaclip.models.backbones import load_vip
    bb = load_vip()
elif name == "mcr":
    # MAE ViT-B/16, 需要自己写 loader 或用 timm
    logger.warning("MCR loader not implemented, skipping")
    continue
elif name == "voltron":
    # 需要 voltron 包
    logger.warning("Voltron loader not implemented, skipping")
    continue
elif name == "theia":
    logger.warning("Theia loader not implemented, skipping")
    continue
```

### Fix 2: 确认 `compute_dynamics_similarity_l2` 已修复

检查 `dynaclip/data/generation.py` 里这个函数是否用的是 full trajectory MSE：

```python
def compute_dynamics_similarity_l2(traj1, traj2):
    mse = np.mean((traj1 - traj2) ** 2)  # 应该是这个（全轨迹）
    return float(np.exp(-mse / 5.0))
```

如果还是 `np.linalg.norm(traj1[-1] - traj2[-1])`（只看最后一帧），必须改。

### Fix 3: 确认 SoftInfoNCE off-diagonal

检查 `dynaclip/losses/contrastive.py` 的 `_build_similarity_matrix()`：

```python
# 应该是：
sim_matrix = torch.zeros(B, B, device=device)
sim_matrix.diagonal().copy_(pair_sim)

# 不应该是：
# sim_matrix = torch.sqrt(pair_sim_i * pair_sim_j + 1e-8) * 0.5  ← 旧的有bug的版本
```

---

## Part 2: 数据生成

如果之前没有用修复后的代码生成过数据，需要重新生成：

```bash
python scripts/generate_v3_ablation_data.py \
    --variant full \
    --output_dir data_cache/phino_data \
    --dataset_root <数据集路径，包含 domainnet/real/> \
    --max_images 20000 --num_physics 5
```

这会产出 metadata.json（100K 条）+ fingerprints/（100K 个 .npz）+ similarity_matrix.npz（50 万对）。

消融数据（random、mass_only、fric_only、rest_only）也需要生成：

```bash
for v in random mass_only fric_only rest_only; do
    python scripts/generate_v3_ablation_data.py --variant $v \
        --output_dir data_cache/phino_${v} \
        --dataset_root <数据集路径>
done
```

---

## Part 3: 训练

```bash
torchrun --nproc_per_node=<GPU数> scripts/pretrain.py \
    --config configs/pretrain_v3.yaml \
    --seed 42 \
    data.data_dir=data_cache/phino_data \
    output.checkpoint_dir=checkpoints/phino
```

约 2-3 小时。Checkpoint 存到 `checkpoints/phino/dynaclip_final.pt`。

---

## Part 4: 需要跑的实验（按优先级排序）

### P0: Linear Probing（所有 baseline）

用训好的 PHINO checkpoint + 所有 frozen baseline，在 PHINO 的 eval data 上跑 probing。

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --data_dir data_cache/phino_data \
    --backbones dynaclip dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip r3m vip \
    --experiments linear_probing \
    --output_dir results/probing_all
```

**关键**：所有 baseline 必须在同一份 eval data 上评估。不要用别的版本的 data。

预期：PHINO 在 mass、friction、restitution 上全部 #1。

### P0: LIBERO-10（完整 per-task + per-seed）

```bash
python scripts/evaluate_libero_v4.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --seeds 3 --epochs 200
```

需要产出：
- 10 个 task 的 per-task success rate
- 3 个 seed 的 per-seed breakdown
- 所有 baseline 的对比（至少 DINOv2、CLIP、SigLIP、VC-1、R3M）

预期：PHINO avg ~60-62%，#1。

### P1: Ablation（probing + 可选 LIBERO）

训 4 个消融 variant（各 10K steps）：

```bash
for v in random mass_only fric_only rest_only; do
    torchrun --nproc_per_node=<GPU数> scripts/pretrain.py \
        --config configs/pretrain_v3.yaml --seed 42 \
        training.total_steps=10000 \
        data.data_dir=data_cache/phino_${v} \
        output.checkpoint_dir=checkpoints/ablation_${v}
done
```

然后每个跑 probing：

```bash
for v in random mass_only fric_only rest_only; do
    python scripts/evaluate.py \
        --checkpoint checkpoints/ablation_${v}/dynaclip_final.pt \
        --data_dir data_cache/phino_${v} \
        --backbones dynaclip dinov2_vitb14 \
        --experiments linear_probing \
        --output_dir results/ablation_${v}
done
```

**必须跑 LIBERO 的**：random physics variant（证明因果性）。

### P1: IntPhys2

```bash
# 先下载数据
wget https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip
unzip IntPhys2.zip -d datasets/intphys2

# 跑评估
python scripts/evaluate_intphys.py \
    --data_dir datasets/intphys2 \
    --split Main \
    --checkpoints checkpoints/phino/dynaclip_final.pt \
    --checkpoint_names PHINO \
    --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16 \
    --output_dir results/intphys
```

预期：PHINO > DINOv2（证明 physics alignment 改善了物理推理）。

### P2: t-SNE Mass 可视化

```bash
python scripts/plot_tsne_mass.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt \
    --data_dir data_cache/phino_data \
    --output figures/tsne_mass_comparison.pdf
```

产出 2 panel 图：DINOv2（随机混色）vs PHINO（mass 渐变）。用于论文 Figure。

### P2: DROID-100

```bash
python scripts/evaluate_droid.py \
    --checkpoint checkpoints/phino/dynaclip_final.pt
```

主要看 PHINO 是否 > DINOv2（anti-correlation finding 已有，重跑确认即可）。

### P3: Within-Category Probing（回应 category confound）

这个需要写新代码：在 linear probing 时，先减去 category 均值（去掉 category 信息），只看同一个 category 内部的 physics 预测能力。如果 R² > 0，说明 model 学到了超越 category 的物理信息。

---

## Part 5: 结果汇总

跑完后把所有结果整理到 `RESULTS.md`，格式参考当前的 RESULTS.md。需要的表：

1. **Linear Probing Table**: PHINO + 所有 baseline 的 mass/friction/restitution R²
2. **LIBERO-10 Table**: per-task × per-backbone success rate
3. **Ablation Table**: random / mass_only / fric_only / rest_only / full 的 probing
4. **IntPhys2 Table**: Overall Acc per backbone
5. **Compute Cost Table**: params / latency / throughput

---

## 文件结构参考

```
DynaCLIP/
├── configs/pretrain_v3.yaml          # 训练配置
├── dynaclip/
│   ├── data/generation.py            # 物理引擎 + 数据生成（确认 bug 已修）
│   ├── data/dataset.py               # Dataset（确认 fast path + trajectory cache）
│   ├── losses/contrastive.py         # SoftInfoNCE（确认 off-diagonal = 0）
│   ├── trainers/pretrain.py          # 训练循环（确认 WiSE-FT dtype fix）
│   └── models/backbones.py           # Backbone registry（加缺少的 loader）
├── scripts/
│   ├── pretrain.py                   # 训练入口
│   ├── evaluate.py                   # Probing 评估（加缺少的 backbone）
│   ├── evaluate_intphys.py           # IntPhys2 评估
│   ├── evaluate_libero_v4.py         # LIBERO 评估
│   ├── generate_v3_ablation_data.py  # 消融数据生成
│   └── plot_tsne_mass.py             # t-SNE 可视化
├── RESULTS.md                        # 结果汇总（跑完后更新）
├── IMPLEMENTATION.md                 # 方法实现说明
└── TODO.md                           # 本文件
```
