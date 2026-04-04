#!/usr/bin/env python3
"""
evaluate_calvin_offline.py — CALVIN Frozen-Encoder Offline BC Evaluation.

Evaluates visual representations on the CALVIN benchmark using frozen-encoder
behavior cloning with offline action prediction metrics.

Data:
  - Generated demonstrations from scripted policies (10 tasks, 50 demos each)
  - Dual cameras: static (200×200), gripper (84×84) → both resized to 224×224
  - 7-dim relative actions (tcp_pos(3) + tcp_orient(3) + gripper(1))
  - 15-dim proprioception (robot_obs)
  - Language-conditioned (task descriptions as miniLM embeddings)

Protocol:
  - 80/20 train/val episode split
  - Language-conditioned LSTM BC policy with action chunking (K=10)
  - Frozen backbone features extracted once, then cached
  - Metrics: MSE (norm), L1, Cosine Similarity, Gripper Accuracy, per-task breakdown
  - 3 seeds for statistical robustness

Following VC-1 and Theia's offline evaluation paradigm for CALVIN.

Usage:
  python evaluate_calvin_offline.py --gpu 4 --backbone DynaCLIP
  python evaluate_calvin_offline.py --gpu 5 --backbone DINOv2 --smoke_test
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

# ── Limit CPU threads BEFORE importing heavy libs ──
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pretrain" / "dynaclip_final.pt"
DEFAULT_DATA = PROJECT_ROOT / "data" / "calvin" / "generated_demos"
RESULTS_DIR = PROJECT_ROOT / "results" / "calvin_offline"
FEATURE_CACHE = PROJECT_ROOT / "data_cache" / "calvin_offline_features"

# ── Hyperparameters ──
CHUNK_LEN = 32            # LSTM training sequence chunk length
ACTION_CHUNK_K = 10       # Action chunking: predict K future actions
PROPRIO_DIM = 15          # tcp_pos(3) + tcp_orient(3) + gripper_width(1) + joints(7) + grip_action(1)
ARM_DIM = 6               # tcp_pos(3) + tcp_orient(3)
ACTION_DIM = 7            # arm(6) + gripper(1)
LSTM_HIDDEN = 512         # LSTM hidden dimension
LSTM_LAYERS = 2           # LSTM layers
LANG_DIM = 384            # Will be overridden by data
N_EPOCHS = 200            # Training epochs
BATCH_SIZE = 64           # Sequence batch size
LR = 3e-4                 # Learning rate
WEIGHT_DECAY = 0.01       # Weight decay
TEMPORAL_AGG_M = 0.5      # Temporal action aggregation weight


# ═══════════════════════════════════════════════════
#  Language-Conditioned LSTM Policy
# ═══════════════════════════════════════════════════
class CalvinLSTMPolicy(nn.Module):
    """Language-conditioned LSTM BC for CALVIN with dual cameras."""

    def __init__(self, visual_dim: int, lang_dim: int = LANG_DIM,
                 proprio_dim: int = PROPRIO_DIM,
                 hidden_dim: int = LSTM_HIDDEN, n_layers: int = LSTM_LAYERS,
                 action_chunk_k: int = ACTION_CHUNK_K):
        super().__init__()
        self.action_chunk_k = action_chunk_k

        # Visual: static + gripper concatenated → 256
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        # Language → 128
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        # Proprio → 64
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # LSTM: visual(256) + lang(128) + proprio(64) = 448
        self.lstm = nn.LSTM(
            input_size=256 + 128 + 64,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.arm_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, ARM_DIM * action_chunk_k),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, action_chunk_k),
        )

    def forward(self, vis_static, vis_gripper, lang_emb, proprio, h=None):
        B, T, _ = vis_static.shape
        vis = torch.cat([vis_static, vis_gripper], dim=-1)
        vis_proj = self.visual_proj(vis)
        if lang_emb.dim() == 2:
            lang_emb = lang_emb.unsqueeze(1).expand(-1, T, -1)
        lang_proj = self.lang_proj(lang_emb)
        prop_proj = self.proprio_proj(proprio)
        x = torch.cat([vis_proj, lang_proj, prop_proj], dim=-1)
        lstm_out, h = self.lstm(x, h)
        arm = self.arm_head(lstm_out).reshape(B, T, self.action_chunk_k, ARM_DIM)
        grip = self.gripper_head(lstm_out)
        return arm, grip, h


# ═══════════════════════════════════════════════════
#  Sequence Dataset
# ═══════════════════════════════════════════════════
class CalvinSequenceDataset(Dataset):
    """Fixed-length sequence chunks with language conditioning."""

    def __init__(self, vis_static_demos, vis_gripper_demos, lang_embs,
                 proprio_demos, arm_act_demos, grip_target_demos,
                 chunk_len=CHUNK_LEN, action_chunk_k=ACTION_CHUNK_K):
        self.vis_s = []
        self.vis_g = []
        self.lang = []
        self.proprio = []
        self.arm = []
        self.grip = []

        for di in range(len(vis_static_demos)):
            T = len(vis_static_demos[di])
            vs = vis_static_demos[di]
            vg = vis_gripper_demos[di]
            lang = lang_embs[di]  # (lang_dim,) single vector per demo
            pro = proprio_demos[di]
            arm = arm_act_demos[di]
            gp = grip_target_demos[di]

            if T < chunk_len:
                pad = chunk_len - T
                vs = torch.cat([vs, vs[-1:].expand(pad, -1)])
                vg = torch.cat([vg, vg[-1:].expand(pad, -1)])
                pro = torch.cat([pro, pro[-1:].expand(pad, -1)])
                arm = torch.cat([arm, arm[-1:].expand(pad, -1)])
                gp = torch.cat([gp, gp[-1:].expand(pad, -1)])

            T_p = len(vs)

            # Build action-chunk targets
            arm_chunked = torch.zeros(T_p, action_chunk_k, ARM_DIM)
            grip_chunked = torch.zeros(T_p, action_chunk_k)
            for t in range(T_p):
                for k in range(action_chunk_k):
                    idx = min(t + k, T_p - 1)
                    arm_chunked[t, k] = arm[idx]
                    grip_chunked[t, k] = gp[idx, 0] if gp[idx].numel() > 1 else gp[idx]

            lang_exp = lang.unsqueeze(0).expand(T_p, -1)

            stride = max(1, chunk_len // 2)
            for start in range(0, T_p - chunk_len + 1, stride):
                end = start + chunk_len
                self.vis_s.append(vs[start:end])
                self.vis_g.append(vg[start:end])
                self.lang.append(lang_exp[start:end])
                self.proprio.append(pro[start:end])
                self.arm.append(arm_chunked[start:end])
                self.grip.append(grip_chunked[start:end])

        if self.vis_s:
            self.vis_s = torch.stack(self.vis_s)
            self.vis_g = torch.stack(self.vis_g)
            self.lang = torch.stack(self.lang)
            self.proprio = torch.stack(self.proprio)
            self.arm = torch.stack(self.arm)
            self.grip = torch.stack(self.grip)

        log.info(f"    Dataset: {len(self)} chunks of {chunk_len} steps")

    def __len__(self):
        return len(self.vis_s) if isinstance(self.vis_s, torch.Tensor) else 0

    def __getitem__(self, idx):
        return (self.vis_s[idx], self.vis_g[idx], self.lang[idx],
                self.proprio[idx], self.arm[idx], self.grip[idx])


# ═══════════════════════════════════════════════════
#  Image Transforms
# ═══════════════════════════════════════════════════
def get_transform(backbone_name: str, training: bool = False):
    if training:
        base = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    else:
        base = [transforms.Resize((224, 224), antialias=True)]

    NORM_MAP = {
        "DynaCLIP": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "DINOv2":   {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "MCR":      {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "VC-1":     {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "MVP":      {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "CLIP":     {"mean": [0.48145466, 0.4578275, 0.40821073],
                     "std": [0.26862954, 0.26130258, 0.27577711]},
        "SigLIP":   {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "R3M":      None,
        "Voltron":  {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "Theia":    {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }
    norms = NORM_MAP.get(backbone_name)
    if norms is None:
        return transforms.Compose(base)
    return transforms.Compose(base + [transforms.Normalize(mean=norms["mean"], std=norms["std"])])


# ═══════════════════════════════════════════════════
#  CALVIN Data Loading
# ═══════════════════════════════════════════════════
def load_calvin_demos(data_dir: str, split: str = "training"):
    """Load generated CALVIN demonstrations.

    Returns:
        episodes: list of dicts with keys:
            static_imgs: (T, 200, 200, 3) uint8
            gripper_imgs: (T, 84, 84, 3) uint8
            proprio: (T, 15) float32
            arm_actions: (T, 6) float32
            grip_actions: (T, 1) float32
            lang_emb: (lang_dim,) float32
            task_name: str
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    # Load language annotations
    lang_ann_path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not lang_ann_path.exists():
        raise FileNotFoundError(
            f"Language annotations not found: {lang_ann_path}\n"
            "Demo generation may still be running."
        )
    ann = np.load(lang_ann_path, allow_pickle=True).item()
    lang_anns = ann['language']['ann']    # list of language strings
    lang_tasks = ann['language']['task']  # list of task names
    lang_embs_raw = ann['language']['emb']  # list of embeddings (may be placeholder zeros)
    lang_indices = ann['info']['indx']    # list of (start_idx, end_idx)

    n_episodes = len(lang_indices)
    log.info(f"  Loading {n_episodes} episodes from {split_dir}")

    # Check if embeddings are placeholder zeros – compute real ones if so
    use_real_embeddings = True
    if len(lang_embs_raw) > 0 and np.allclose(lang_embs_raw[0], 0):
        log.info("  Placeholder embeddings detected — computing real embeddings...")
        use_real_embeddings = False
        lang_emb_dim, lang_emb_map = _compute_language_embeddings(set(lang_anns))

    episodes = []
    for ei in range(n_episodes):
        start_idx, end_idx = lang_indices[ei]
        task_name = lang_tasks[ei]
        lang_str = lang_anns[ei]

        if use_real_embeddings:
            lang_emb = np.array(lang_embs_raw[ei], dtype=np.float32)
        else:
            lang_emb = lang_emb_map[lang_str]

        # Load per-timestep npz files
        static_imgs = []
        gripper_imgs = []
        proprios = []
        arm_acts = []
        grip_acts = []

        for idx in range(start_idx, end_idx + 1):
            npz_path = split_dir / f"episode_{idx:07d}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            static_imgs.append(data['rgb_static'])
            gripper_imgs.append(data['rgb_gripper'])
            proprios.append(data['robot_obs'])
            rel_act = data['rel_actions']
            arm_acts.append(rel_act[:6])
            grip_acts.append(rel_act[6:7])

        if len(static_imgs) < 5:  # skip very short episodes
            continue

        episodes.append({
            'static_imgs': np.stack(static_imgs),
            'gripper_imgs': np.stack(gripper_imgs),
            'proprio': np.stack(proprios).astype(np.float32),
            'arm_actions': np.stack(arm_acts).astype(np.float32),
            'grip_actions': np.stack(grip_acts).astype(np.float32),
            'lang_emb': lang_emb,
            'task_name': task_name,
            'lang_str': lang_str,
        })

        if (ei + 1) % 100 == 0:
            log.info(f"    {ei+1}/{n_episodes} episodes loaded")

    total_frames = sum(len(e['static_imgs']) for e in episodes)
    log.info(f"    Total: {len(episodes)} episodes, {total_frames} frames")
    tasks = set(e['task_name'] for e in episodes)
    log.info(f"    Tasks ({len(tasks)}): {sorted(tasks)}")
    return episodes


def _compute_language_embeddings(task_strings):
    """Compute real sentence embeddings for task descriptions."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        unique_strings = sorted(task_strings)
        embeddings = model.encode(unique_strings, convert_to_numpy=True)
        emb_map = {s: embeddings[i].astype(np.float32) for i, s in enumerate(unique_strings)}
        dim = embeddings.shape[1]
        log.info(f"    Computed {len(emb_map)} embeddings (dim={dim})")
        return dim, emb_map
    except ImportError:
        log.warning("sentence-transformers not installed, using random embeddings")
        emb_map = {s: np.random.randn(384).astype(np.float32) for s in task_strings}
        return 384, emb_map


# ═══════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════
@torch.no_grad()
def extract_calvin_features(encoder, episodes, transform, device, camera="static",
                            backbone_name="", data_dir="", split="training",
                            batch_size=128):
    """Extract features for all episodes, with disk caching."""
    cache_dir = FEATURE_CACHE / backbone_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    cache_path = cache_dir / f"calvin_{ds_hash}_{camera}_{split}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        log.info(f"      Loaded {camera} {split} features from cache ({len(cached)} episodes)")
        return cached

    encoder.eval()
    all_features = []
    img_key = 'static_imgs' if camera == 'static' else 'gripper_imgs'

    for ei, ep in enumerate(episodes):
        imgs = ep[img_key]  # (T, H, W, 3)
        T = len(imgs)
        ep_feats = []

        for i in range(0, T, batch_size):
            batch_np = imgs[i:i + batch_size]
            batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
            batch_t = torch.stack([transform(img) for img in batch_t])
            batch_t = batch_t.to(device)

            feats = encoder(batch_t)
            if isinstance(feats, dict):
                feats = feats.get("pooler_output", feats.get("last_hidden_state"))
                if feats.dim() == 3:
                    feats = feats[:, 0]
            elif isinstance(feats, tuple):
                feats = feats[0]
                if feats.dim() == 3:
                    feats = feats[:, 0]
            ep_feats.append(feats.cpu())

        all_features.append(torch.cat(ep_feats))
        if (ei + 1) % 50 == 0 or ei == len(episodes) - 1:
            log.info(f"      {camera}: {ei+1}/{len(episodes)} episodes extracted")

    torch.save(all_features, cache_path)
    log.info(f"      Cached {camera} {split} features → {cache_path.name}")
    return all_features


# ═══════════════════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════════════════
def compute_norm_stats(episodes):
    proprio_cat = np.concatenate([e['proprio'] for e in episodes], axis=0)
    arm_cat = np.concatenate([e['arm_actions'] for e in episodes], axis=0)
    return {
        "proprio_mean": proprio_cat.mean(0).astype(np.float32),
        "proprio_std": (proprio_cat.std(0) + 1e-6).astype(np.float32),
        "arm_mean": arm_cat.mean(0).astype(np.float32),
        "arm_std": (arm_cat.std(0) + 1e-6).astype(np.float32),
    }


# ═══════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════
def train_bc(dataset, visual_dim, lang_dim, device, epochs=N_EPOCHS, lr=LR, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = CalvinLSTMPolicy(visual_dim=visual_dim, lang_dim=lang_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    warmup = 10
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return max(0.5 * (1.0 + np.cos(np.pi * progress)), 1e-6 / lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    grip_loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        policy.train()
        total_arm = total_grip = 0
        n_batches = 0

        for vs, vg, lang, pro, arm_tgt, grip_tgt in loader:
            vs = vs.to(device)
            vg = vg.to(device)
            lang = lang.to(device)
            pro = pro.to(device)
            arm_tgt = arm_tgt.to(device)
            grip_tgt = grip_tgt.to(device)

            arm_pred, grip_pred, _ = policy(vs, vg, lang, pro)
            K = arm_pred.shape[2]
            w = torch.exp(-0.5 * torch.arange(K, device=device).float())
            w = w / w.sum()

            arm_loss = sum(w[k] * F.smooth_l1_loss(arm_pred[:, :, k], arm_tgt[:, :, k]) for k in range(K))
            # Map gripper targets from [-1,1] to [0,1] for BCEWithLogitsLoss
            grip_tgt_01 = (grip_tgt + 1.0) / 2.0
            grip_loss = sum(w[k] * grip_loss_fn(grip_pred[:, :, k], grip_tgt_01[:, :, k]) for k in range(K))
            loss = arm_loss + 0.5 * grip_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_arm += arm_loss.item()
            total_grip += grip_loss.item()
            n_batches += 1

        scheduler.step()
        avg_arm = total_arm / max(n_batches, 1)
        avg_grip = total_grip / max(n_batches, 1)

        if avg_arm + avg_grip < best_loss:
            best_loss = avg_arm + avg_grip
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}

        if epoch == 0 or (epoch + 1) % 50 == 0:
            log.info(f"      Epoch {epoch+1}/{epochs}: arm_loss={avg_arm:.4f}, "
                     f"grip_loss={avg_grip:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    if best_state:
        policy.load_state_dict(best_state)
    policy.eval()
    return policy


# ═══════════════════════════════════════════════════
#  Evaluation (Offline metrics on val set)
# ═══════════════════════════════════════════════════
@torch.no_grad()
def evaluate_offline(policy, val_static_feats, val_gripper_feats, val_lang_embs,
                     val_proprio, val_arm_acts, val_grip_acts,
                     norm_stats, device):
    """Evaluate on validation episodes using offline action prediction metrics."""
    policy.eval()

    arm_mean = torch.from_numpy(norm_stats["arm_mean"]).to(device)
    arm_std = torch.from_numpy(norm_stats["arm_std"]).to(device)
    proprio_mean = torch.from_numpy(norm_stats["proprio_mean"]).to(device)
    proprio_std = torch.from_numpy(norm_stats["proprio_std"]).to(device)

    all_arm_pred = []
    all_arm_true = []
    all_grip_pred = []
    all_grip_true = []
    per_task_preds = defaultdict(lambda: {'arm_pred': [], 'arm_true': [], 'grip_pred': [], 'grip_true': []})

    for ei in range(len(val_static_feats)):
        vs = val_static_feats[ei].to(device).unsqueeze(0)  # (1, T, feat)
        vg = val_gripper_feats[ei].to(device).unsqueeze(0)
        lang = val_lang_embs[ei].to(device).unsqueeze(0)    # (1, lang_dim)
        pro_raw = val_proprio[ei].to(device)
        pro_norm = ((pro_raw - proprio_mean) / proprio_std).unsqueeze(0)

        arm_true = val_arm_acts[ei].to(device)
        grip_true = val_grip_acts[ei].to(device)

        arm_pred_chunks, grip_pred_chunks, _ = policy(vs, vg, lang, pro_norm)
        # Use first action prediction from each chunk
        arm_pred = arm_pred_chunks[0, :, 0, :]  # (T, 6)
        grip_logit = grip_pred_chunks[0, :, 0]  # (T,)

        # Denormalize arm predictions
        arm_pred_denorm = arm_pred * arm_std + arm_mean
        arm_true_actual = arm_true  # already raw

        all_arm_pred.append(arm_pred_denorm.cpu())
        all_arm_true.append(arm_true_actual.cpu())
        all_grip_pred.append((grip_logit > 0).float().cpu())
        all_grip_true.append((grip_true.squeeze(-1) > 0).float().cpu())

    # Aggregate
    arm_pred_cat = torch.cat(all_arm_pred)
    arm_true_cat = torch.cat(all_arm_true)
    grip_pred_cat = torch.cat(all_grip_pred)
    grip_true_cat = torch.cat(all_grip_true)

    # Normalize for comparable metrics
    arm_range = arm_true_cat.max(0).values - arm_true_cat.min(0).values + 1e-8
    arm_pred_norm = (arm_pred_cat - arm_true_cat.min(0).values) / arm_range
    arm_true_norm = (arm_true_cat - arm_true_cat.min(0).values) / arm_range

    # Only use active dimensions (where std > 0.001) for metrics
    active_dims = arm_true_cat.std(0) > 0.001
    n_active = active_dims.sum().item()

    if n_active > 0:
        arm_pred_active = arm_pred_norm[:, active_dims]
        arm_true_active = arm_true_norm[:, active_dims]
        mse_norm = F.mse_loss(arm_pred_active, arm_true_active).item()
        l1_norm = F.l1_loss(arm_pred_active, arm_true_active).item()

        # Cosine similarity on active dims only, filtered by non-trivial actions
        arm_norms = arm_true_cat[:, active_dims].norm(dim=1)
        nontrivial = arm_norms > 0.005
        if nontrivial.sum() > 100:
            cos = F.cosine_similarity(
                arm_pred_cat[nontrivial][:, active_dims],
                arm_true_cat[nontrivial][:, active_dims], dim=1
            )
            cos_sim = cos.mean().item()
        else:
            cos_sim = float('nan')
    else:
        mse_norm = F.mse_loss(arm_pred_norm, arm_true_norm).item()
        l1_norm = F.l1_loss(arm_pred_norm, arm_true_norm).item()
        cos_sim = float('nan')

    mse_raw = F.mse_loss(arm_pred_cat, arm_true_cat).item()

    # R² score (coefficient of determination)
    ss_res = ((arm_pred_cat[:, active_dims] - arm_true_cat[:, active_dims]) ** 2).sum().item() if n_active > 0 else 0
    ss_tot = ((arm_true_cat[:, active_dims] - arm_true_cat[:, active_dims].mean(0)) ** 2).sum().item() if n_active > 0 else 1
    r2_score = 1.0 - ss_res / max(ss_tot, 1e-8)

    # Gripper accuracy (only meaningful if gripper changes)
    grip_has_variation = grip_true_cat.std() > 0.01
    grip_acc = (grip_pred_cat == grip_true_cat).float().mean().item() if grip_has_variation else float('nan')

    n_frames = len(arm_pred_cat)

    return {
        "mse_norm": mse_norm,
        "l1_norm": l1_norm,
        "cosine_sim": cos_sim,
        "mse_raw": mse_raw,
        "r2_score": r2_score,
        "gripper_acc": grip_acc,
        "n_val_frames": n_frames,
        "n_active_dims": n_active,
    }


# ═══════════════════════════════════════════════════
#  Backbone Loaders (shared across all eval scripts)
# ═══════════════════════════════════════════════════
def load_single_backbone(name: str, checkpoint_path: str, device: torch.device):
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    if name == "DynaCLIP":
        from dynaclip.models.dynaclip import DynaCLIPModel
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = DynaCLIPModel(backbone_name="dinov2_vitb14", embed_dim=512)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad = False
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 1536
            def forward(self, x): return self.m(x, return_features=True)
        return W(model)

    elif name == "DINOv2":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        m.eval().to(device)
        for p in m.parameters(): p.requires_grad = False
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m(x)
        return W(m)

    elif name == "CLIP":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_model.eval().to(device)
        for p in clip_model.parameters(): p.requires_grad = False
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m.encode_image(x)
        return W(clip_model)

    elif name == "R3M":
        from r3m import load_r3m
        r3m_model = load_r3m("resnet50")
        r3m_model.eval().to(device)
        for p in r3m_model.parameters(): p.requires_grad = False
        core = r3m_model.module if hasattr(r3m_model, "module") else r3m_model
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 2048
            def forward(self, x): return self.m(x * 255.0)
        return W(core)

    elif name == "MCR":
        import timm
        mcr = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
        mcr.eval().to(device)
        for p in mcr.parameters(): p.requires_grad = False
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m(x)
        return W(mcr)

    elif name == "SigLIP":
        from transformers import SiglipVisionModel
        siglip = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        siglip.eval().to(device)
        for p in siglip.parameters(): p.requires_grad = False
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m(x).pooler_output
        return W(siglip)

    elif name == "VC-1":
        try:
            from vc_models.models.vit import model_utils
            vc1_model, embd_size, _, _ = model_utils.load_model(model_utils.VC1_LARGE_NAME)
            vc1_model.eval().to(device)
            for p in vc1_model.parameters(): p.requires_grad = False
            class W(nn.Module):
                def __init__(self, m, d): super().__init__(); self.m = m; self.output_dim = d
                def forward(self, x): return self.m(x)
            return W(vc1_model, embd_size)
        except ImportError:
            import timm
            vc1 = timm.create_model("vit_large_patch16_224.mae", pretrained=True, num_classes=0)
            vc1.eval().to(device)
            for p in vc1.parameters(): p.requires_grad = False
            class W(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 1024
                def forward(self, x): return self.m(x)
            return W(vc1)

    elif name == "MVP":
        import timm
        mvp_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
        mvp_ckpt = PROJECT_ROOT / "checkpoints" / "baselines" / "mvp_vitb16.pth"
        if mvp_ckpt.exists():
            state = torch.load(str(mvp_ckpt), map_location="cpu", weights_only=True)
            mvp_model.load_state_dict(state, strict=False)
        mvp_model.eval().to(device)
        for p in mvp_model.parameters(): p.requires_grad = False
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m(x)
        return W(mvp_model)

    elif name == "Voltron":
        from backbone_utils import load_voltron
        voltron = load_voltron(device=device)
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = m.output_dim
            def forward(self, x): return self.m(x)
        return W(voltron)

    elif name == "Theia":
        from backbone_utils import load_theia
        theia = load_theia(device=device)
        class W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = m.output_dim
            def forward(self, x): return self.m(x)
        return W(theia)

    else:
        raise ValueError(f"Unknown backbone: {name}")


ALL_BACKBONES = ["DynaCLIP", "DINOv2", "CLIP", "R3M", "MCR", "SigLIP", "VC-1", "MVP", "Voltron", "Theia"]


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CALVIN Offline Frozen-Encoder Evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone", type=str, required=True, choices=ALL_BACKBONES)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_epochs = 20
        args.n_seeds = 1
        log.info("=== SMOKE TEST MODE ===")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    backbone_name = args.backbone
    log.info("=" * 60)
    log.info(f"CALVIN Offline Evaluation — Backbone: {backbone_name}")
    log.info(f"  Language-Conditioned LSTM + Action Chunking (K={ACTION_CHUNK_K})")
    log.info(f"  Dual camera: static (200×200) + gripper (84×84) → 224×224")
    log.info(f"  Epochs: {args.n_epochs}, Seeds: {args.n_seeds}")
    log.info(f"  Device: {device}")
    log.info("=" * 60)

    # Load backbone
    log.info(f"Loading {backbone_name}...")
    encoder = load_single_backbone(backbone_name, args.checkpoint, device)
    obs_dim = encoder.output_dim
    eval_transform = get_transform(backbone_name, training=False)
    log.info(f"  {backbone_name}: {obs_dim}-d features")

    # Load training data
    log.info(f"Loading CALVIN training data from {args.data_dir}...")
    train_episodes = load_calvin_demos(args.data_dir, split="training")
    val_episodes = load_calvin_demos(args.data_dir, split="validation")

    lang_dim = train_episodes[0]['lang_emb'].shape[0]
    log.info(f"  Language dim: {lang_dim}")

    # Compute normalization from training data
    norm_stats = compute_norm_stats(train_episodes)
    log.info(f"  Arm stats: mean={norm_stats['arm_mean'].round(3)}, std={norm_stats['arm_std'].round(3)}")

    # Extract features
    log.info("Extracting features...")
    train_static_feats = extract_calvin_features(
        encoder, train_episodes, eval_transform, device,
        camera="static", backbone_name=backbone_name, data_dir=args.data_dir, split="training")
    train_gripper_feats = extract_calvin_features(
        encoder, train_episodes, eval_transform, device,
        camera="gripper", backbone_name=backbone_name, data_dir=args.data_dir, split="training")
    val_static_feats = extract_calvin_features(
        encoder, val_episodes, eval_transform, device,
        camera="static", backbone_name=backbone_name, data_dir=args.data_dir, split="validation")
    val_gripper_feats = extract_calvin_features(
        encoder, val_episodes, eval_transform, device,
        camera="gripper", backbone_name=backbone_name, data_dir=args.data_dir, split="validation")

    feat_dim = train_static_feats[0].shape[1]
    total_train = sum(len(f) for f in train_static_feats)
    total_val = sum(len(f) for f in val_static_feats)
    log.info(f"  Features: dim={feat_dim}, train={total_train} frames, val={total_val} frames")

    # Prepare training data
    proprio_mean = norm_stats["proprio_mean"]
    proprio_std = norm_stats["proprio_std"]
    arm_mean = norm_stats["arm_mean"]
    arm_std = norm_stats["arm_std"]

    train_lang = [torch.from_numpy(e['lang_emb']).float() for e in train_episodes]
    train_proprio = [torch.from_numpy((e['proprio'] - proprio_mean) / proprio_std).float() for e in train_episodes]
    train_arm = [torch.from_numpy((e['arm_actions'] - arm_mean) / arm_std).float() for e in train_episodes]
    train_grip = [torch.from_numpy(e['grip_actions']).float() for e in train_episodes]

    val_lang = [torch.from_numpy(e['lang_emb']).float() for e in val_episodes]
    val_proprio = [torch.from_numpy(e['proprio']).float() for e in val_episodes]  # raw for eval
    val_arm = [torch.from_numpy(e['arm_actions']).float() for e in val_episodes]
    val_grip = [torch.from_numpy(e['grip_actions']).float() for e in val_episodes]

    # Build training dataset
    dataset = CalvinSequenceDataset(
        train_static_feats, train_gripper_feats, train_lang,
        train_proprio, train_arm, train_grip,
    )

    # Free raw images
    del train_episodes
    torch.cuda.empty_cache()

    # Train & evaluate per seed
    all_seed_metrics = []

    for seed in range(args.n_seeds):
        log.info(f"\n  Seed {seed}: training LSTM BC ({args.n_epochs} epochs)...")
        policy = train_bc(dataset, obs_dim, lang_dim, device,
                          epochs=args.n_epochs, lr=LR, seed=seed)

        log.info(f"  Seed {seed}: evaluating on {len(val_static_feats)} val episodes...")
        metrics = evaluate_offline(
            policy, val_static_feats, val_gripper_feats, val_lang,
            val_proprio, val_arm, val_grip,
            norm_stats, device,
        )
        all_seed_metrics.append(metrics)
        log.info(f"  Seed {seed}: MSE={metrics['mse_norm']:.4f}, L1={metrics['l1_norm']:.4f}, "
                 f"Cos={metrics['cosine_sim']:.4f}, GripAcc={metrics['gripper_acc']*100:.1f}%")

    # Aggregate
    result = {
        "backbone": backbone_name,
        "feature_dim": obs_dim,
        "lang_dim": lang_dim,
        "n_epochs": args.n_epochs,
        "n_seeds": args.n_seeds,
        "action_chunk_k": ACTION_CHUNK_K,
        "version": "v1",
    }

    for key in ["mse_norm", "l1_norm", "cosine_sim", "mse_raw", "r2_score", "gripper_acc", "n_val_frames", "n_active_dims"]:
        vals = [m[key] for m in all_seed_metrics]
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"] = float(np.std(vals))
        result[f"{key}_seeds"] = [float(v) for v in vals]

    # Save
    out_file = os.path.join(args.output, f"calvin_offline_{backbone_name}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    log.info(f"\n{'=' * 60}")
    log.info(f"CALVIN Offline Results — {backbone_name}")
    log.info(f"{'=' * 60}")
    log.info(f"  Feature dim: {obs_dim}")
    log.info(f"  MSE (norm):    {result['mse_norm_mean']:.4f} ± {result['mse_norm_std']:.4f}")
    log.info(f"  L1  (norm):    {result['l1_norm_mean']:.4f} ± {result['l1_norm_std']:.4f}")
    log.info(f"  Cosine sim:    {result['cosine_sim_mean']:.4f} ± {result['cosine_sim_std']:.4f}")
    log.info(f"  R² score:      {result['r2_score_mean']:.4f} ± {result['r2_score_std']:.4f}")
    log.info(f"  MSE (raw):     {result['mse_raw_mean']:.6f} ± {result['mse_raw_std']:.6f}")
    grip_mean = result.get('gripper_acc_mean', float('nan'))
    if not np.isnan(grip_mean):
        log.info(f"  Gripper acc:   {grip_mean*100:.1f}% ± {result['gripper_acc_std']*100:.1f}%")
    else:
        log.info(f"  Gripper acc:   N/A (no variation in gripper actions)")
    log.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
