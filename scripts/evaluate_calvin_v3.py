#!/usr/bin/env python3
"""
evaluate_calvin_v3.py — CALVIN Simulator-Based Frozen-Encoder BC Evaluation.

Evaluates frozen visual encoders on the CALVIN benchmark using the official
LH-MTLC (Long-Horizon Multi-Task Language Control) protocol:
  - Train language-conditioned LSTM BC policy on frozen features
  - Evaluate via simulator rollout on 1000 chains of 5 tasks each
  - Report: avg chain length, per-chain completion rates (1/5 through 5/5)

Architecture:
  - Dual cameras: static (200x200) + gripper (84x84), resized to 224x224
  - Language embedding: pre-computed miniLM or CLIP embeddings
  - Policy: language-conditioned LSTM BC with action chunking (K=10)
  - Actions: 7-dim relative (tcp_pos(3), tcp_orient(3), gripper(1))

Following the frozen-encoder evaluation paradigm of VC-1 and Theia:
  - Backbone weights are FROZEN (no gradient)
  - Only the policy head is trained
  - This isolates the encoder's contribution to downstream performance

Usage:
  python evaluate_calvin_v3.py --gpu 4 --backbone DynaCLIP --dataset_path /path/to/task_D_D
  python evaluate_calvin_v3.py --gpu 5 --backbone DINOv2 --dataset_path /path/to/task_D_D
  python evaluate_calvin_v3.py --gpu 7 --backbone CLIP --smoke_test
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

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
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pretrain" / "dynaclip_final.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "calvin_v3"
FEATURE_CACHE = PROJECT_ROOT / "data_cache" / "calvin_features_v3"

# ── Hyperparameters ──
CHUNK_LEN = 32            # LSTM training sequence chunk length (CALVIN has shorter episodes)
ACTION_CHUNK_K = 10       # Action chunking: predict K future actions
PROPRIO_DIM = 15          # tcp_pos(3) + tcp_orient(3) + gripper_width(1) + joints(7) + grip_action(1)
ACTION_DIM = 7            # rel: tcp_pos(3) + tcp_orient(3) + gripper(1)
ARM_DIM = 6               # tcp_pos(3) + tcp_orient(3)
LSTM_HIDDEN = 512         # LSTM hidden dimension
LSTM_LAYERS = 2           # LSTM layers
LANG_DIM = 384            # miniLM embedding dimension (default)
N_EPOCHS = 200            # Training epochs
BATCH_SIZE = 64           # Sequence batch size
LR = 3e-4                 # Learning rate
WEIGHT_DECAY = 0.01       # Weight decay
EP_LEN = 360              # Max steps per subtask (CALVIN standard)
NUM_SEQUENCES = 1000      # Number of evaluation chains (CALVIN standard)
TEMPORAL_AGG_M = 0.5      # Temporal action aggregation weight


# ═══════════════════════════════════════════════════
#  Language-Conditioned LSTM Policy with Action Chunking
# ═══════════════════════════════════════════════════
class CalvinLSTMPolicy(nn.Module):
    """Language-conditioned LSTM BC policy for CALVIN.
    
    Takes: static visual features + gripper visual features + language embedding + proprio
    Outputs: K-step action chunks (6-dim arm + 1-dim gripper)
    """

    def __init__(self, visual_dim: int, lang_dim: int = LANG_DIM,
                 proprio_dim: int = PROPRIO_DIM,
                 hidden_dim: int = LSTM_HIDDEN, n_layers: int = LSTM_LAYERS,
                 action_chunk_k: int = ACTION_CHUNK_K):
        super().__init__()
        self.action_chunk_k = action_chunk_k

        # Visual projection: static + gripper cameras → 256
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        # Language projection → 128
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        # Proprio projection → 64
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
        # Action heads with chunking
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
        """
        Args:
            vis_static:  (B, T, feat_dim)
            vis_gripper: (B, T, feat_dim)
            lang_emb:    (B, T, lang_dim) or (B, lang_dim) — broadcast across T
            proprio:     (B, T, proprio_dim)
            h:           optional LSTM hidden state
        Returns:
            arm_pred:  (B, T, K, 6)
            grip_pred: (B, T, K) logits
            h:         updated hidden state
        """
        B, T, _ = vis_static.shape
        vis = torch.cat([vis_static, vis_gripper], dim=-1)
        vis_proj = self.visual_proj(vis)

        # Handle lang broadcast
        if lang_emb.dim() == 2:
            lang_emb = lang_emb.unsqueeze(1).expand(-1, T, -1)
        lang_proj = self.lang_proj(lang_emb)

        prop_proj = self.proprio_proj(proprio)
        x = torch.cat([vis_proj, lang_proj, prop_proj], dim=-1)
        lstm_out, h = self.lstm(x, h)
        arm = self.arm_head(lstm_out).reshape(B, T, self.action_chunk_k, ARM_DIM)
        grip = self.gripper_head(lstm_out)
        return arm, grip, h

    def forward_step(self, vis_static, vis_gripper, lang_emb, proprio, h=None):
        """Single-step forward for evaluation."""
        if vis_static.dim() == 1:
            vis_static = vis_static.unsqueeze(0)
        if vis_gripper.dim() == 1:
            vis_gripper = vis_gripper.unsqueeze(0)
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        if lang_emb.dim() == 1:
            lang_emb = lang_emb.unsqueeze(0)
        arm, grip, h = self.forward(
            vis_static.unsqueeze(1), vis_gripper.unsqueeze(1),
            lang_emb, proprio.unsqueeze(1), h
        )
        return arm.squeeze(1), grip.squeeze(1), h


# ═══════════════════════════════════════════════════
#  CALVIN Sequence Dataset
# ═══════════════════════════════════════════════════
class CalvinSequenceDataset(Dataset):
    """Fixed-length sequence chunks from CALVIN demonstrations with language conditioning."""

    def __init__(self, vis_static_demos, vis_gripper_demos, lang_emb_demos,
                 proprio_demos, arm_act_demos, grip_target_demos,
                 chunk_len=CHUNK_LEN, action_chunk_k=ACTION_CHUNK_K):
        self.vis_s_chunks = []
        self.vis_g_chunks = []
        self.lang_chunks = []
        self.proprio_chunks = []
        self.arm_chunks = []
        self.grip_chunks = []

        for di in range(len(vis_static_demos)):
            T = len(vis_static_demos[di])
            if T < chunk_len:
                pad_len = chunk_len - T
                vis_s = torch.cat([vis_static_demos[di], vis_static_demos[di][-1:].expand(pad_len, -1)])
                vis_g = torch.cat([vis_gripper_demos[di], vis_gripper_demos[di][-1:].expand(pad_len, -1)])
                lang = lang_emb_demos[di]  # same for whole demo
                proprio = torch.cat([proprio_demos[di], proprio_demos[di][-1:].expand(pad_len, -1)])
                arm = torch.cat([arm_act_demos[di], arm_act_demos[di][-1:].expand(pad_len, -1)])
                grip = torch.cat([grip_target_demos[di], grip_target_demos[di][-1:].expand(pad_len, -1)])
            else:
                vis_s = vis_static_demos[di]
                vis_g = vis_gripper_demos[di]
                lang = lang_emb_demos[di]
                proprio = proprio_demos[di]
                arm = arm_act_demos[di]
                grip = grip_target_demos[di]

            T_padded = len(vis_s)

            # Build action-chunk targets
            arm_chunked = torch.zeros(T_padded, action_chunk_k, ARM_DIM)
            grip_chunked = torch.zeros(T_padded, action_chunk_k)
            for t in range(T_padded):
                for k in range(action_chunk_k):
                    idx = min(t + k, T_padded - 1)
                    arm_chunked[t, k] = arm[idx]
                    grip_chunked[t, k] = grip[idx, 0] if grip[idx].dim() > 0 and grip[idx].numel() > 1 else grip[idx]

            # Expand language to match T
            lang_expanded = lang.unsqueeze(0).expand(T_padded, -1) if lang.dim() == 1 else lang

            # Sliding window with 50% overlap
            stride = max(1, chunk_len // 2)
            for start in range(0, T_padded - chunk_len + 1, stride):
                end = start + chunk_len
                self.vis_s_chunks.append(vis_s[start:end])
                self.vis_g_chunks.append(vis_g[start:end])
                self.lang_chunks.append(lang_expanded[start:end])
                self.proprio_chunks.append(proprio[start:end])
                self.arm_chunks.append(arm_chunked[start:end])
                self.grip_chunks.append(grip_chunked[start:end])

        if len(self.vis_s_chunks) > 0:
            self.vis_s_chunks = torch.stack(self.vis_s_chunks)
            self.vis_g_chunks = torch.stack(self.vis_g_chunks)
            self.lang_chunks = torch.stack(self.lang_chunks)
            self.proprio_chunks = torch.stack(self.proprio_chunks)
            self.arm_chunks = torch.stack(self.arm_chunks)
            self.grip_chunks = torch.stack(self.grip_chunks)

        log.info(f"    Dataset: {len(self)} chunks of {chunk_len} steps")

    def __len__(self):
        return len(self.vis_s_chunks) if isinstance(self.vis_s_chunks, torch.Tensor) else 0

    def __getitem__(self, idx):
        return (self.vis_s_chunks[idx], self.vis_g_chunks[idx],
                self.lang_chunks[idx], self.proprio_chunks[idx],
                self.arm_chunks[idx], self.grip_chunks[idx])


# ═══════════════════════════════════════════════════
#  Image Transforms (same as LIBERO v4)
# ═══════════════════════════════════════════════════
def get_transform(backbone_name: str, training: bool = False):
    """Return the correct normalization transform for each backbone."""
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
    }

    norms = NORM_MAP.get(backbone_name)
    if norms is None:
        return transforms.Compose(base)
    else:
        return transforms.Compose(base + [
            transforms.Normalize(mean=norms["mean"], std=norms["std"]),
        ])


# ═══════════════════════════════════════════════════
#  CALVIN Data Loading
# ═══════════════════════════════════════════════════
def load_calvin_training_data(dataset_path: str, max_demos: int = None):
    """Load CALVIN training demonstrations from numpy files.
    
    Returns lists of per-demo arrays for:
      - static_imgs, gripper_imgs, robot_obs, rel_actions, lang_embeddings
    """
    train_dir = Path(dataset_path) / "training"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    # Load language annotations
    lang_ann_dir = train_dir / "lang_annotations"
    auto_lang = np.load(lang_ann_dir / "auto_lang_ann.npy", allow_pickle=True).item()
    
    # auto_lang['language']['ann'] = list of raw language strings
    # auto_lang['language']['task'] = list of task IDs
    # auto_lang['language']['emb'] = precomputed miniLM embeddings
    # auto_lang['info']['indx'] = list of (start_idx, end_idx) tuples
    
    lang_embs = auto_lang['language']['emb']  # (N, 384) miniLM embeddings
    lang_indices = auto_lang['info']['indx']   # list of (start, end)
    lang_tasks = auto_lang['language']['task'] if 'task' in auto_lang['language'] else None
    
    n_demos = len(lang_indices)
    if max_demos is not None:
        n_demos = min(n_demos, max_demos)
    
    log.info(f"  Loading {n_demos} annotated demonstrations from {train_dir}")
    
    all_static = []
    all_gripper = []
    all_proprio = []
    all_arm = []
    all_grip = []
    all_lang = []
    
    for di in range(n_demos):
        start_idx, end_idx = lang_indices[di]
        lang_emb = torch.from_numpy(lang_embs[di]).float()
        
        # Load individual numpy files for each timestep
        demo_static = []
        demo_gripper = []
        demo_proprio = []
        demo_arm = []
        demo_grip = []
        
        for idx in range(start_idx, end_idx + 1):
            npz_path = train_dir / f"episode_{idx:07d}.npz"
            if not npz_path.exists():
                # Try .npy format
                npy_path = train_dir / f"episode_{idx:07d}.npy"
                if npy_path.exists():
                    data = np.load(npy_path, allow_pickle=True).item()
                else:
                    continue
            else:
                data = np.load(npz_path, allow_pickle=True)
                # Convert to dict if needed
                if hasattr(data, 'files'):
                    data = {k: data[k] for k in data.files}
                else:
                    data = dict(data)
            
            demo_static.append(data['rgb_static'])      # (200, 200, 3)
            demo_gripper.append(data['rgb_gripper'])     # (84, 84, 3)
            demo_proprio.append(data['robot_obs'])       # (15,)
            rel_act = data['rel_actions']                # (7,)
            demo_arm.append(rel_act[:6])                 # arm: first 6
            demo_grip.append(rel_act[6:7])               # gripper: last 1
        
        if len(demo_static) < 2:
            continue
            
        all_static.append(np.stack(demo_static))
        all_gripper.append(np.stack(demo_gripper))
        all_proprio.append(np.stack(demo_proprio).astype(np.float32))
        all_arm.append(np.stack(demo_arm).astype(np.float32))
        all_grip.append(np.stack(demo_grip).astype(np.float32))
        all_lang.append(lang_emb)
        
        if (di + 1) % 100 == 0:
            log.info(f"    Loaded {di+1}/{n_demos} demos")
    
    total_frames = sum(len(x) for x in all_static)
    log.info(f"    Total: {len(all_static)} demos, {total_frames} frames, "
             f"lang_dim={all_lang[0].shape[-1] if all_lang else 0}")
    return all_static, all_gripper, all_proprio, all_arm, all_grip, all_lang


# ═══════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════
def _get_cache_path(backbone_name: str, dataset_path: str, camera: str) -> Path:
    cache_dir = FEATURE_CACHE / backbone_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds_hash = hashlib.md5(dataset_path.encode()).hexdigest()[:8]
    return cache_dir / f"calvin_{ds_hash}_{camera}.pt"


@torch.no_grad()
def extract_features_batch(encoder, all_images, transform, device, batch_size=128,
                           backbone_name="", dataset_path="", camera="static"):
    """Extract visual features for all frames with disk caching."""
    if backbone_name and dataset_path:
        cache_path = _get_cache_path(backbone_name, dataset_path, camera)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            log.info(f"      Loaded {camera} features from cache ({len(cached)} demos)")
            return cached

    encoder.eval()
    all_features = []

    for di, demo_imgs in enumerate(all_images):
        T = len(demo_imgs)
        demo_feats = []

        for i in range(0, T, batch_size):
            batch_np = demo_imgs[i:i + batch_size]
            batch_t = torch.from_numpy(np.stack(batch_np)).permute(0, 3, 1, 2).float() / 255.0
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
            demo_feats.append(feats.cpu())

        all_features.append(torch.cat(demo_feats))

        if (di + 1) % 50 == 0 or di == len(all_images) - 1:
            log.info(f"      {camera}: {di+1}/{len(all_images)} demos extracted")

    if backbone_name and dataset_path:
        cache_path = _get_cache_path(backbone_name, dataset_path, camera)
        torch.save(all_features, cache_path)
        log.info(f"      Cached {camera} features → {cache_path.name}")

    return all_features


# ═══════════════════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════════════════
def compute_norm_stats(all_proprio, all_arm_acts):
    """Compute normalization stats."""
    proprio_cat = np.concatenate(all_proprio, axis=0)
    arm_cat = np.concatenate(all_arm_acts, axis=0)
    return {
        "proprio_mean": proprio_cat.mean(0).astype(np.float32),
        "proprio_std": (proprio_cat.std(0) + 1e-6).astype(np.float32),
        "arm_mean": arm_cat.mean(0).astype(np.float32),
        "arm_std": (arm_cat.std(0) + 1e-6).astype(np.float32),
    }


def normalize_list(arrays, mean, std):
    return [(a - mean) / std for a in arrays]


def to_tensor_list(arrays):
    return [torch.from_numpy(a).float() for a in arrays]


# ═══════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════
def train_bc_lstm(dataset: CalvinSequenceDataset, visual_dim: int, lang_dim: int,
                  device: torch.device, epochs: int = N_EPOCHS, lr: float = LR, seed: int = 0):
    """Train language-conditioned LSTM BC policy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = CalvinLSTMPolicy(visual_dim=visual_dim, lang_dim=lang_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    warmup_epochs = 10
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)

    grip_loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience_limit = 40

    for epoch in range(epochs):
        policy.train()
        total_loss = 0
        n_batches = 0

        for vis_s, vis_g, lang, proprio, arm_act, grip_target in loader:
            vis_s = vis_s.to(device)
            vis_g = vis_g.to(device)
            lang = lang.to(device)
            proprio = proprio.to(device)
            arm_act = arm_act.to(device)
            grip_target = grip_target.to(device)

            arm_pred, grip_pred, _ = policy(vis_s, vis_g, lang, proprio)

            K = arm_pred.shape[2]
            chunk_weights = torch.exp(-0.5 * torch.arange(K, device=device).float())
            chunk_weights = chunk_weights / chunk_weights.sum()

            arm_loss = 0
            for k in range(K):
                arm_loss += chunk_weights[k] * F.smooth_l1_loss(arm_pred[:, :, k], arm_act[:, :, k])

            grip_loss = 0
            for k in range(K):
                grip_loss += chunk_weights[k] * grip_loss_fn(grip_pred[:, :, k], grip_target[:, :, k])

            loss = arm_loss + 0.5 * grip_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in policy.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            log.info(f"      Epoch {epoch+1}/{epochs}  loss={avg_loss:.5f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}  patience={patience_counter}/{patience_limit}")

        if patience_counter >= patience_limit and epoch > 50:
            log.info(f"      Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    log.info(f"      Training done. Best loss: {best_loss:.5f}")
    return policy


# ═══════════════════════════════════════════════════
#  Encoder Helper
# ═══════════════════════════════════════════════════
def _encode(encoder, img_tensor):
    """Helper to encode a single image tensor. Returns (1, feat_dim)."""
    feat = encoder(img_tensor)
    if isinstance(feat, dict):
        feat = feat.get("pooler_output", feat.get("last_hidden_state"))
        if feat.dim() == 3:
            feat = feat[:, 0]
    elif isinstance(feat, tuple):
        feat = feat[0]
        if feat.dim() == 3:
            feat = feat[:, 0]
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)
    return feat


# ═══════════════════════════════════════════════════
#  CALVIN Simulator Evaluation (LH-MTLC Protocol)
# ═══════════════════════════════════════════════════
def get_calvin_env(dataset_path):
    """Create CALVIN environment from dataset path."""
    from calvin_env.envs.play_table_env import get_env
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)
    return env


def get_eval_sequences(num_sequences=NUM_SEQUENCES):
    """Get evaluation sequences from CALVIN."""
    # Try to use official eval sequences
    try:
        calvin_models_dir = Path(__file__).resolve().parent.parent.parent / "calvin" / "calvin_models"
        sys.path.insert(0, str(calvin_models_dir))
        from calvin_agent.evaluation.multistep_sequences import get_sequences
        return get_sequences(num_sequences)
    except ImportError:
        log.warning("CALVIN agent not found, generating evaluation sequences")
        return _generate_eval_sequences(num_sequences)


def _generate_eval_sequences(num_sequences):
    """Fallback: generate evaluation sequences if CALVIN agent is not installed."""
    # Standard CALVIN eval tasks
    all_tasks = [
        "rotate_red_block_right", "rotate_red_block_left",
        "rotate_blue_block_right", "rotate_blue_block_left",
        "push_red_block_right", "push_red_block_left",
        "push_blue_block_right", "push_blue_block_left",
        "move_slider_left", "move_slider_right",
        "open_drawer", "close_drawer",
        "lift_red_block_table", "lift_blue_block_table", "lift_pink_block_table",
        "lift_red_block_slider", "lift_blue_block_slider", "lift_pink_block_slider",
        "place_in_slider", "place_in_drawer",
        "turn_on_lightbulb", "turn_off_lightbulb",
        "turn_on_led", "turn_off_led",
        "push_into_drawer", "stack_block", "unstack_block",
    ]
    rng = np.random.RandomState(0)
    sequences = []
    for _ in range(num_sequences):
        # Sample 5 random tasks (with replacement across categories)
        chain = rng.choice(all_tasks, size=5, replace=True).tolist()
        # Initial state = random integer (used as random seed for env reset)
        initial_state = rng.randint(0, 100)
        sequences.append((initial_state, chain))
    return sequences


def load_validation_annotations(dataset_path):
    """Load validation language annotations (task → lang embedding mapping)."""
    val_dir = Path(dataset_path) / "validation"
    lang_ann_dir = val_dir / "lang_annotations"
    
    # Load precomputed embeddings for evaluation
    embeddings_path = lang_ann_dir / "embeddings.npy"
    if embeddings_path.exists():
        emb_data = np.load(embeddings_path, allow_pickle=True).item()
        return emb_data
    
    # Fallback: use auto_lang_ann
    auto_lang = np.load(lang_ann_dir / "auto_lang_ann.npy", allow_pickle=True).item()
    return auto_lang


def get_lang_embedding_for_task(task_name, val_annotations, device):
    """Get language embedding for a specific task."""
    if isinstance(val_annotations, dict):
        if 'language' in val_annotations:
            # auto_lang_ann format
            tasks = val_annotations['language'].get('task', [])
            embs = val_annotations['language']['emb']
            for i, t in enumerate(tasks):
                if t == task_name:
                    return torch.from_numpy(embs[i]).float().to(device)
        # embeddings.npy format - may have task_name as key
        if task_name in val_annotations:
            emb = val_annotations[task_name]
            if isinstance(emb, np.ndarray):
                return torch.from_numpy(emb).float().to(device)
    
    # Fallback: compute from CLIP text encoder
    log.warning(f"  No cached embedding for '{task_name}', using CLIP text encoder")
    return _compute_lang_embedding(task_name, device)


def _compute_lang_embedding(text, device):
    """Compute language embedding using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode([text], convert_to_tensor=True).squeeze(0).to(device)
        return emb
    except ImportError:
        log.warning("sentence-transformers not installed, returning random embedding")
        return torch.randn(384, device=device)


def get_env_state_for_initial_condition(initial_state):
    """Get robot and scene state for initial condition."""
    try:
        calvin_models_dir = Path(__file__).resolve().parent.parent.parent / "calvin" / "calvin_models"
        sys.path.insert(0, str(calvin_models_dir))
        from calvin_agent.evaluation.utils import get_env_state_for_initial_condition as _get_state
        return _get_state(initial_state)
    except ImportError:
        # Fallback: return None to use default env reset
        return None, None


def get_task_oracle(dataset_path):
    """Get the CALVIN task oracle for checking success."""
    try:
        calvin_models_dir = Path(__file__).resolve().parent.parent.parent / "calvin" / "calvin_models"
        sys.path.insert(0, str(calvin_models_dir))
        conf_dir = calvin_models_dir / "calvin_agent" / "conf"
        
        import hydra
        from omegaconf import OmegaConf
        task_cfg = OmegaConf.load(conf_dir / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml")
        task_oracle = hydra.utils.instantiate(task_cfg)
        
        val_annotations = OmegaConf.load(conf_dir / "annotations" / "new_playtable_validation.yaml")
        return task_oracle, val_annotations
    except Exception as e:
        log.error(f"Failed to load task oracle: {e}")
        return None, None


@torch.no_grad()
def evaluate_calvin_lhmtlc(encoder, policy, dataset_path, transform, device,
                            norm_stats, num_sequences=NUM_SEQUENCES):
    """Evaluate using the official CALVIN LH-MTLC protocol.
    
    Returns:
        results: list of chain lengths (0-5)
        metrics: dict with avg_len, per-chain completion rates
    """
    env = get_calvin_env(dataset_path)
    eval_sequences = get_eval_sequences(num_sequences)
    task_oracle, val_annotations = get_task_oracle(dataset_path)
    val_embeddings = load_validation_annotations(dataset_path)
    
    if task_oracle is None:
        log.error("Cannot evaluate without task oracle. Please install calvin_models.")
        return [], {}

    proprio_mean = torch.from_numpy(norm_stats["proprio_mean"]).to(device)
    proprio_std = torch.from_numpy(norm_stats["proprio_std"]).to(device)
    arm_mean = torch.from_numpy(norm_stats["arm_mean"]).to(device)
    arm_std = torch.from_numpy(norm_stats["arm_std"]).to(device)

    K = policy.action_chunk_k
    results = []
    
    pbar = tqdm(eval_sequences, desc="CALVIN LH-MTLC", position=0, leave=True)
    
    for initial_state, eval_sequence in pbar:
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        if robot_obs is not None:
            env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        else:
            env.reset()
        
        chain_success = 0
        
        for subtask in eval_sequence:
            # Get language embedding for this subtask
            lang_annotation = val_annotations[subtask][0] if val_annotations and subtask in val_annotations else subtask
            lang_emb = get_lang_embedding_for_task(subtask, val_embeddings, device)
            
            # Reset LSTM hidden state for each subtask
            h = None
            action_queue = []
            obs = env.get_obs()
            start_info = env.get_info()
            success = False
            
            for step in range(EP_LEN):
                # Encode static camera
                static_img = obs['rgb_static']  # (200, 200, 3)
                static_t = torch.from_numpy(static_img).permute(2, 0, 1).float() / 255.0
                static_t = transform(static_t).unsqueeze(0).to(device)
                static_feat = _encode(encoder, static_t)

                # Encode gripper camera
                gripper_img = obs['rgb_gripper']  # (84, 84, 3)
                gripper_t = torch.from_numpy(gripper_img).permute(2, 0, 1).float() / 255.0
                gripper_t = transform(gripper_t).unsqueeze(0).to(device)
                gripper_feat = _encode(encoder, gripper_t)

                # Proprio
                proprio = obs['robot_obs']  # (15,)
                proprio_t = torch.from_numpy(proprio).float().to(device)
                proprio_norm = (proprio_t - proprio_mean) / proprio_std

                # Policy step
                arm_pred, grip_logit, h = policy.forward_step(
                    static_feat, gripper_feat, lang_emb, proprio_norm, h
                )
                arm_chunk = arm_pred[0].cpu()
                grip_chunk = grip_logit[0].cpu()

                # Temporal aggregation
                action_queue.insert(0, (arm_chunk, grip_chunk))
                if len(action_queue) > K:
                    action_queue = action_queue[:K]

                arm_agg = torch.zeros(ARM_DIM)
                grip_agg = 0.0
                total_weight = 0.0

                for i, (ac, gc) in enumerate(action_queue):
                    if i < ac.shape[0]:
                        w = np.exp(-TEMPORAL_AGG_M * i)
                        arm_agg += w * ac[i]
                        grip_agg += w * gc[i].item()
                        total_weight += w

                arm_agg /= total_weight
                grip_agg /= total_weight

                # CALVIN expects relative actions as 7-tuple (already scaled)
                arm_action = arm_agg.numpy()
                grip_action = 1.0 if grip_agg > 0 else -1.0
                action = np.concatenate([arm_action, [grip_action]])
                action = np.clip(action, -1, 1)

                obs, _, _, current_info = env.step(action)

                # Check success via task oracle
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(current_task_info) > 0:
                    success = True
                    break

            if success:
                chain_success += 1
            else:
                break  # Chain breaks on first failure
        
        results.append(chain_success)
        
        # Update progress bar
        if len(results) > 0:
            avg_len = np.mean(results)
            pbar.set_description(
                f"Avg Len: {avg_len:.2f} | " +
                " ".join([f"{i+1}/5: {np.mean([r >= i+1 for r in results])*100:.1f}%" for i in range(5)])
            )
    
    env.close()
    
    # Compute metrics
    metrics = {
        "avg_chain_length": float(np.mean(results)),
        "std_chain_length": float(np.std(results)),
    }
    for i in range(1, 6):
        metrics[f"chain_{i}_of_5"] = float(np.mean([r >= i for r in results]))
    
    return results, metrics


# ═══════════════════════════════════════════════════
#  Backbone Loaders (shared with LIBERO v4)
# ═══════════════════════════════════════════════════
def load_single_backbone(name: str, checkpoint_path: str, device: torch.device):
    """Load a single visual encoder by name.
    
    Same backbone loaders as evaluate_libero_v4.py for consistency.
    """
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

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 1536
            def forward(self, x):
                return self.m(x, return_features=True)
        return Wrapper(model)

    elif name == "DINOv2":
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        dinov2.eval().to(device)
        for p in dinov2.parameters():
            p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768
            def forward(self, x):
                return self.m(x)
        return Wrapper(dinov2)

    elif name == "CLIP":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_model.eval().to(device)
        for p in clip_model.parameters():
            p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768
            def forward(self, x):
                return self.m.encode_image(x)
        return Wrapper(clip_model)

    elif name == "R3M":
        from r3m import load_r3m
        r3m_model = load_r3m("resnet50")
        r3m_model.eval().to(device)
        for p in r3m_model.parameters():
            p.requires_grad = False
        core = r3m_model.module if hasattr(r3m_model, "module") else r3m_model

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 2048
            def forward(self, x):
                return self.m(x * 255.0)
        return Wrapper(core)

    elif name == "MCR":
        import timm
        mcr = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
        mcr.eval().to(device)
        for p in mcr.parameters():
            p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768
            def forward(self, x):
                return self.m(x)
        return Wrapper(mcr)

    elif name == "SigLIP":
        from transformers import SiglipVisionModel
        siglip = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        siglip.eval().to(device)
        for p in siglip.parameters():
            p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768
            def forward(self, x):
                return self.m(x).pooler_output
        return Wrapper(siglip)

    elif name == "VC-1":
        try:
            from vc_models.models.vit import model_utils
            vc1_model, embd_size, _, _ = model_utils.load_model(model_utils.VC1_LARGE_NAME)
            vc1_model.eval().to(device)
            for p in vc1_model.parameters():
                p.requires_grad = False

            class Wrapper(nn.Module):
                def __init__(self, m, dim):
                    super().__init__()
                    self.m = m
                    self.output_dim = dim
                def forward(self, x):
                    return self.m(x)
            return Wrapper(vc1_model, embd_size)
        except ImportError:
            import timm
            vc1 = timm.create_model("vit_large_patch16_224.mae", pretrained=True, num_classes=0)
            vc1.eval().to(device)
            for p in vc1.parameters():
                p.requires_grad = False

            class Wrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                    self.output_dim = 1024
                def forward(self, x):
                    return self.m(x)
            return Wrapper(vc1)

    elif name == "MVP":
        import timm
        mvp_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
        mvp_ckpt = PROJECT_ROOT / "checkpoints" / "baselines" / "mvp_vitb16.pth"
        if mvp_ckpt.exists():
            state = torch.load(str(mvp_ckpt), map_location="cpu", weights_only=True)
            mvp_model.load_state_dict(state, strict=False)
        mvp_model.eval().to(device)
        for p in mvp_model.parameters():
            p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768
            def forward(self, x):
                return self.m(x)
        return Wrapper(mvp_model)

    elif name == "Voltron":
        from backbone_utils import load_voltron
        voltron = load_voltron(device=device)

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = m.output_dim
            def forward(self, x):
                return self.m(x)
        return Wrapper(voltron)

    elif name == "Theia":
        from backbone_utils import load_theia
        theia = load_theia(device=device)

        class Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = m.output_dim
            def forward(self, x):
                return self.m(x)
        return Wrapper(theia)

    else:
        raise ValueError(f"Unknown backbone: {name}")


ALL_BACKBONES = ["DynaCLIP", "DINOv2", "CLIP", "R3M", "MCR", "SigLIP", "VC-1", "MVP", "Voltron", "Theia"]


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="CALVIN LH-MTLC Frozen-Encoder Evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to CALVIN dataset (e.g., task_D_D or calvin_debug_dataset)")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone", type=str, required=True, choices=ALL_BACKBONES)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--num_sequences", type=int, default=NUM_SEQUENCES)
    parser.add_argument("--max_demos", type=int, default=None,
                        help="Max training demos to load (None = all)")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick smoke test: 20 epochs, 50 sequences, 1 seed")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_epochs = 20
        args.num_sequences = 50
        args.n_seeds = 1
        args.max_demos = 200
        log.info("=== SMOKE TEST MODE ===")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    backbone_name = args.backbone
    log.info("=" * 60)
    log.info(f"CALVIN v3 Evaluation — Backbone: {backbone_name}")
    log.info(f"  Language-Conditioned LSTM + Action Chunking (K={ACTION_CHUNK_K})")
    log.info(f"  Dual camera: static (200×200) + gripper (84×84) → 224×224")
    log.info(f"  Epochs: {args.n_epochs}, Sequences: {args.num_sequences}, Seeds: {args.n_seeds}")
    log.info(f"  Device: {device}")
    log.info("=" * 60)

    # Load backbone
    log.info(f"Loading {backbone_name}...")
    encoder = load_single_backbone(backbone_name, args.checkpoint, device)
    obs_dim = encoder.output_dim
    eval_transform = get_transform(backbone_name, training=False)
    log.info(f"  {backbone_name}: {obs_dim}-d features")

    # Load training data
    log.info(f"Loading CALVIN training data from {args.dataset_path}...")
    all_static, all_gripper, all_proprio, all_arm, all_grip, all_lang = \
        load_calvin_training_data(args.dataset_path, max_demos=args.max_demos)

    lang_dim = all_lang[0].shape[-1] if all_lang else LANG_DIM

    # Compute normalization
    norm_stats = compute_norm_stats(all_proprio, all_arm)
    log.info(f"  Arm stats: mean={norm_stats['arm_mean'].round(3)}, std={norm_stats['arm_std'].round(3)}")

    # Normalize
    all_proprio_norm = normalize_list(all_proprio, norm_stats["proprio_mean"], norm_stats["proprio_std"])
    all_arm_norm = normalize_list(all_arm, norm_stats["arm_mean"], norm_stats["arm_std"])

    # Extract features
    log.info(f"  Extracting static camera features...")
    all_static_feats = extract_features_batch(
        encoder, all_static, eval_transform, device,
        backbone_name=backbone_name, dataset_path=args.dataset_path, camera="static")
    log.info(f"  Extracting gripper camera features...")
    all_gripper_feats = extract_features_batch(
        encoder, all_gripper, eval_transform, device,
        backbone_name=backbone_name, dataset_path=args.dataset_path, camera="gripper")

    feat_dim = all_static_feats[0].shape[1]
    log.info(f"  Features: dim={feat_dim}, total={sum(len(f) for f in all_static_feats)} per camera")

    # Convert to tensors
    all_proprio_t = to_tensor_list(all_proprio_norm)
    all_arm_t = to_tensor_list(all_arm_norm)
    all_grip_t = [torch.from_numpy(g).float() for g in all_grip]

    # Build dataset
    dataset = CalvinSequenceDataset(
        all_static_feats, all_gripper_feats, all_lang,
        all_proprio_t, all_arm_t, all_grip_t,
        chunk_len=CHUNK_LEN,
    )

    # Free raw images
    del all_static, all_gripper
    torch.cuda.empty_cache()

    # Train & evaluate per seed
    all_seed_results = []
    all_seed_metrics = []

    for seed in range(args.n_seeds):
        log.info(f"\n  Seed {seed}: training LSTM BC ({args.n_epochs} epochs)...")
        policy = train_bc_lstm(
            dataset, obs_dim, lang_dim, device,
            epochs=args.n_epochs, lr=LR, seed=seed,
        )

        log.info(f"  Seed {seed}: evaluating ({args.num_sequences} chains)...")
        results, metrics = evaluate_calvin_lhmtlc(
            encoder, policy, args.dataset_path, eval_transform, device,
            norm_stats, num_sequences=args.num_sequences,
        )
        all_seed_results.append(results)
        all_seed_metrics.append(metrics)

        log.info(f"  Seed {seed} results:")
        log.info(f"    Avg chain length: {metrics['avg_chain_length']:.2f} ± {metrics['std_chain_length']:.2f}")
        for i in range(1, 6):
            log.info(f"    {i}/5 tasks: {metrics[f'chain_{i}_of_5']:.1%}")

    # Aggregate across seeds
    avg_metrics = {}
    for key in all_seed_metrics[0]:
        values = [m[key] for m in all_seed_metrics]
        avg_metrics[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    final_results = {
        backbone_name: {
            "avg_chain_length": avg_metrics["avg_chain_length"],
            "per_chain_completion": {
                f"{i}/5": avg_metrics[f"chain_{i}_of_5"]
                for i in range(1, 6)
            },
            "per_seed": all_seed_metrics,
            "feature_dim": obs_dim,
            "lang_dim": lang_dim,
            "n_epochs": args.n_epochs,
            "num_sequences": args.num_sequences,
            "n_seeds": args.n_seeds,
            "action_chunk_k": ACTION_CHUNK_K,
            "version": "v3",
        }
    }

    # Save
    output_file = os.path.join(args.output, f"calvin_v3_{backbone_name}.json")
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    log.info(f"\n{'=' * 60}")
    log.info(f"CALVIN v3 Results — {backbone_name}")
    log.info(f"{'=' * 60}")
    log.info(f"  Avg Chain Length: {avg_metrics['avg_chain_length']['mean']:.2f} ± {avg_metrics['avg_chain_length']['std']:.2f}")
    for i in range(1, 6):
        m = avg_metrics[f"chain_{i}_of_5"]
        log.info(f"  {i}/5 tasks: {m['mean']:.1%} ± {m['std']:.1%}")
    log.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
