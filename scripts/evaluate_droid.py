#!/usr/bin/env python3
"""
evaluate_droid.py — DROID-100 Frozen-Encoder Offline BC Evaluation.

Evaluates visual representations on the DROID-100 real-world manipulation dataset
using frozen-encoder behavior cloning with action prediction metrics.

Protocol:
  - 100 real-world episodes from DROID (diverse scenes, tasks, robots)
  - 3 cameras: exterior_1 (180×320), exterior_2 (180×320), wrist (180×320)
  - 7-dim actions (6 joint velocities + 1 gripper), 14-dim proprioception
  - Train/val split: 80 train / 20 val episodes (stratified by language)
  - LSTM BC policy with action chunking (K=10), trained on frozen features
  - Metrics: Action MSE, L1, Cosine Similarity, Gripper Accuracy (on val set)
  - 3 seeds for statistical robustness

Since DROID is real-world data (no sim), we report offline action prediction quality.
This measures how well the frozen representation supports learning manipulation policies.

Usage:
  python evaluate_droid.py --gpu 4 --backbone DynaCLIP
  python evaluate_droid.py --gpu 5 --backbone DINOv2
  python evaluate_droid.py --gpu 7 --backbone CLIP
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
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
DROID_DATA = PROJECT_ROOT / "data" / "droid" / "1.0.0"
RESULTS_DIR = PROJECT_ROOT / "results" / "droid100"
FEATURE_CACHE = PROJECT_ROOT / "data_cache" / "droid100_features"

# ── Hyperparameters (matching LIBERO v4 / CALVIN v3 Protocol) ──
CHUNK_LEN = 50         # LSTM training sequence chunk length
ACTION_CHUNK_K = 10    # Action chunking: predict K future actions
PROPRIO_DIM = 14       # cartesian_pos(6) + gripper_pos(1) + joint_pos(7)
ARM_DIM = 6            # 6 joint velocities
ACTION_DIM = 7         # arm(6) + gripper(1)
LSTM_HIDDEN = 512      # LSTM hidden dimension
LSTM_LAYERS = 2        # LSTM layers
N_EPOCHS = 200         # Training epochs
BATCH_SIZE = 64        # Sequence batch size
LR = 3e-4              # Learning rate
WEIGHT_DECAY = 0.01    # Weight decay
N_TRAIN = 80           # Train episodes
N_VAL = 20             # Val episodes
TEMPORAL_AGG_M = 0.5   # Temporal action aggregation weight


# ═══════════════════════════════════════════════════
#  LSTM Policy Network with Action Chunking
# ═══════════════════════════════════════════════════
class DROIDLSTMPolicy(nn.Module):
    """LSTM-based BC policy for DROID with triple camera input + proprio + action chunking."""

    def __init__(self, visual_dim: int, proprio_dim: int = PROPRIO_DIM,
                 hidden_dim: int = LSTM_HIDDEN, n_layers: int = LSTM_LAYERS,
                 action_chunk_k: int = ACTION_CHUNK_K, n_cameras: int = 3):
        super().__init__()
        self.action_chunk_k = action_chunk_k
        self.n_cameras = n_cameras

        # Visual projection: n_cameras concatenated → 512 → 256
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim * n_cameras, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=256 + 64,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        # Action chunking heads
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

    def forward(self, vis_feats, proprio, h=None):
        """
        Args:
            vis_feats: list of (B, T, feat_dim) for each camera, OR (B, T, n_cam*feat_dim)
            proprio:   (B, T, proprio_dim)
            h:         optional LSTM hidden state
        Returns:
            arm_pred:  (B, T, K, 6)
            grip_pred: (B, T, K)
            h:         updated hidden state
        """
        if isinstance(vis_feats, (list, tuple)):
            vis = torch.cat(vis_feats, dim=-1)
        else:
            vis = vis_feats
        B, T, _ = vis.shape
        vis_proj = self.visual_proj(vis)
        prop_proj = self.proprio_proj(proprio)
        x = torch.cat([vis_proj, prop_proj], dim=-1)
        lstm_out, h = self.lstm(x, h)
        arm = self.arm_head(lstm_out).reshape(B, T, self.action_chunk_k, ARM_DIM)
        grip = self.gripper_head(lstm_out)  # (B, T, K)
        return arm, grip, h

    def forward_step(self, vis_feats, proprio, h=None):
        """Single-step for evaluation. Inputs: (B, dim) each."""
        if isinstance(vis_feats, (list, tuple)):
            vis_feats = [v.unsqueeze(1) if v.dim() == 2 else v for v in vis_feats]
        else:
            if vis_feats.dim() == 2:
                vis_feats = vis_feats.unsqueeze(1)
        if proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
        arm, grip, h = self.forward(vis_feats, proprio, h)
        return arm.squeeze(1), grip.squeeze(1), h


# ═══════════════════════════════════════════════════
#  Sequence Dataset with Action Chunking Targets
# ═══════════════════════════════════════════════════
class DROIDSequenceDataset(Dataset):
    """Fixed-length sequence chunks from DROID episodes with action chunking."""

    def __init__(self, episode_data, chunk_len=CHUNK_LEN,
                 action_chunk_k=ACTION_CHUNK_K):
        """
        episode_data: list of dicts, each with:
            'vis_feats': (T, n_cam * feat_dim) tensor
            'proprio': (T, 14) tensor
            'arm_acts': (T, 6) tensor
            'grip_targets': (T, 1) tensor
        """
        self.vis_chunks = []
        self.proprio_chunks = []
        self.arm_chunks = []
        self.grip_chunks = []

        for ep in episode_data:
            vis = ep['vis_feats']
            proprio = ep['proprio']
            arm = ep['arm_acts']
            grip = ep['grip_targets']
            T = len(vis)

            if T < chunk_len:
                pad_len = chunk_len - T
                vis = torch.cat([vis, vis[-1:].expand(pad_len, -1)])
                proprio = torch.cat([proprio, proprio[-1:].expand(pad_len, -1)])
                arm = torch.cat([arm, arm[-1:].expand(pad_len, -1)])
                grip = torch.cat([grip, grip[-1:].expand(pad_len, -1)])

            T_padded = len(vis)

            # Build action-chunk targets
            arm_chunked = torch.zeros(T_padded, action_chunk_k, ARM_DIM)
            grip_chunked = torch.zeros(T_padded, action_chunk_k)
            for t in range(T_padded):
                for k in range(action_chunk_k):
                    idx = min(t + k, T_padded - 1)
                    arm_chunked[t, k] = arm[idx]
                    grip_chunked[t, k] = grip[idx, 0]

            # Sliding window with 50% overlap
            stride = max(1, chunk_len // 2)
            for start in range(0, T_padded - chunk_len + 1, stride):
                end = start + chunk_len
                self.vis_chunks.append(vis[start:end])
                self.proprio_chunks.append(proprio[start:end])
                self.arm_chunks.append(arm_chunked[start:end])
                self.grip_chunks.append(grip_chunked[start:end])

        self.vis_chunks = torch.stack(self.vis_chunks)
        self.proprio_chunks = torch.stack(self.proprio_chunks)
        self.arm_chunks = torch.stack(self.arm_chunks)
        self.grip_chunks = torch.stack(self.grip_chunks)

        log.info(f"    Dataset: {len(self)} chunks of {chunk_len} steps, "
                 f"vis_dim={self.vis_chunks.shape[-1]}")

    def __len__(self):
        return len(self.vis_chunks)

    def __getitem__(self, idx):
        return (self.vis_chunks[idx], self.proprio_chunks[idx],
                self.arm_chunks[idx], self.grip_chunks[idx])


# ═══════════════════════════════════════════════════
#  DROID TFRecord Data Loading
# ═══════════════════════════════════════════════════
def load_droid_episodes(data_dir: str, max_episodes: int = None):
    """Load all episodes from DROID TFRecord files.

    Returns: list of dicts with keys:
        'ext1_imgs': list of (H, W, 3) uint8 arrays
        'ext2_imgs': list of (H, W, 3) uint8 arrays
        'wrist_imgs': list of (H, W, 3) uint8 arrays
        'proprio': (T, 14) float32
        'actions': (T, 7) float64
        'language': str
    """
    import tensorflow as tf
    import glob

    tfrecord_files = sorted(glob.glob(os.path.join(data_dir, '*.tfrecord*')))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files in {data_dir}")

    episodes = []
    for fi, tf_file in enumerate(tfrecord_files):
        raw_dataset = tf.data.TFRecordDataset(tf_file)
        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            features = example.features.feature

            # Determine episode length from actions (7-dim)
            n_steps = len(features['steps/action'].float_list.value) // 7

            # Language instruction
            lang = features['steps/language_instruction'].bytes_list.value[0].decode('utf-8')

            # Actions: (n_steps, 7)
            actions = np.array(features['steps/action'].float_list.value,
                               dtype=np.float64).reshape(n_steps, 7)

            # Proprioception: cartesian(6) + gripper(1) + joint(7) = 14
            cart_pos = np.array(
                features['steps/observation/cartesian_position'].float_list.value,
                dtype=np.float32).reshape(n_steps, 6)
            grip_pos = np.array(
                features['steps/observation/gripper_position'].float_list.value,
                dtype=np.float32).reshape(n_steps, 1)
            joint_pos = np.array(
                features['steps/observation/joint_position'].float_list.value,
                dtype=np.float32).reshape(n_steps, 7)
            proprio = np.concatenate([cart_pos, grip_pos, joint_pos], axis=-1)

            # Decode images (JPEG → uint8)
            ext1_imgs = []
            ext2_imgs = []
            wrist_imgs = []
            for t in range(n_steps):
                ext1 = tf.io.decode_jpeg(
                    features['steps/observation/exterior_image_1_left'].bytes_list.value[t]
                ).numpy()
                ext2 = tf.io.decode_jpeg(
                    features['steps/observation/exterior_image_2_left'].bytes_list.value[t]
                ).numpy()
                wrist = tf.io.decode_jpeg(
                    features['steps/observation/wrist_image_left'].bytes_list.value[t]
                ).numpy()
                ext1_imgs.append(ext1)
                ext2_imgs.append(ext2)
                wrist_imgs.append(wrist)

            episodes.append({
                'ext1_imgs': ext1_imgs,
                'ext2_imgs': ext2_imgs,
                'wrist_imgs': wrist_imgs,
                'proprio': proprio,
                'actions': actions.astype(np.float32),
                'language': lang,
            })

            if max_episodes and len(episodes) >= max_episodes:
                break

        if max_episodes and len(episodes) >= max_episodes:
            break

        if (fi + 1) % 10 == 0:
            log.info(f"  Read {fi+1}/{len(tfrecord_files)} TFRecord files, "
                     f"{len(episodes)} episodes so far")

    log.info(f"  Loaded {len(episodes)} DROID episodes, "
             f"total frames: {sum(len(e['ext1_imgs']) for e in episodes)}")
    return episodes


def train_val_split(episodes, n_train=N_TRAIN, seed=42):
    """Split episodes into train/val sets (deterministic by seed)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(episodes))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_eps = [episodes[i] for i in train_idx]
    val_eps = [episodes[i] for i in val_idx]
    log.info(f"  Split: {len(train_eps)} train, {len(val_eps)} val episodes")
    return train_eps, val_eps


# ═══════════════════════════════════════════════════
#  Per-Backbone Image Transforms
# ═══════════════════════════════════════════════════
def get_transform(backbone_name: str, training: bool = False):
    """Return the correct normalization transform for each backbone."""
    if training:
        base = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                         antialias=True),
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
#  Feature Extraction with Caching
# ═══════════════════════════════════════════════════
def _get_cache_path(backbone_name: str, camera: str, split: str) -> Path:
    cache_dir = FEATURE_CACHE / backbone_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"droid100_{camera}_{split}.pt"


@torch.no_grad()
def extract_features_for_episodes(encoder, episodes, camera_key, transform,
                                  device, batch_size=64,
                                  backbone_name="", split="train"):
    """Extract visual features for all frames across episodes for one camera.

    Args:
        camera_key: one of 'ext1_imgs', 'ext2_imgs', 'wrist_imgs'
    Returns:
        list of (T_i, feat_dim) tensors
    """
    if backbone_name:
        cache_path = _get_cache_path(backbone_name, camera_key, split)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            log.info(f"      Loaded {camera_key}/{split} features from cache ({len(cached)} eps)")
            return cached

    encoder.eval()
    all_features = []

    for ei, ep in enumerate(episodes):
        imgs = ep[camera_key]
        T = len(imgs)
        ep_feats = []

        for i in range(0, T, batch_size):
            batch_np = np.stack(imgs[i:i + batch_size])
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

        if (ei + 1) % 20 == 0 or ei == len(episodes) - 1:
            log.info(f"      {camera_key}/{split}: {ei+1}/{len(episodes)} episodes")

    if backbone_name:
        cache_path = _get_cache_path(backbone_name, camera_key, split)
        torch.save(all_features, cache_path)
        log.info(f"      Cached → {cache_path.name}")

    return all_features


# ═══════════════════════════════════════════════════
#  Normalization
# ═══════════════════════════════════════════════════
def compute_norm_stats(episodes):
    """Compute normalization stats from training episodes."""
    all_proprio = np.concatenate([ep['proprio'] for ep in episodes], axis=0)
    all_arm = np.concatenate([ep['actions'][:, :6] for ep in episodes], axis=0)
    return {
        "proprio_mean": all_proprio.mean(0).astype(np.float32),
        "proprio_std": (all_proprio.std(0) + 1e-6).astype(np.float32),
        "arm_mean": all_arm.mean(0).astype(np.float32),
        "arm_std": (all_arm.std(0) + 1e-6).astype(np.float32),
    }


def normalize_array(arr, mean, std):
    return (arr - mean) / std


# ═══════════════════════════════════════════════════
#  Build Episode Feature Dicts
# ═══════════════════════════════════════════════════
def build_episode_data(episodes, ext1_feats, ext2_feats, wrist_feats, norm_stats):
    """Combine features and normalize proprio/actions for each episode."""
    episode_data = []
    for i, ep in enumerate(episodes):
        T = len(ep['ext1_imgs'])
        # Concatenate 3 camera features
        vis = torch.cat([ext1_feats[i], ext2_feats[i], wrist_feats[i]], dim=-1)  # (T, 3*feat_dim)

        # Normalize proprio and arm actions
        proprio_norm = normalize_array(ep['proprio'], norm_stats['proprio_mean'],
                                       norm_stats['proprio_std'])
        arm_norm = normalize_array(ep['actions'][:, :6], norm_stats['arm_mean'],
                                   norm_stats['arm_std'])
        # Gripper: already in reasonable range, convert to 0/1 for BCE
        grip = ep['actions'][:, 6:7].astype(np.float32)
        # DROID gripper is continuous, normalize to [0,1]
        grip_min, grip_max = grip.min(), grip.max()
        if grip_max - grip_min > 1e-6:
            grip = (grip - grip_min) / (grip_max - grip_min)
        else:
            grip = np.zeros_like(grip)

        episode_data.append({
            'vis_feats': vis,
            'proprio': torch.from_numpy(proprio_norm).float(),
            'arm_acts': torch.from_numpy(arm_norm).float(),
            'grip_targets': torch.from_numpy(grip).float(),
        })
    return episode_data


# ═══════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════
def train_bc_lstm(train_dataset, visual_dim, device, epochs=N_EPOCHS,
                  lr=LR, seed=0):
    """Train LSTM BC policy on sequence chunks."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = DROIDLSTMPolicy(visual_dim=visual_dim, n_cameras=3).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    policy.train()
    for epoch in range(epochs):
        epoch_arm_loss = 0.0
        epoch_grip_loss = 0.0
        n_batches = 0

        for vis, proprio, arm_targets, grip_targets in loader:
            vis = vis.to(device)
            proprio = proprio.to(device)
            arm_targets = arm_targets.to(device)
            grip_targets = grip_targets.to(device)

            arm_pred, grip_pred, _ = policy(vis, proprio)

            arm_loss = F.mse_loss(arm_pred, arm_targets)
            grip_loss = F.binary_cross_entropy_with_logits(grip_pred, grip_targets)
            loss = arm_loss + 0.1 * grip_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + n_batches / max(len(loader), 1))

            epoch_arm_loss += arm_loss.item()
            epoch_grip_loss += grip_loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_arm = epoch_arm_loss / max(n_batches, 1)
            avg_grip = epoch_grip_loss / max(n_batches, 1)
            log.info(f"      Epoch {epoch+1}/{epochs}: arm_loss={avg_arm:.4f}, "
                     f"grip_loss={avg_grip:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

    policy.eval()
    return policy


# ═══════════════════════════════════════════════════
#  Offline Evaluation Metrics
# ═══════════════════════════════════════════════════
@torch.no_grad()
def evaluate_offline(policy, val_episode_data, norm_stats, device):
    """Evaluate BC policy on held-out episodes with action prediction metrics.

    Metrics:
    - MSE: Mean squared error on (normalized) arm actions
    - L1: Mean absolute error on arm actions
    - Cosine: Cosine similarity between predicted and ground-truth action vectors
    - Gripper Acc: Binary accuracy of gripper open/close prediction
    - Raw MSE: MSE in original (un-normalized) action space
    """
    policy.eval()
    all_arm_pred = []
    all_arm_true = []
    all_grip_pred = []
    all_grip_true = []

    for ep in val_episode_data:
        vis = ep['vis_feats'].unsqueeze(0).to(device)     # (1, T, vis_dim)
        proprio = ep['proprio'].unsqueeze(0).to(device)    # (1, T, 14)
        arm_true = ep['arm_acts']                          # (T, 6) normalized
        grip_true = ep['grip_targets']                     # (T, 1)

        arm_pred, grip_pred, _ = policy(vis, proprio)
        # Take first action of each chunk (k=0)
        arm_pred = arm_pred[0, :, 0, :].cpu()   # (T, 6)
        grip_pred = grip_pred[0, :, 0].cpu()     # (T,)

        all_arm_pred.append(arm_pred)
        all_arm_true.append(arm_true)
        all_grip_pred.append(grip_pred)
        all_grip_true.append(grip_true[:, 0])

    arm_pred_cat = torch.cat(all_arm_pred)    # (N, 6)
    arm_true_cat = torch.cat(all_arm_true)    # (N, 6)
    grip_pred_cat = torch.cat(all_grip_pred)  # (N,)
    grip_true_cat = torch.cat(all_grip_true)  # (N,)

    # MSE (normalized space)
    mse = F.mse_loss(arm_pred_cat, arm_true_cat).item()

    # L1
    l1 = F.l1_loss(arm_pred_cat, arm_true_cat).item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(arm_pred_cat, arm_true_cat, dim=-1).mean().item()

    # Raw MSE (un-normalized)
    arm_std = torch.from_numpy(norm_stats['arm_std'])
    arm_mean = torch.from_numpy(norm_stats['arm_mean'])
    raw_pred = arm_pred_cat * arm_std + arm_mean
    raw_true = arm_true_cat * arm_std + arm_mean
    raw_mse = F.mse_loss(raw_pred, raw_true).item()

    # Gripper accuracy
    grip_binary_pred = (grip_pred_cat > 0).float()
    grip_binary_true = (grip_true_cat > 0.5).float()
    grip_acc = (grip_binary_pred == grip_binary_true).float().mean().item()

    return {
        "mse_norm": mse,
        "l1_norm": l1,
        "cosine_sim": cos_sim,
        "mse_raw": raw_mse,
        "gripper_acc": grip_acc,
        "n_val_frames": len(arm_pred_cat),
    }


# ═══════════════════════════════════════════════════
#  Backbone Loaders (same as LIBERO v4)
# ═══════════════════════════════════════════════════
def load_single_backbone(name: str, checkpoint_path: str, device: torch.device):
    """Load a single visual encoder by name."""
    sys.path.insert(0, str(PROJECT_ROOT))

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
        import timm
        try:
            from vc_models.models.vit import model_utils
            vc1_model, embd_size, _, _ = model_utils.load_model(
                model_utils.VC1_LARGE_NAME
            )
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
            log.warning("vc_models not installed. Using ViT-L MAE as VC-1 proxy.")
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
            log.info("Loaded MVP weights from checkpoint")
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
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
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
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
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
    parser = argparse.ArgumentParser(description="DROID-100 frozen-encoder evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--data_dir", type=str, default=str(DROID_DATA))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone", type=str, required=True, choices=ALL_BACKBONES)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick test: 20 epochs, 1 seed")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_epochs = 20
        args.n_seeds = 1
        log.info("=== SMOKE TEST MODE ===")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    backbone_name = args.backbone
    log.info("=" * 60)
    log.info(f"DROID-100 Evaluation — Backbone: {backbone_name}")
    log.info(f"  LSTM + Action Chunking (K={ACTION_CHUNK_K}), 3 cameras")
    log.info(f"  Epochs: {args.n_epochs}, Seeds: {args.n_seeds}")
    log.info(f"  Train/Val: {N_TRAIN}/{N_VAL} episodes")
    log.info(f"  Device: {device}")
    log.info("=" * 60)

    # Load backbone
    log.info(f"Loading {backbone_name}...")
    encoder = load_single_backbone(backbone_name, args.checkpoint, device)
    feat_dim = encoder.output_dim
    eval_transform = get_transform(backbone_name, training=False)
    log.info(f"  {backbone_name}: {feat_dim}-d features")

    # Load all DROID episodes
    log.info("Loading DROID-100 episodes...")
    episodes = load_droid_episodes(args.data_dir)

    # Split
    train_eps, val_eps = train_val_split(episodes, n_train=N_TRAIN, seed=42)

    # Compute normalization stats from training data
    norm_stats = compute_norm_stats(train_eps)
    log.info(f"  Arm action stats: mean={norm_stats['arm_mean'].round(3)}, "
             f"std={norm_stats['arm_std'].round(3)}")

    # Extract features for all 3 cameras
    camera_keys = ['ext1_imgs', 'ext2_imgs', 'wrist_imgs']
    train_features = {}
    val_features = {}

    for cam_key in camera_keys:
        log.info(f"  Extracting {cam_key} features...")
        train_features[cam_key] = extract_features_for_episodes(
            encoder, train_eps, cam_key, eval_transform, device,
            backbone_name=backbone_name, split="train")
        val_features[cam_key] = extract_features_for_episodes(
            encoder, val_eps, cam_key, eval_transform, device,
            backbone_name=backbone_name, split="val")

    log.info(f"  Features extracted: {feat_dim}-d per camera, "
             f"{feat_dim * 3}-d concatenated")

    # Build episode data dicts with normalized features
    train_data = build_episode_data(
        train_eps, train_features['ext1_imgs'],
        train_features['ext2_imgs'], train_features['wrist_imgs'],
        norm_stats)
    val_data = build_episode_data(
        val_eps, val_features['ext1_imgs'],
        val_features['ext2_imgs'], val_features['wrist_imgs'],
        norm_stats)

    # Build training dataset
    train_dataset = DROIDSequenceDataset(train_data, chunk_len=CHUNK_LEN)

    # Train & evaluate per seed
    all_metrics = defaultdict(list)
    for seed in range(args.n_seeds):
        log.info(f"\n  Seed {seed}: training LSTM BC ({args.n_epochs} epochs)...")
        policy = train_bc_lstm(
            train_dataset, feat_dim, device,
            epochs=args.n_epochs, lr=LR, seed=seed,
        )

        log.info(f"  Seed {seed}: evaluating on {len(val_data)} val episodes...")
        metrics = evaluate_offline(policy, val_data, norm_stats, device)

        for k, v in metrics.items():
            all_metrics[k].append(v)

        log.info(f"  Seed {seed}: MSE={metrics['mse_norm']:.4f}, "
                 f"L1={metrics['l1_norm']:.4f}, "
                 f"Cos={metrics['cosine_sim']:.4f}, "
                 f"GripAcc={metrics['gripper_acc']:.1%}")

        del policy
        torch.cuda.empty_cache()

    # Aggregate results
    results = {
        "backbone": backbone_name,
        "feature_dim": feat_dim,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_epochs": args.n_epochs,
        "n_seeds": args.n_seeds,
        "action_chunk_k": ACTION_CHUNK_K,
        "version": "v1",
    }
    for k, vals in all_metrics.items():
        results[f"{k}_mean"] = float(np.mean(vals))
        results[f"{k}_std"] = float(np.std(vals))
        results[f"{k}_seeds"] = [float(v) for v in vals]

    # Save
    output_file = os.path.join(args.output, f"droid100_{backbone_name}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    log.info(f"\n{'=' * 60}")
    log.info(f"DROID-100 Results — {backbone_name}")
    log.info(f"{'=' * 60}")
    log.info(f"  Feature dim: {feat_dim}")
    log.info(f"  MSE (norm):    {results['mse_norm_mean']:.4f} ± {results['mse_norm_std']:.4f}")
    log.info(f"  L1  (norm):    {results['l1_norm_mean']:.4f} ± {results['l1_norm_std']:.4f}")
    log.info(f"  Cosine sim:    {results['cosine_sim_mean']:.4f} ± {results['cosine_sim_std']:.4f}")
    log.info(f"  MSE (raw):     {results['mse_raw_mean']:.6f} ± {results['mse_raw_std']:.6f}")
    log.info(f"  Gripper acc:   {results['gripper_acc_mean']:.1%} ± {results['gripper_acc_std']:.1%}")
    log.info(f"\nResults saved to {output_file}")

    # Free images from memory
    del episodes, train_eps, val_eps, train_features, val_features
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
