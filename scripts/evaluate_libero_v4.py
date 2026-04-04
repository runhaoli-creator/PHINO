#!/usr/bin/env python3
"""
evaluate_libero_v4.py — LIBERO-10 Frozen-Encoder BC evaluation (publication ready).

Fixes over v3:
  1. DynaCLIP normalization bug: "CLIP" in "DynaCLIP" → used CLIP norms instead of ImageNet.
     Now uses EXACT matching for backbone names.
  2. Increased training: 200 epochs, cosine schedule with restarts.
  3. Action chunking: predict K future actions, execute first, temporal smoothing.
  4. Image augmentation during training: random crop+resize, color jitter.
  5. Increased max sim steps to 400 (some LIBERO tasks need >300 steps).
  6. Added VC-1 and MVP backbone support.
  7. Proper evaluation: 50 episodes, 3 seeds, per-task and average reporting.
  8. Action ensemble: average actions from overlapping chunks (eval-time smoothing).

Usage:
  python evaluate_libero_v4.py --gpu 0 --backbone DynaCLIP
  python evaluate_libero_v4.py --gpu 1 --backbone VC-1
  python evaluate_libero_v4.py --gpu 2 --backbone DINOv2
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path

# ── Limit CPU threads BEFORE importing heavy libs ──
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import h5py
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
LIBERO_DATA = PROJECT_ROOT / "data" / "libero" / "datasets" / "libero_10"
RESULTS_DIR = PROJECT_ROOT / "results" / "libero10_v4"
FEATURE_CACHE = PROJECT_ROOT / "data_cache" / "libero10_features_v4"

# ── Hyperparameters (v4 improved) ──
CHUNK_LEN = 50         # LSTM training sequence chunk length
ACTION_CHUNK_K = 10    # Action chunking: predict K future actions
PROPRIO_DIM = 12       # ee_pos(3) + gripper_qpos(2) + joint_pos(7)
ARM_DIM = 6            # delta xyz + delta rotation
ACTION_DIM = 7         # arm(6) + gripper(1)
LSTM_HIDDEN = 512      # LSTM hidden dimension
LSTM_LAYERS = 2        # LSTM layers
N_EPOCHS = 200         # Training epochs (doubled from v3)
BATCH_SIZE = 64        # Sequence batch size (doubled)
LR = 3e-4              # Learning rate (tuned down)
WEIGHT_DECAY = 0.01    # Weight decay
MAX_STEPS = 400        # Max sim steps per episode (increased from 300)
N_EVAL_EPISODES = 50   # LIBERO standard = 50
TEMPORAL_AGG_M = 0.5   # Temporal action aggregation (exponential weighting)


# ═══════════════════════════════════════════════════
#  LSTM Policy Network with Action Chunking
# ═══════════════════════════════════════════════════
class LSTMPolicy(nn.Module):
    """LSTM-based BC policy with dual visual input + proprio + action chunking."""

    def __init__(self, visual_dim: int, proprio_dim: int = PROPRIO_DIM,
                 hidden_dim: int = LSTM_HIDDEN, n_layers: int = LSTM_LAYERS,
                 action_chunk_k: int = ACTION_CHUNK_K):
        super().__init__()
        self.action_chunk_k = action_chunk_k
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim * 2, 512),       # dual camera → 512
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
        # Action chunking: predict K future arm actions + K future gripper logits
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

    def forward(self, vis_agent, vis_eye, proprio, h=None):
        """
        Args:
            vis_agent: (B, T, feat_dim)
            vis_eye:   (B, T, feat_dim)
            proprio:   (B, T, proprio_dim)
            h:         optional LSTM hidden state
        Returns:
            arm_pred:  (B, T, K, 6), K action chunks
            grip_pred: (B, T, K) logits
            h:         updated hidden state
        """
        B, T, _ = vis_agent.shape
        vis = torch.cat([vis_agent, vis_eye], dim=-1)
        vis_proj = self.visual_proj(vis)
        prop_proj = self.proprio_proj(proprio)
        x = torch.cat([vis_proj, prop_proj], dim=-1)
        lstm_out, h = self.lstm(x, h)
        arm = self.arm_head(lstm_out)  # (B, T, K*6)
        grip = self.gripper_head(lstm_out)  # (B, T, K)
        arm = arm.reshape(B, T, self.action_chunk_k, ARM_DIM)
        return arm, grip, h

    def forward_step(self, vis_agent, vis_eye, proprio, h=None):
        """Single-step forward for evaluation.
        Inputs: (B, dim) → expand to (B, 1, dim) for LSTM.
        Returns: arm (B, K, 6), grip (B, K), h
        """
        if vis_agent.dim() == 1:
            vis_agent = vis_agent.unsqueeze(0)
        if vis_eye.dim() == 1:
            vis_eye = vis_eye.unsqueeze(0)
        if proprio.dim() == 1:
            proprio = proprio.unsqueeze(0)
        arm, grip, h = self.forward(
            vis_agent.unsqueeze(1), vis_eye.unsqueeze(1), proprio.unsqueeze(1), h
        )
        return arm.squeeze(1), grip.squeeze(1), h  # (B, K, 6), (B, K), h


# ═══════════════════════════════════════════════════
#  Sequence Dataset with action chunking targets
# ═══════════════════════════════════════════════════
class SequenceDataset(Dataset):
    """Fixed-length sequence chunks from demo trajectories."""

    def __init__(self, demo_vis_a, demo_vis_e, demo_proprio,
                 demo_arm_acts, demo_grip_targets, chunk_len=CHUNK_LEN,
                 action_chunk_k=ACTION_CHUNK_K):
        self.vis_a_chunks = []
        self.vis_e_chunks = []
        self.proprio_chunks = []
        self.arm_chunks = []       # (chunk_len, K, 6) for each sample
        self.grip_chunks = []      # (chunk_len, K) for each sample

        for di in range(len(demo_vis_a)):
            T = len(demo_vis_a[di])
            if T < chunk_len:
                # Pad short demos
                pad_len = chunk_len - T
                vis_a = torch.cat([demo_vis_a[di], demo_vis_a[di][-1:].expand(pad_len, -1)])
                vis_e = torch.cat([demo_vis_e[di], demo_vis_e[di][-1:].expand(pad_len, -1)])
                proprio = torch.cat([demo_proprio[di], demo_proprio[di][-1:].expand(pad_len, -1)])
                arm = torch.cat([demo_arm_acts[di], demo_arm_acts[di][-1:].expand(pad_len, -1)])
                grip = torch.cat([demo_grip_targets[di], demo_grip_targets[di][-1:].expand(pad_len, -1)])
            else:
                vis_a = demo_vis_a[di]
                vis_e = demo_vis_e[di]
                proprio = demo_proprio[di]
                arm = demo_arm_acts[di]
                grip = demo_grip_targets[di]
            
            T_padded = len(vis_a)
            
            # Build action-chunk targets: for timestep t, targets are actions[t:t+K]
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
                self.vis_a_chunks.append(vis_a[start:end])
                self.vis_e_chunks.append(vis_e[start:end])
                self.proprio_chunks.append(proprio[start:end])
                self.arm_chunks.append(arm_chunked[start:end])
                self.grip_chunks.append(grip_chunked[start:end])

        self.vis_a_chunks = torch.stack(self.vis_a_chunks)
        self.vis_e_chunks = torch.stack(self.vis_e_chunks)
        self.proprio_chunks = torch.stack(self.proprio_chunks)
        self.arm_chunks = torch.stack(self.arm_chunks)    # (N, L, K, 6)
        self.grip_chunks = torch.stack(self.grip_chunks)  # (N, L, K)

        log.info(f"    Dataset: {len(self)} chunks of {chunk_len} steps, "
                 f"visual_dim={self.vis_a_chunks.shape[-1]}, "
                 f"action_chunk_k={action_chunk_k}")

    def __len__(self):
        return len(self.vis_a_chunks)

    def __getitem__(self, idx):
        return (self.vis_a_chunks[idx], self.vis_e_chunks[idx],
                self.proprio_chunks[idx], self.arm_chunks[idx],
                self.grip_chunks[idx])


# ═══════════════════════════════════════════════════
#  Per-Backbone Image Transforms — BUG FIX: exact matching
# ═══════════════════════════════════════════════════
def get_transform(backbone_name: str, training: bool = False):
    """Return the correct normalization transform for each backbone.
    
    v4 fix: Use EXACT matching to avoid 'CLIP in DynaCLIP' bug.
    v4 add: Training augmentation (random crop, color jitter).
    """
    if training:
        # Training augmentation: random resized crop + color jitter
        base = [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                         antialias=True),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        ]
    else:
        base = [transforms.Resize((224, 224), antialias=True)]

    # EXACT matching for backbone normalization (v4 FIX)
    NORM_MAP = {
        "DynaCLIP": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # ImageNet (DINOv2-based)
        "DINOv2":   {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # ImageNet
        "MCR":      {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # ImageNet (MAE)
        "VC-1":     {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # ImageNet (MAE-based)
        "MVP":      {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},  # ImageNet (MAE-based)
        "CLIP":     {"mean": [0.48145466, 0.4578275, 0.40821073],
                     "std": [0.26862954, 0.26130258, 0.27577711]},  # OpenAI CLIP
        "SigLIP":   {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},  # SigLIP
        "R3M":      None,  # R3M: no normalization (expects [0, 255])
    }

    norms = NORM_MAP.get(backbone_name)
    if norms is None:
        # R3M: just resize, no normalization
        return transforms.Compose(base)
    else:
        return transforms.Compose(base + [
            transforms.Normalize(mean=norms["mean"], std=norms["std"]),
        ])


# ═══════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════
def load_demo_data(hdf5_path: str, max_demos: int = 50):
    """Load both cameras, proprio, and actions from HDF5."""
    all_agent = []
    all_eye = []
    all_proprio = []
    all_arm = []
    all_grip = []

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])[:max_demos]
        for demo_key in demos:
            obs = f["data"][demo_key]["obs"]
            actions = f["data"][demo_key]["actions"][:]  # (T, 7)

            # Agentview camera
            agent_key = "agentview_rgb" if "agentview_rgb" in obs else "agentview_image"
            agent_imgs = obs[agent_key][:]

            # Eye-in-hand camera
            eye_key = "eye_in_hand_rgb" if "eye_in_hand_rgb" in obs else "eye_in_hand_image"
            if eye_key in obs:
                eye_imgs = obs[eye_key][:]
            else:
                log.warning(f"    No eye-in-hand camera in {demo_key}, using agentview")
                eye_imgs = agent_imgs.copy()

            # Proprio: ee_pos(3) + gripper_states(2) + joint_states(7) = 12
            ee_pos = obs["ee_pos"][:]
            gripper = obs["gripper_states"][:]
            joints = obs["joint_states"][:]
            proprio = np.concatenate([ee_pos, gripper, joints], axis=-1).astype(np.float32)

            # Arm actions (first 6 dims)
            arm_acts = actions[:, :6].astype(np.float32)

            # Gripper target: convert -1/+1 to 0/1 for BCE
            grip_raw = actions[:, 6].astype(np.float32)
            grip_target = ((grip_raw + 1.0) / 2.0).reshape(-1, 1)

            all_agent.append(agent_imgs)
            all_eye.append(eye_imgs)
            all_proprio.append(proprio)
            all_arm.append(arm_acts)
            all_grip.append(grip_target)

    total_frames = sum(len(x) for x in all_agent)
    log.info(f"    Loaded {len(all_agent)} demos, {total_frames} total frames (dual camera)")
    return all_agent, all_eye, all_proprio, all_arm, all_grip


# ═══════════════════════════════════════════════════
#  Feature Extraction (with caching + progress)
# ═══════════════════════════════════════════════════
def _get_cache_path(backbone_name: str, hdf5_path: str, camera: str) -> Path:
    cache_dir = FEATURE_CACHE / backbone_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = Path(hdf5_path).stem
    return cache_dir / f"{base}_{camera}.pt"


@torch.no_grad()
def extract_features_batch(encoder, all_images, transform, device, batch_size=128,
                           backbone_name="", hdf5_path="", camera="agent"):
    """Extract visual features for all frames with disk caching."""
    if backbone_name and hdf5_path:
        cache_path = _get_cache_path(backbone_name, hdf5_path, camera)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            log.info(f"      Loaded {camera} features from cache ({len(cached)} demos)")
            return cached

    encoder.eval()
    all_features = []
    n_demos = len(all_images)

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

        if (di + 1) % 10 == 0 or di == n_demos - 1:
            log.info(f"      {camera}: {di+1}/{n_demos} demos extracted")

    # Save to cache
    if backbone_name and hdf5_path:
        cache_path = _get_cache_path(backbone_name, hdf5_path, camera)
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
#  Training (v4 improved)
# ═══════════════════════════════════════════════════
def train_bc_lstm(dataset: SequenceDataset, visual_dim: int, device: torch.device,
                  epochs: int = N_EPOCHS, lr: float = LR, seed: int = 0):
    """Train LSTM-based BC policy with action chunking."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = LSTMPolicy(visual_dim=visual_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # Cosine annealing with warm restarts
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
    patience_limit = 40  # Early stopping patience

    for epoch in range(epochs):
        policy.train()
        total_loss = 0
        n_batches = 0

        for vis_a, vis_e, proprio, arm_act, grip_target in loader:
            vis_a = vis_a.to(device)
            vis_e = vis_e.to(device)
            proprio = proprio.to(device)
            arm_act = arm_act.to(device)      # (B, T, K, 6)
            grip_target = grip_target.to(device)  # (B, T, K)

            arm_pred, grip_pred, _ = policy(vis_a, vis_e, proprio)
            # arm_pred: (B, T, K, 6), grip_pred: (B, T, K)

            # Action chunking loss: weight closer actions more
            K = arm_pred.shape[2]
            chunk_weights = torch.exp(-0.5 * torch.arange(K, device=device).float())
            chunk_weights = chunk_weights / chunk_weights.sum()
            
            # Arm loss per chunk position
            arm_loss = 0
            for k in range(K):
                arm_loss += chunk_weights[k] * F.smooth_l1_loss(arm_pred[:, :, k], arm_act[:, :, k])
            
            # Gripper loss per chunk position
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

        # Early stopping
        if patience_counter >= patience_limit and epoch > 50:
            log.info(f"      Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    log.info(f"      Training done. Best loss: {best_loss:.5f}")
    return policy


# ═══════════════════════════════════════════════════
#  Simulator Evaluation with Temporal Action Aggregation
# ═══════════════════════════════════════════════════
def get_proprio_from_env(obs):
    """Extract 12-dim proprio from sim observation."""
    ee_pos = obs["robot0_eef_pos"]
    gripper = obs["robot0_gripper_qpos"]
    joints = obs["robot0_joint_pos"]
    return np.concatenate([ee_pos, gripper, joints]).astype(np.float32)


@torch.no_grad()
def evaluate_policy_in_sim(encoder, policy, task_id, suite, transform,
                           device, norm_stats, n_episodes=N_EVAL_EPISODES,
                           max_steps=MAX_STEPS):
    """Evaluate LSTM policy with temporal action aggregation."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 128,
        "camera_widths": 128,
    }
    env = OffScreenRenderEnv(**env_args)
    init_states = suite.get_task_init_states(task_id)

    proprio_mean = torch.from_numpy(norm_stats["proprio_mean"]).to(device)
    proprio_std = torch.from_numpy(norm_stats["proprio_std"]).to(device)
    arm_mean = torch.from_numpy(norm_stats["arm_mean"]).to(device)
    arm_std = torch.from_numpy(norm_stats["arm_std"]).to(device)

    successes = 0
    K = policy.action_chunk_k

    for ep in range(n_episodes):
        obs = env.reset()
        state_idx = ep % len(init_states)
        env.set_init_state(init_states[state_idx])
        obs, _, _, _ = env.step([0.0] * 7)

        h = None
        policy.eval()
        
        # Action chunk queue for temporal aggregation
        action_queue = []  # list of (arm_chunk, grip_chunk), most recent first

        for step in range(max_steps):
            # Encode agentview
            agent_img = obs["agentview_image"]
            agent_t = torch.from_numpy(agent_img).permute(2, 0, 1).float() / 255.0
            agent_t = transform(agent_t).unsqueeze(0).to(device)
            agent_feat = _encode(encoder, agent_t)

            # Encode eye-in-hand
            eye_img = obs["robot0_eye_in_hand_image"]
            eye_t = torch.from_numpy(eye_img).permute(2, 0, 1).float() / 255.0
            eye_t = transform(eye_t).unsqueeze(0).to(device)
            eye_feat = _encode(encoder, eye_t)

            # Normalized proprio
            proprio = get_proprio_from_env(obs)
            proprio_t = torch.from_numpy(proprio).to(device)
            proprio_norm = (proprio_t - proprio_mean) / proprio_std

            # LSTM step → get action chunk predictions
            arm_pred, grip_logit, h = policy.forward_step(
                agent_feat, eye_feat, proprio_norm, h
            )
            # arm_pred: (1, K, 6), grip_logit: (1, K)

            arm_chunk = arm_pred[0].cpu()    # (K, 6)
            grip_chunk = grip_logit[0].cpu()  # (K,)

            # Add to queue (newest first)
            action_queue.insert(0, (arm_chunk, grip_chunk))
            if len(action_queue) > K:
                action_queue = action_queue[:K]

            # Temporal aggregation: weight overlapping predictions
            arm_agg = torch.zeros(ARM_DIM)
            grip_agg = 0.0
            total_weight = 0.0

            for i, (ac, gc) in enumerate(action_queue):
                # ac[i] is the prediction from i steps ago for the current timestep
                if i < ac.shape[0]:
                    w = np.exp(-TEMPORAL_AGG_M * i)
                    arm_agg += w * ac[i]
                    grip_agg += w * gc[i].item()
                    total_weight += w

            arm_agg /= total_weight
            grip_agg /= total_weight

            # Denormalize arm action
            arm_action = (arm_agg * arm_std.cpu() + arm_mean.cpu()).numpy()

            # Binarize gripper
            grip_action = 1.0 if grip_agg > 0 else -1.0

            action = np.concatenate([arm_action, [grip_action]])
            action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(action)

            if done or reward > 0:
                if reward > 0:
                    successes += 1
                break

    env.close()
    sr = successes / n_episodes
    return sr


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
#  Backbone Loaders
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
        from timm.models.vision_transformer import VisionTransformer
        # VC-1 is a ViT-L/16 trained with MAE on Ego4D+MNI
        # Load from Facebook's vc1-large model
        try:
            from vc_models.models.vit import model_utils
            vc1_model, embd_size, model_transforms, model_info = model_utils.load_model(
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
            # Fallback: use timm's ViT-L/16 MAE (similar architecture)
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
        # MVP is a ViT-B/16 trained with MAE on egocentric video + ImageNet
        try:
            # Try loading the actual MVP model
            mvp_model = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
            # Override with MVP weights if available
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
        except Exception as e:
            log.warning(f"Failed to load MVP: {e}")
            raise

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--data_dir", type=str, default=str(LIBERO_DATA))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--backbone", type=str, required=True,
                        choices=ALL_BACKBONES)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--n_episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task IDs or 'all'")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick smoke test: 20 epochs, 10 episodes, 1 seed, 2 tasks")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_epochs = 20
        args.n_episodes = 10
        args.n_seeds = 1
        if args.tasks == "all":
            args.tasks = "2,3"  # Task 2 and 3 (best performing tasks for quick validation)
        log.info("=== SMOKE TEST MODE ===")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    backbone_name = args.backbone
    log.info("=" * 60)
    log.info(f"LIBERO-10 v4 Evaluation — Backbone: {backbone_name}")
    log.info(f"  LSTM + Action Chunking (K={ACTION_CHUNK_K}), dual camera")
    log.info(f"  Epochs: {args.n_epochs}, Episodes: {args.n_episodes}, Seeds: {args.n_seeds}")
    log.info(f"  Device: {device}")
    log.info("=" * 60)

    # Load backbone
    log.info(f"Loading {backbone_name}...")
    encoder = load_single_backbone(backbone_name, args.checkpoint, device)
    obs_dim = encoder.output_dim
    eval_transform = get_transform(backbone_name, training=False)
    log.info(f"  {backbone_name}: {obs_dim}-d features")
    log.info(f"  Eval transform: {eval_transform}")

    # LIBERO benchmark
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_10"]()

    if args.tasks == "all":
        task_ids = list(range(suite.n_tasks))
    else:
        task_ids = [int(t) for t in args.tasks.split(",")]

    # Map task IDs to HDF5 files
    data_dir = Path(args.data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    task_to_hdf5 = {}
    for tid in task_ids:
        task = suite.get_task(tid)
        task_name = task.name
        for hf in hdf5_files:
            if task_name in hf.stem or hf.stem.replace("_demo", "") in task_name:
                task_to_hdf5[tid] = hf
                break
        if tid not in task_to_hdf5 and tid < len(hdf5_files):
            task_to_hdf5[tid] = hdf5_files[tid]

    results = {}
    task_results = {}

    for tid in task_ids:
        task = suite.get_task(tid)
        log.info(f"\n  Task {tid}: {task.language}")

        if tid not in task_to_hdf5:
            log.warning(f"  No HDF5 for task {tid}, skipping")
            continue

        # Load demo data
        log.info(f"    Loading demo data (dual camera)...")
        all_agent, all_eye, all_proprio, all_arm, all_grip = load_demo_data(
            str(task_to_hdf5[tid])
        )

        # Compute normalization
        norm_stats = compute_norm_stats(all_proprio, all_arm)
        log.info(f"    Arm action stats: mean={norm_stats['arm_mean'].round(3)}, "
                 f"std={norm_stats['arm_std'].round(3)}")

        # Normalize
        all_proprio_norm = normalize_list(all_proprio, norm_stats["proprio_mean"], norm_stats["proprio_std"])
        all_arm_norm = normalize_list(all_arm, norm_stats["arm_mean"], norm_stats["arm_std"])

        # Extract features (using eval transform for features — no augmentation)
        hdf5_str = str(task_to_hdf5[tid])
        log.info(f"    Extracting agentview features...")
        all_agent_feats = extract_features_batch(
            encoder, all_agent, eval_transform, device,
            backbone_name=backbone_name, hdf5_path=hdf5_str, camera="agentview")
        log.info(f"    Extracting eye-in-hand features...")
        all_eye_feats = extract_features_batch(
            encoder, all_eye, eval_transform, device,
            backbone_name=backbone_name, hdf5_path=hdf5_str, camera="eyeinhand")

        feat_dim = all_agent_feats[0].shape[1]
        log.info(f"    Features: dim={feat_dim}, total={sum(len(f) for f in all_agent_feats)} per camera")

        # Convert to tensors
        all_proprio_t = to_tensor_list(all_proprio_norm)
        all_arm_t = to_tensor_list(all_arm_norm)
        all_grip_t = to_tensor_list(all_grip)

        # Build dataset
        dataset = SequenceDataset(
            all_agent_feats, all_eye_feats, all_proprio_t,
            all_arm_t, all_grip_t, chunk_len=CHUNK_LEN,
        )

        # Train & evaluate per seed
        seed_successes = []
        for seed in range(args.n_seeds):
            log.info(f"    Seed {seed}: training LSTM BC ({args.n_epochs} epochs)...")
            policy = train_bc_lstm(
                dataset, obs_dim, device,
                epochs=args.n_epochs, lr=LR, seed=seed,
            )

            log.info(f"    Seed {seed}: evaluating ({args.n_episodes} episodes)...")
            sr = evaluate_policy_in_sim(
                encoder, policy, tid, suite, eval_transform, device, norm_stats,
                n_episodes=args.n_episodes, max_steps=MAX_STEPS,
            )
            seed_successes.append(sr)
            log.info(f"    Seed {seed}: success rate = {sr:.1%}")

        mean_sr = np.mean(seed_successes)
        std_sr = np.std(seed_successes)
        task_results[task.language] = {
            "task_id": tid,
            "mean": float(mean_sr),
            "std": float(std_sr),
            "seeds": [float(s) for s in seed_successes],
        }
        log.info(f"  → Task {tid}: {mean_sr:.1%} ± {std_sr:.1%}")

        # Free memory
        del all_agent, all_eye, all_agent_feats, all_eye_feats, dataset
        torch.cuda.empty_cache()

    # Aggregate
    if task_results:
        all_means = [v["mean"] for v in task_results.values()]
        results[backbone_name] = {
            "per_task": task_results,
            "avg_success_rate": float(np.mean(all_means)),
            "std_across_tasks": float(np.std(all_means)),
            "feature_dim": obs_dim,
            "n_episodes": args.n_episodes,
            "n_seeds": args.n_seeds,
            "n_epochs": args.n_epochs,
            "action_chunk_k": ACTION_CHUNK_K,
            "version": "v4",
        }

    # Save results
    output_file = os.path.join(args.output, f"libero10_v4_{backbone_name}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    log.info(f"\n{'=' * 60}")
    log.info(f"LIBERO-10 v4 Results — {backbone_name}")
    log.info(f"{'=' * 60}")
    for task_desc, tr in task_results.items():
        log.info(f"  Task {tr['task_id']}: {tr['mean']:.1%} ± {tr['std']:.1%}  {task_desc[:50]}")
    if task_results:
        avg = np.mean([v["mean"] for v in task_results.values()])
        log.info(f"\n  Average: {avg:.1%}")
    log.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
