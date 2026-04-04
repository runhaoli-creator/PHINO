#!/usr/bin/env python3
"""
evaluate_libero_v3.py — LIBERO-10 BC evaluation with LSTM policy.

Critical bug fixes over v2:
  1. Per-backbone image normalization (R3M, SigLIP, CLIP were WRONG)
  2. Dual camera input (agentview + eye-in-hand)
  3. LSTM policy for temporal reasoning (replaces MLP + frame stacking)
  4. Separate gripper handling (BCE loss for binary gripper)
  5. 50 evaluation episodes (LIBERO standard, was 20)
  6. SmoothL1 loss for arm actions (replaces MSE)
  7. Sequence-based LSTM training with proper chunks

Usage:
  python evaluate_libero_v3.py --gpu 0 --backbone DynaCLIP
  python evaluate_libero_v3.py --gpu 1 --backbone DINOv2
  ...
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
torch.set_num_threads(4)  # Limit PyTorch intra-op threads
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pretrain" / "dynaclip_final.pt"
LIBERO_DATA = PROJECT_ROOT / "data" / "libero" / "datasets" / "libero_10"
RESULTS_DIR = PROJECT_ROOT / "results" / "libero10_v3"
FEATURE_CACHE = PROJECT_ROOT / "data_cache" / "libero10_features"

# ── Hyperparameters ──
CHUNK_LEN = 50         # LSTM training sequence chunk length
PROPRIO_DIM = 12       # ee_pos(3) + gripper_qpos(2) + joint_pos(7)
ARM_DIM = 6            # delta xyz + delta rotation
ACTION_DIM = 7         # arm(6) + gripper(1)
LSTM_HIDDEN = 512      # LSTM hidden dimension
LSTM_LAYERS = 2        # LSTM layers
N_EPOCHS = 100         # Training epochs
BATCH_SIZE = 32        # Sequence batch size
LR = 1e-3              # Learning rate
MAX_STEPS = 300        # Max sim steps per episode
N_EVAL_EPISODES = 50   # LIBERO standard: 50 initial states


# ═══════════════════════════════════════════════════
#  LSTM Policy Network
# ═══════════════════════════════════════════════════
class LSTMPolicy(nn.Module):
    """LSTM-based BC policy with dual visual input + proprio."""

    def __init__(self, visual_dim: int, proprio_dim: int = PROPRIO_DIM,
                 hidden_dim: int = LSTM_HIDDEN, n_layers: int = LSTM_LAYERS):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim * 2, 256),  # dual camera
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=256 + 64,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.arm_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, ARM_DIM),
        )
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, vis_agent, vis_eye, proprio, h=None):
        """
        Full-sequence forward for training.
        Args:
            vis_agent: (B, T, feat_dim) agentview features
            vis_eye:   (B, T, feat_dim) eye-in-hand features
            proprio:   (B, T, proprio_dim)
            h:         optional LSTM hidden state
        Returns:
            arm_pred:  (B, T, 6)
            grip_pred: (B, T, 1) logits
            h:         updated hidden state
        """
        vis = torch.cat([vis_agent, vis_eye], dim=-1)
        vis_proj = self.visual_proj(vis)
        prop_proj = self.proprio_proj(proprio)
        x = torch.cat([vis_proj, prop_proj], dim=-1)
        lstm_out, h = self.lstm(x, h)
        arm = self.arm_head(lstm_out)
        grip = self.gripper_head(lstm_out)
        return arm, grip, h

    def forward_step(self, vis_agent, vis_eye, proprio, h=None):
        """Single-step forward for evaluation.
        Inputs: (B, dim) → expand to (B, 1, dim) for LSTM.
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
        return arm.squeeze(1), grip.squeeze(1), h


# ═══════════════════════════════════════════════════
#  Sequence Dataset
# ═══════════════════════════════════════════════════
class SequenceDataset(Dataset):
    """Fixed-length sequence chunks from demo trajectories."""

    def __init__(self, demo_vis_a, demo_vis_e, demo_proprio,
                 demo_arm_acts, demo_grip_targets, chunk_len=CHUNK_LEN):
        self.vis_a_chunks = []
        self.vis_e_chunks = []
        self.proprio_chunks = []
        self.arm_chunks = []
        self.grip_chunks = []

        for di in range(len(demo_vis_a)):
            T = len(demo_vis_a[di])
            # If demo shorter than chunk_len, pad it
            if T < chunk_len:
                pad_len = chunk_len - T
                vis_a = torch.cat([demo_vis_a[di], demo_vis_a[di][-1:].expand(pad_len, -1)])
                vis_e = torch.cat([demo_vis_e[di], demo_vis_e[di][-1:].expand(pad_len, -1)])
                proprio = torch.cat([demo_proprio[di], demo_proprio[di][-1:].expand(pad_len, -1)])
                arm = torch.cat([demo_arm_acts[di], demo_arm_acts[di][-1:].expand(pad_len, -1)])
                grip = torch.cat([demo_grip_targets[di], demo_grip_targets[di][-1:].expand(pad_len, -1)])
                self.vis_a_chunks.append(vis_a)
                self.vis_e_chunks.append(vis_e)
                self.proprio_chunks.append(proprio)
                self.arm_chunks.append(arm)
                self.grip_chunks.append(grip)
                continue

            # Sliding window with 50% overlap
            stride = max(1, chunk_len // 2)
            for start in range(0, T - chunk_len + 1, stride):
                end = start + chunk_len
                self.vis_a_chunks.append(demo_vis_a[di][start:end])
                self.vis_e_chunks.append(demo_vis_e[di][start:end])
                self.proprio_chunks.append(demo_proprio[di][start:end])
                self.arm_chunks.append(demo_arm_acts[di][start:end])
                self.grip_chunks.append(demo_grip_targets[di][start:end])

        self.vis_a_chunks = torch.stack(self.vis_a_chunks)
        self.vis_e_chunks = torch.stack(self.vis_e_chunks)
        self.proprio_chunks = torch.stack(self.proprio_chunks)
        self.arm_chunks = torch.stack(self.arm_chunks)
        self.grip_chunks = torch.stack(self.grip_chunks)

        log.info(f"    Dataset: {len(self)} chunks of {chunk_len} steps, "
                 f"visual_dim={self.vis_a_chunks.shape[-1]}")

    def __len__(self):
        return len(self.vis_a_chunks)

    def __getitem__(self, idx):
        return (self.vis_a_chunks[idx], self.vis_e_chunks[idx],
                self.proprio_chunks[idx], self.arm_chunks[idx],
                self.grip_chunks[idx])


# ═══════════════════════════════════════════════════
#  Per-Backbone Image Transforms (BUG FIX!)
# ═══════════════════════════════════════════════════
def get_transform(backbone_name: str):
    """Return the correct normalization transform for each backbone."""
    resize = [transforms.Resize((224, 224), antialias=True)]

    if backbone_name == "R3M":
        # R3M expects images in [0, 255]. We pass [0,1] and multiply by 255 in wrapper.
        # NO ImageNet normalization!
        return transforms.Compose(resize)

    elif "SigLIP" in backbone_name:
        # SigLIP uses mean=0.5, std=0.5
        return transforms.Compose(resize + [
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    elif "CLIP" in backbone_name:
        # OpenAI CLIP normalization
        return transforms.Compose(resize + [
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    else:
        # DynaCLIP, DINOv2, MCR (MAE) → ImageNet normalization
        return transforms.Compose(resize + [
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ═══════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════
def load_demo_data(hdf5_path: str, max_demos: int = 50):
    """Load both cameras, proprio, and actions from HDF5.

    Returns:
        all_agentview: list of (T, H, W, C) uint8 arrays
        all_eyeinhand: list of (T, H, W, C) uint8 arrays
        all_proprio:   list of (T, 12) float32 arrays
        all_arm_acts:  list of (T, 6) float32 arrays
        all_grip_targets: list of (T, 1) float32 arrays (0 or 1 for BCE)
    """
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
                # Fallback: duplicate agentview if no eye-in-hand
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
            grip_target = ((grip_raw + 1.0) / 2.0).reshape(-1, 1)  # -1→0, +1→1

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
    """Compute a deterministic cache path for extracted features."""
    cache_dir = FEATURE_CACHE / backbone_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use HDF5 filename + camera as key
    base = Path(hdf5_path).stem
    return cache_dir / f"{base}_{camera}.pt"


@torch.no_grad()
def extract_features_batch(encoder, all_images, transform, device, batch_size=128,
                           backbone_name="", hdf5_path="", camera="agent"):
    """Extract visual features for all frames. Returns list of (T, feat_dim) tensors.
    Supports disk caching to avoid re-extraction."""

    # Try loading from cache
    if backbone_name and hdf5_path:
        cache_path = _get_cache_path(backbone_name, hdf5_path, camera)
        if cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            log.info(f"      Loaded {camera} features from cache ({len(cached)} demos)")
            return cached

    encoder.eval()
    all_features = []
    n_demos = len(all_images)
    total_frames = sum(len(d) for d in all_images)

    for di, demo_imgs in enumerate(all_images):
        T = len(demo_imgs)
        demo_feats = []

        for i in range(0, T, batch_size):
            batch_np = demo_imgs[i:i + batch_size]
            # Vectorized preprocessing
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
    """Compute normalization stats for proprio and arm actions."""
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
def train_bc_lstm(dataset: SequenceDataset, visual_dim: int, device: torch.device,
                  epochs: int = N_EPOCHS, lr: float = LR, seed: int = 0):
    """Train LSTM-based BC policy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = LSTMPolicy(visual_dim=visual_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=False)

    grip_loss_fn = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        policy.train()
        total_loss = 0
        n_batches = 0

        for vis_a, vis_e, proprio, arm_act, grip_target in loader:
            vis_a = vis_a.to(device)
            vis_e = vis_e.to(device)
            proprio = proprio.to(device)
            arm_act = arm_act.to(device)
            grip_target = grip_target.to(device)

            arm_pred, grip_pred, _ = policy(vis_a, vis_e, proprio)

            # SmoothL1 loss for arm actions (more robust than MSE)
            arm_loss = F.smooth_l1_loss(arm_pred, arm_act)

            # BCE loss for binary gripper
            grip_loss = grip_loss_fn(grip_pred, grip_target)

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

        if (epoch + 1) % 20 == 0 or epoch == 0:
            log.info(f"      Epoch {epoch+1}/{epochs}  loss={avg_loss:.5f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}")

    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    log.info(f"      Training done. Best loss: {best_loss:.5f}")
    return policy


# ═══════════════════════════════════════════════════
#  Simulator Evaluation
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
    """Evaluate LSTM policy in LIBERO sim with dual camera."""
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

    # Normalization tensors
    proprio_mean = torch.from_numpy(norm_stats["proprio_mean"]).to(device)
    proprio_std = torch.from_numpy(norm_stats["proprio_std"]).to(device)
    arm_mean = torch.from_numpy(norm_stats["arm_mean"]).to(device)
    arm_std = torch.from_numpy(norm_stats["arm_std"]).to(device)

    successes = 0

    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        state_idx = ep % len(init_states)
        env.set_init_state(init_states[state_idx])
        # Step once to get valid obs after setting init state
        obs, _, _, _ = env.step([0.0] * 7)

        # Reset LSTM hidden state for each episode
        h = None
        policy.eval()

        for step in range(max_steps):
            # 1. Encode agentview
            agent_img = obs["agentview_image"]
            agent_t = torch.from_numpy(agent_img).permute(2, 0, 1).float() / 255.0
            agent_t = transform(agent_t).unsqueeze(0).to(device)
            agent_feat = _encode(encoder, agent_t)

            # 2. Encode eye-in-hand
            eye_img = obs["robot0_eye_in_hand_image"]
            eye_t = torch.from_numpy(eye_img).permute(2, 0, 1).float() / 255.0
            eye_t = transform(eye_t).unsqueeze(0).to(device)
            eye_feat = _encode(encoder, eye_t)

            # 3. Get normalized proprio
            proprio = get_proprio_from_env(obs)
            proprio_t = torch.from_numpy(proprio).to(device)
            proprio_norm = (proprio_t - proprio_mean) / proprio_std

            # 4. LSTM step
            arm_pred, grip_logit, h = policy.forward_step(
                agent_feat, eye_feat, proprio_norm, h
            )

            # 5. Denormalize arm action
            arm_action = arm_pred * arm_std + arm_mean
            arm_action = arm_action.cpu().numpy().flatten()

            # 6. Binarize gripper
            grip_action = 1.0 if grip_logit.item() > 0 else -1.0

            # 7. Compose and clip
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
    """Helper to encode a single image tensor through any backbone.
    Returns (1, feat_dim) keeping batch dimension."""
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
    return feat  # (1, feat_dim)


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

        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 1536
            def forward(self, x): return self.m(x, return_features=True)

        return Wrapper(model)

    elif name == "DINOv2":
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        dinov2.eval().to(device)
        for p in dinov2.parameters(): p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m(x)

        return Wrapper(dinov2)

    elif name == "CLIP":
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_model.eval().to(device)
        for p in clip_model.parameters(): p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m.encode_image(x)

        return Wrapper(clip_model)

    elif name == "R3M":
        from r3m import load_r3m
        r3m_model = load_r3m("resnet50")
        r3m_model.eval().to(device)
        for p in r3m_model.parameters(): p.requires_grad = False
        core = r3m_model.module if hasattr(r3m_model, "module") else r3m_model

        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 2048
            def forward(self, x): return self.m(x * 255.0)  # R3M expects [0, 255]

        return Wrapper(core)

    elif name == "MCR":
        import timm
        mcr = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
        mcr.eval().to(device)
        for p in mcr.parameters(): p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m(x)

        return Wrapper(mcr)

    elif name == "SigLIP":
        from transformers import SiglipVisionModel
        siglip = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        siglip.eval().to(device)
        for p in siglip.parameters(): p.requires_grad = False

        class Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m; self.output_dim = 768
            def forward(self, x): return self.m(x).pooler_output

        return Wrapper(siglip)

    else:
        raise ValueError(f"Unknown backbone: {name}")


ALL_BACKBONES = ["DynaCLIP", "DINOv2", "CLIP", "R3M", "MCR", "SigLIP"]


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
                        choices=ALL_BACKBONES,
                        help="Which backbone to evaluate")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--n_episodes", type=int, default=N_EVAL_EPISODES)
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task IDs or 'all'")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick smoke test: 10 epochs, 5 episodes, 1 seed")
    args = parser.parse_args()

    if args.smoke_test:
        args.n_epochs = 10
        args.n_episodes = 5
        args.n_seeds = 1
        log.info("=== SMOKE TEST MODE ===")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    backbone_name = args.backbone
    log.info("=" * 60)
    log.info(f"LIBERO-10 v3 Evaluation — Backbone: {backbone_name}")
    log.info(f"  LSTM policy, dual camera, per-backbone normalization")
    log.info(f"  Epochs: {args.n_epochs}, Episodes: {args.n_episodes}, Seeds: {args.n_seeds}")
    log.info(f"  Device: {device}")
    log.info("=" * 60)

    # Load backbone
    log.info(f"Loading {backbone_name}...")
    encoder = load_single_backbone(backbone_name, args.checkpoint, device)
    obs_dim = encoder.output_dim
    transform = get_transform(backbone_name)
    log.info(f"  {backbone_name}: {obs_dim}-d features, transform={transform}")

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

        # ── Load demo data (dual camera) ──
        log.info(f"    Loading demo data (dual camera)...")
        all_agent, all_eye, all_proprio, all_arm, all_grip = load_demo_data(
            str(task_to_hdf5[tid])
        )

        # ── Compute normalization (proprio + arm actions) ──
        norm_stats = compute_norm_stats(all_proprio, all_arm)
        log.info(f"    Arm action stats: mean={norm_stats['arm_mean'].round(3)}, "
                 f"std={norm_stats['arm_std'].round(3)}")

        # ── Normalize ──
        all_proprio_norm = normalize_list(all_proprio, norm_stats["proprio_mean"], norm_stats["proprio_std"])
        all_arm_norm = normalize_list(all_arm, norm_stats["arm_mean"], norm_stats["arm_std"])

        # ── Extract features for both cameras ──
        hdf5_str = str(task_to_hdf5[tid])
        log.info(f"    Extracting agentview features...")
        all_agent_feats = extract_features_batch(
            encoder, all_agent, transform, device,
            backbone_name=backbone_name, hdf5_path=hdf5_str, camera="agentview")
        log.info(f"    Extracting eye-in-hand features...")
        all_eye_feats = extract_features_batch(
            encoder, all_eye, transform, device,
            backbone_name=backbone_name, hdf5_path=hdf5_str, camera="eyeinhand")

        feat_dim = all_agent_feats[0].shape[1]
        log.info(f"    Features: dim={feat_dim}, total={sum(len(f) for f in all_agent_feats)} per camera")

        # Convert to tensors
        all_proprio_t = to_tensor_list(all_proprio_norm)
        all_arm_t = to_tensor_list(all_arm_norm)
        all_grip_t = to_tensor_list(all_grip)

        # ── Build dataset ──
        dataset = SequenceDataset(
            all_agent_feats, all_eye_feats, all_proprio_t,
            all_arm_t, all_grip_t, chunk_len=CHUNK_LEN,
        )

        # ── Train & evaluate per seed ──
        seed_successes = []
        for seed in range(args.n_seeds):
            log.info(f"    Seed {seed}: training LSTM BC ({args.n_epochs} epochs)...")
            policy = train_bc_lstm(
                dataset, obs_dim, device,
                epochs=args.n_epochs, lr=LR, seed=seed,
            )

            log.info(f"    Seed {seed}: evaluating ({args.n_episodes} episodes)...")
            sr = evaluate_policy_in_sim(
                encoder, policy, tid, suite, transform, device, norm_stats,
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
        }

    # Save results
    output_file = os.path.join(args.output, f"libero10_v3_{backbone_name}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    log.info(f"\n{'=' * 60}")
    log.info(f"LIBERO-10 v3 Results — {backbone_name}")
    log.info(f"{'=' * 60}")
    for task_desc, tr in task_results.items():
        log.info(f"  Task {tr['task_id']}: {tr['mean']:.1%} ± {tr['std']:.1%}  {task_desc[:50]}")
    if task_results:
        avg = np.mean([v["mean"] for v in task_results.values()])
        log.info(f"\n  Average: {avg:.1%}")
    log.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
