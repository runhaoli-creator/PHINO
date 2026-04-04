#!/usr/bin/env python3
"""
evaluate_libero_v2.py — Improved LIBERO-10 Behavioral Cloning evaluation.

Key improvements over v1:
  1. Proprioceptive state input: ee_pos(3) + gripper(2) + joints(7) = 12 dims
  2. Frame stacking: last 3 encoded frames concatenated
  3. Action chunking: predict 10 future actions, execute first (receding horizon)
  4. Deeper policy: 4-layer MLP with 1024 hidden dims
  5. Full data: all frames from all 50 demos per task (no subsampling)
  6. 300 training epochs with warmup + cosine decay
  7. Proper normalization of proprioceptive state and actions

Protocol:
  - Freeze visual encoder (no fine-tuning)
  - Pre-extract features for all demo frames
  - Train lightweight BC policy head on features + proprio → action chunks
  - Evaluate trained policy in LIBERO simulator
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pretrain" / "dynaclip_final.pt"
LIBERO_DATA = PROJECT_ROOT / "data" / "libero" / "datasets" / "libero_10"
RESULTS_DIR = PROJECT_ROOT / "results" / "libero10_v2"

# Hyperparameters
HISTORY_LEN = 3       # Number of stacked frames
ACTION_CHUNK = 10     # Predict this many future actions
PROPRIO_DIM = 12      # ee_pos(3) + gripper(2) + joints(7)
ACTION_DIM = 7        # 7-DOF actions
HIDDEN_DIM = 1024     # Policy hidden size
N_EPOCHS = 300        # Training epochs
BATCH_SIZE = 512      # Training batch size
LR = 3e-4             # Learning rate
WARMUP_EPOCHS = 10    # LR warmup
MAX_STEPS = 300       # Max sim steps per episode


# ─────────────────────────────────────────────────
#  Policy Network
# ─────────────────────────────────────────────────
class ActionChunkingPolicy(nn.Module):
    """MLP policy: stacked visual features + proprio → action chunk."""

    def __init__(self, obs_dim: int, proprio_dim: int = PROPRIO_DIM,
                 history_len: int = HISTORY_LEN, action_chunk: int = ACTION_CHUNK,
                 action_dim: int = ACTION_DIM, hidden: int = HIDDEN_DIM):
        super().__init__()
        visual_in = obs_dim * history_len

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, action_chunk * action_dim),
        )
        self.action_chunk = action_chunk
        self.action_dim = action_dim

    def forward(self, visual_feats: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_feats: (B, history_len * obs_dim)
            proprio: (B, proprio_dim)
        Returns:
            (B, action_chunk, action_dim)
        """
        v = self.visual_proj(visual_feats)
        p = self.proprio_proj(proprio)
        x = torch.cat([v, p], dim=-1)
        out = self.policy(x)
        return out.view(-1, self.action_chunk, self.action_dim)


# ─────────────────────────────────────────────────
#  Dataset for Pre-Extracted Features
# ─────────────────────────────────────────────────
class FeatureWindowDataset(Dataset):
    """Dataset over sliding windows of pre-extracted features."""

    def __init__(self, all_features, all_proprio, all_actions,
                 history_len=HISTORY_LEN, action_chunk=ACTION_CHUNK):
        self.feat_windows = []
        self.proprio_data = []
        self.action_chunks = []

        for di in range(len(all_features)):
            T = len(all_features[di])
            feats = all_features[di]    # (T, feat_dim) tensor
            proprio = all_proprio[di]   # (T, proprio_dim) numpy
            actions = all_actions[di]   # (T, action_dim) numpy

            for t in range(history_len - 1, T):
                # Feature window: stack last history_len frames
                feat_w = feats[t - history_len + 1: t + 1]  # (H, feat_dim)
                feat_flat = feat_w.reshape(-1)  # (H * feat_dim,)

                # Proprio at current timestep
                prop = torch.from_numpy(proprio[t]).float()

                # Action chunk: next action_chunk actions
                end = min(t + action_chunk, T)
                act = actions[t:end]

                # Pad if near end of demo
                if len(act) < action_chunk:
                    pad = np.tile(act[-1:], (action_chunk - len(act), 1))
                    act = np.concatenate([act, pad], axis=0)

                act = torch.from_numpy(act).float()  # (chunk, 7)

                self.feat_windows.append(feat_flat)
                self.proprio_data.append(prop)
                self.action_chunks.append(act)

        self.feat_windows = torch.stack(self.feat_windows)
        self.proprio_data = torch.stack(self.proprio_data)
        self.action_chunks = torch.stack(self.action_chunks)

        log.info(f"    Dataset: {len(self)} windows, feat={self.feat_windows.shape[1]}, "
                 f"proprio={self.proprio_data.shape[1]}, actions={self.action_chunks.shape[1:]}")

    def __len__(self):
        return len(self.feat_windows)

    def __getitem__(self, idx):
        return self.feat_windows[idx], self.proprio_data[idx], self.action_chunks[idx]


# ─────────────────────────────────────────────────
#  Load Demo Data from HDF5
# ─────────────────────────────────────────────────
def load_demo_data(hdf5_path: str, max_demos: int = 50):
    """Load images, proprioceptive state, and actions from an HDF5 file.

    Returns:
        all_images: list of (T, H, W, C) uint8 arrays
        all_proprio: list of (T, 12) float32 arrays
        all_actions: list of (T, 7) float32 arrays
    """
    all_images = []
    all_proprio = []
    all_actions = []

    with h5py.File(hdf5_path, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])[:max_demos]
        for demo_key in demos:
            obs = f["data"][demo_key]["obs"]
            acts = f["data"][demo_key]["actions"][:]

            # Image key
            img_key = "agentview_rgb" if "agentview_rgb" in obs else "agentview_image"
            imgs = obs[img_key][:]  # (T, H, W, C)

            # Proprioceptive state: ee_pos(3) + gripper(2) + joints(7)
            ee_pos = obs["ee_pos"][:]             # (T, 3)
            gripper = obs["gripper_states"][:]     # (T, 2)
            joints = obs["joint_states"][:]        # (T, 7)
            proprio = np.concatenate([ee_pos, gripper, joints], axis=-1).astype(np.float32)

            all_images.append(imgs)
            all_proprio.append(proprio)
            all_actions.append(acts.astype(np.float32))

    log.info(f"    Loaded {len(all_images)} demos, "
             f"{sum(len(x) for x in all_images)} total frames")
    return all_images, all_proprio, all_actions


# ─────────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────────
@torch.no_grad()
def extract_features_all(encoder, all_images, transform, device, batch_size=64):
    """Extract visual features for every frame across all demos.

    Returns:
        list of (T, feat_dim) tensors (on CPU)
    """
    encoder.eval()
    all_features = []

    for demo_imgs in all_images:
        T = len(demo_imgs)
        demo_feats = []

        for i in range(0, T, batch_size):
            batch_np = demo_imgs[i:i + batch_size]
            imgs = []
            for img in batch_np:
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_t = transform(img_t)
                imgs.append(img_t)
            batch = torch.stack(imgs).to(device)

            feats = encoder(batch)
            if isinstance(feats, dict):
                feats = feats.get("pooler_output", feats.get("last_hidden_state"))
                if feats.dim() == 3:
                    feats = feats[:, 0]
            elif isinstance(feats, tuple):
                feats = feats[0]
                if feats.dim() == 3:
                    feats = feats[:, 0]

            demo_feats.append(feats.cpu())

        all_features.append(torch.cat(demo_feats))  # (T, feat_dim)

    return all_features


# ─────────────────────────────────────────────────
#  Normalization Statistics
# ─────────────────────────────────────────────────
def compute_normalization(all_proprio, all_actions):
    """Compute mean/std for proprio and actions across all demos."""
    proprio_cat = np.concatenate(all_proprio, axis=0)  # (N, 12)
    actions_cat = np.concatenate(all_actions, axis=0)   # (N, 7)

    stats = {
        "proprio_mean": proprio_cat.mean(0).astype(np.float32),
        "proprio_std": (proprio_cat.std(0) + 1e-6).astype(np.float32),
        "action_mean": actions_cat.mean(0).astype(np.float32),
        "action_std": (actions_cat.std(0) + 1e-6).astype(np.float32),
    }
    return stats


def normalize_proprio_list(all_proprio, stats):
    """Normalize proprioceptive state."""
    return [
        (p - stats["proprio_mean"]) / stats["proprio_std"]
        for p in all_proprio
    ]


def normalize_actions_list(all_actions, stats):
    """Normalize actions."""
    return [
        (a - stats["action_mean"]) / stats["action_std"]
        for a in all_actions
    ]


# ─────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────
def train_bc(dataset: FeatureWindowDataset, obs_dim: int, device: torch.device,
             epochs: int = N_EPOCHS, lr: float = LR, seed: int = 0):
    """Train a BC policy on pre-extracted features with warmup + cosine."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = ActionChunkingPolicy(obs_dim=obs_dim).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return epoch / max(WARMUP_EPOCHS, 1)
        progress = (epoch - WARMUP_EPOCHS) / max(epochs - WARMUP_EPOCHS, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=False)

    # Temporal weights: closer actions matter more
    chunk_weights = 0.95 ** torch.arange(ACTION_CHUNK, dtype=torch.float32).to(device)
    chunk_weights = chunk_weights / chunk_weights.sum()  # (C,)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        policy.train()
        total_loss = 0
        n_batches = 0

        for feat_w, prop, act_chunk in loader:
            feat_w = feat_w.to(device)
            prop = prop.to(device)
            act_chunk = act_chunk.to(device)  # (B, C, 7)

            pred = policy(feat_w, prop)  # (B, C, 7)

            # Weighted MSE over action chunk
            mse = (pred - act_chunk) ** 2  # (B, C, 7)
            mse = mse.mean(dim=-1)  # (B, C)
            loss = (mse * chunk_weights.unsqueeze(0)).sum(dim=-1).mean()

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

        if (epoch + 1) % 50 == 0 or epoch == 0:
            log.info(f"      Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Restore best
    if best_state is not None:
        policy.load_state_dict(best_state)
    policy.eval()
    log.info(f"      Training done. Best loss: {best_loss:.6f}")
    return policy


# ─────────────────────────────────────────────────
#  Simulator Evaluation
# ─────────────────────────────────────────────────
def get_proprio_from_env(obs):
    """Extract 12-dim proprioceptive state from simulator observation."""
    ee_pos = obs["robot0_eef_pos"]         # (3,)
    gripper = obs["robot0_gripper_qpos"]   # (2,)
    joints = obs["robot0_joint_pos"]       # (7,)
    proprio = np.concatenate([ee_pos, gripper, joints]).astype(np.float32)
    return proprio


@torch.no_grad()
def evaluate_policy_in_sim(encoder, policy, task_id, suite, transform,
                           device, norm_stats, n_episodes=20, max_steps=MAX_STEPS):
    """Evaluate BC policy in LIBERO simulator with frame stacking + proprio."""
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
    action_mean = torch.from_numpy(norm_stats["action_mean"]).to(device)
    action_std = torch.from_numpy(norm_stats["action_std"]).to(device)

    successes = 0

    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        state_idx = ep % len(init_states)
        env.set_init_state(init_states[state_idx])
        # Step once to get valid obs after setting init state
        dummy_action = [0.0] * 7
        obs, _, _, _ = env.step(dummy_action)

        # Frame buffer for stacking
        feat_buffer = deque(maxlen=HISTORY_LEN)

        for step in range(max_steps):
            # 1. Encode current frame
            img = obs["agentview_image"]  # (H, W, C) uint8
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = transform(img_t).unsqueeze(0).to(device)

            feat = encoder(img_t)
            if isinstance(feat, dict):
                feat = feat.get("pooler_output", feat.get("last_hidden_state"))
                if feat.dim() == 3:
                    feat = feat[:, 0]
            elif isinstance(feat, tuple):
                feat = feat[0]
                if feat.dim() == 3:
                    feat = feat[:, 0]
            feat = feat.squeeze(0)  # (feat_dim,)

            feat_buffer.append(feat)

            # Pad buffer at episode start
            while len(feat_buffer) < HISTORY_LEN:
                feat_buffer.appendleft(feat_buffer[0])

            # 2. Stack features
            stacked = torch.cat(list(feat_buffer))  # (H * feat_dim,)

            # 3. Get normalized proprioceptive state
            proprio = get_proprio_from_env(obs)
            proprio_t = torch.from_numpy(proprio).to(device)
            proprio_norm = (proprio_t - proprio_mean) / proprio_std

            # 4. Predict action chunk
            action_chunk = policy(
                stacked.unsqueeze(0),
                proprio_norm.unsqueeze(0),
            )  # (1, C, 7)

            # 5. Execute first action (receding horizon)
            action = action_chunk[0, 0]  # (7,)
            action = action * action_std + action_mean  # denormalize
            action = action.cpu().numpy()
            action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(action)

            if done or reward > 0:
                if reward > 0:
                    successes += 1
                break

    env.close()
    sr = successes / n_episodes
    return sr


# ─────────────────────────────────────────────────
#  Load Backbones (same as v1)
# ─────────────────────────────────────────────────
def load_backbones(checkpoint_path, device):
    """Load all visual encoders."""
    backbones = {}
    sys.path.insert(0, str(PROJECT_ROOT))
    from dynaclip.models.dynaclip import DynaCLIPModel

    # 1. DynaCLIP
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = DynaCLIPModel(backbone_name="dinov2_vitb14", embed_dim=512)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)

        class DynaCLIPBackbone(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 1536

            def forward(self, x):
                return self.m(x, return_features=True)

        backbones["DynaCLIP"] = DynaCLIPBackbone(model)
        log.info("  Loaded DynaCLIP (1536-d)")
    except Exception as e:
        log.warning(f"  DynaCLIP load failed: {e}")

    # 2. DINOv2-ViT-B/14
    try:
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        dinov2.eval().to(device)
        for p in dinov2.parameters():
            p.requires_grad = False

        class DINOv2Wrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768

            def forward(self, x):
                return self.m(x)

        backbones["DINOv2-B/14"] = DINOv2Wrapper(dinov2)
        log.info("  Loaded DINOv2-ViT-B/14 (768-d)")
    except Exception as e:
        log.warning(f"  DINOv2 failed: {e}")

    # 3. CLIP ViT-L/14
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        clip_model.eval().to(device)
        for p in clip_model.parameters():
            p.requires_grad = False

        class CLIPWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768

            def forward(self, x):
                return self.m.encode_image(x)

        backbones["CLIP-L/14"] = CLIPWrapper(clip_model)
        log.info("  Loaded CLIP-ViT-L/14 (768-d)")
    except Exception as e:
        log.warning(f"  CLIP failed: {e}")

    # 4. R3M (ResNet-50)
    try:
        from r3m import load_r3m
        r3m_model = load_r3m("resnet50")
        r3m_model.eval().to(device)
        for p in r3m_model.parameters():
            p.requires_grad = False

        # Unwrap DataParallel if needed
        core = r3m_model
        if hasattr(core, "module"):
            core = core.module

        class R3MWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 2048

            def forward(self, x):
                return self.m(x * 255.0)

        backbones["R3M"] = R3MWrapper(core)
        log.info("  Loaded R3M-ResNet50 (2048-d)")
    except Exception as e:
        log.warning(f"  R3M failed: {e}")

    # 5. MCR (MAE ViT-B/16 — strong self-supervised baseline)
    try:
        import timm
        mcr = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
        mcr.eval().to(device)
        for p in mcr.parameters():
            p.requires_grad = False

        class MCRWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768

            def forward(self, x):
                return self.m(x)

        backbones["MCR"] = MCRWrapper(mcr)
        log.info("  Loaded MCR/MAE-ViT-B/16 (768-d)")
    except Exception as e:
        log.warning(f"  MCR failed: {e}")

    # 6. SigLIP
    try:
        from transformers import SiglipVisionModel
        siglip_vision = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        siglip_vision.eval().to(device)
        for p in siglip_vision.parameters():
            p.requires_grad = False

        class SigLIPWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768

            def forward(self, x):
                out = self.m(x)
                return out.pooler_output

        backbones["SigLIP-B/16"] = SigLIPWrapper(siglip_vision)
        log.info("  Loaded SigLIP-ViT-B/16 (768-d)")
    except Exception as e:
        log.warning(f"  SigLIP failed: {e}")

    return backbones


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--data_dir", type=str, default=str(LIBERO_DATA))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task IDs or 'all'")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    log.info("=" * 60)
    log.info("LIBERO-10 Improved BC Evaluation (v2)")
    log.info(f"  Frame stacking: {HISTORY_LEN}")
    log.info(f"  Action chunking: {ACTION_CHUNK}")
    log.info(f"  Proprio dim: {PROPRIO_DIM}")
    log.info(f"  Epochs: {args.n_epochs}")
    log.info(f"  Device: {device}")
    log.info("=" * 60)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load backbones
    log.info("Loading visual encoders...")
    backbones = load_backbones(args.checkpoint, device)
    log.info(f"Loaded {len(backbones)} backbones: {list(backbones.keys())}")

    # Get LIBERO benchmark
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_10"]()

    # Get task list
    if args.tasks == "all":
        task_ids = list(range(suite.n_tasks))
    else:
        task_ids = [int(t) for t in args.tasks.split(",")]

    # Find HDF5 files
    data_dir = Path(args.data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    assert len(hdf5_files) >= len(task_ids), \
        f"Need {len(task_ids)} HDF5 files, found {len(hdf5_files)}"

    # Map task_id → hdf5 file
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

    for backbone_name, encoder in backbones.items():
        log.info(f"\n{'=' * 60}")
        log.info(f"Backbone: {backbone_name}")
        log.info(f"{'=' * 60}")

        obs_dim = getattr(encoder, "output_dim", 768)
        task_results = {}

        for tid in task_ids:
            task = suite.get_task(tid)
            log.info(f"\n  Task {tid}: {task.language}")

            if tid not in task_to_hdf5:
                log.warning(f"  No HDF5 for task {tid}, skipping")
                continue

            # ── Load demo data ──
            log.info(f"    Loading demo data...")
            all_images, all_proprio, all_actions = load_demo_data(
                str(task_to_hdf5[tid])
            )

            # ── Compute normalization ──
            norm_stats = compute_normalization(all_proprio, all_actions)

            # ── Normalize proprio and actions ──
            all_proprio_norm = normalize_proprio_list(all_proprio, norm_stats)
            all_actions_norm = normalize_actions_list(all_actions, norm_stats)

            # ── Extract features ──
            log.info(f"    Extracting features with {backbone_name}...")
            all_features = extract_features_all(encoder, all_images, transform, device)
            feat_dim = all_features[0].shape[1]
            log.info(f"    Features: dim={feat_dim}, "
                     f"{sum(len(f) for f in all_features)} total vectors")

            # ── Build dataset ──
            dataset = FeatureWindowDataset(
                all_features, all_proprio_norm, all_actions_norm,
                history_len=HISTORY_LEN, action_chunk=ACTION_CHUNK
            )

            # ── Train & evaluate per seed ──
            seed_successes = []
            for seed in range(args.n_seeds):
                log.info(f"    Seed {seed}: training BC ({args.n_epochs} epochs)...")
                policy = train_bc(dataset, obs_dim, device,
                                  epochs=args.n_epochs, seed=seed)

                log.info(f"    Seed {seed}: evaluating ({args.n_episodes} episodes)...")
                sr = evaluate_policy_in_sim(
                    encoder, policy, tid, suite, transform,
                    device, norm_stats,
                    n_episodes=args.n_episodes,
                    max_steps=MAX_STEPS,
                )
                seed_successes.append(sr)
                log.info(f"    Seed {seed}: success rate = {sr:.1%}")

            mean_sr = np.mean(seed_successes)
            std_sr = np.std(seed_successes)
            task_results[task.language] = {
                "mean": float(mean_sr),
                "std": float(std_sr),
                "seeds": [float(s) for s in seed_successes],
            }
            log.info(f"  → {backbone_name} Task {tid}: {mean_sr:.1%} ± {std_sr:.1%}")

            # Free memory
            del all_images, all_features, dataset
            torch.cuda.empty_cache()

        # Aggregate
        if task_results:
            all_means = [v["mean"] for v in task_results.values()]
            results[backbone_name] = {
                "per_task": task_results,
                "avg_success_rate": float(np.mean(all_means)),
                "std_success_rate": float(np.std(all_means)),
                "feature_dim": obs_dim,
            }
            log.info(f"\n  {backbone_name} overall: {np.mean(all_means):.1%} avg SR")

        # Save intermediate results
        output_file = os.path.join(args.output, f"libero10_v2_gpu{args.gpu}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("LIBERO-10 v2 Results Summary")
    log.info("=" * 60)
    log.info(f"{'Backbone':<20} {'Avg SR':>10} {'Std':>10}")
    log.info("-" * 40)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["avg_success_rate"]):
        log.info(f"{name:<20} {res['avg_success_rate']:>10.1%} "
                 f"{res['std_success_rate']:>10.1%}")

    log.info(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
