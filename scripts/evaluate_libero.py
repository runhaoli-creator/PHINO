#!/usr/bin/env python3
"""
LIBERO-10 Downstream Evaluation for DynaCLIP.

Trains a simple BC (Behavioral Cloning) policy on top of frozen visual features
from DynaCLIP and baseline encoders, then evaluates in the LIBERO simulator.

Protocol:
  - 10 LIBERO tasks, 50 demos each
  - Frozen visual encoder → MLP action head
  - 50 epochs BC training per task
  - 20 evaluation episodes per task × 3 seeds
  - Reports average success rate
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Setup ──
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pretrain" / "dynaclip_final.pt"
LIBERO_DATA = PROJECT_ROOT / "data" / "libero" / "datasets" / "libero_10"
RESULTS_DIR = PROJECT_ROOT / "results" / "libero10"


# ── BC Policy Head ──
class BCPolicyHead(nn.Module):
    """Simple MLP behavioral cloning head."""

    def __init__(self, obs_dim: int, action_dim: int = 7, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, features):
        return self.net(features)


# ── Demo Dataset ──
class LIBERODemoDataset(Dataset):
    """Load LIBERO demonstration HDF5 data."""

    def __init__(self, hdf5_path: str, transform=None, max_demos: int = 50):
        self.transform = transform
        self.images = []
        self.actions = []

        with h5py.File(hdf5_path, "r") as f:
            demos = sorted([k for k in f["data"].keys() if k.startswith("demo")])[:max_demos]
            for demo_key in demos:
                demo = f["data"][demo_key]
                obs = demo["obs"]

                # Get agentview images
                if "agentview_image" in obs:
                    imgs = obs["agentview_image"][:]  # (T, H, W, C)
                elif "agentview_rgb" in obs:
                    imgs = obs["agentview_rgb"][:]
                else:
                    # Try first available image key
                    img_keys = [k for k in obs.keys() if "image" in k or "rgb" in k]
                    if img_keys:
                        imgs = obs[img_keys[0]][:]
                    else:
                        continue

                acts = demo["actions"][:]  # (T, 7)

                # Subsample to keep memory reasonable (every 2 frames)
                step = max(1, len(imgs) // 100)
                for i in range(0, len(imgs) - 1, step):
                    self.images.append(imgs[i])
                    self.actions.append(acts[i])

        self.images = np.array(self.images)
        self.actions = np.array(self.actions, dtype=np.float32)
        log.info(f"  Loaded {len(self.images)} frames from {len(demos)} demos")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, C) uint8
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        if self.transform:
            img = self.transform(img)
        action = torch.from_numpy(self.actions[idx])
        return img, action


# ── Feature Extraction ──
@torch.no_grad()
def extract_features_batch(encoder, images, device, batch_size=64):
    """Extract features from a batch of images using a frozen encoder."""
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        feat = encoder(batch)
        if isinstance(feat, dict):
            feat = feat.get("pooler_output", feat.get("last_hidden_state", None))
            if feat is not None and feat.dim() == 3:
                feat = feat[:, 0]  # CLS token
        elif isinstance(feat, tuple):
            feat = feat[0]
            if feat.dim() == 3:
                feat = feat[:, 0]
        features.append(feat.cpu())
    return torch.cat(features, dim=0)


# ── Load Backbones ──
def load_backbones(checkpoint_path: str, device: torch.device):
    """Load DynaCLIP and baseline visual encoders."""
    backbones = {}

    # 1. DynaCLIP
    sys.path.insert(0, str(PROJECT_ROOT))
    from dynaclip.models.dynaclip import DynaCLIPModel

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

    # 2. DINOv2-ViT-B/14 (frozen, no fine-tuning)
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
        log.warning(f"  DINOv2 load failed: {e}")

    # 3. CLIP ViT-L/14
    try:
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
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
        log.warning(f"  CLIP load failed: {e}")

    # 4. R3M (ResNet-50)
    try:
        from r3m import load_r3m

        r3m_model = load_r3m("resnet50")
        # R3M loads as DataParallel; unwrap to avoid device mismatch
        if isinstance(r3m_model, nn.DataParallel):
            r3m_model = r3m_model.module
        r3m_model.eval().to(device)
        for p in r3m_model.parameters():
            p.requires_grad = False

        class R3MWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 2048

            def forward(self, x):
                return self.m(x * 255.0)  # R3M expects [0, 255]

        backbones["R3M"] = R3MWrapper(r3m_model)
        log.info("  Loaded R3M (2048-d)")
    except Exception as e:
        log.warning(f"  R3M load failed: {e}")

    # 5. MCR (MAE ViT-B/16)
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
        log.warning(f"  MCR load failed: {e}")

    # 6. SigLIP ViT-B/16
    try:
        from transformers import AutoModel, AutoProcessor

        siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        siglip_vision = siglip.vision_model
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
        log.warning(f"  SigLIP load failed: {e}")

    return backbones


# ── Train BC ──
def train_bc(features: torch.Tensor, actions: torch.Tensor, obs_dim: int,
             device: torch.device, epochs: int = 50, lr: float = 1e-3, seed: int = 0):
    """Train a BC policy head on precomputed features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = BCPolicyHead(obs_dim=obs_dim, action_dim=actions.shape[1]).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    features = features.to(device)
    actions = actions.to(device)

    n = len(features)
    batch_size = min(256, n)

    for epoch in range(epochs):
        policy.train()
        perm = torch.randperm(n)
        total_loss = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = policy(features[idx])
            loss = F.mse_loss(pred, actions[idx])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

    policy.eval()
    return policy


# ── Evaluate in Simulator ──
@torch.no_grad()
def evaluate_policy_in_sim(encoder, policy, task_id, suite, transform,
                           device, n_episodes=20, max_steps=300):
    """Evaluate a BC policy in the LIBERO simulator."""
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

    successes = 0
    for ep in range(n_episodes):
        env.seed(ep)
        obs = env.reset()
        state_idx = ep % len(init_states)
        env.set_init_state(init_states[state_idx])
        # Step once to get observations after setting init state
        dummy_action = [0.0] * 7
        obs, _, _, _ = env.step(dummy_action)

        for step in range(max_steps):
            # Get image
            img = obs["agentview_image"]  # (H, W, C)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_t = transform(img_t).unsqueeze(0).to(device)

            # Encode
            feat = encoder(img_t)
            if isinstance(feat, dict):
                feat = feat.get("pooler_output", feat.get("last_hidden_state"))
                if feat.dim() == 3:
                    feat = feat[:, 0]
            elif isinstance(feat, tuple):
                feat = feat[0]
                if feat.dim() == 3:
                    feat = feat[:, 0]

            # Predict action
            action = policy(feat).cpu().numpy()[0]
            action = np.clip(action, -1, 1)

            obs, reward, done, info = env.step(action)

            if done or reward > 0:
                if reward > 0:
                    successes += 1
                break

    env.close()
    return successes / n_episodes


# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--data_dir", type=str, default=str(LIBERO_DATA))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated task IDs or 'all'")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    log.info("=" * 60)
    log.info("LIBERO-10 Downstream Evaluation")
    log.info("=" * 60)

    # Transform for all encoders (resize to 224x224)
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
    assert len(hdf5_files) >= len(task_ids), f"Need {len(task_ids)} HDF5 files, found {len(hdf5_files)}"

    # Map task_id to hdf5 files by task name matching
    task_to_hdf5 = {}
    for tid in task_ids:
        task = suite.get_task(tid)
        task_name = task.name
        # Find matching HDF5
        for hf in hdf5_files:
            if task_name in hf.stem or hf.stem.replace("_demo", "") in task_name:
                task_to_hdf5[tid] = hf
                break
        if tid not in task_to_hdf5:
            # Fallback: use index
            if tid < len(hdf5_files):
                task_to_hdf5[tid] = hdf5_files[tid]

    results = {}

    for backbone_name, encoder in backbones.items():
        log.info(f"\n{'='*60}")
        log.info(f"Backbone: {backbone_name}")
        log.info(f"{'='*60}")

        obs_dim = getattr(encoder, "output_dim", 768)
        task_results = {}

        for tid in task_ids:
            task = suite.get_task(tid)
            log.info(f"\n  Task {tid}: {task.language}")

            if tid not in task_to_hdf5:
                log.warning(f"  No HDF5 for task {tid}, skipping")
                continue

            # Load demo data
            dataset = LIBERODemoDataset(str(task_to_hdf5[tid]), transform=transform)

            if len(dataset) == 0:
                log.warning(f"  Empty dataset for task {tid}")
                continue

            # Extract features from all demo images
            log.info(f"  Extracting features with {backbone_name}...")
            all_images = torch.stack([dataset[i][0] for i in range(len(dataset))])
            all_actions = torch.stack([dataset[i][1] for i in range(len(dataset))])
            features = extract_features_batch(encoder, all_images, device)
            log.info(f"  Features shape: {features.shape}")

            seed_successes = []
            for seed in range(args.n_seeds):
                log.info(f"  Seed {seed}: training BC ({args.n_epochs} epochs)...")
                policy = train_bc(features, all_actions, obs_dim, device,
                                  epochs=args.n_epochs, seed=seed)

                log.info(f"  Seed {seed}: evaluating ({args.n_episodes} episodes)...")
                sr = evaluate_policy_in_sim(
                    encoder, policy, tid, suite, transform,
                    device, n_episodes=args.n_episodes
                )
                seed_successes.append(sr)
                log.info(f"  Seed {seed}: success rate = {sr:.1%}")

            mean_sr = np.mean(seed_successes)
            std_sr = np.std(seed_successes)
            task_results[task.language] = {
                "mean": float(mean_sr),
                "std": float(std_sr),
                "seeds": [float(s) for s in seed_successes],
            }
            log.info(f"  → {backbone_name} Task {tid}: {mean_sr:.1%} ± {std_sr:.1%}")

        # Aggregate
        if task_results:
            all_means = [v["mean"] for v in task_results.values()]
            results[backbone_name] = {
                "per_task": task_results,
                "avg_success_rate": float(np.mean(all_means)),
                "std_success_rate": float(np.std(all_means)),
                "feature_dim": obs_dim,
            }
            log.info(f"\n  {backbone_name} overall: {np.mean(all_means):.1%} avg success rate")

    # Save results
    output_file = os.path.join(args.output, "libero10_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {output_file}")

    # Print summary table
    log.info("\n" + "=" * 60)
    log.info("LIBERO-10 Results Summary")
    log.info("=" * 60)
    log.info(f"{'Backbone':<20} {'Avg SR':>10} {'Std':>10}")
    log.info("-" * 40)
    for name, res in sorted(results.items(), key=lambda x: -x[1]["avg_success_rate"]):
        log.info(f"{name:<20} {res['avg_success_rate']:>10.1%} {res['std_success_rate']:>10.1%}")


if __name__ == "__main__":
    main()
