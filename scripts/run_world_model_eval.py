#!/usr/bin/env python
"""
World Model Evaluation: Train a small RSSM (Recurrent State-Space Model) on
frozen backbone features and measure next-state prediction quality.

This tests whether DynaCLIP features capture dynamics-relevant information
by measuring how well an RSSM can predict the next physics state from
current embeddings.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_world_model_eval.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, ".")

from dynaclip.models.dynaclip import DynaCLIPModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/world_model_eval.log"),
    ],
)
logger = logging.getLogger(__name__)


# ======================================================================
# Backbone wrappers
# ======================================================================
class BackboneFeatureExtractor(nn.Module):
    def __init__(self, backbone, name):
        super().__init__()
        self.backbone = backbone
        self.name = name

    @torch.no_grad()
    def forward(self, x):
        out = self.backbone(x)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]
        if isinstance(out, dict):
            return out.get("x_norm_clstoken", list(out.values())[0])
        if isinstance(out, torch.Tensor):
            return out[:, 0] if out.dim() == 3 else out
        if isinstance(out, (tuple, list)):
            return out[0] if out[0].dim() == 2 else out[0][:, 0]
        return out


class DynaCLIPFeatureExtractor(nn.Module):
    def __init__(self, model, use_projection=True):
        super().__init__()
        self.model = model
        self.use_projection = use_projection

    @torch.no_grad()
    def forward(self, x):
        return self.model(x, return_features=not self.use_projection)


# ======================================================================
# Simple RSSM World Model
# ======================================================================
class SimpleRSSM(nn.Module):
    """
    Simplified RSSM that:
      - Takes: current visual embedding + current physics state
      - Predicts: next physics state (mass, friction, restitution)

    Architecture: GRU-based recurrent model with physics prediction head.
    """
    def __init__(self, obs_dim, state_dim=128, hidden_dim=256, physics_dim=3):
        super().__init__()
        self.state_dim = state_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim + physics_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU for temporal dynamics
        self.gru = nn.GRUCell(hidden_dim, state_dim)

        # Physics prediction head
        self.physics_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, physics_dim),
        )

        # State prediction (for latent consistency)
        self.state_prior = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, obs_embed, physics_current, hidden_state=None):
        """
        obs_embed: (B, obs_dim) - visual features
        physics_current: (B, 3) - current [mass, friction, restitution]
        hidden_state: (B, state_dim) or None
        """
        B = obs_embed.shape[0]
        if hidden_state is None:
            hidden_state = torch.zeros(B, self.state_dim, device=obs_embed.device)

        # Encode observation + physics
        x = torch.cat([obs_embed, physics_current], dim=-1)
        encoded = self.obs_encoder(x)

        # Update hidden state
        new_hidden = self.gru(encoded, hidden_state)

        # Predict physics
        physics_pred = self.physics_head(new_hidden)

        return physics_pred, new_hidden


# ======================================================================
# Training dataset: create synthetic "trajectories" from metadata
# ======================================================================
class PhysicsTrajectoryDataset(Dataset):
    """
    Creates synthetic physics trajectories from the metadata.
    
    For each "trajectory", we sample entries from the same image_group
    (same base image, different physics). The task is to predict the
    physics of the next sample given current embedding + physics.
    """
    def __init__(self, meta, transform, traj_len=5):
        self.transform = transform
        self.traj_len = traj_len

        # Group by image_group
        groups = defaultdict(list)
        for e in meta:
            groups[e["image_group"]].append(e)

        # Only keep groups with enough entries
        self.groups = [v for v in groups.values() if len(v) >= traj_len]
        logger.info(f"Created {len(self.groups)} trajectory groups (len>={traj_len})")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]
        # Sample traj_len entries (or use all if exactly right)
        if len(group) > self.traj_len:
            indices = np.random.choice(len(group), self.traj_len, replace=False)
            entries = [group[i] for i in sorted(indices)]
        else:
            entries = group[:self.traj_len]

        images = []
        physics = []
        for e in entries:
            try:
                img = Image.open(e["image_path"]).convert("RGB")
                img_t = self.transform(img)
            except Exception:
                img_t = torch.zeros(3, 224, 224)
            images.append(img_t)
            physics.append([e["mass"], e["static_friction"], e["restitution"]])

        return {
            "images": torch.stack(images),  # (T, 3, H, W)
            "physics": torch.tensor(physics, dtype=torch.float32),  # (T, 3)
        }


def train_world_model(backbone, train_loader, val_loader, device,
                       obs_dim, epochs=20, lr=1e-3):
    """Train RSSM on frozen backbone features and return validation metrics."""
    model = SimpleRSSM(obs_dim=obs_dim, state_dim=128, hidden_dim=256, physics_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    backbone.eval().to(device)

    best_val_loss = float('inf')
    best_metrics = {}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            images = batch["images"]  # (B, T, 3, H, W)
            physics = batch["physics"].to(device)  # (B, T, 3)
            B, T = images.shape[:2]

            # Encode all images
            with torch.no_grad():
                imgs_flat = images.view(-1, 3, 224, 224).to(device)
                feats_flat = backbone(imgs_flat)
                feats = feats_flat.view(B, T, -1)

            # Forward through RSSM: predict next physics from current
            total_loss = 0
            hidden = None
            for t in range(T - 1):
                physics_pred, hidden = model(feats[:, t], physics[:, t], hidden)
                loss = F.mse_loss(physics_pred, physics[:, t + 1])
                total_loss += loss

            total_loss = total_loss / (T - 1)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_preds = {"mass": [], "friction": [], "restitution": []}
        val_targets = {"mass": [], "friction": [], "restitution": []}
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"]
                physics = batch["physics"].to(device)
                B, T = images.shape[:2]

                imgs_flat = images.view(-1, 3, 224, 224).to(device)
                feats_flat = backbone(imgs_flat)
                feats = feats_flat.view(B, T, -1)

                hidden = None
                for t in range(T - 1):
                    physics_pred, hidden = model(feats[:, t], physics[:, t], hidden)
                    loss = F.mse_loss(physics_pred, physics[:, t + 1])
                    val_losses.append(loss.item())

                    val_preds["mass"].append(physics_pred[:, 0].cpu().numpy())
                    val_preds["friction"].append(physics_pred[:, 1].cpu().numpy())
                    val_preds["restitution"].append(physics_pred[:, 2].cpu().numpy())
                    val_targets["mass"].append(physics[:, t + 1, 0].cpu().numpy())
                    val_targets["friction"].append(physics[:, t + 1, 1].cpu().numpy())
                    val_targets["restitution"].append(physics[:, t + 1, 2].cpu().numpy())

        val_loss = np.mean(val_losses)
        from sklearn.metrics import r2_score

        metrics = {"val_loss": float(val_loss)}
        for prop in ["mass", "friction", "restitution"]:
            preds = np.concatenate(val_preds[prop])
            targets = np.concatenate(val_targets[prop])
            r2 = r2_score(targets, preds)
            metrics[f"{prop}_r2"] = float(r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics.copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs}: train_loss={np.mean(train_losses):.4f}, "
                         f"val_loss={val_loss:.4f}, mass_r2={metrics['mass_r2']:.4f}, "
                         f"fric_r2={metrics['friction_r2']:.4f}, rest_r2={metrics['restitution_r2']:.4f}")

    backbone.cpu()
    return best_metrics


def load_all_backbones(checkpoint_path, device):
    import torchvision.models as tvm
    backbones = {}

    if checkpoint_path and Path(checkpoint_path).exists():
        model = DynaCLIPModel(backbone_name="dinov2_vitb14", embed_dim=512, freeze_backbone=False)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        backbones["DynaCLIP"] = (DynaCLIPFeatureExtractor(model, True), 512)

    try:
        dinov2_b = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        dinov2_b.eval()
        backbones["DINOv2-B/14"] = (BackboneFeatureExtractor(dinov2_b, "DINOv2-B/14"), 768)
    except Exception as e:
        logger.warning(f"DINOv2-B/14: {e}")

    try:
        dinov2_l = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        dinov2_l.eval()
        backbones["DINOv2-L/14"] = (BackboneFeatureExtractor(dinov2_l, "DINOv2-L/14"), 1024)
    except Exception as e:
        logger.warning(f"DINOv2-L/14: {e}")

    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_visual = clip_model.visual
        clip_visual.eval()
        backbones["CLIP-L/14"] = (BackboneFeatureExtractor(clip_visual, "CLIP-L/14"), 768)
    except Exception as e:
        logger.warning(f"CLIP-L/14: {e}")

    try:
        class R3MStyleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
                self.features = nn.Sequential(*list(resnet.children())[:-1])
            def forward(self, x):
                return self.features(x).flatten(1)
        r3m = R3MStyleEncoder()
        r3m.eval()
        backbones["R3M"] = (BackboneFeatureExtractor(r3m, "R3M"), 2048)
    except Exception as e:
        logger.warning(f"R3M: {e}")

    return backbones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pretrain/dynaclip_final.pt")
    parser.add_argument("--data_dir", default="data_cache/dynaclip_data")
    parser.add_argument("--output_dir", default="results/world_model")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    with open(Path(args.data_dir) / "metadata.json") as f:
        all_meta = json.load(f)

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create trajectory datasets
    rng_state = np.random.RandomState(42)
    np.random.seed(42)

    full_dataset = PhysicsTrajectoryDataset(all_meta, transform, traj_len=5)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train: {n_train} trajectories, Val: {n_val} trajectories")

    # Load backbones
    backbones = load_all_backbones(args.checkpoint, device)
    logger.info(f"Loaded {len(backbones)} backbones")

    results = {}

    for bb_name, (bb, obs_dim) in backbones.items():
        logger.info(f"Training world model for {bb_name} (obs_dim={obs_dim})...")
        t0 = time.time()

        metrics = train_world_model(
            bb, train_loader, val_loader, device,
            obs_dim=obs_dim, epochs=args.epochs, lr=1e-3,
        )

        results[bb_name] = metrics
        elapsed = time.time() - t0
        logger.info(f"  {bb_name} done in {elapsed:.0f}s: "
                     f"mass_r2={metrics['mass_r2']:.4f}, "
                     f"fric_r2={metrics['friction_r2']:.4f}, "
                     f"rest_r2={metrics['restitution_r2']:.4f}")

        torch.cuda.empty_cache()

    # Save results
    out_path = output_dir / "world_model_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("  WORLD MODEL EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Backbone':<15} {'Mass R²':>10} {'Friction R²':>12} {'Rest. R²':>10} {'Val Loss':>10}")
    print("-" * 60)
    for bb_name, m in sorted(results.items(), key=lambda x: x[1].get("mass_r2", 0)):
        print(f"{bb_name:<15} {m['mass_r2']:>10.4f} {m['friction_r2']:>12.4f} "
              f"{m['restitution_r2']:>10.4f} {m['val_loss']:>10.4f}")

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
