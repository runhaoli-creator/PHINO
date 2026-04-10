#!/usr/bin/env python3
"""
t-SNE visualization: PHINO vs DINOv2, colored by continuous mass.

Generates a 2-panel figure:
  Left:  DINOv2-B/14 (frozen) embeddings colored by mass → random colors (no physics)
  Right: PHINO embeddings colored by mass → smooth gradient (physics encoded)

Usage:
  python scripts/plot_tsne_mass.py \
      --checkpoint checkpoints/phino/dynaclip_final.pt \
      --data_dir data_cache/phino_data \
      --output figures/tsne_mass_comparison.pdf \
      --max_samples 3000
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_phino(checkpoint_path, device):
    """Load PHINO encoder (fine-tuned DINOv2)."""
    from dynaclip.models.dynaclip import DynaCLIPEncoder
    encoder = DynaCLIPEncoder(checkpoint_path=checkpoint_path, feature_type="cls")
    encoder = encoder.to(device).eval()
    return encoder


def load_dinov2(device):
    """Load frozen DINOv2-B/14."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    def extract(images):
        out = model.forward_features(images)
        if isinstance(out, dict):
            return out.get("x_norm_clstoken", out["x"][:, 0])
        return out[:, 0]

    return extract


def extract_features(model_fn, loader, device, max_samples=3000):
    """Extract features and mass values."""
    feats, masses = [], []
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= max_samples:
                break
            img = batch["image"].to(device)
            f = model_fn(img)
            if isinstance(f, torch.Tensor) and f.dim() == 3:
                f = f[:, 0]
            feats.append(f.cpu())
            masses.append(batch["mass"])
            count += len(img)

    feats = torch.cat(feats)[:max_samples]
    masses = torch.cat(masses)[:max_samples]
    return feats.numpy(), masses.numpy()


def plot_tsne_panel(ax, embeddings_2d, masses, title, show_colorbar=False):
    """Plot a single t-SNE panel colored by mass."""
    norm = Normalize(vmin=masses.min(), vmax=masses.max())
    cmap = cm.coolwarm

    scatter = ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=masses,
        cmap=cmap,
        norm=norm,
        s=8,
        alpha=0.6,
        edgecolors='none',
        rasterized=True,
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if show_colorbar:
        return scatter
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="PHINO checkpoint path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to PHINO-Bench data")
    parser.add_argument("--output", type=str, default="figures/tsne_mass_comparison.pdf")
    parser.add_argument("--max_samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load data
    from dynaclip.data.dataset import PhysicsProbeDataset
    dataset = PhysicsProbeDataset(data_dir=args.data_dir, split="test", seed=42)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=4
    )

    print("Loading DINOv2-B/14...")
    dinov2_fn = load_dinov2(device)
    print("Extracting DINOv2 features...")
    dinov2_feats, masses = extract_features(dinov2_fn, loader, device, args.max_samples)

    print(f"Loading PHINO from {args.checkpoint}...")
    phino_model = load_phino(args.checkpoint, device)
    print("Extracting PHINO features...")
    phino_feats, _ = extract_features(phino_model, loader, device, args.max_samples)

    # Run t-SNE with same seed for both
    print("Running t-SNE for DINOv2...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, n_iter=1000)
    dinov2_2d = tsne.fit_transform(dinov2_feats)

    print("Running t-SNE for PHINO...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=args.seed, n_iter=1000)
    phino_2d = tsne.fit_transform(phino_feats)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_tsne_panel(ax1, dinov2_2d, masses, "DINOv2-B/14 (Frozen)")
    scatter = plot_tsne_panel(ax2, phino_2d, masses, "PHINO (Ours)", show_colorbar=True)

    # Shared colorbar
    cbar = fig.colorbar(scatter, ax=[ax1, ax2], location='bottom', shrink=0.6,
                        aspect=30, pad=0.08)
    cbar.set_label("Mass (kg)", fontsize=12)

    plt.suptitle("t-SNE Embeddings Colored by Object Mass", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved to {args.output}")

    # Also save PNG
    png_path = args.output.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {png_path}")


if __name__ == "__main__":
    main()
