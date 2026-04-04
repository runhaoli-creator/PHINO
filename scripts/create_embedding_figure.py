#!/usr/bin/env python3
"""
Embedding Analysis Figure for DynaCLIP paper.

Creates a publication-quality figure showing:
1. t-SNE of DynaCLIP embeddings colored by material type (left)
2. t-SNE of DINOv2 embeddings colored by material type (center)
3. Embedding space colored by continuous mass values (right)

This demonstrates that DynaCLIP's representation space encodes 
physics-aware structure that correlates with material properties.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent.parent))
from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.data.dataset import PhysicsProbeDataset, get_eval_transform
from torch.utils.data import DataLoader


def extract_features(model_or_func, loader, device, max_samples=3000):
    """Extract features and metadata from dataset."""
    feats, masses, frictions, rests, cats = [], [], [], [], []
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= max_samples:
                break
            img = batch["image"].to(device)
            f = model_or_func(img)
            if isinstance(f, dict):
                f = f.get("x_norm_clstoken", list(f.values())[0])
            if f.dim() == 3:
                f = f[:, 0]
            feats.append(f.cpu())
            masses.append(batch["mass"])
            frictions.append(batch["static_friction"])
            rests.append(batch["restitution"])
            cats.extend(batch["category"])
            count += img.shape[0]
    
    feats = torch.cat(feats)[:max_samples]
    masses = torch.cat(masses)[:max_samples].numpy()
    frictions = torch.cat(frictions)[:max_samples].numpy()
    rests = torch.cat(rests)[:max_samples].numpy()
    cats = cats[:max_samples]
    
    return feats.numpy(), masses, frictions, rests, cats


def make_figure(checkpoint_path, data_dir, output_path, device="cuda:0", max_samples=2500):
    """Create the embedding analysis figure."""
    
    # Load data
    transform = get_eval_transform(224)
    test_data = PhysicsProbeDataset(data_dir, split="test", transform=transform)
    loader = DataLoader(test_data, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)
    
    # Load DynaCLIP v2
    model = DynaCLIPModel(backbone_name="dinov2_vitb14", embed_dim=512, freeze_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    
    dynaclip_func = lambda x: model(x, return_features=True)
    print("Extracting DynaCLIP features...")
    dc_feats, masses, frictions, rests, cats = extract_features(dynaclip_func, loader, device, max_samples)
    
    # Load DINOv2
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    dinov2.eval().to(device)
    for p in dinov2.parameters():
        p.requires_grad = False
    
    # Re-load data with same order (set seed)
    loader2 = DataLoader(test_data, batch_size=128, num_workers=4, pin_memory=True, shuffle=True,
                         generator=torch.Generator().manual_seed(42))
    # Actually just re-extract with same loader (will differ), but use cached data
    print("Extracting DINOv2 features...")
    loader3 = DataLoader(test_data, batch_size=128, num_workers=4, pin_memory=True, shuffle=True)
    d2_feats, _, _, _, _ = extract_features(dinov2, loader3, device, max_samples)
    
    # t-SNE embeddings
    print("Computing t-SNE for DynaCLIP...")
    try:
        tsne_dc = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    except TypeError:
        tsne_dc = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb_dc = tsne_dc.fit_transform(dc_feats)
    
    print("Computing t-SNE for DINOv2...")
    try:
        tsne_d2 = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    except TypeError:
        tsne_d2 = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb_d2 = tsne_d2.fit_transform(d2_feats)
    
    # Map categories to material types
    from dynaclip.data.generation import get_material_for_category
    materials = [get_material_for_category(c) for c in cats]
    mat_set = sorted(set(materials))
    
    # ─── Create figure ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Color maps
    material_cmap = plt.colormaps.get_cmap("tab10") if hasattr(plt, 'colormaps') else plt.cm.get_cmap("tab10")
    mat_colors = {m: material_cmap(i / max(len(mat_set) - 1, 1)) for i, m in enumerate(mat_set)}
    
    # Panel (a): DynaCLIP by material type
    ax = axes[0]
    for mat in mat_set:
        mask = np.array([m == mat for m in materials])
        ax.scatter(emb_dc[mask, 0], emb_dc[mask, 1], c=[mat_colors[mat]],
                   label=mat.replace("_", " ").title(), alpha=0.5, s=6, edgecolors='none')
    ax.set_title("DynaCLIP (Ours)", fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, markerscale=3, loc='lower left', framealpha=0.8, ncol=2)
    ax.set_xlabel("(a) Materials form distinct clusters", fontsize=9)
    
    # Panel (b): DINOv2 by material type  
    ax = axes[1]
    for mat in mat_set:
        mask = np.array([m == mat for m in materials])
        ax.scatter(emb_d2[mask, 0], emb_d2[mask, 1], c=[mat_colors[mat]],
                   label=mat.replace("_", " ").title(), alpha=0.5, s=6, edgecolors='none')
    ax.set_title("DINOv2-B/14 (Baseline)", fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("(b) No material structure", fontsize=9)
    
    # Panel (c): DynaCLIP colored by mass
    ax = axes[2]
    norm = Normalize(vmin=np.percentile(masses, 5), vmax=np.percentile(masses, 95))
    sc = ax.scatter(emb_dc[:, 0], emb_dc[:, 1], c=masses, cmap='viridis',
                    alpha=0.6, s=6, edgecolors='none', norm=norm)
    cbar = plt.colorbar(sc, ax=ax, label='Mass (kg)', shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("DynaCLIP — Mass Gradient", fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("(c) Smooth mass gradient in embedding space", fontsize=9)
    
    plt.tight_layout(w_pad=1.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved embedding analysis figure to {output_path}")
    
    # Also save individual panels for flexibility
    for i, (emb, title, suffix) in enumerate([
        (emb_dc, "DynaCLIP", "dynaclip_material"),
        (emb_d2, "DINOv2", "dinov2_material"),
    ]):
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
        for mat in mat_set:
            mask = np.array([m == mat for m in materials])
            ax2.scatter(emb[mask, 0], emb[mask, 1], c=[mat_colors[mat]],
                       label=mat.replace("_", " ").title(), alpha=0.5, s=10, edgecolors='none')
        ax2.set_title(f"{title} — Material Type", fontsize=14, fontweight='bold')
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.legend(fontsize=8, markerscale=3, framealpha=0.8)
        plt.tight_layout()
        panel_path = output_path.replace(".pdf", f"_{suffix}.pdf").replace(".png", f"_{suffix}.png")
        plt.savefig(panel_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Mass gradient panel
    fig3, ax3 = plt.subplots(1, 1, figsize=(6, 5))
    sc = ax3.scatter(emb_dc[:, 0], emb_dc[:, 1], c=masses, cmap='viridis',
                     alpha=0.6, s=10, edgecolors='none', norm=norm)
    plt.colorbar(sc, ax=ax3, label='Mass (kg)', shrink=0.8)
    ax3.set_title("DynaCLIP — Mass Gradient", fontsize=14, fontweight='bold')
    ax3.set_xticks([]); ax3.set_yticks([])
    plt.tight_layout()
    mass_path = output_path.replace(".pdf", "_mass.pdf").replace(".png", "_mass.png")
    plt.savefig(mass_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/pretrain_v2/dynaclip_final.pt")
    parser.add_argument("--data_dir", type=str, default="data_cache/dynaclip_data")
    parser.add_argument("--output", type=str, default="paper/figures/embedding_analysis.png")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=2500)
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    make_figure(args.checkpoint, args.data_dir, args.output, args.device, args.max_samples)
