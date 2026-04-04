#!/usr/bin/env python3
"""
Language-Grounded Physics Queries for DynaCLIP.

This experiment evaluates whether physics-aware embeddings can answer
natural language queries about physical properties:

1. "Find the heaviest object" — rank images by predicted mass
2. "Find the most slippery surface" — rank by inverse friction
3. "Find the bounciest object" — rank by predicted restitution
4. "Which objects feel similar?" — clustering by physics similarity

Protocol:
  - Extract features from all test images using each backbone
  - Train a linear probe on physics properties
  - Use the probe to answer language queries (retrieval ranking)
  - Evaluate: Recall@K for top-k heaviest/lightest/bounciest/slipperiest
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.data.dataset import PhysicsProbeDataset, get_eval_transform
from torch.utils.data import DataLoader


def load_backbones(checkpoint_path, device):
    """Load DynaCLIP + baselines."""
    backbones = {}
    
    # DynaCLIP v2
    model = DynaCLIPModel(backbone_name="dinov2_vitb14", embed_dim=512, freeze_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    
    class DCWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        @torch.no_grad()
        def forward(self, x):
            return self.m(x, return_features=True)
    
    backbones["DynaCLIP"] = DCWrapper(model)
    
    # DINOv2
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    dinov2.eval().to(device)
    for p in dinov2.parameters():
        p.requires_grad = False
    backbones["DINOv2"] = dinov2
    
    # CLIP
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
            @torch.no_grad()
            def forward(self, x):
                return self.m.encode_image(x)
        
        backbones["CLIP"] = CLIPWrapper(clip_model)
    except Exception as e:
        print(f"CLIP load failed: {e}")
    
    # SigLIP
    try:
        from transformers import SiglipVisionModel
        siglip = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        siglip.eval().to(device)
        for p in siglip.parameters():
            p.requires_grad = False
        
        class SigLIPWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            @torch.no_grad()
            def forward(self, x):
                return self.m(x).pooler_output
        
        backbones["SigLIP"] = SigLIPWrapper(siglip)
    except Exception as e:
        print(f"SigLIP load failed: {e}")
    
    return backbones


@torch.no_grad()
def extract_all_features(backbone, loader, device, max_samples=5000):
    """Extract features and properties."""
    feats, masses, frictions, rests, cats = [], [], [], [], []
    count = 0
    for batch in loader:
        if count >= max_samples:
            break
        img = batch["image"].to(device)
        f = backbone(img)
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
    
    return (torch.cat(feats).numpy(), 
            torch.cat(masses).numpy(),
            torch.cat(frictions).numpy(),
            torch.cat(rests).numpy(),
            cats)


def evaluate_physics_queries(feats_train, props_train, feats_test, props_test, prop_name):
    """Evaluate language-grounded physics queries for a property."""
    
    # Train a linear probe to predict the property from features
    probe = Ridge(alpha=1.0)
    probe.fit(feats_train, props_train)
    pred = probe.predict(feats_test)
    r2 = r2_score(props_test, pred)
    
    # Query 1: "Find the heaviest/most-X/bounciest" → rank by predicted property
    # Ground truth ranking
    true_rank = np.argsort(-props_test)  # descending
    pred_rank = np.argsort(-pred)
    
    results = {"r2": float(r2)}
    
    # Recall@K: what fraction of the true top-K are in the predicted top-K?
    n = len(props_test)
    for k_frac in [0.01, 0.05, 0.10, 0.20]:
        k = max(1, int(k_frac * n))
        true_topk = set(true_rank[:k])
        pred_topk = set(pred_rank[:k])
        recall = len(true_topk & pred_topk) / k
        results[f"recall@{int(k_frac*100)}%"] = float(recall)
    
    # NDCG@K
    for k_frac in [0.05, 0.10]:
        k = max(1, int(k_frac * n))
        # Relevance = actual property value (higher = more relevant for "heaviest" query)
        relevance = props_test[pred_rank[:k]]
        ideal_relevance = props_test[true_rank[:k]]
        
        # DCG
        discounts = 1.0 / np.log2(np.arange(k) + 2)
        dcg = np.sum(relevance * discounts)
        idcg = np.sum(ideal_relevance * discounts)
        ndcg = dcg / max(idcg, 1e-10)
        results[f"ndcg@{int(k_frac*100)}%"] = float(ndcg)
    
    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(props_test, pred)
    results["spearman_rho"] = float(rho)
    results["spearman_pval"] = float(pval)
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, 
                        default="checkpoints/pretrain_v2/dynaclip_final.pt")
    parser.add_argument("--data_dir", type=str, default="data_cache/dynaclip_data")
    parser.add_argument("--output", type=str, default="results/physics_queries")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    
    device = args.device
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Load data
    transform = get_eval_transform(224)
    train_data = PhysicsProbeDataset(args.data_dir, split="train", transform=transform)
    test_data = PhysicsProbeDataset(args.data_dir, split="test", transform=transform)
    train_loader = DataLoader(train_data, batch_size=256, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, num_workers=4, pin_memory=True)
    
    # Load backbones
    backbones = load_backbones(args.checkpoint, device)
    print(f"Loaded {len(backbones)} backbones: {list(backbones.keys())}")
    
    # Queries map to properties
    queries = {
        "Find the heaviest object": ("mass", "higher"),
        "Find the lightest object": ("mass", "lower"),
        "Find the most slippery surface": ("static_friction", "lower"),
        "Find the roughest surface": ("static_friction", "higher"),
        "Find the bounciest object": ("restitution", "higher"),
        "Find the least bouncy object": ("restitution", "lower"),
    }
    
    all_results = {}
    
    for bname, backbone in backbones.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {bname}")
        print(f"{'='*50}")
        
        backbone = backbone.to(device)
        
        # Extract features
        print("  Extracting train features...")
        tr_feats, tr_mass, tr_fric, tr_rest, _ = extract_all_features(backbone, train_loader, device)
        print("  Extracting test features...")
        te_feats, te_mass, te_fric, te_rest, te_cats = extract_all_features(backbone, test_loader, device)
        
        backbone.cpu()
        
        bk_results = {}
        props_map = {
            "mass": (tr_mass, te_mass),
            "static_friction": (tr_fric, te_fric),
            "restitution": (tr_rest, te_rest),
        }
        
        for query, (prop_name, direction) in queries.items():
            tr_prop, te_prop = props_map[prop_name]
            if direction == "lower":
                # Invert for "find lowest" queries
                res = evaluate_physics_queries(tr_feats, -tr_prop, te_feats, -te_prop, prop_name)
            else:
                res = evaluate_physics_queries(tr_feats, tr_prop, te_feats, te_prop, prop_name)
            bk_results[query] = res
            print(f"  {query}: R²={res['r2']:.3f}, Recall@5%={res['recall@5%']:.3f}, ρ={res['spearman_rho']:.3f}")
        
        all_results[bname] = bk_results
    
    # Save
    with open(Path(args.output) / "physics_queries.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("LANGUAGE-GROUNDED PHYSICS QUERIES — SUMMARY")
    print(f"{'='*80}")
    print(f"{'Backbone':<15} {'Query':<35} {'R²':>6} {'Rec@5%':>8} {'NDCG@5%':>8} {'ρ':>6}")
    print("-" * 80)
    for bname, bres in all_results.items():
        for query, res in bres.items():
            r2 = res['r2']
            rec = res['recall@5%']
            ndcg = res.get('ndcg@5%', 0)
            rho = res['spearman_rho']
            print(f"{bname:<15} {query:<35} {r2:>6.3f} {rec:>8.3f} {ndcg:>8.3f} {rho:>6.3f}")
        print()


if __name__ == "__main__":
    main()
