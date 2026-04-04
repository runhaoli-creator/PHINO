#!/usr/bin/env python
"""
DynaCLIP Ablation Evaluation.

Loads trained ablation checkpoints and runs linear probing to compare.
Produces a single JSON + formatted table comparing all variants.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.data.dataset import PhysicsProbeDataset, get_eval_transform
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/eval_ablations.log"),
    ],
)
logger = logging.getLogger(__name__)


ABLATION_VARIANTS = {
    "DynaCLIP (full)": {
        "checkpoint_dir": "checkpoints/pretrain",
        "freeze_backbone": False,
    },
    "Frozen backbone": {
        "checkpoint_dir": "checkpoints/ablation_frozen",
        "freeze_backbone": True,
    },
    "No hard mining": {
        "checkpoint_dir": "checkpoints/ablation_nohardneg",
        "freeze_backbone": False,
    },
    "Random physics": {
        "checkpoint_dir": "checkpoints/ablation_random_physics",
        "freeze_backbone": False,
    },
    "Standard InfoNCE": {
        "checkpoint_dir": "checkpoints/ablation_infonce",
        "freeze_backbone": False,
    },
    "Triplet loss": {
        "checkpoint_dir": "checkpoints/ablation_triplet",
        "freeze_backbone": False,
    },
    "BYOL loss": {
        "checkpoint_dir": "checkpoints/ablation_byol",
        "freeze_backbone": False,
    },
    "Last-2 blocks": {
        "checkpoint_dir": "checkpoints/ablation_last2_blocks",
        "freeze_backbone": False,
        "unfreeze_last_n_blocks": 2,
    },
    "Last-4 blocks": {
        "checkpoint_dir": "checkpoints/ablation_last4_blocks",
        "freeze_backbone": False,
        "unfreeze_last_n_blocks": 4,
    },
}


def find_best_checkpoint(ckpt_dir: str) -> str:
    """Find the best checkpoint - prefer dynaclip_final.pt, then highest step."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    final = ckpt_dir / "dynaclip_final.pt"
    if final.exists():
        return str(final)
    ckpts = sorted(ckpt_dir.glob("dynaclip_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    if not ckpts:
        return None
    return str(ckpts[-1])


def load_ablation_model(variant_cfg: dict, device: str) -> torch.nn.Module:
    """Load a DynaCLIP variant from checkpoint."""
    ckpt_path = find_best_checkpoint(variant_cfg["checkpoint_dir"])
    if ckpt_path is None:
        return None

    model = DynaCLIPModel(
        backbone_name="dinov2_vitb14",
        embed_dim=512,
        freeze_backbone=variant_cfg["freeze_backbone"],
        unfreeze_last_n_blocks=variant_cfg.get("unfreeze_last_n_blocks", -1),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded from {ckpt_path}")
    return model


@torch.no_grad()
def extract_features(model, loader, device):
    """Extract backbone features + labels from a loader."""
    model = model.to(device)
    all_feats, all_mass, all_fric, all_rest, all_cats = [], [], [], [], []
    for batch in loader:
        img = batch["image"].to(device)
        feats = model(img, return_features=True)  # backbone features (1536-d)
        all_feats.append(feats.cpu())
        all_mass.append(batch["mass"])
        all_fric.append(batch["static_friction"])
        all_rest.append(batch["restitution"])
        all_cats.extend(batch["category"])
    model.cpu()
    return {
        "features": torch.cat(all_feats),
        "mass": torch.cat(all_mass).numpy(),
        "friction": torch.cat(all_fric).numpy(),
        "restitution": torch.cat(all_rest).numpy(),
        "categories": all_cats,
    }


def linear_probe(train_data, test_data, property_name, num_seeds=5, num_epochs=100, lr=1e-3):
    """Run linear probing for a single property."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import r2_score, accuracy_score
    from sklearn.preprocessing import StandardScaler

    X_train = train_data["features"].numpy()
    X_test = test_data["features"].numpy()

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if property_name == "category":
        # Classification
        from dynaclip.data.generation import get_material_for_category
        y_train = [get_material_for_category(c) for c in train_data["categories"]]
        y_test = [get_material_for_category(c) for c in test_data["categories"]]

        scores = []
        for seed in range(num_seeds):
            clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
    else:
        # Regression
        y_train = train_data[property_name]
        y_test = test_data[property_name]

        scores = []
        for seed in range(num_seeds):
            alpha = 10 ** (seed - 2)  # Vary regularization slightly
            reg = Ridge(alpha=alpha)
            reg.fit(X_train, y_train)
            pred = reg.predict(X_test)
            scores.append(r2_score(y_test, pred))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci95": float(1.96 * np.std(scores) / np.sqrt(len(scores))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_cache/dynaclip_data")
    parser.add_argument("--output_dir", default="results/ablations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    transform = get_eval_transform(224)
    train_data = PhysicsProbeDataset(args.data_dir, split="train", transform=transform)
    test_data = PhysicsProbeDataset(args.data_dir, split="test", transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    results = {}

    for variant_name, variant_cfg in ABLATION_VARIANTS.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Evaluating: {variant_name}")
        logger.info(f"{'='*40}")

        model = load_ablation_model(variant_cfg, device)
        if model is None:
            logger.warning(f"  No checkpoint found for '{variant_name}', skipping")
            continue

        # Extract features
        train_feats = extract_features(model, train_loader, device)
        test_feats = extract_features(model, test_loader, device)

        # Linear probing
        variant_results = {}
        for prop in ["mass", "friction", "restitution", "category"]:
            variant_results[prop] = linear_probe(train_feats, test_feats, prop)
            logger.info(f"  {prop}: {variant_results[prop]['mean']:.4f} ± {variant_results[prop]['std']:.4f}")

        results[variant_name] = variant_results

    # Save results
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print formatted table
    print("\n" + "=" * 90)
    print("ABLATION RESULTS")
    print("=" * 90)
    print(f"{'Variant':<25} {'Mass R²':<14} {'Friction R²':<14} {'Restitution R²':<16} {'Material Acc':<14}")
    print("-" * 90)
    for vname, vres in results.items():
        m = vres.get("mass", {}).get("mean", 0)
        f = vres.get("friction", {}).get("mean", 0)
        r = vres.get("restitution", {}).get("mean", 0)
        c = vres.get("category", {}).get("mean", 0)
        m_ci = vres.get("mass", {}).get("ci95", 0)
        f_ci = vres.get("friction", {}).get("ci95", 0)
        r_ci = vres.get("restitution", {}).get("ci95", 0)
        c_ci = vres.get("category", {}).get("ci95", 0)
        print(f"{vname:<25} {m:.4f}±{m_ci:.3f}  {f:.4f}±{f_ci:.3f}  {r:.4f}±{r_ci:.3f}    {c:.4f}±{c_ci:.3f}")

    logger.info(f"\nResults saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
