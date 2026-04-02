"""
DynaCLIP Evaluation Pipeline.

Runs all feasible experiments on the trained DynaCLIP model and baseline backbones:
  1. Linear Probing (mass, friction, restitution, category)
  2. Invisible Physics Test (embedding sensitivity)
  3. Zero-Shot Physics Inference (k-NN retrieval)

Compares DynaCLIP against:
  - DINOv2-ViT-B/14 (frozen baseline, same architecture)
  - DINOv2-ViT-L/14 (larger frozen baseline)
  - SigLIP-ViT-B/16 (vision-language model)
  - CLIP-ViT-L/14 (vision-language model)

Usage:
  CUDA_VISIBLE_DEVICES=6 python scripts/evaluate.py \
    --checkpoint checkpoints/pretrain/dynaclip_step_50000.pt \
    --data_dir data_cache/dynaclip_data \
    --output_dir results/eval
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dynaclip.models.backbones import (
    load_dynaclip,
    load_dinov2_vitb14,
    load_dinov2_vitl14,
    load_siglip,
    load_clip_vitl14,
)
from dynaclip.data.dataset import create_probe_dataloader
from dynaclip.eval.linear_probing import PhysicsLinearProbing
from dynaclip.eval.invisible_physics import InvisiblePhysicsEvaluator
from dynaclip.eval.zero_shot import ZeroShotPhysicsInference, create_real_library

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_backbones(checkpoint_path: str, backbones_to_load: list, device: str) -> dict:
    """Load all requested backbones."""
    backbones = {}

    for name in backbones_to_load:
        try:
            t0 = time.time()
            if name == "dynaclip":
                bb = load_dynaclip(checkpoint_path)
            elif name == "dinov2_vitb14":
                bb = load_dinov2_vitb14()
            elif name == "dinov2_vitl14":
                bb = load_dinov2_vitl14()
            elif name == "siglip":
                bb = load_siglip()
            elif name == "clip_vitl14":
                bb = load_clip_vitl14()
            else:
                logger.warning(f"Unknown backbone: {name}, skipping")
                continue

            bb = bb.to(device)
            bb.eval()
            backbones[name] = bb
            logger.info(f"Loaded {name} ({bb.output_dim}d) in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue

    return backbones


def run_linear_probing(backbones, data_dir, device, num_seeds=5, num_epochs=100):
    """Experiment 1: Linear probing for physics property prediction."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Linear Probing")
    logger.info("=" * 60)

    train_loader = create_probe_dataloader(data_dir, split="train", batch_size=256, num_workers=4)
    val_loader = create_probe_dataloader(data_dir, split="val", batch_size=256, num_workers=4)
    test_loader = create_probe_dataloader(data_dir, split="test", batch_size=256, num_workers=4)

    evaluator = PhysicsLinearProbing(
        backbones=backbones,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_seeds=num_seeds,
        num_epochs=num_epochs,
        device=device,
        probe_category=True,
    )

    return evaluator.run()


def run_invisible_physics(backbones, data_dir, device):
    """Experiment 2: Invisible physics test."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Invisible Physics Test")
    logger.info("=" * 60)

    # Load invisible physics pairs from dataset
    # We create pairs from the same image with different physics properties
    from PIL import Image
    from torchvision import transforms as T
    from torch.utils.data import Dataset, DataLoader as DL

    meta_path = Path(data_dir) / "metadata.json"
    with open(meta_path) as f:
        all_meta = json.load(f)

    # Group by image_group (same base image, different physics)
    groups = {}
    for entry in all_meta:
        gid = entry["image_group"]
        if gid not in groups:
            groups[gid] = []
        groups[gid].append(entry)

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build pairs: same image appearance, different physics
    pairs = []
    max_pairs = 500
    for gid, entries in sorted(groups.items()):
        if len(pairs) >= max_pairs:
            break
        if len(entries) < 2:
            continue
        e_a, e_b = entries[0], entries[1]
        pairs.append((e_a, e_b))

    if not pairs:
        logger.error("No invisible physics pairs found!")
        return {}

    logger.info(f"Built {len(pairs)} invisible physics pairs")

    class InvisiblePairsDataset(Dataset):
        def __init__(self, pairs, transform):
            self.pairs = pairs
            self.transform = transform
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            e_a, e_b = self.pairs[idx]
            img = Image.open(e_a["image_path"]).convert("RGB")
            img_tensor = self.transform(img)
            mass_a = e_a["mass"]
            mass_b = e_b["mass"]
            return {
                "img_a": img_tensor,
                "img_b": img_tensor,  # Same image, different physics
                "mass_a": torch.tensor(mass_a, dtype=torch.float32),
                "mass_b": torch.tensor(mass_b, dtype=torch.float32),
                "heavier_label": torch.tensor(1 if mass_a > mass_b else 0, dtype=torch.long),
            }

    dataset = InvisiblePairsDataset(pairs, transform)
    loader = DL(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    evaluator = InvisiblePhysicsEvaluator(
        backbones=backbones,
        test_loader=loader,
        device=device,
    )

    return evaluator.run_full_evaluation()


def run_zero_shot(backbones, data_dir, device, n_lib=2000, n_query=500):
    """Experiment 5: Zero-shot physics inference via k-NN retrieval."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Zero-Shot Physics Inference")
    logger.info("=" * 60)

    lib_images, lib_props, query_images, query_props = create_real_library(
        data_dir=data_dir,
        n_lib=n_lib,
        n_query=n_query,
        seed=42,
    )

    evaluator = ZeroShotPhysicsInference(
        backbones=backbones,
        library_images=lib_images,
        library_properties=lib_props,
        query_images=query_images,
        query_properties=query_props,
        k=5,
        device=device,
    )

    return evaluator.run()


def format_results_table(results: dict, experiment_name: str) -> str:
    """Format results as a readable table."""
    lines = [f"\n{'='*70}", f"  {experiment_name} Results", f"{'='*70}"]

    if not results:
        lines.append("  No results available.")
        return "\n".join(lines)

    # Get all properties across all backbones
    all_props = set()
    for bb_results in results.values():
        all_props.update(bb_results.keys())
    all_props = sorted(all_props)

    # Header
    header = f"{'Backbone':<20}"
    for prop in all_props:
        header += f" | {prop:<20}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for bb_name, bb_results in results.items():
        row = f"{bb_name:<20}"
        for prop in all_props:
            if prop in bb_results:
                r = bb_results[prop]
                metric = r.get("metric", "r2")
                mean = r.get("mean", r.get("r2", r.get("accuracy", 0)))
                std = r.get("std", 0)
                if isinstance(mean, (int, float)):
                    row += f" | {mean:.4f} +/- {std:.4f} "
                else:
                    row += f" | {mean!s:<20}"
            else:
                row += f" | {'N/A':<20}"
        lines.append(row)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="DynaCLIP Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DynaCLIP checkpoint")
    parser.add_argument("--data_dir", type=str, default="data_cache/dynaclip_data")
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--backbones", type=str, nargs="+",
                        default=["dynaclip", "dinov2_vitb14", "siglip", "clip_vitl14"],
                        help="Backbones to evaluate")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["linear_probing", "invisible_physics", "zero_shot"],
                        help="Experiments to run")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Epochs for linear probing")
    parser.add_argument("--n_lib", type=int, default=2000,
                        help="Library size for zero-shot")
    parser.add_argument("--n_query", type=int, default=500,
                        help="Query size for zero-shot")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir / "eval.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Backbones: {args.backbones}")
    logger.info(f"Experiments: {args.experiments}")

    # Load backbones
    backbones = load_backbones(args.checkpoint, args.backbones, device)
    if not backbones:
        logger.error("No backbones loaded! Exiting.")
        return

    all_results = {}

    # Run experiments
    if "linear_probing" in args.experiments:
        t0 = time.time()
        lp_results = run_linear_probing(
            backbones, args.data_dir, device,
            num_seeds=args.num_seeds, num_epochs=args.num_epochs,
        )
        all_results["linear_probing"] = lp_results
        print(format_results_table(lp_results, "Linear Probing"))
        logger.info(f"Linear probing took {time.time()-t0:.0f}s")

    if "invisible_physics" in args.experiments:
        t0 = time.time()
        ip_results = run_invisible_physics(backbones, args.data_dir, device)
        all_results["invisible_physics"] = ip_results
        print(format_results_table(ip_results, "Invisible Physics"))
        logger.info(f"Invisible physics took {time.time()-t0:.0f}s")

    if "zero_shot" in args.experiments:
        t0 = time.time()
        zs_results = run_zero_shot(
            backbones, args.data_dir, device,
            n_lib=args.n_lib, n_query=args.n_query,
        )
        all_results["zero_shot"] = zs_results
        print(format_results_table(zs_results, "Zero-Shot Physics Inference"))
        logger.info(f"Zero-shot took {time.time()-t0:.0f}s")

    # Save all results
    results_path = output_dir / "results.json"

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"Results saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"  All evaluations complete! Results: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
