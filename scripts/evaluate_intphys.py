#!/usr/bin/env python3
"""
IntPhys2 Benchmark Evaluation for DynaCLIP.

Evaluates whether physics-aligned visual encoders (DynaCLIP V2/V3) can detect
violations of intuitive physics better than frozen baselines (DINOv2, CLIP, SigLIP).

Uses IntPhys2 (Bordes et al., 2025) violation-of-expectation benchmark:
  - Plausible videos → smooth embedding trajectories
  - Implausible videos → embedding discontinuities (object vanishing, teleporting)

Three surprise metrics:
  1. embedding_diff:   Sum of squared L2 distances between consecutive frames
  2. max_jump:         Largest single frame-to-frame L2 distance
  3. prediction_error: Linear predictor (emb[t] → emb[t+1]) trained on plausible,
                       measured on all videos

Pairwise accuracy: for each (plausible, implausible) pair, check if implausible
has higher surprise. Random chance = 50%.

Usage:
    python scripts/evaluate_intphys.py \
        --data_dir /path/to/intphys2 \
        --split Main \
        --checkpoints checkpoints/pretrain_v2/dynaclip_final.pt \
                      checkpoints/pretrain_v3/dynaclip_final.pt \
        --checkpoint_names DynaCLIP-V2 DynaCLIP-V3 \
        --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16 \
        --output_dir results/intphys \
        --gpu 0
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Video loading
# ============================================================================
def load_video_frames(
    video_path: str,
    frame_stride: int = 2,
    max_frames: int = 100,
    size: int = 224,
) -> Optional[torch.Tensor]:
    """Load video frames using decord, returning (T, 3, H, W) tensor."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = list(range(0, total_frames, frame_stride))[:max_frames]
        if len(indices) < 2:
            indices = list(range(min(total_frames, max_frames)))
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8

        # Resize and normalize
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        tensors = []
        for i in range(frames.shape[0]):
            img = Image.fromarray(frames[i])
            tensors.append(transform(img))
        return torch.stack(tensors)  # (T, 3, 224, 224)
    except Exception as e:
        logger.warning(f"Failed to load {video_path}: {e}")
        return None


# ============================================================================
# Backbone loading
# ============================================================================
def load_backbone(name: str, checkpoint_path: Optional[str] = None, device: str = "cuda"):
    """Load a visual backbone encoder."""
    from dynaclip.models.backbones import (
        load_dynaclip, load_dinov2_vitb14, load_dinov2_vitl14,
        load_siglip, load_clip_vitl14,
    )

    loaders = {
        "dinov2_vitb14": load_dinov2_vitb14,
        "dinov2_vitl14": load_dinov2_vitl14,
        "siglip_vitb16": load_siglip,
        "siglip": load_siglip,
        "clip_vitl14": load_clip_vitl14,
    }

    if checkpoint_path:
        backbone = load_dynaclip(checkpoint_path)
    elif name in loaders:
        backbone = loaders[name]()
    else:
        raise ValueError(f"Unknown backbone: {name}")

    backbone = backbone.to(device)
    backbone.eval()
    return backbone


def encode_frames(
    backbone, frames: torch.Tensor, device: str, batch_size: int = 32
) -> np.ndarray:
    """Encode video frames into embeddings. Returns (T, D) numpy array."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size].to(device)
            emb = backbone(batch)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# ============================================================================
# Surprise metrics
# ============================================================================
def compute_embedding_diff(embeddings: np.ndarray) -> float:
    """Total squared L2 distance between consecutive frame embeddings."""
    diffs = np.diff(embeddings, axis=0)
    return float(np.sum(np.sum(diffs ** 2, axis=1)))


def compute_max_jump(embeddings: np.ndarray) -> float:
    """Largest single frame-to-frame L2 distance."""
    diffs = np.diff(embeddings, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.max(distances)) if len(distances) > 0 else 0.0


def compute_prediction_error(
    embeddings: np.ndarray,
    predictor_weights: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> float:
    """Prediction error using linear predictor emb[t] → emb[t+1]."""
    if predictor_weights is None or embeddings.shape[0] < 2:
        return 0.0
    W, b = predictor_weights
    X = embeddings[:-1]
    Y = embeddings[1:]
    pred = X @ W + b
    errors = np.sum((pred - Y) ** 2, axis=1)
    return float(np.mean(errors))


def train_linear_predictor(
    plausible_embeddings_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a linear predictor: emb[t] → emb[t+1] on plausible videos."""
    X_all, Y_all = [], []
    for emb in plausible_embeddings_list:
        if emb.shape[0] < 2:
            continue
        X_all.append(emb[:-1])
        Y_all.append(emb[1:])

    if not X_all:
        dim = plausible_embeddings_list[0].shape[1] if plausible_embeddings_list else 768
        return np.zeros((dim, dim)), np.zeros(dim)

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)

    # Ridge regression: W = (X^T X + λI)^{-1} X^T Y
    dim = X.shape[1]
    lam = 1.0
    XtX = X.T @ X + lam * np.eye(dim)
    XtY = X.T @ Y
    W = np.linalg.solve(XtX, XtY)
    b = np.mean(Y - X @ W, axis=0)
    return W, b


# ============================================================================
# IntPhys2 data loading
# ============================================================================
def load_intphys2_metadata(
    data_dir: str, split: str = "Main"
) -> List[Dict]:
    """Load IntPhys2 metadata CSV."""
    csv_path = os.path.join(data_dir, split, "metadata.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata not found: {csv_path}")

    entries = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["video_path"] = os.path.join(data_dir, split, row["file_name"])
            row["is_possible"] = "Possible" in row.get("type", "")
            entries.append(row)

    logger.info(f"Loaded {len(entries)} entries from {csv_path}")
    return entries


def group_into_pairs(
    entries: List[Dict],
) -> List[Dict]:
    """Group videos into (plausible, implausible) pairs by SceneIndex + condition.

    Each scene has multiple possible/impossible variants. We create all
    valid (possible, impossible) pairs from the same scene.
    """
    scenes = defaultdict(lambda: {"possible": [], "impossible": []})
    for e in entries:
        key = (e["SceneIndex"], e.get("condition", "unknown"))
        if e["is_possible"]:
            scenes[key]["possible"].append(e)
        else:
            scenes[key]["impossible"].append(e)

    pairs = []
    for key, group in scenes.items():
        for pos in group["possible"]:
            for neg in group["impossible"]:
                pairs.append({
                    "scene": key[0],
                    "condition": key[1],
                    "possible": pos,
                    "impossible": neg,
                    "difficulty": pos.get("Difficulty", "unknown"),
                })

    logger.info(
        f"Created {len(pairs)} evaluation pairs from "
        f"{len(scenes)} scenes"
    )
    return pairs


# ============================================================================
# Main evaluation
# ============================================================================
def evaluate_backbone(
    backbone,
    pairs: List[Dict],
    device: str,
    frame_stride: int = 2,
    max_frames: int = 100,
    batch_size: int = 32,
    use_predictor: bool = True,
) -> Dict:
    """Evaluate a single backbone on IntPhys2 pairs."""
    # Phase 1: Encode all unique videos
    video_paths = set()
    for p in pairs:
        video_paths.add(p["possible"]["video_path"])
        video_paths.add(p["impossible"]["video_path"])

    logger.info(f"Encoding {len(video_paths)} unique videos...")
    video_embeddings = {}
    t0 = time.time()
    for i, vpath in enumerate(sorted(video_paths)):
        frames = load_video_frames(vpath, frame_stride, max_frames)
        if frames is None:
            continue
        emb = encode_frames(backbone, frames, device, batch_size)
        video_embeddings[vpath] = emb
        if (i + 1) % 50 == 0:
            logger.info(f"  Encoded {i+1}/{len(video_paths)} videos")

    encode_time = time.time() - t0
    logger.info(f"Encoding complete in {encode_time:.1f}s")

    # Phase 2: Train linear predictor on plausible videos
    predictor_weights = None
    if use_predictor:
        plausible_embs = []
        for p in pairs:
            vp = p["possible"]["video_path"]
            if vp in video_embeddings:
                plausible_embs.append(video_embeddings[vp])
        if plausible_embs:
            predictor_weights = train_linear_predictor(plausible_embs)
            logger.info("Linear predictor trained on plausible videos")

    # Phase 3: Compute surprise metrics for all videos
    video_surprises = {}
    for vpath, emb in video_embeddings.items():
        video_surprises[vpath] = {
            "embedding_diff": compute_embedding_diff(emb),
            "max_jump": compute_max_jump(emb),
            "prediction_error": compute_prediction_error(emb, predictor_weights),
        }

    # Phase 4: Pairwise classification
    metrics = ["embedding_diff", "max_jump", "prediction_error"]
    results_by_metric = {}

    for metric in metrics:
        correct = 0
        total = 0
        by_condition = defaultdict(lambda: {"correct": 0, "total": 0})
        by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})

        for p in pairs:
            pos_path = p["possible"]["video_path"]
            neg_path = p["impossible"]["video_path"]
            if pos_path not in video_surprises or neg_path not in video_surprises:
                continue

            pos_surprise = video_surprises[pos_path][metric]
            neg_surprise = video_surprises[neg_path][metric]

            # Implausible should have HIGHER surprise
            is_correct = neg_surprise > pos_surprise
            total += 1
            if is_correct:
                correct += 1

            cond = p["condition"]
            by_condition[cond]["total"] += 1
            if is_correct:
                by_condition[cond]["correct"] += 1

            diff = p["difficulty"]
            by_difficulty[diff]["total"] += 1
            if is_correct:
                by_difficulty[diff]["correct"] += 1

        overall_acc = correct / total if total > 0 else 0.0
        results_by_metric[metric] = {
            "overall": {"accuracy": overall_acc, "correct": correct, "total": total},
            "by_condition": {
                k: {
                    "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                    **v,
                }
                for k, v in by_condition.items()
            },
            "by_difficulty": {
                k: {
                    "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0,
                    **v,
                }
                for k, v in by_difficulty.items()
            },
        }

    # Find best metric
    best_metric = max(
        metrics,
        key=lambda m: results_by_metric[m]["overall"]["accuracy"],
    )

    return {
        "all_metrics": results_by_metric,
        "best_metric": best_metric,
        "best_accuracy": results_by_metric[best_metric]["overall"]["accuracy"],
        "encoding_time_sec": encode_time,
        "num_videos": len(video_embeddings),
        "num_pairs": len(pairs),
    }


# ============================================================================
# Multi-GPU parallel evaluation
# ============================================================================
def evaluate_backbone_on_gpu(args_tuple):
    """Worker function for parallel GPU evaluation."""
    backbone_name, checkpoint_path, pairs, gpu_id, frame_stride, max_frames, batch_size = args_tuple
    device = f"cuda:{gpu_id}"
    logger.info(f"[GPU {gpu_id}] Loading {backbone_name}...")

    try:
        backbone = load_backbone(backbone_name, checkpoint_path, device)
        result = evaluate_backbone(
            backbone, pairs, device,
            frame_stride=frame_stride,
            max_frames=max_frames,
            batch_size=batch_size,
        )
        # Free GPU memory
        del backbone
        torch.cuda.empty_cache()
        return backbone_name, result
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed evaluating {backbone_name}: {e}")
        import traceback
        traceback.print_exc()
        return backbone_name, {"error": str(e)}


def run_sequential_evaluation(
    backbone_configs: List[Tuple[str, Optional[str]]],
    pairs: List[Dict],
    gpu_ids: List[int],
    frame_stride: int = 2,
    max_frames: int = 100,
    batch_size: int = 32,
) -> Dict:
    """Run evaluation sequentially, one backbone per available GPU (round-robin)."""
    all_results = {}
    for i, (name, ckpt) in enumerate(backbone_configs):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {name} on GPU {gpu_id}")
        logger.info(f"{'='*60}")

        _, result = evaluate_backbone_on_gpu(
            (name, ckpt, pairs, gpu_id, frame_stride, max_frames, batch_size)
        )
        all_results[name] = result

    return all_results


def run_parallel_evaluation(
    backbone_configs: List[Tuple[str, Optional[str]]],
    pairs: List[Dict],
    gpu_ids: List[int],
    frame_stride: int = 2,
    max_frames: int = 100,
    batch_size: int = 32,
) -> Dict:
    """Run evaluation in parallel across multiple GPUs using multiprocessing."""
    import torch.multiprocessing as mp

    # Assign backbones to GPUs round-robin
    tasks = []
    for i, (name, ckpt) in enumerate(backbone_configs):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((name, ckpt, pairs, gpu_id, frame_stride, max_frames, batch_size))

    # Use spawn context to avoid CUDA fork issues
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(len(tasks), len(gpu_ids))) as pool:
            results_list = pool.map(evaluate_backbone_on_gpu, tasks)
    except Exception as e:
        logger.warning(f"Parallel evaluation failed ({e}), falling back to sequential")
        return run_sequential_evaluation(
            backbone_configs, pairs, gpu_ids, frame_stride, max_frames, batch_size
        )

    return {name: result for name, result in results_list}


# ============================================================================
# Reporting
# ============================================================================
def print_results_table(all_results: Dict):
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("IntPhys2 Pairwise Classification Accuracy")
    print("=" * 80)
    print(f"{'Backbone':<25} {'Best Metric':<20} {'Overall Acc':>12}")
    print("-" * 60)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1].get("best_accuracy", 0),
        reverse=True,
    )

    for name, result in sorted_results:
        if "error" in result:
            print(f"{name:<25} {'ERROR':<20} {'N/A':>12}")
            continue
        best = result["best_metric"]
        acc = result["best_accuracy"]
        print(f"{name:<25} {best:<20} {acc:>11.1%}")

    print("=" * 80)

    # Per-condition breakdown
    print("\nPer-Condition Breakdown (best metric per backbone):")
    print("-" * 80)

    conditions = set()
    for result in all_results.values():
        if "error" in result:
            continue
        best = result["best_metric"]
        conditions.update(result["all_metrics"][best].get("by_condition", {}).keys())

    conditions = sorted(conditions)
    header = f"{'Backbone':<25}" + "".join(f"{c[:12]:>14}" for c in conditions)
    print(header)
    print("-" * len(header))

    for name, result in sorted_results:
        if "error" in result:
            continue
        best = result["best_metric"]
        by_cond = result["all_metrics"][best].get("by_condition", {})
        row = f"{name:<25}"
        for c in conditions:
            if c in by_cond:
                row += f"{by_cond[c]['accuracy']:>13.1%}"
            else:
                row += f"{'N/A':>14}"
        print(row)

    # Per-difficulty breakdown
    print("\nPer-Difficulty Breakdown:")
    print("-" * 80)

    difficulties = set()
    for result in all_results.values():
        if "error" in result:
            continue
        best = result["best_metric"]
        difficulties.update(result["all_metrics"][best].get("by_difficulty", {}).keys())

    difficulties = sorted(difficulties)
    header = f"{'Backbone':<25}" + "".join(f"{d:>14}" for d in difficulties)
    print(header)
    print("-" * len(header))

    for name, result in sorted_results:
        if "error" in result:
            continue
        best = result["best_metric"]
        by_diff = result["all_metrics"][best].get("by_difficulty", {})
        row = f"{name:<25}"
        for d in difficulties:
            if d in by_diff:
                row += f"{by_diff[d]['accuracy']:>13.1%}"
            else:
                row += f"{'N/A':>14}"
        print(row)

    print()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="IntPhys2 Benchmark Evaluation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to IntPhys2 data")
    parser.add_argument("--split", type=str, default="Main", choices=["Main", "Debug"],
                        help="Which split to evaluate on")
    parser.add_argument("--checkpoints", nargs="+", default=[], help="DynaCLIP checkpoint paths")
    parser.add_argument("--checkpoint_names", nargs="+", default=[], help="Display names")
    parser.add_argument("--baselines", nargs="+",
                        default=["dinov2_vitb14", "dinov2_vitl14"],
                        help="Frozen baseline names")
    parser.add_argument("--output_dir", type=str, default="results/intphys",
                        help="Output directory")
    parser.add_argument("--frame_stride", type=int, default=2, help="Sample every N-th frame")
    parser.add_argument("--max_frames", type=int, default=100, help="Max frames per video")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--gpu", type=int, nargs="+", default=[0], help="GPU ID(s)")
    parser.add_argument("--no_predictor", action="store_true", help="Skip linear predictor")
    parser.add_argument("--parallel", action="store_true", help="Run backbones in parallel")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["Easy", "Medium", "Hard"],
                        help="Filter by difficulty level")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    entries = load_intphys2_metadata(args.data_dir, args.split)

    # Optional difficulty filter
    if args.difficulty:
        entries = [e for e in entries if e.get("Difficulty") == args.difficulty]
        logger.info(f"Filtered to {len(entries)} entries with difficulty={args.difficulty}")

    # Group into pairs
    pairs = group_into_pairs(entries)
    if not pairs:
        logger.error("No valid pairs found!")
        return

    # Build backbone list
    backbone_configs = []

    # Checkpoints (DynaCLIP V2, V3, etc.)
    for i, ckpt in enumerate(args.checkpoints):
        name = args.checkpoint_names[i] if i < len(args.checkpoint_names) else f"DynaCLIP-{i}"
        backbone_configs.append((name, ckpt))

    # Baselines
    for baseline in args.baselines:
        backbone_configs.append((baseline, None))

    logger.info(f"Evaluating {len(backbone_configs)} backbones: "
                f"{[c[0] for c in backbone_configs]}")

    # Run evaluation
    if args.parallel and len(args.gpu) > 1:
        all_results = run_parallel_evaluation(
            backbone_configs, pairs, args.gpu,
            args.frame_stride, args.max_frames, args.batch_size,
        )
    else:
        all_results = run_sequential_evaluation(
            backbone_configs, pairs, args.gpu,
            args.frame_stride, args.max_frames, args.batch_size,
        )

    # Print results
    print_results_table(all_results)

    # Save results
    output_path = os.path.join(args.output_dir, "intphys_results.json")
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    logger.info(f"Results saved to {output_path}")

    # Also save a compact summary
    summary = {}
    for name, result in all_results.items():
        if "error" in result:
            summary[name] = {"error": result["error"]}
            continue
        summary[name] = {
            "best_metric": result["best_metric"],
            "best_accuracy": result["best_accuracy"],
            "all_accuracies": {
                m: result["all_metrics"][m]["overall"]["accuracy"]
                for m in result["all_metrics"]
            },
        }

    summary_path = os.path.join(args.output_dir, "intphys_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
