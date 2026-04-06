"""
IntPhys Benchmark Evaluation for DynaCLIP.

Evaluates whether physics-aligned visual encoders can detect violations of
intuitive physics (object permanence, continuity, shape constancy) using
the IntPhys benchmark (Riochet et al., 2022).

Protocol:
  1. Encode each video frame with a frozen visual encoder
  2. Compute temporal "surprise" as embedding discontinuity between frames
  3. For each (plausible, implausible) video pair, check if the implausible
     video has higher surprise
  4. Report pairwise classification accuracy per physics property

Adapted for static image encoders (DynaCLIP, DINOv2, CLIP, etc.) that
process frames independently, unlike V-JEPA which uses temporal prediction.

Three surprise metrics:
  - embedding_diff: sum of ||emb[t+1] - emb[t]||² (raw discontinuity)
  - max_jump: max ||emb[t+1] - emb[t]||² (largest single discontinuity)
  - prediction_error: fit linear predictor on plausible videos, measure
    prediction error on all videos

Usage:
  python scripts/evaluate_intphys.py \
      --data_dir /path/to/intphys/dev \
      --checkpoints checkpoints/pretrain_v2/dynaclip_final.pt \
                    checkpoints/pretrain_v3/dynaclip_final.pt \
      --baselines dinov2_vitb14 dinov2_vitl14 clip_vitl14 siglip_vitb16 \
      --output_dir results/intphys
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import Ridge

sys.path.insert(0, ".")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# IntPhys Video Dataset
# ============================================================================
class IntPhysVideoDataset(Dataset):
    """Load IntPhys videos as frame sequences.

    Expected directory structure (IntPhys dev set):
        data_dir/
        ├── {property}/          # e.g., object_permanence, continuity, shape_constancy
        │   ├── {scene_id}/
        │   │   ├── possible/
        │   │   │   ├── video_001/
        │   │   │   │   ├── scene/
        │   │   │   │   │   ├── 00001.png
        │   │   │   │   │   ├── 00002.png
        │   │   │   │   │   └── ...
        │   │   ├── impossible/
        │   │   │   ├── video_001/
        │   │   │   │   ├── scene/
        │   │   │   │   │   ├── 00001.png
        │   │   │   │   │   └── ...

    Also supports flat video format:
        data_dir/
        ├── {property}/
        │   ├── {video_id}.mp4    (will be decoded frame by frame)

    The dataset returns (video_frames, label, property, video_id) where
    label=0 for possible and label=1 for impossible.
    """

    def __init__(
        self,
        data_dir: str,
        transform=None,
        max_frames: int = 100,
        frame_stride: int = 1,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.max_frames = max_frames
        self.frame_stride = frame_stride

        if transform is None:
            from torchvision import transforms as T
            self.transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

        # Discover videos
        self.videos = []  # List of (frames_path_or_video_path, label, property, video_id)
        self._discover_videos()
        logger.info(f"Found {len(self.videos)} videos in {data_dir}")

    def _discover_videos(self):
        """Auto-detect IntPhys directory structure.

        Supports two layouts:
          Structure 1: {property}/{scene_id}/{possible|impossible}/{video_id}/scene/*.png
          Structure 2: {property}/{possible|impossible}/{video_id}/scene/*.png
        Uses a seen-set to avoid double-counting.
        """
        seen = set()

        def _add_video(frames_dir, label, prop_name, video_id):
            key = str(frames_dir.resolve())
            if key in seen:
                return
            frame_files = sorted(
                list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
            )
            if len(frame_files) > 0:
                self.videos.append((frames_dir, label, prop_name, video_id))
                seen.add(key)

        for prop_dir in sorted(self.data_dir.iterdir()):
            if not prop_dir.is_dir():
                continue
            prop_name = prop_dir.name

            for sub_dir in sorted(prop_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue

                # Check if sub_dir is a label directory (structure 2)
                if sub_dir.name in ("possible", "impossible"):
                    label = 0 if sub_dir.name == "possible" else 1
                    for video_dir in sorted(sub_dir.iterdir()):
                        if video_dir.is_dir():
                            frames_dir = video_dir / "scene"
                            if not frames_dir.exists():
                                frames_dir = video_dir
                            _add_video(frames_dir, label, prop_name,
                                       f"{prop_name}/{sub_dir.name}/{video_dir.name}")
                        elif video_dir.suffix in (".mp4", ".avi", ".mov"):
                            self.videos.append((
                                video_dir, label, prop_name,
                                f"{prop_name}/{sub_dir.name}/{video_dir.stem}"
                            ))
                            seen.add(str(video_dir.resolve()))
                    continue

                # Otherwise sub_dir is a scene directory (structure 1)
                scene_dir = sub_dir
                for label_name in ["possible", "impossible"]:
                    label_dir = scene_dir / label_name
                    if not label_dir.exists():
                        continue
                    label = 0 if label_name == "possible" else 1
                    for video_dir in sorted(label_dir.iterdir()):
                        if video_dir.is_dir():
                            frames_dir = video_dir / "scene"
                            if not frames_dir.exists():
                                frames_dir = video_dir
                            _add_video(frames_dir, label, prop_name,
                                       f"{prop_name}/{scene_dir.name}/{label_name}/{video_dir.name}")
                        elif video_dir.suffix in (".mp4", ".avi", ".mov"):
                            self.videos.append((
                                video_dir, label, prop_name,
                                f"{prop_name}/{scene_dir.name}/{label_name}/{video_dir.stem}"
                            ))
                            seen.add(str(video_dir.resolve()))

        if len(self.videos) == 0:
            logger.warning(
                f"No videos found in {self.data_dir}. "
                "Please check the directory structure. Expected: "
                "{property}/{scene_id}/possible|impossible/{video_id}/scene/*.png"
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        path, label, prop_name, video_id = self.videos[idx]

        if path.is_dir():
            frames = self._load_frames_from_dir(path)
        else:
            frames = self._load_frames_from_video(path)

        return {
            "frames": frames,       # (T, 3, 224, 224)
            "label": label,         # 0=possible, 1=impossible
            "property": prop_name,
            "video_id": video_id,
            "num_frames": frames.shape[0],
        }

    def _load_frames_from_dir(self, frames_dir: Path) -> torch.Tensor:
        """Load frames from a directory of images."""
        from PIL import Image
        frame_paths = sorted(
            list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        )
        frame_paths = frame_paths[::self.frame_stride][:self.max_frames]

        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            frames.append(self.transform(img))

        if len(frames) == 0:
            return torch.zeros(1, 3, 224, 224)
        return torch.stack(frames)

    def _load_frames_from_video(self, video_path: Path) -> torch.Tensor:
        """Load frames from a video file."""
        try:
            import decord
            decord.bridge.set_bridge("torch")
            vr = decord.VideoReader(str(video_path))
            total_frames = len(vr)
            indices = list(range(0, total_frames, self.frame_stride))[:self.max_frames]
            frames_raw = vr.get_batch(indices)  # (T, H, W, 3)

            from PIL import Image
            frames = []
            for i in range(frames_raw.shape[0]):
                img = Image.fromarray(frames_raw[i].numpy())
                frames.append(self.transform(img))
            return torch.stack(frames)
        except ImportError:
            logger.warning("decord not installed. Install with: pip install decord")
            return torch.zeros(1, 3, 224, 224)


# ============================================================================
# Backbone Loading
# ============================================================================
def load_backbone(name: str, checkpoint: Optional[str] = None, device: str = "cuda") -> nn.Module:
    """Load a visual backbone for evaluation."""

    if checkpoint and Path(checkpoint).exists():
        # Load DynaCLIP checkpoint
        from dynaclip.models.dynaclip import DynaCLIPEncoder
        encoder = DynaCLIPEncoder(
            checkpoint_path=checkpoint,
            backbone_name="dinov2_vitb14",
            feature_type="cls_mean",
        )
        encoder = encoder.to(device).eval()
        logger.info(f"Loaded DynaCLIP checkpoint: {checkpoint}")
        return encoder

    # Load frozen baseline
    if name == "dinov2_vitb14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    elif name == "dinov2_vitl14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=True)
    elif name == "clip_vitl14":
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            # Keep full model so encode_image works; wrap to only expose visual
            class CLIPVisualWrapper(nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.clip_model = clip_model
                def forward(self, x):
                    return self.clip_model.encode_image(x)
            model = CLIPVisualWrapper(model)
        except ImportError:
            logger.warning("open_clip not installed. Skipping CLIP.")
            return None
    elif name == "siglip_vitb16":
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained("google/siglip-base-patch16-224").vision_model
        except ImportError:
            logger.warning("transformers not installed. Skipping SigLIP.")
            return None
    else:
        logger.warning(f"Unknown backbone: {name}")
        return None

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"Loaded baseline: {name}")
    return model


def extract_features(model: nn.Module, images: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Extract features from a batch of images. Returns (B, D) features."""
    with torch.no_grad():
        images = images.to(device)
        # Handle different model types
        if hasattr(model, 'forward_features'):
            # DINOv2-style: extract CLS + mean-pooled patches
            out = model.forward_features(images)
            if isinstance(out, dict):
                cls = out.get("x_norm_clstoken", None)
                patch = out.get("x_norm_patchtokens", None)
                if cls is None or patch is None:
                    x = out.get("x", None)
                    if x is not None:
                        cls = x[:, 0]
                        patch = x[:, 1:]
                    else:
                        raise RuntimeError(f"Unexpected DINOv2 output keys: {out.keys()}")
                features = torch.cat([cls, patch.mean(dim=1)], dim=-1)
            else:
                features = torch.cat([out[:, 0], out[:, 1:].mean(dim=1)], dim=-1)
        else:
            # CLIP wrapper, SigLIP, DynaCLIPEncoder, or any generic model
            features = model(images)

        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 1:
            features = features.unsqueeze(0)

    return features.float().cpu()


# ============================================================================
# Surprise Metrics
# ============================================================================
def compute_embedding_diff_surprise(embeddings: torch.Tensor) -> float:
    """Sum of squared L2 distances between consecutive frame embeddings.

    Higher = more temporal discontinuity = more "surprising".
    """
    if embeddings.shape[0] < 2:
        return 0.0
    diffs = (embeddings[1:] - embeddings[:-1]).norm(dim=-1)  # (T-1,)
    return float(diffs.pow(2).sum())


def compute_max_jump_surprise(embeddings: torch.Tensor) -> float:
    """Maximum L2 distance between consecutive frames.

    Captures the single largest discontinuity (e.g., object vanishing).
    """
    if embeddings.shape[0] < 2:
        return 0.0
    diffs = (embeddings[1:] - embeddings[:-1]).norm(dim=-1)
    return float(diffs.max())


def compute_cosine_surprise(embeddings: torch.Tensor) -> float:
    """1 - mean cosine similarity between consecutive frames.

    Higher = less temporal coherence = more surprising.
    """
    if embeddings.shape[0] < 2:
        return 0.0
    cos_sims = F.cosine_similarity(embeddings[:-1], embeddings[1:], dim=-1)
    return float(1.0 - cos_sims.mean())


def compute_prediction_error_surprise(
    embeddings: torch.Tensor,
    predictor: Optional[Ridge] = None,
) -> float:
    """Prediction error using a linear temporal predictor.

    If predictor is provided, use it. Otherwise, use simple frame differencing.
    """
    if embeddings.shape[0] < 2:
        return 0.0

    if predictor is not None:
        X = embeddings[:-1].numpy()
        Y = embeddings[1:].numpy()
        pred = predictor.predict(X)
        errors = np.sum((Y - pred) ** 2, axis=-1)
        return float(np.sum(errors))
    else:
        return compute_embedding_diff_surprise(embeddings)


# ============================================================================
# Main Evaluation
# ============================================================================
class IntPhysEvaluator:
    """Evaluate visual backbones on IntPhys benchmark."""

    def __init__(
        self,
        data_dir: str,
        device: str = "cuda",
        max_frames: int = 100,
        frame_stride: int = 2,
        batch_size: int = 32,
    ):
        self.device = device
        self.batch_size = batch_size
        self.dataset = IntPhysVideoDataset(
            data_dir=data_dir,
            max_frames=max_frames,
            frame_stride=frame_stride,
        )
        self.surprise_fns = {
            "embedding_diff": compute_embedding_diff_surprise,
            "max_jump": compute_max_jump_surprise,
            "cosine": compute_cosine_surprise,
        }

    @torch.no_grad()
    def encode_video(self, model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
        """Encode all frames of a video. Returns (T, D) embeddings."""
        all_embs = []
        for i in range(0, frames.shape[0], self.batch_size):
            batch = frames[i:i + self.batch_size]
            embs = extract_features(model, batch, self.device)
            all_embs.append(embs)
        return torch.cat(all_embs, dim=0)

    def compute_all_surprises(
        self,
        model: nn.Module,
        predictor: Optional[Ridge] = None,
    ) -> List[dict]:
        """Compute surprise for all videos."""
        results = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            frames = sample["frames"]
            embeddings = self.encode_video(model, frames)

            surprises = {}
            for metric_name, fn in self.surprise_fns.items():
                surprises[metric_name] = fn(embeddings)

            if predictor is not None:
                surprises["prediction_error"] = compute_prediction_error_surprise(
                    embeddings, predictor
                )

            results.append({
                "video_id": sample["video_id"],
                "label": sample["label"],  # 0=possible, 1=impossible
                "property": sample["property"],
                "num_frames": sample["num_frames"],
                "surprises": surprises,
            })

            if (idx + 1) % 50 == 0:
                logger.info(f"  Encoded {idx + 1}/{len(self.dataset)} videos")

        return results

    def train_predictor(
        self,
        model: nn.Module,
    ) -> Ridge:
        """Train a linear temporal predictor on possible (plausible) videos only."""
        logger.info("Training linear temporal predictor on plausible videos...")
        all_X, all_Y = [], []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            if sample["label"] != 0:  # Only train on possible videos
                continue

            frames = sample["frames"]
            embeddings = self.encode_video(model, frames)

            if embeddings.shape[0] >= 2:
                all_X.append(embeddings[:-1].numpy())
                all_Y.append(embeddings[1:].numpy())

        if len(all_X) == 0:
            logger.warning("No plausible videos found for predictor training!")
            return Ridge(alpha=1.0)

        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0)
        logger.info(f"  Training predictor on {X.shape[0]} frame transitions")

        predictor = Ridge(alpha=1.0)
        predictor.fit(X, Y)
        return predictor

    def compute_pairwise_accuracy(
        self,
        results: List[dict],
        metric_name: str = "embedding_diff",
    ) -> dict:
        """Compute pairwise classification accuracy.

        For each property, pair possible and impossible videos.
        If impossible video has higher surprise → correct classification.
        """
        # Group by property
        by_property = {}
        for r in results:
            prop = r["property"]
            by_property.setdefault(prop, {"possible": [], "impossible": []})
            label_str = "possible" if r["label"] == 0 else "impossible"
            by_property[prop][label_str].append(r["surprises"][metric_name])

        accuracies = {}
        for prop, groups in by_property.items():
            possible = groups["possible"]
            impossible = groups["impossible"]

            if len(possible) == 0 or len(impossible) == 0:
                logger.warning(f"  {prop}: no pairs (possible={len(possible)}, impossible={len(impossible)})")
                continue

            # Pairwise comparison: for each (p, i) pair, check if surprise(i) > surprise(p)
            correct = 0
            total = 0
            for p_surp in possible:
                for i_surp in impossible:
                    if i_surp > p_surp:
                        correct += 1
                    elif i_surp == p_surp:
                        correct += 0.5  # tie → random
                    total += 1

            acc = correct / total if total > 0 else 0.5
            accuracies[prop] = {
                "accuracy": float(acc),
                "n_possible": len(possible),
                "n_impossible": len(impossible),
                "n_pairs": total,
                "mean_surprise_possible": float(np.mean(possible)),
                "mean_surprise_impossible": float(np.mean(impossible)),
            }
            logger.info(f"  {prop}: {acc:.1%} ({total} pairs)")

        # Overall accuracy
        if accuracies:
            all_accs = [v["accuracy"] for v in accuracies.values()]
            accuracies["overall"] = {
                "accuracy": float(np.mean(all_accs)),
                "per_property": {k: v["accuracy"] for k, v in accuracies.items() if k != "overall"},
            }

        return accuracies

    def evaluate_backbone(
        self,
        model: nn.Module,
        backbone_name: str,
        use_predictor: bool = True,
    ) -> dict:
        """Full evaluation of a single backbone."""
        logger.info(f"=== Evaluating {backbone_name} on IntPhys ===")

        # Encode all videos once (used for both predictor training and surprise)
        logger.info("  Encoding all videos...")
        all_embeddings = {}
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            embs = self.encode_video(model, sample["frames"])
            all_embeddings[idx] = embs
            if (idx + 1) % 50 == 0:
                logger.info(f"  Encoded {idx + 1}/{len(self.dataset)} videos")

        # Train predictor on plausible videos using cached embeddings
        predictor = None
        if use_predictor:
            logger.info("  Training linear temporal predictor on plausible videos...")
            all_X, all_Y = [], []
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                if sample["label"] != 0:
                    continue
                embs = all_embeddings[idx]
                if embs.shape[0] >= 2:
                    all_X.append(embs[:-1].numpy())
                    all_Y.append(embs[1:].numpy())
            if len(all_X) > 0:
                X = np.concatenate(all_X, axis=0)
                Y = np.concatenate(all_Y, axis=0)
                predictor = Ridge(alpha=1.0)
                predictor.fit(X, Y)
                logger.info(f"  Predictor trained on {X.shape[0]} frame transitions")

        # Compute surprises using cached embeddings
        results = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            embeddings = all_embeddings[idx]
            surprises = {}
            for metric_name, fn in self.surprise_fns.items():
                surprises[metric_name] = fn(embeddings)
            if predictor is not None:
                surprises["prediction_error"] = compute_prediction_error_surprise(
                    embeddings, predictor
                )
            results.append({
                "video_id": sample["video_id"],
                "label": sample["label"],
                "property": sample["property"],
                "num_frames": sample["num_frames"],
                "surprises": surprises,
            })

        # Compute pairwise accuracy for each surprise metric
        all_metrics = {}
        metric_names = list(self.surprise_fns.keys())
        if predictor is not None:
            metric_names.append("prediction_error")

        for metric_name in metric_names:
            logger.info(f"--- Metric: {metric_name} ---")
            acc = self.compute_pairwise_accuracy(results, metric_name)
            all_metrics[metric_name] = acc

        # Pick best metric
        best_metric = max(
            metric_names,
            key=lambda m: all_metrics[m].get("overall", {}).get("accuracy", 0)
        )
        best_acc = all_metrics[best_metric].get("overall", {}).get("accuracy", 0)

        logger.info(f"  Best metric: {best_metric} → {best_acc:.1%}")

        return {
            "backbone": backbone_name,
            "best_metric": best_metric,
            "best_accuracy": best_acc,
            "all_metrics": all_metrics,
        }


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="IntPhys Benchmark Evaluation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to IntPhys dev set directory")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=[],
                        help="DynaCLIP checkpoint paths (V2, V3, etc.)")
    parser.add_argument("--checkpoint_names", type=str, nargs="+", default=[],
                        help="Names for checkpoints (e.g., DynaCLIP-V2 DynaCLIP-V3)")
    parser.add_argument("--baselines", type=str, nargs="+",
                        default=["dinov2_vitb14", "dinov2_vitl14"],
                        help="Baseline backbone names to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/intphys")
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_predictor", action="store_true",
                        help="Skip training linear temporal predictor")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = IntPhysEvaluator(
        data_dir=args.data_dir,
        device=args.device,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
    )

    all_results = {}

    # Evaluate DynaCLIP checkpoints
    for i, ckpt_path in enumerate(args.checkpoints):
        name = args.checkpoint_names[i] if i < len(args.checkpoint_names) else f"DynaCLIP-ckpt{i}"
        model = load_backbone(name, checkpoint=ckpt_path, device=args.device)
        if model is not None:
            result = evaluator.evaluate_backbone(model, name, use_predictor=not args.no_predictor)
            all_results[name] = result
            del model
            torch.cuda.empty_cache()

    # Evaluate baselines
    for baseline_name in args.baselines:
        model = load_backbone(baseline_name, device=args.device)
        if model is not None:
            result = evaluator.evaluate_backbone(
                model, baseline_name, use_predictor=not args.no_predictor
            )
            all_results[baseline_name] = result
            del model
            torch.cuda.empty_cache()

    # Save results
    results_path = output_dir / "intphys_results.json"
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    logger.info(f"Results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("IntPhys Pairwise Classification Accuracy")
    print("=" * 80)
    print(f"{'Backbone':<25} {'Best Metric':<20} {'Overall Acc':>12}")
    print("-" * 60)
    for name, result in all_results.items():
        best_metric = result["best_metric"]
        best_acc = result["best_accuracy"]
        print(f"{name:<25} {best_metric:<20} {best_acc:>11.1%}")

        # Per-property breakdown
        best_data = result["all_metrics"][best_metric]
        for prop, prop_data in best_data.items():
            if prop != "overall" and isinstance(prop_data, dict) and "accuracy" in prop_data:
                print(f"  {prop:<23} {'':20} {prop_data['accuracy']:>11.1%}")

    print("=" * 80)


if __name__ == "__main__":
    main()
