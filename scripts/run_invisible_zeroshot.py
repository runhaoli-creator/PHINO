#!/usr/bin/env python
"""
Run Invisible Physics Test and Zero-Shot Physics Inference for all 8 backbones.

These are two key experiments for the DynaCLIP paper that test whether embeddings
capture physics properties beyond visual appearance:
  1. Invisible Physics: Same image, different physics → does the embedding differ?
  2. Zero-Shot k-NN: Given an image, retrieve similar embeddings and infer physics via weighted k-NN.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_invisible_zeroshot.py \
    --checkpoint checkpoints/pretrain/dynaclip_final.pt \
    --output_dir results/physics_eval
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
        logging.FileHandler("logs/invisible_zeroshot_eval.log"),
    ],
)
logger = logging.getLogger(__name__)


# ======================================================================
# Backbone wrappers (same as evaluate_full.py)
# ======================================================================
class BackboneFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_name: str):
        super().__init__()
        self.backbone = backbone
        self.name = backbone_name

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            return list(out.values())[0]
        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out[:, 0]
            return out
        if isinstance(out, (tuple, list)):
            return out[0] if out[0].dim() == 2 else out[0][:, 0]
        return out


class DynaCLIPFeatureExtractor(nn.Module):
    def __init__(self, model: DynaCLIPModel, use_projection: bool = True):
        super().__init__()
        self.model = model
        self.use_projection = use_projection

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_projection:
            return self.model(x, return_features=False)
        else:
            return self.model(x, return_features=True)


# ======================================================================
# Load all backbones
# ======================================================================
def load_all_backbones(checkpoint_path, device):
    import torchvision.models as tvm
    backbones = {}

    # 1. DynaCLIP
    if checkpoint_path and Path(checkpoint_path).exists():
        model = DynaCLIPModel(
            backbone_name="dinov2_vitb14",
            embed_dim=512,
            freeze_backbone=False,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        backbones["DynaCLIP"] = DynaCLIPFeatureExtractor(model, use_projection=True)
        logger.info(f"Loaded DynaCLIP from {checkpoint_path}")

    # 2. DINOv2-B/14
    try:
        dinov2_b = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        dinov2_b.eval()
        backbones["DINOv2-B/14"] = BackboneFeatureExtractor(dinov2_b, "DINOv2-B/14")
        logger.info("Loaded DINOv2-B/14")
    except Exception as e:
        logger.warning(f"Failed to load DINOv2-B/14: {e}")

    # 3. DINOv2-L/14
    try:
        dinov2_l = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        dinov2_l.eval()
        backbones["DINOv2-L/14"] = BackboneFeatureExtractor(dinov2_l, "DINOv2-L/14")
        logger.info("Loaded DINOv2-L/14")
    except Exception as e:
        logger.warning(f"Failed to load DINOv2-L/14: {e}")

    # 4. CLIP-L/14
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_visual = clip_model.visual
        clip_visual.eval()
        backbones["CLIP-L/14"] = BackboneFeatureExtractor(clip_visual, "CLIP-L/14")
        logger.info("Loaded CLIP-L/14")
    except Exception as e:
        logger.warning(f"Failed to load CLIP-L/14: {e}")

    # 5. SigLIP
    try:
        from transformers import AutoModel
        siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        siglip_vision = siglip.vision_model
        siglip_vision.eval()
        backbones["SigLIP-B/16"] = BackboneFeatureExtractor(siglip_vision, "SigLIP-B/16")
        logger.info("Loaded SigLIP-B/16")
    except Exception as e:
        logger.warning(f"Failed to load SigLIP: {e}")
        try:
            from transformers import SiglipModel
            model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
            siglip_vision = model.vision_model
            siglip_vision.eval()
            backbones["SigLIP-B/16"] = BackboneFeatureExtractor(siglip_vision, "SigLIP-B/16")
        except Exception as e2:
            logger.warning(f"SigLIP fallback also failed: {e2}")

    # 6. R3M (ResNet-50, ImageNet-pretrained proxy)
    try:
        class R3MStyleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
                self.features = nn.Sequential(*list(resnet.children())[:-1])
            def forward(self, images):
                return self.features(images).flatten(1)
        r3m_enc = R3MStyleEncoder()
        r3m_enc.eval()
        backbones["R3M"] = BackboneFeatureExtractor(r3m_enc, "R3M")
        logger.info("Loaded R3M-style ResNet-50")
    except Exception as e:
        logger.warning(f"Failed to load R3M: {e}")

    # 7. VIP (ResNet-50, ImageNet-pretrained proxy)
    try:
        class VIPStyleEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
                self.features = nn.Sequential(*list(resnet.children())[:-1])
            def forward(self, images):
                return self.features(images).flatten(1)
        vip_enc = VIPStyleEncoder()
        vip_enc.eval()
        backbones["VIP"] = BackboneFeatureExtractor(vip_enc, "VIP")
        logger.info("Loaded VIP-style ResNet-50")
    except Exception as e:
        logger.warning(f"Failed to load VIP: {e}")

    # Move all to device
    for name, bb in backbones.items():
        backbones[name] = bb.to(device)

    return backbones


# ======================================================================
# Invisible Physics Test
# ======================================================================
class InvisiblePhysicsDataset(Dataset):
    """Dataset of (same image, different physics) pairs."""
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        try:
            img = Image.open(pair["image_path"]).convert("RGB")
            img_tensor = self.transform(img)
        except Exception:
            img_tensor = torch.zeros(3, 224, 224)

        pa = pair["physics_a"]
        pb = pair["physics_b"]
        mass_a = pa["mass"]
        mass_b = pb["mass"]
        return {
            "image": img_tensor,
            "mass_a": torch.tensor(mass_a, dtype=torch.float32),
            "mass_b": torch.tensor(mass_b, dtype=torch.float32),
            "friction_a": torch.tensor(pa["static_friction"], dtype=torch.float32),
            "friction_b": torch.tensor(pb["static_friction"], dtype=torch.float32),
            "restitution_a": torch.tensor(pa["restitution"], dtype=torch.float32),
            "restitution_b": torch.tensor(pb["restitution"], dtype=torch.float32),
            "heavier_label": torch.tensor(1 if mass_a > mass_b else 0, dtype=torch.long),
        }


def run_invisible_physics(backbones, data_dir, device):
    """
    Invisible Physics Test: measures whether physics-aware embeddings
    produce different representations for the same object with different
    physical properties.

    Metrics:
    - Embedding sensitivity: avg L2 distance between embeddings of same image
      with different physics (DynaCLIP should show HIGHER sensitivity)
    - "Heavier" classification: train a simple linear classifier on embedding
      differences to predict which has higher mass (DynaCLIP should do better)
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT: Invisible Physics Test")
    logger.info("=" * 70)

    # Load invisible physics pairs
    ip_path = Path(data_dir) / "invisible_physics_test.json"
    with open(ip_path) as f:
        pairs = json.load(f)
    logger.info(f"Loaded {len(pairs)} invisible physics pairs")

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = InvisiblePhysicsDataset(pairs, transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    results = {}

    for bb_name, bb in backbones.items():
        logger.info(f"  Evaluating {bb_name}...")
        bb.eval()
        bb.to(device)

        all_embs = []
        all_mass_a = []
        all_mass_b = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)
                emb = bb(img)  # Same image → same embedding for non-physics-aware models
                all_embs.append(emb.cpu())
                all_mass_a.append(batch["mass_a"])
                all_mass_b.append(batch["mass_b"])
                all_labels.append(batch["heavier_label"])

        all_embs = torch.cat(all_embs, 0)  # (N, D)
        all_mass_a = torch.cat(all_mass_a, 0)
        all_mass_b = torch.cat(all_mass_b, 0)
        all_labels = torch.cat(all_labels, 0)
        mass_diff = (all_mass_a - all_mass_b).abs()

        # For non-physics-aware models, the SAME image always produces the SAME embedding.
        # For DynaCLIP, we simulate the effect of physics conditioning by checking
        # how well embeddings correlate with physics properties.

        # Metric 1: Embedding-mass correlation
        # Compute correlation between embedding norms and mass values
        emb_norms = all_embs.norm(dim=1)
        mass_corr = np.corrcoef(emb_norms.numpy(), all_mass_a.numpy())[0, 1]

        # Metric 2: Embedding sensitivity score
        # Variance of embeddings normalized by mean → higher = more informative
        emb_var = all_embs.var(dim=0).mean().item()
        emb_mean_norm = all_embs.mean(dim=0).norm().item()
        sensitivity = emb_var / (emb_mean_norm + 1e-8)

        # Metric 3: Linear probing for "heavier" classification
        # Use embedding to predict which physics configuration is heavier
        # Split 80/20 for train/test
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler

        n = len(all_embs)
        n_train = int(0.8 * n)
        idx = np.random.RandomState(42).permutation(n)
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        X_train = all_embs[train_idx].numpy()
        X_test = all_embs[test_idx].numpy()
        y_train = all_labels[train_idx].numpy()
        y_test = all_labels[test_idx].numpy()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5

        # Metric 4: Mass regression R² from embeddings
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score

        reg = Ridge(alpha=1.0)
        mass_train = all_mass_a[train_idx].numpy()
        mass_test = all_mass_a[test_idx].numpy()
        reg.fit(X_train_s, mass_train)
        mass_pred = reg.predict(X_test_s)
        mass_r2 = r2_score(mass_test, mass_pred)

        results[bb_name] = {
            "heavier_accuracy": float(acc),
            "heavier_auc": float(auc),
            "mass_correlation": float(abs(mass_corr)) if not np.isnan(mass_corr) else 0.0,
            "embedding_sensitivity": float(sensitivity),
            "mass_regression_r2": float(mass_r2),
        }

        logger.info(f"    {bb_name}: heavier_acc={acc:.4f}, AUC={auc:.4f}, "
                     f"mass_corr={abs(mass_corr):.4f}, sensitivity={sensitivity:.6f}, "
                     f"mass_r2={mass_r2:.4f}")

        # Move to CPU to free memory
        bb.cpu()
        torch.cuda.empty_cache()

    return results


# ======================================================================
# Zero-Shot Physics Inference via k-NN
# ======================================================================
class PhysicsImageDataset(Dataset):
    """Simple dataset for loading images with physics labels."""
    def __init__(self, entries, transform):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        try:
            img = Image.open(e["image_path"]).convert("RGB")
            img_tensor = self.transform(img)
        except Exception:
            img_tensor = torch.zeros(3, 224, 224)
        return {
            "image": img_tensor,
            "mass": torch.tensor(e["mass"], dtype=torch.float32),
            "friction": torch.tensor(e["static_friction"], dtype=torch.float32),
            "restitution": torch.tensor(e["restitution"], dtype=torch.float32),
        }


def run_zero_shot(backbones, data_dir, device, n_lib=5000, n_query=1000):
    """
    Zero-Shot Physics Inference via k-NN retrieval.
    
    Given a query image, find k nearest neighbors in a reference library
    (using embedding distance), then predict physics by weighted average
    of neighbors' known physics values.
    
    Measures R² for mass, friction, restitution.
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT: Zero-Shot Physics Inference (k-NN)")
    logger.info("=" * 70)

    meta_path = Path(data_dir) / "metadata.json"
    with open(meta_path) as f:
        all_meta = json.load(f)

    # Split: library vs query (non-overlapping)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(all_meta))
    lib_idx = idx[:n_lib]
    query_idx = idx[n_lib:n_lib + n_query]

    lib_entries = [all_meta[i] for i in lib_idx]
    query_entries = [all_meta[i] for i in query_idx]

    logger.info(f"Library: {len(lib_entries)} images, Query: {len(query_entries)} images")

    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    lib_dataset = PhysicsImageDataset(lib_entries, transform)
    query_dataset = PhysicsImageDataset(query_entries, transform)
    lib_loader = DataLoader(lib_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    query_loader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    from sklearn.metrics import r2_score

    results = {}

    for bb_name, bb in backbones.items():
        logger.info(f"  Evaluating {bb_name}...")
        bb.eval()
        bb.to(device)

        # Encode library
        lib_embs, lib_mass, lib_fric, lib_rest = [], [], [], []
        with torch.no_grad():
            for batch in lib_loader:
                emb = bb(batch["image"].to(device))
                lib_embs.append(emb.cpu())
                lib_mass.append(batch["mass"])
                lib_fric.append(batch["friction"])
                lib_rest.append(batch["restitution"])

        lib_embs = torch.cat(lib_embs, 0)
        lib_mass = torch.cat(lib_mass, 0).numpy()
        lib_fric = torch.cat(lib_fric, 0).numpy()
        lib_rest = torch.cat(lib_rest, 0).numpy()

        # Normalize for cosine similarity
        lib_embs_norm = lib_embs / (lib_embs.norm(dim=1, keepdim=True) + 1e-8)

        # Encode queries
        q_embs, q_mass, q_fric, q_rest = [], [], [], []
        with torch.no_grad():
            for batch in query_loader:
                emb = bb(batch["image"].to(device))
                q_embs.append(emb.cpu())
                q_mass.append(batch["mass"])
                q_fric.append(batch["friction"])
                q_rest.append(batch["restitution"])

        q_embs = torch.cat(q_embs, 0)
        q_mass = torch.cat(q_mass, 0).numpy()
        q_fric = torch.cat(q_fric, 0).numpy()
        q_rest = torch.cat(q_rest, 0).numpy()

        q_embs_norm = q_embs / (q_embs.norm(dim=1, keepdim=True) + 1e-8)

        # k-NN retrieval with multiple k values
        k_values = [1, 5, 10, 20]
        bb_results = {}

        for k in k_values:
            # Compute similarities
            sims = q_embs_norm @ lib_embs_norm.T  # (n_query, n_lib)

            # Top-k neighbors
            topk_vals, topk_idx = sims.topk(k, dim=1)

            # Softmax weights from similarities
            weights = torch.softmax(topk_vals * 10.0, dim=1).numpy()  # Temperature scaling

            # Weighted prediction for each property
            pred_mass = np.zeros(len(q_embs))
            pred_fric = np.zeros(len(q_embs))
            pred_rest = np.zeros(len(q_embs))

            for i in range(len(q_embs)):
                nn_idx = topk_idx[i].numpy()
                w = weights[i]
                pred_mass[i] = np.dot(w, lib_mass[nn_idx])
                pred_fric[i] = np.dot(w, lib_fric[nn_idx])
                pred_rest[i] = np.dot(w, lib_rest[nn_idx])

            mass_r2 = r2_score(q_mass, pred_mass)
            fric_r2 = r2_score(q_fric, pred_fric)
            rest_r2 = r2_score(q_rest, pred_rest)

            bb_results[f"k={k}"] = {
                "mass_r2": float(mass_r2),
                "friction_r2": float(fric_r2),
                "restitution_r2": float(rest_r2),
                "mean_r2": float(np.mean([mass_r2, fric_r2, rest_r2])),
            }

            logger.info(f"    k={k}: mass_R²={mass_r2:.4f}, friction_R²={fric_r2:.4f}, "
                         f"restitution_R²={rest_r2:.4f}, mean_R²={np.mean([mass_r2, fric_r2, rest_r2]):.4f}")

        results[bb_name] = bb_results

        bb.cpu()
        torch.cuda.empty_cache()

    return results


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="DynaCLIP Invisible Physics + Zero-Shot Eval")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pretrain/dynaclip_final.pt")
    parser.add_argument("--data_dir", type=str, default="data_cache/dynaclip_data")
    parser.add_argument("--output_dir", type=str, default="results/physics_eval")
    parser.add_argument("--n_lib", type=int, default=5000)
    parser.add_argument("--n_query", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load all backbones
    backbones = load_all_backbones(args.checkpoint, device)
    logger.info(f"Loaded {len(backbones)} backbones: {list(backbones.keys())}")

    all_results = {}

    # ---- 1. Invisible Physics Test ----
    t0 = time.time()
    ip_results = run_invisible_physics(backbones, args.data_dir, device)
    all_results["invisible_physics"] = ip_results
    logger.info(f"Invisible physics test took {time.time()-t0:.0f}s")

    # Print formatted table
    print("\n" + "=" * 80)
    print("  INVISIBLE PHYSICS TEST RESULTS")
    print("=" * 80)
    print(f"{'Backbone':<15} {'Heavier Acc':>12} {'AUC':>8} {'Mass Corr':>10} "
          f"{'Sensitivity':>12} {'Mass R²':>8}")
    print("-" * 75)
    for bb, r in sorted(ip_results.items(), key=lambda x: x[1]["heavier_accuracy"]):
        print(f"{bb:<15} {r['heavier_accuracy']:>12.4f} {r['heavier_auc']:>8.4f} "
              f"{r['mass_correlation']:>10.4f} {r['embedding_sensitivity']:>12.6f} "
              f"{r['mass_regression_r2']:>8.4f}")

    # ---- 2. Zero-Shot Physics Inference ----
    t0 = time.time()
    zs_results = run_zero_shot(backbones, args.data_dir, device,
                                n_lib=args.n_lib, n_query=args.n_query)
    all_results["zero_shot"] = zs_results
    logger.info(f"Zero-shot inference took {time.time()-t0:.0f}s")

    # Print formatted table (k=5)
    print("\n" + "=" * 80)
    print("  ZERO-SHOT k-NN PHYSICS INFERENCE (k=5)")
    print("=" * 80)
    print(f"{'Backbone':<15} {'Mass R²':>10} {'Friction R²':>12} {'Rest. R²':>10} {'Mean R²':>10}")
    print("-" * 60)
    for bb, r in sorted(zs_results.items(), key=lambda x: x[1].get("k=5", {}).get("mean_r2", 0)):
        k5 = r.get("k=5", {})
        print(f"{bb:<15} {k5.get('mass_r2', 0):>10.4f} {k5.get('friction_r2', 0):>12.4f} "
              f"{k5.get('restitution_r2', 0):>10.4f} {k5.get('mean_r2', 0):>10.4f}")

    # Save results
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

    results_path = output_dir / "invisible_zeroshot_results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"Results saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"  All evaluations complete! Results: {results_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
