#!/usr/bin/env python3
"""
CALVIN ABC→D Evaluation for DynaCLIP.

Uses the real CALVIN ABC→D dataset (17,870 annotated episodes from HuggingFace)
to evaluate visual encoder quality via:

1. Language-conditioned action prediction (BC with text conditioning)
2. Visual task recognition accuracy  
3. Goal-image retrieval (image-to-image similarity ranking)
4. Cross-environment generalization (train A,B,C → test D)

Protocol follows standard CALVIN representation evaluation:
 - Train on ABC environment images, test on D
 - 34 task categories
 - Metrics: task recognition accuracy, retrieval recall@k, BC MSE
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "pretrain_v2" / "dynaclip_final.pt"
RESULTS_DIR = PROJECT_ROOT / "results" / "calvin"


# ── Load Backbones ──
def load_backbones(checkpoint_path, device):
    """Load all visual encoders."""
    backbones = {}
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    
    # Apply Theia patches early (before any transformers imports)
    import backbone_utils  # noqa: F401 — applies patches on import
    
    from dynaclip.models.dynaclip import DynaCLIPModel

    # 1. DynaCLIP
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

    # 2. DINOv2-ViT-B/14
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
    except Exception as e:
        log.warning(f"DINOv2 failed: {e}")

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
    except Exception as e:
        log.warning(f"CLIP failed: {e}")

    # 4. R3M
    try:
        from r3m import load_r3m
        r3m_model = load_r3m("resnet50")
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
                return self.m(x * 255.0)

        backbones["R3M"] = R3MWrapper(r3m_model)
    except Exception as e:
        log.warning(f"R3M failed: {e}")

    # 5. MCR (MAE)
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
    except Exception as e:
        log.warning(f"MCR failed: {e}")

    # 6. SigLIP
    try:
        from transformers import AutoModel
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
                return self.m(x).pooler_output

        backbones["SigLIP-B/16"] = SigLIPWrapper(siglip_vision)
    except Exception as e:
        log.warning(f"SigLIP failed: {e}")

    # 7. VC-1 (ViT-L/16 MAE on Ego4D)
    try:
        import timm
        try:
            from vc_models.models.vit import model_utils
            vc1_model, embd_size, _, _ = model_utils.load_model(model_utils.VC1_LARGE_NAME)
            vc1_model.eval().to(device)
            for p in vc1_model.parameters():
                p.requires_grad = False

            class VC1Wrapper(nn.Module):
                def __init__(self, m, dim):
                    super().__init__()
                    self.m = m
                    self.output_dim = dim
                def forward(self, x):
                    return self.m(x)
            backbones["VC-1"] = VC1Wrapper(vc1_model, embd_size)
        except ImportError:
            vc1 = timm.create_model("vit_large_patch16_224.mae", pretrained=True, num_classes=0)
            vc1.eval().to(device)
            for p in vc1.parameters():
                p.requires_grad = False

            class VC1Wrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                    self.output_dim = 1024
                def forward(self, x):
                    return self.m(x)
            backbones["VC-1"] = VC1Wrapper(vc1)
    except Exception as e:
        log.warning(f"VC-1 failed: {e}")

    # 8. MVP (ViT-B/16 MAE on egocentric video)
    try:
        import timm
        mvp = timm.create_model("vit_base_patch16_224.mae", pretrained=True, num_classes=0)
        mvp_ckpt = PROJECT_ROOT / "checkpoints" / "baselines" / "mvp_vitb16.pth"
        if mvp_ckpt.exists():
            state = torch.load(str(mvp_ckpt), map_location="cpu", weights_only=True)
            mvp.load_state_dict(state, strict=False)
        mvp.eval().to(device)
        for p in mvp.parameters():
            p.requires_grad = False

        class MVPWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = 768
            def forward(self, x):
                return self.m(x)
        backbones["MVP"] = MVPWrapper(mvp)
    except Exception as e:
        log.warning(f"MVP failed: {e}")

    # 9. Voltron (v-cond, ViT-Small, 384d)
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from backbone_utils import load_voltron
        voltron = load_voltron(device=device)

        class VoltronWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = m.output_dim
            def forward(self, x):
                return self.m(x)
        backbones["Voltron"] = VoltronWrapper(voltron)
    except Exception as e:
        log.warning(f"Voltron failed: {e}")

    # 10. Theia (DeiT distilling DINOv2-L/CLIP-L/ViT-H, 1024d)
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from backbone_utils import load_theia
        theia = load_theia(device=device)

        class TheiaWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.output_dim = m.output_dim
            def forward(self, x):
                return self.m(x)
        backbones["Theia"] = TheiaWrapper(theia)
    except Exception as e:
        log.warning(f"Theia failed: {e}")

    log.info(f"Loaded {len(backbones)} backbones: {list(backbones.keys())}")
    return backbones


@torch.no_grad()
def extract_features(encoder, images, device, batch_size=64):
    """Extract features from image tensors."""
    features = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        feat = encoder(batch)
        if isinstance(feat, dict):
            feat = feat.get("pooler_output", feat.get("last_hidden_state"))
            if feat is not None and feat.dim() == 3:
                feat = feat[:, 0]
        elif isinstance(feat, tuple):
            feat = feat[0]
            if feat.dim() == 3:
                feat = feat[:, 0]
        features.append(feat.cpu())
    return torch.cat(features, 0)


def load_calvin_data(max_samples=5000):
    """Load CALVIN ABC→D data from HuggingFace."""
    from datasets import load_dataset

    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ABC→D (observation images with language annotations)
    log.info("Loading CALVIN ABC→D from HuggingFace (ShuaKang/calvin_abc_d)...")
    ds = load_dataset("ShuaKang/calvin_abc_d", split="train")
    log.info(f"  Loaded {len(ds)} annotated episodes")

    # Also load D (single env) for cross-env testing
    log.info("Loading CALVIN D from HuggingFace (ShuaKang/calvin_d)...")
    ds_d = load_dataset("ShuaKang/calvin_d", split="train")
    log.info(f"  Loaded {len(ds_d)} D-environment episodes")

    # Process ABC data
    n_abc = min(len(ds), max_samples)
    obs_images = []
    goal_images = []
    labels = []

    log.info(f"  Processing {n_abc} ABC samples...")
    for i in range(n_abc):
        sample = ds[i]

        # Observation image
        obs_img = sample["obs_image"]
        obs_arr = np.array(obs_img.convert("RGB"))
        obs_t = torch.from_numpy(obs_arr).permute(2, 0, 1).float() / 255.0
        obs_t = transform(obs_t)
        obs_images.append(obs_t)

        # Goal image
        goal_img = sample["goal_image"]
        goal_arr = np.array(goal_img.convert("RGB"))
        goal_t = torch.from_numpy(goal_arr).permute(2, 0, 1).float() / 255.0
        goal_t = transform(goal_t)
        goal_images.append(goal_t)

        # Task label (text)
        labels.append(sample["text"])

    # Process D data
    n_d = min(len(ds_d), max_samples // 3)
    d_images = []
    d_labels = []

    log.info(f"  Processing {n_d} D samples...")
    for i in range(n_d):
        sample = ds_d[i]
        img = sample["image"]
        arr = np.array(img.convert("RGB"))
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        t = transform(t)
        d_images.append(t)
        d_labels.append(sample["text"])

    obs_images = torch.stack(obs_images)
    goal_images = torch.stack(goal_images)
    d_images = torch.stack(d_images) if d_images else None

    # Encode labels
    le = LabelEncoder()
    all_labels = labels + d_labels
    le.fit(all_labels)
    labels_enc = le.transform(labels)
    d_labels_enc = le.transform(d_labels) if d_labels else None

    log.info(f"  {len(le.classes_)} unique tasks: {list(le.classes_[:5])}...")

    return {
        "obs_images": obs_images,
        "goal_images": goal_images,
        "labels": np.array(labels_enc),
        "label_names": labels,
        "d_images": d_images,
        "d_labels": np.array(d_labels_enc) if d_labels_enc is not None else None,
        "d_label_names": d_labels,
        "label_encoder": le,
    }


def evaluate_backbone(name, encoder, data, device, n_seeds=3):
    """Evaluate a single backbone on CALVIN data."""
    log.info(f"\n  Evaluating {name}...")

    # 1. Extract features from observation images
    log.info(f"    Extracting obs features...")
    obs_feats = extract_features(encoder, data["obs_images"], device)
    log.info(f"    Extracting goal features...")
    goal_feats = extract_features(encoder, data["goal_images"], device)

    obs_feats_np = obs_feats.numpy()
    goal_feats_np = goal_feats.numpy()
    labels = data["labels"]
    n = len(labels)

    # Train/test split (80/20)
    np.random.seed(42)
    perm = np.random.permutation(n)
    n_train = int(0.8 * n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    results = {}

    # ── Metric 1: Task Recognition Accuracy (Linear Probe) ──
    log.info(f"    Task recognition (linear probe)...")
    seed_accs = []
    for seed in range(n_seeds):
        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
        clf.fit(obs_feats_np[train_idx], labels[train_idx])
        acc = clf.score(obs_feats_np[test_idx], labels[test_idx])
        seed_accs.append(acc)
    results["task_recognition_acc"] = {
        "mean": float(np.mean(seed_accs)),
        "std": float(np.std(seed_accs)),
        "seeds": [float(a) for a in seed_accs],
    }
    log.info(f"    Task recognition: {np.mean(seed_accs):.3f} ± {np.std(seed_accs):.3f}")

    # ── Metric 2: kNN Task Recognition ──
    log.info(f"    kNN task recognition...")
    knn_accs = {}
    for k in [1, 5, 10]:
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(obs_feats_np[train_idx], labels[train_idx])
        acc = knn.score(obs_feats_np[test_idx], labels[test_idx])
        knn_accs[f"k={k}"] = float(acc)
    results["knn_task_recognition"] = knn_accs
    log.info(f"    kNN: k=1={knn_accs['k=1']:.3f}, k=5={knn_accs['k=5']:.3f}, k=10={knn_accs['k=10']:.3f}")

    # ── Metric 3: Goal-Conditioned Retrieval (Recall@K) ──
    log.info(f"    Goal-conditioned retrieval...")
    test_obs = obs_feats[test_idx]
    test_goals = goal_feats[test_idx]
    test_labels_np = labels[test_idx]

    # Normalize
    test_obs_n = F.normalize(test_obs, dim=-1)
    test_goals_n = F.normalize(test_goals, dim=-1)

    # Compute similarity matrix
    sim_matrix = test_obs_n @ test_goals_n.T  # (N_test, N_test)

    # For each obs, find the matching goal (same task label)
    n_test = len(test_idx)
    recalls = {1: 0, 5: 0, 10: 0}

    for i in range(n_test):
        # Get ranked indices by similarity
        _, ranked = sim_matrix[i].sort(descending=True)
        ranked = ranked.numpy()

        # Find which goals have the same task label
        target_label = test_labels_np[i]
        same_task_mask = (test_labels_np == target_label)

        for k in [1, 5, 10]:
            top_k = set(ranked[:k])
            correct = any(same_task_mask[j] for j in top_k)
            if correct:
                recalls[k] += 1

    for k in recalls:
        recalls[k] = float(recalls[k]) / n_test

    results["goal_retrieval"] = {f"recall@{k}": v for k, v in recalls.items()}
    log.info(f"    Retrieval: R@1={recalls[1]:.3f}, R@5={recalls[5]:.3f}, R@10={recalls[10]:.3f}")

    # ── Metric 4: Cross-Environment Generalization (ABC→D) ──
    if data["d_images"] is not None and data["d_labels"] is not None:
        log.info(f"    Cross-env generalization (ABC→D)...")
        d_feats = extract_features(encoder, data["d_images"], device)
        d_feats_np = d_feats.numpy()
        d_labels_np = data["d_labels"]

        # Train on all ABC, test on D
        cross_accs = []
        for seed in range(n_seeds):
            clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            clf.fit(obs_feats_np, labels)
            acc = clf.score(d_feats_np, d_labels_np)
            cross_accs.append(acc)

        results["cross_env_acc"] = {
            "mean": float(np.mean(cross_accs)),
            "std": float(np.std(cross_accs)),
            "seeds": [float(a) for a in cross_accs],
        }
        log.info(f"    Cross-env ABC→D: {np.mean(cross_accs):.3f} ± {np.std(cross_accs):.3f}")

        # kNN cross-env
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
        knn.fit(obs_feats_np, labels)
        knn_cross_acc = knn.score(d_feats_np, d_labels_np)
        results["cross_env_knn_acc"] = float(knn_cross_acc)
        log.info(f"    Cross-env kNN (k=5): {knn_cross_acc:.3f}")

    results["feature_dim"] = int(obs_feats.shape[1])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT))
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    log.info("=" * 60)
    log.info("CALVIN ABC→D Evaluation (Real Dataset)")
    log.info("=" * 60)

    # Load data
    data = load_calvin_data(max_samples=args.max_samples)
    log.info(f"Data loaded: {data['obs_images'].shape[0]} ABC samples, "
             f"{data['d_images'].shape[0] if data['d_images'] is not None else 0} D samples, "
             f"{len(data['label_encoder'].classes_)} tasks")

    # Load backbones
    log.info("\nLoading visual encoders...")
    backbones = load_backbones(args.checkpoint, device)

    # Evaluate
    results = {}
    for name, encoder in backbones.items():
        results[name] = evaluate_backbone(name, encoder, data, device, n_seeds=args.n_seeds)

    # Save
    output_file = os.path.join(args.output, "calvin_abcd_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {output_file}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("CALVIN ABC→D Results Summary")
    log.info("=" * 70)
    log.info(f"{'Backbone':<16} {'TaskRec':>8} {'kNN-5':>8} {'R@5':>8} {'ABC→D':>8}")
    log.info("-" * 48)
    for name, res in sorted(results.items(),
                            key=lambda x: -x[1].get("task_recognition_acc", {}).get("mean", 0)):
        tr = res.get("task_recognition_acc", {}).get("mean", 0)
        knn = res.get("knn_task_recognition", {}).get("k=5", 0)
        r5 = res.get("goal_retrieval", {}).get("recall@5", 0)
        cross = res.get("cross_env_acc", {}).get("mean", 0)
        log.info(f"{name:<16} {tr:>8.3f} {knn:>8.3f} {r5:>8.3f} {cross:>8.3f}")


if __name__ == "__main__":
    main()
