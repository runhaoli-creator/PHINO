#!/usr/bin/env python
"""
DynaCLIP Full Evaluation Pipeline.

Runs all real experiments on trained DynaCLIP + baseline backbones:
1. Linear probing (mass, friction, restitution R²; category accuracy)
2. Zero-shot k-NN physics inference
3. Material clustering quality (NMI, ARI)
4. t-SNE visualization by material type

Baselines: DINOv2-B/14, DINOv2-L/14, CLIP-L/14, SigLIP, Random Init
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

sys.path.insert(0, ".")

from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.models.backbones import load_backbone
from dynaclip.data.dataset import PhysicsProbeDataset, get_eval_transform
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/eval_full.log"),
    ],
)
logger = logging.getLogger(__name__)


# ======================================================================
# Backbone wrapper: map any backbone to a feature extractor
# ======================================================================
class BackboneFeatureExtractor(nn.Module):
    """Wraps any backbone to output a flat feature vector."""

    def __init__(self, backbone: nn.Module, backbone_name: str):
        super().__init__()
        self.backbone = backbone
        self.name = backbone_name

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        # HuggingFace model outputs (BaseModelOutputWithPooling, etc.)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state[:, 0]  # CLS token
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            return list(out.values())[0]
        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out[:, 0]  # CLS token
            return out
        # Tuple output (some models return tuple)
        if isinstance(out, (tuple, list)):
            return out[0] if out[0].dim() == 2 else out[0][:, 0]
        return out


class DynaCLIPFeatureExtractor(nn.Module):
    """Wraps trained DynaCLIP → encode images → projection head output."""

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
# Experiment 1: Linear Probing
# ======================================================================
def run_linear_probing(backbones, train_loader, val_loader, test_loader,
                       device, num_seeds=5, num_epochs=100, lr=1e-3):
    """Run linear probing for each backbone and property."""
    from dynaclip.eval.linear_probing import PhysicsLinearProbing

    probing = PhysicsLinearProbing(
        backbones=backbones,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_seeds=num_seeds,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        probe_category=True,
    )
    return probing.run()


# ======================================================================
# Experiment 2: Zero-Shot k-NN Physics Inference
# ======================================================================
@torch.no_grad()
def run_knn_eval(backbones, train_loader, test_loader, device, k_values=[1, 5, 10]):
    """k-NN probe: use nearest neighbors to predict physics properties."""
    from sklearn.metrics import r2_score

    results = {}
    for bname, backbone in backbones.items():
        logger.info(f"k-NN eval: {bname}")
        backbone = backbone.to(device).eval()

        # Extract library features
        lib_feats, lib_mass, lib_fric, lib_rest = [], [], [], []
        for batch in train_loader:
            img = batch["image"].to(device)
            f = backbone(img)
            lib_feats.append(f.cpu())
            lib_mass.append(batch["mass"])
            lib_fric.append(batch["static_friction"])
            lib_rest.append(batch["restitution"])
        lib_feats = torch.cat(lib_feats)
        lib_mass = torch.cat(lib_mass).numpy()
        lib_fric = torch.cat(lib_fric).numpy()
        lib_rest = torch.cat(lib_rest).numpy()

        # Normalize
        lib_feats = nn.functional.normalize(lib_feats, dim=-1)

        # Evaluate on test set
        test_feats, test_mass, test_fric, test_rest = [], [], [], []
        for batch in test_loader:
            img = batch["image"].to(device)
            f = backbone(img)
            test_feats.append(f.cpu())
            test_mass.append(batch["mass"])
            test_fric.append(batch["static_friction"])
            test_rest.append(batch["restitution"])
        test_feats = torch.cat(test_feats)
        test_mass = torch.cat(test_mass).numpy()
        test_fric = torch.cat(test_fric).numpy()
        test_rest = torch.cat(test_rest).numpy()
        test_feats = nn.functional.normalize(test_feats, dim=-1)

        # k-NN
        bk_results = {}
        sims = test_feats @ lib_feats.T  # (N_test, N_lib)
        for k in k_values:
            topk_idx = sims.topk(k, dim=-1).indices.numpy()

            pred_mass = np.mean(lib_mass[topk_idx], axis=1)
            pred_fric = np.mean(lib_fric[topk_idx], axis=1)
            pred_rest = np.mean(lib_rest[topk_idx], axis=1)

            bk_results[f"k={k}"] = {
                "mass_r2": float(r2_score(test_mass, pred_mass)),
                "friction_r2": float(r2_score(test_fric, pred_fric)),
                "restitution_r2": float(r2_score(test_rest, pred_rest)),
            }
            logger.info(f"  k={k}: mass_r2={bk_results[f'k={k}']['mass_r2']:.4f}")

        results[bname] = bk_results
        backbone.cpu()

    return results


# ======================================================================
# Experiment 3: Material Clustering (NMI, ARI)
# ======================================================================
@torch.no_grad()
def run_clustering_eval(backbones, loader, device, n_clusters=10):
    """Evaluate clustering quality on material types."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    results = {}
    for bname, backbone in backbones.items():
        logger.info(f"Clustering eval: {bname}")
        backbone = backbone.to(device).eval()

        all_feats, all_cats, all_materials = [], [], []
        for batch in loader:
            img = batch["image"].to(device)
            f = backbone(img)
            all_feats.append(f.cpu())
            all_cats.extend(batch["category"])

        all_feats = torch.cat(all_feats).numpy()
        # Map categories to material types
        from dynaclip.data.generation import get_material_for_category
        all_materials = [get_material_for_category(c) for c in all_cats]

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(all_feats)

        # Convert materials to integer labels
        mat_set = sorted(set(all_materials))
        mat_to_idx = {m: i for i, m in enumerate(mat_set)}
        true_labels = [mat_to_idx[m] for m in all_materials]

        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        ari = adjusted_rand_score(true_labels, pred_labels)

        results[bname] = {"nmi": float(nmi), "ari": float(ari)}
        logger.info(f"  NMI={nmi:.4f}, ARI={ari:.4f}")
        backbone.cpu()

    return results


# ======================================================================
# Experiment 4: t-SNE Visualization
# ======================================================================
@torch.no_grad()
def run_tsne(backbone, loader, device, output_path, max_samples=3000):
    """Generate t-SNE visualization colored by material type."""
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    backbone = backbone.to(device).eval()

    feats, materials, cats = [], [], []
    count = 0
    for batch in loader:
        if count >= max_samples:
            break
        img = batch["image"].to(device)
        f = backbone(img)
        feats.append(f.cpu())
        cats.extend(batch["category"])
        count += img.shape[0]

    feats = torch.cat(feats)[:max_samples].numpy()

    from dynaclip.data.generation import get_material_for_category
    materials = [get_material_for_category(c) for c in cats[:max_samples]]

    # t-SNE
    try:
        tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    emb = tsne.fit_transform(feats)

    # Plot
    mat_set = sorted(set(materials))
    try:
        colors_map = plt.colormaps.get_cmap("tab10")
    except AttributeError:
        colors_map = plt.cm.get_cmap("tab10", len(mat_set))
    mat_to_color = {m: colors_map(i) for i, m in enumerate(mat_set)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for mat in mat_set:
        mask = [m == mat for m in materials]
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[mat_to_color[mat]],
                   label=mat, alpha=0.5, s=8)
    ax.legend(fontsize=8, markerscale=3)
    ax.set_title("t-SNE of DynaCLIP Embeddings by Material Type")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved t-SNE plot to {output_path}")
    backbone.cpu()


# ======================================================================
# Load backbones
# ======================================================================
def load_all_backbones(checkpoint_path=None, device="cuda"):
    """Load DynaCLIP + all baseline backbones."""
    backbones = {}

    # 1. DynaCLIP (trained)
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
        backbones["DynaCLIP-backbone"] = DynaCLIPFeatureExtractor(model, use_projection=False)
        logger.info(f"Loaded DynaCLIP from {checkpoint_path}")

    # 2. DINOv2-B/14 (frozen, no fine-tuning)
    try:
        dinov2_b = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        dinov2_b.eval()
        backbones["DINOv2-B/14"] = BackboneFeatureExtractor(dinov2_b, "DINOv2-B/14")
    except Exception as e:
        logger.warning(f"Failed to load DINOv2-B/14: {e}")

    # 3. DINOv2-L/14
    try:
        dinov2_l = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        dinov2_l.eval()
        backbones["DINOv2-L/14"] = BackboneFeatureExtractor(dinov2_l, "DINOv2-L/14")
    except Exception as e:
        logger.warning(f"Failed to load DINOv2-L/14: {e}")

    # 4. CLIP-L/14
    try:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        clip_visual = clip_model.visual
        clip_visual.eval()
        backbones["CLIP-L/14"] = BackboneFeatureExtractor(clip_visual, "CLIP-L/14")
    except Exception as e:
        logger.warning(f"Failed to load CLIP-L/14: {e}")

    # 5. SigLIP
    try:
        from transformers import AutoModel, AutoProcessor
        siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        siglip_vision = siglip.vision_model
        siglip_vision.eval()
        backbones["SigLIP-B/16"] = BackboneFeatureExtractor(siglip_vision, "SigLIP-B/16")
        logger.info("Loaded SigLIP-B/16")
    except Exception as e:
        logger.warning(f"Failed to load SigLIP: {e}")
        # Fallback: try SiglipModel
        try:
            from transformers import SiglipModel
            model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
            siglip_vision = model.vision_model
            siglip_vision.eval()
            backbones["SigLIP-B/16"] = BackboneFeatureExtractor(siglip_vision, "SigLIP-B/16")
            logger.info("Loaded SigLIP-B/16 (via SiglipModel)")
        except Exception as e2:
            logger.warning(f"SigLIP fallback also failed: {e2}")

    # 6. R3M (ResNet-50 backbone, robot manipulation features)
    # Since R3M's Google Drive weights are access-restricted, we use
    # ImageNet-pretrained ResNet-50 (same architecture as R3M) as the baseline.
    # R3M trains ResNet-50 with time-contrastive learning on Ego4D.
    try:
        import torchvision.models as tvm

        class R3MStyleEncoder(nn.Module):
            """ResNet-50 encoder matching R3M architecture (output_dim=2048)."""
            def __init__(self):
                super().__init__()
                resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
                self.features = nn.Sequential(*list(resnet.children())[:-1])
            def forward(self, images):
                x = self.features(images)
                return x.flatten(1)  # (B, 2048)

        r3m_enc = R3MStyleEncoder()
        r3m_enc.eval()
        backbones["R3M"] = BackboneFeatureExtractor(r3m_enc, "R3M")
        logger.info("Loaded R3M-style ResNet-50 (ImageNet-pretrained)")
    except Exception as e:
        logger.warning(f"Failed to load R3M encoder: {e}")

    # 7. VIP (ResNet-50, value-implicit pre-training)
    # Same approach: ResNet-50 architecture matching VIP's backbone
    try:
        class VIPStyleEncoder(nn.Module):
            """ResNet-50 encoder matching VIP architecture (output_dim=2048)."""
            def __init__(self):
                super().__init__()
                resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
                self.features = nn.Sequential(*list(resnet.children())[:-1])
            def forward(self, images):
                x = self.features(images)
                return x.flatten(1)  # (B, 2048)

        vip_enc = VIPStyleEncoder()
        vip_enc.eval()
        backbones["VIP"] = BackboneFeatureExtractor(vip_enc, "VIP")
        logger.info("Loaded VIP-style ResNet-50 (ImageNet-pretrained)")
    except Exception as e:
        logger.warning(f"Failed to load VIP encoder: {e}")

    # 8. MCR / MVP (Masked Visual Pre-training, ViT-B/16)
    # MCR (arXiv 2410.22325) trains a ViT with masked reconstruction on DROID robot data.
    # Since MCR weights aren't publicly released, we use MAE ViT-B/16 (same architecture
    # and training objective) as a representative masked-reconstruction baseline.
    try:
        import timm as _timm

        class MCRStyleEncoder(nn.Module):
            """MAE ViT-B/16 encoder as MCR-equivalent baseline (output_dim=768)."""
            def __init__(self):
                super().__init__()
                self.vit = _timm.create_model("vit_base_patch16_224.mae", pretrained=True)
            def forward(self, images):
                feats = self.vit.forward_features(images)  # (B, 197, 768)
                return feats[:, 0]  # CLS token → (B, 768)

        mcr_enc = MCRStyleEncoder()
        mcr_enc.eval()
        backbones["MCR"] = BackboneFeatureExtractor(mcr_enc, "MCR")
        logger.info("Loaded MCR-style MAE ViT-B/16 (masked reconstruction baseline)")
    except Exception as e:
        logger.warning(f"Failed to load MCR encoder: {e}")

    # 9. Voltron (v-cond, ViT-Small, 384d)
    try:
        from backbone_utils import load_voltron
        voltron = load_voltron(device="cpu")
        backbones["Voltron"] = BackboneFeatureExtractor(voltron, "Voltron")
        logger.info(f"Loaded Voltron v-cond (384d)")
    except Exception as e:
        logger.warning(f"Failed to load Voltron: {e}")

    # 10. Theia (DeiT backbone distilling DINOv2-L/CLIP-L/ViT-H, 1024d)
    try:
        from backbone_utils import load_theia
        theia = load_theia(device="cpu")
        backbones["Theia"] = BackboneFeatureExtractor(theia, "Theia")
        logger.info(f"Loaded Theia (1024d)")
    except Exception as e:
        logger.warning(f"Failed to load Theia: {e}")

    return backbones


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="DynaCLIP Full Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DynaCLIP checkpoint")
    parser.add_argument("--data_dir", type=str, default="data_cache/dynaclip_data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    t0 = time.time()
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DynaCLIP Full Evaluation Pipeline")
    logger.info("=" * 60)

    # Load backbones
    backbones = load_all_backbones(args.checkpoint, device)
    logger.info(f"Loaded {len(backbones)} backbones: {list(backbones.keys())}")

    # Load data
    transform = get_eval_transform(224)
    train_data = PhysicsProbeDataset(args.data_dir, split="train", transform=transform)
    val_data = PhysicsProbeDataset(args.data_dir, split="val", transform=transform)
    test_data = PhysicsProbeDataset(args.data_dir, split="test", transform=transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    all_results = {}

    # === Experiment 1: Linear Probing ===
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT 1: Linear Probing")
    logger.info("=" * 40)
    lp_results = run_linear_probing(
        backbones, train_loader, val_loader, test_loader,
        device, num_seeds=args.num_seeds, num_epochs=100, lr=1e-3,
    )
    all_results["linear_probing"] = lp_results
    with open(output_dir / "linear_probing.json", "w") as f:
        json.dump(lp_results, f, indent=2)

    # === Experiment 2: k-NN Physics Inference ===
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT 2: k-NN Physics Inference")
    logger.info("=" * 40)
    knn_results = run_knn_eval(
        backbones, train_loader, test_loader, device, k_values=[1, 5, 10, 20],
    )
    all_results["knn"] = knn_results
    with open(output_dir / "knn_results.json", "w") as f:
        json.dump(knn_results, f, indent=2)

    # === Experiment 3: Material Clustering ===
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT 3: Material Clustering")
    logger.info("=" * 40)
    cluster_results = run_clustering_eval(backbones, test_loader, device)
    all_results["clustering"] = cluster_results
    with open(output_dir / "clustering_results.json", "w") as f:
        json.dump(cluster_results, f, indent=2)

    # Save all results (before t-SNE so crash doesn't lose them)
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # === Experiment 4: t-SNE Visualization ===
    logger.info("\n" + "=" * 40)
    logger.info("EXPERIMENT 4: t-SNE Visualization")
    logger.info("=" * 40)
    tsne_dir = output_dir / "tsne"
    tsne_dir.mkdir(exist_ok=True)
    for bname, backbone in backbones.items():
        try:
            safe_name = bname.replace("/", "_").replace(" ", "_")
            run_tsne(backbone, test_loader, device,
                     str(tsne_dir / f"tsne_{safe_name}.png"))
        except Exception as e:
            logger.warning(f"t-SNE failed for {bname}: {e}")

    elapsed = (time.time() - t0) / 60
    logger.info(f"\n{'=' * 60}")
    logger.info(f"All evaluations completed in {elapsed:.1f} minutes")
    logger.info(f"Results saved to {output_dir}")

    # Print summary table
    print("\n" + "=" * 80)
    print("LINEAR PROBING RESULTS")
    print("=" * 80)
    print(f"{'Backbone':<25} {'Mass R²':<12} {'Friction R²':<14} {'Restitution R²':<16} {'Category Acc':<14}")
    print("-" * 80)
    for bname, bres in lp_results.items():
        mass = bres.get("mass", {}).get("mean", 0)
        fric = bres.get("static_friction", {}).get("mean", 0)
        rest = bres.get("restitution", {}).get("mean", 0)
        cat = bres.get("category", {}).get("mean", 0)
        print(f"{bname:<25} {mass:<12.4f} {fric:<14.4f} {rest:<16.4f} {cat:<14.4f}")

    print("\nk-NN RESULTS (k=5)")
    print("-" * 60)
    for bname, bres in knn_results.items():
        k5 = bres.get("k=5", {})
        print(f"{bname:<25} mass_r2={k5.get('mass_r2', 0):.4f}  "
              f"fric_r2={k5.get('friction_r2', 0):.4f}  "
              f"rest_r2={k5.get('restitution_r2', 0):.4f}")

    print("\nCLUSTERING RESULTS")
    print("-" * 40)
    for bname, bres in cluster_results.items():
        print(f"{bname:<25} NMI={bres['nmi']:.4f}  ARI={bres['ari']:.4f}")


if __name__ == "__main__":
    main()
