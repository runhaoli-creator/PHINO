"""
Experiment 5: Zero-Shot Physics Inference.

Build library of 1000 objects with known properties.
For a query image, find 5 nearest neighbors in each backbone's embedding space,
predict properties as similarity-weighted average.
Report R² for mass, friction, restitution.
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class ZeroShotPhysicsInference:
    """Zero-shot physics property prediction via nearest-neighbor retrieval."""

    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        library_images: torch.Tensor,
        library_properties: dict,
        query_images: torch.Tensor,
        query_properties: dict,
        k: int = 5,
        device: str = "cuda",
    ):
        """
        Args:
            library_images: (N_lib, 3, 224, 224) library of known objects
            library_properties: dict with 'mass', 'friction', 'restitution' arrays of shape (N_lib,)
            query_images: (N_query, 3, 224, 224)
            query_properties: same structure as library_properties
            k: number of nearest neighbors
        """
        self.backbones = backbones
        self.library_images = library_images
        self.library_properties = library_properties
        self.query_images = query_images
        self.query_properties = query_properties
        self.k = k
        self.device = device

    @torch.no_grad()
    def encode_all(self, backbone: nn.Module, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        """Encode all images through a backbone."""
        backbone = backbone.to(self.device).eval()
        embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            emb = backbone(batch)
            embeddings.append(emb.cpu())
        return torch.cat(embeddings, dim=0)

    def predict_properties(
        self,
        query_embeddings: torch.Tensor,
        library_embeddings: torch.Tensor,
    ) -> dict:
        """Predict physics properties via k-NN weighted average."""
        # Normalize
        q_norm = F.normalize(query_embeddings, dim=-1)
        l_norm = F.normalize(library_embeddings, dim=-1)

        # Cosine similarity
        sim_matrix = torch.mm(q_norm, l_norm.T)  # (N_query, N_lib)

        # Top-k
        topk_sim, topk_idx = sim_matrix.topk(self.k, dim=-1)

        # Softmax weights
        weights = F.softmax(topk_sim / 0.1, dim=-1)  # (N_query, k)

        predictions = {}
        for prop_name in ["mass", "static_friction", "restitution"]:
            lib_vals = torch.tensor(self.library_properties[prop_name], dtype=torch.float32)
            nn_vals = lib_vals[topk_idx]  # (N_query, k)
            pred = (weights * nn_vals).sum(dim=-1)
            predictions[prop_name] = pred.numpy()

        return predictions

    def run(self) -> dict:
        """Run zero-shot physics inference experiment."""
        logger.info("=== Experiment 5: Zero-Shot Physics Inference ===")
        results = {}

        for name, backbone in self.backbones.items():
            logger.info(f"--- {name} ---")

            lib_emb = self.encode_all(backbone, self.library_images)
            query_emb = self.encode_all(backbone, self.query_images)

            predictions = self.predict_properties(query_emb, lib_emb)

            backbone_results = {}
            for prop_name in ["mass", "static_friction", "restitution"]:
                pred = predictions[prop_name]
                true = self.query_properties[prop_name]
                r2 = r2_score(true, pred)
                backbone_results[prop_name] = {
                    "r2": float(r2),
                    "mse": float(np.mean((pred - true) ** 2)),
                }
                logger.info(f"  {prop_name}: R² = {r2:.4f}")

            results[name] = backbone_results

        return results


def create_real_library(
    data_dir: str,
    n_lib: int = 1000,
    n_query: int = 200,
    seed: int = 42,
):
    """Create library and query set from real generated data.

    Loads images from the DynaCLIP data directory and splits into
    library (known properties) and query (to predict).

    Returns:
        lib_images: (N_lib, 3, 224, 224) tensor
        lib_props: dict with 'mass', 'static_friction', 'restitution' arrays
        query_images: (N_query, 3, 224, 224) tensor
        query_props: same structure
    """
    import json
    from pathlib import Path
    from PIL import Image
    from torchvision import transforms as T

    rng = np.random.default_rng(seed)

    meta_path = Path(data_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at {meta_path}")
    with open(meta_path) as f:
        all_meta = json.load(f)

    total_needed = n_lib + n_query
    if total_needed > len(all_meta):
        total_needed = len(all_meta)
        n_lib = int(0.83 * total_needed)
        n_query = total_needed - n_lib
        logger.warning(f"Only {len(all_meta)} entries available. Using lib={n_lib}, query={n_query}")

    # Random sample
    indices = rng.permutation(len(all_meta))[:total_needed]
    lib_indices = indices[:n_lib]
    query_indices = indices[n_lib:n_lib + n_query]

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def load_subset(idxs):
        images = []
        props = {"mass": [], "static_friction": [], "restitution": []}
        for i in idxs:
            entry = all_meta[i]
            img_path = Path(entry["image_path"])
            if not img_path.is_absolute():
                img_path = Path(data_dir) / img_path
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img)
                images.append(img_tensor)
                props["mass"].append(entry["mass"])
                props["static_friction"].append(entry["static_friction"])
                props["restitution"].append(entry["restitution"])
            except Exception as e:
                logger.warning(f"Skipping {img_path}: {e}")
                continue
        return torch.stack(images), {k: np.array(v) for k, v in props.items()}

    logger.info(f"Loading {n_lib} library + {n_query} query images from {data_dir}")
    lib_images, lib_props = load_subset(lib_indices)
    query_images, query_props = load_subset(query_indices)
    logger.info(f"Loaded library: {lib_images.shape}, query: {query_images.shape}")

    return lib_images, lib_props, query_images, query_props
