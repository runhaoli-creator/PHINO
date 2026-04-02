"""
DynaCLIP Pre-computation: Extract and cache DINOv2 embeddings for pair mining.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageListDataset(Dataset):
    """Simple dataset that loads images from a list of paths."""

    def __init__(self, image_paths: list, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        from PIL import Image
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {path}: {e}")

        if self.transform:
            img = self.transform(img)
        return img, idx


def precompute_dino_embeddings(
    image_paths: list,
    output_path: str,
    model_name: str = "dinov2_vitb14",
    batch_size: int = 128,
    device: str = "cuda",
    num_workers: int = 8,
) -> np.ndarray:
    """Pre-compute DINOv2 embeddings for all images.

    Returns: (N, D) embedding array.
    """
    from dynaclip.data.dataset import get_eval_transform

    logger.info(f"Pre-computing DINOv2 embeddings for {len(image_paths)} images")

    # Load DINOv2 model
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()

    transform = get_eval_transform()
    dataset = ImageListDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    all_embeddings = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for imgs, indices in tqdm(loader, desc="Computing DINOv2 embeddings"):
            imgs = imgs.to(device)
            features = model(imgs)  # CLS token
            all_embeddings.append(features.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), embeddings=embeddings)
    logger.info(f"Saved {embeddings.shape} embeddings to {output_path}")

    return embeddings


def mine_hard_pairs(
    dino_embeddings: np.ndarray,
    dynamics_similarities: np.ndarray,
    pair_indices: np.ndarray,
    num_hard_neg: int = 60000,
    num_hard_pos: int = 60000,
    num_random: int = 80000,
    dino_neg_threshold: float = 0.9,
    dyn_neg_threshold: float = 0.3,
    dino_pos_threshold: float = 0.5,
    dyn_pos_threshold: float = 0.7,
) -> list:
    """Mine hard positive/negative pairs based on DINOv2 and dynamics similarity.

    Hard negatives: DINOv2 cos_sim > 0.9, dynamics_sim < 0.3
    Hard positives: DINOv2 cos_sim < 0.5, dynamics_sim > 0.7
    """
    logger.info("Mining hard pairs...")

    n = len(dino_embeddings)
    # Normalize embeddings
    norms = np.linalg.norm(dino_embeddings, axis=1, keepdims=True)
    norm_emb = dino_embeddings / (norms + 1e-8)

    hard_negatives = []
    hard_positives = []
    random_pairs = []

    # Process pre-computed pairs using REAL similarity values
    for idx in range(len(pair_indices)):
        i, j = int(pair_indices[idx][0]), int(pair_indices[idx][1])
        if i >= n or j >= n:
            continue
        dino_sim = float(np.dot(norm_emb[i], norm_emb[j]))
        dyn_sim = float(dynamics_similarities[idx])

        if dino_sim > dino_neg_threshold and dyn_sim < dyn_neg_threshold:
            hard_negatives.append((i, j, dyn_sim))
        elif dino_sim < dino_pos_threshold and dyn_sim > dyn_pos_threshold:
            hard_positives.append((i, j, dyn_sim))
        else:
            random_pairs.append((i, j, dyn_sim))

    logger.info(f"Found {len(hard_negatives)} hard negatives, "
                f"{len(hard_positives)} hard positives, "
                f"{len(random_pairs)} random from pre-computed pairs")

    # Fill random pairs if needed (using ACTUAL proxy similarities)
    rng = np.random.default_rng(42)
    while len(random_pairs) < num_random:
        i, j = rng.integers(0, n, size=2)
        dino_sim = float(np.dot(norm_emb[i], norm_emb[j]))
        random_pairs.append((i, j, dino_sim))  # Use dino_sim as rough proxy

    all_pairs = (
        hard_negatives[:num_hard_neg]
        + hard_positives[:num_hard_pos]
        + random_pairs[:num_random]
    )

    rng.shuffle(all_pairs)
    logger.info(f"Mined {len(all_pairs)} pairs: "
                f"{min(len(hard_negatives), num_hard_neg)} hard neg, "
                f"{min(len(hard_positives), num_hard_pos)} hard pos, "
                f"{min(len(random_pairs), num_random)} random")
    return all_pairs
