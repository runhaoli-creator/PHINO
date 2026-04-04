"""
DynaCLIP Ablation Studies.

Four ablations comparing trained DynaCLIP variants:
1. Frozen backbone: Only train projection head (1.6M params)
2. No hard mining: Random pair sampling (no hard neg/pos)
3. Random physics: Randomly assigned physics (old data, no category correlation)
4. Standard InfoNCE: Binary positive/negative (no soft similarity labels)

Each variant is trained via configs/ablation_*.yaml and checkpoints are evaluated
via linear probing using scripts/evaluate_ablations.py.

This module provides the AblationStudy class that loads checkpoints and runs
linear probing comparison.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.eval.linear_probing import PhysicsLinearProbing

logger = logging.getLogger(__name__)


# Ablation variant definitions
ABLATION_CONFIGS = {
    "DynaCLIP (full)": {
        "config": "configs/pretrain.yaml",
        "checkpoint_dir": "checkpoints/pretrain",
        "freeze_backbone": False,
        "description": "Full DynaCLIP: fine-tuned backbone + soft InfoNCE + hard mining + category-grounded physics",
    },
    "Frozen backbone": {
        "config": "configs/ablation_frozen.yaml",
        "checkpoint_dir": "checkpoints/ablation_frozen",
        "freeze_backbone": True,
        "description": "Only projection head trained (1.6M params), backbone frozen",
    },
    "No hard mining": {
        "config": "configs/ablation_nohardneg.yaml",
        "checkpoint_dir": "checkpoints/ablation_nohardneg",
        "freeze_backbone": False,
        "description": "All random pairs, no hard negative/positive mining",
    },
    "Random physics": {
        "config": "configs/ablation_random_physics.yaml",
        "checkpoint_dir": "checkpoints/ablation_random_physics",
        "freeze_backbone": False,
        "description": "Physics assigned randomly (no category correlation)",
    },
    "Standard InfoNCE": {
        "config": "configs/ablation_infonce.yaml",
        "checkpoint_dir": "checkpoints/ablation_infonce",
        "freeze_backbone": False,
        "description": "Binary InfoNCE loss instead of Soft InfoNCE",
    },
}


class AblationStudy:
    """Load ablation checkpoints and compare via linear probing."""

    def __init__(
        self,
        data_dir: str = "data_cache/dynaclip_data",
        output_dir: str = "results/ablations",
        device: str = "cuda",
        num_seeds: int = 5,
        num_epochs: int = 100,
    ):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_seeds = num_seeds
        self.num_epochs = num_epochs

    def find_best_checkpoint(self, ckpt_dir: str) -> Optional[str]:
        """Find checkpoint with highest step number."""
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            return None
        ckpts = sorted(
            ckpt_dir.glob("dynaclip_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        return str(ckpts[-1]) if ckpts else None

    def load_variant(self, variant_name: str) -> Optional[nn.Module]:
        """Load a DynaCLIP variant from its checkpoint."""
        cfg = ABLATION_CONFIGS[variant_name]
        ckpt_path = self.find_best_checkpoint(cfg["checkpoint_dir"])
        if ckpt_path is None:
            logger.warning(f"No checkpoint found for '{variant_name}' in {cfg['checkpoint_dir']}")
            return None

        model = DynaCLIPModel(
            backbone_name="dinov2_vitb14",
            embed_dim=512,
            freeze_backbone=cfg["freeze_backbone"],
        )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        logger.info(f"Loaded '{variant_name}' from {ckpt_path}")
        return model

    def run_all(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> dict:
        """Run linear probing for all ablation variants that have checkpoints."""
        backbones = {}
        for variant_name in ABLATION_CONFIGS:
            model = self.load_variant(variant_name)
            if model is not None:
                backbones[variant_name] = model

        if not backbones:
            logger.error("No ablation checkpoints found!")
            return {}

        logger.info(f"Running ablation comparison for {len(backbones)} variants")

        probing = PhysicsLinearProbing(
            backbones=backbones,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_seeds=self.num_seeds,
            num_epochs=self.num_epochs,
            lr=1e-3,
            device=self.device,
            probe_category=True,
        )
        results = probing.run()

        # Save
        with open(self.output_dir / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {self.output_dir / 'ablation_results.json'}")

        return results
