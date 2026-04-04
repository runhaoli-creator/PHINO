#!/usr/bin/env python
"""
Run additional ablation trainings for DynaCLIP.

Ablations:
1. Triplet loss
2. BYOL loss  
3. Unfreeze last 2 blocks only
4. Unfreeze last 4 blocks only

Each runs sequentially on a single GPU.
"""
import argparse
import logging
import yaml
import sys
import copy
from pathlib import Path

sys.path.insert(0, ".")

import torch
from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.losses.contrastive import build_loss
from dynaclip.data.dataset import create_contrastive_dataloader
from dynaclip.trainers.pretrain import DynaCLIPTrainer
from dynaclip.utils.helpers import setup_logging, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_CONFIG = {
    "model": {
        "backbone": "dinov2_vitb14",
        "embed_dim": 512,
        "freeze_backbone": False,
        "unfreeze_last_n_blocks": -1,
        "gradient_checkpointing": True,
    },
    "loss": {
        "name": "soft_infonce",
        "temperature": 0.07,
        "similarity_temperature": 0.1,
        "learnable_temperature": True,
    },
    "training": {
        "backbone_lr": 1e-5,
        "head_lr": 1e-3,
        "weight_decay": 0.05,
        "warmup_steps": 200,
        "total_steps": 10000,
        "batch_size": 128,
        "grad_accum_steps": 2,
        "use_bf16": True,
    },
    "data": {
        "data_dir": "data_cache/dynaclip_data",
        "num_pairs": 200000,
        "hard_neg_ratio": 0.3,
        "hard_pos_ratio": 0.3,
        "num_workers": 4,
    },
    "logging": {
        "log_every": 100,
        "eval_every": 5000,
        "save_every": 5000,
        "use_wandb": False,
        "project_name": "dynaclip",
    },
}

ABLATIONS = {
    "triplet": {
        "desc": "Triplet loss (margin=1.0)",
        "overrides": {
            "loss": {"name": "triplet", "temperature": 0.07, "similarity_temperature": 0.1, "learnable_temperature": False},
        },
    },
    "byol": {
        "desc": "BYOL loss (no negatives)",
        "overrides": {
            "loss": {"name": "byol", "temperature": 0.07, "similarity_temperature": 0.1, "learnable_temperature": False},
        },
    },
    "last2_blocks": {
        "desc": "Only unfreeze last 2 transformer blocks",
        "overrides": {
            "model": {"unfreeze_last_n_blocks": 2},
        },
    },
    "last4_blocks": {
        "desc": "Only unfreeze last 4 transformer blocks",
        "overrides": {
            "model": {"unfreeze_last_n_blocks": 4},
        },
    },
}


def merge_config(base, overrides):
    """Merge overrides into base config (deep)."""
    cfg = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def train_ablation(name, config, device="cuda:0"):
    """Train a single ablation."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION: {name} - {ABLATIONS[name]['desc']}")
    logger.info(f"{'='*60}")
    
    checkpoint_dir = f"checkpoints/ablation_{name}"
    log_dir = f"logs/ablation_{name}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if already done
    final_ckpt = Path(checkpoint_dir) / "dynaclip_final.pt"
    if final_ckpt.exists():
        logger.info(f"Ablation '{name}' already completed, skipping")
        return
    
    set_seed(42)
    
    # Model
    model_cfg = config["model"]
    model = DynaCLIPModel(
        backbone_name=model_cfg["backbone"],
        embed_dim=model_cfg["embed_dim"],
        freeze_backbone=model_cfg["freeze_backbone"],
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", -1),
    )
    if model_cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
    
    # Loss
    loss_cfg = config["loss"]
    loss_fn = build_loss(
        loss_cfg["name"],
        temperature=loss_cfg.get("temperature", 0.07),
        similarity_temperature=loss_cfg.get("similarity_temperature", 0.1),
        learnable_temperature=loss_cfg.get("learnable_temperature", True),
    )
    
    # Data
    data_cfg = config["data"]
    train_loader = create_contrastive_dataloader(
        data_dir=data_cfg["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=data_cfg["num_workers"],
        distributed=False,
        num_pairs=data_cfg["num_pairs"],
        hard_neg_ratio=data_cfg["hard_neg_ratio"],
        hard_pos_ratio=data_cfg["hard_pos_ratio"],
        seed=42,
    )
    
    # Trainer
    train_cfg = config["training"]
    trainer = DynaCLIPTrainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        backbone_lr=train_cfg["backbone_lr"],
        head_lr=train_cfg["head_lr"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=train_cfg["total_steps"],
        grad_accum_steps=train_cfg.get("grad_accum_steps", 1),
        use_bf16=train_cfg.get("use_bf16", True),
        checkpoint_dir=checkpoint_dir,
        use_wandb=False,
        device=device,
        local_rank=-1,
        log_every=config["logging"]["log_every"],
        eval_every=config["logging"]["eval_every"],
        save_every=config["logging"]["save_every"],
    )
    
    trainer.train()
    logger.info(f"Ablation '{name}' complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    for name in args.ablations:
        if name not in ABLATIONS:
            logger.warning(f"Unknown ablation: {name}")
            continue
        config = merge_config(BASE_CONFIG, ABLATIONS[name]["overrides"])
        train_ablation(name, config, device=args.device)
    
    logger.info("All ablations complete!")


if __name__ == "__main__":
    main()
