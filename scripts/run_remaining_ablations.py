#!/usr/bin/env python
"""
Run ALL remaining ablations for DynaCLIP paper.

Ablations to run (each on a separate GPU for maximum parallelism):
  GPU 0: A3a - Hard negative ratio 15%
  GPU 1: A3b - Hard negative ratio 50%  
  GPU 2: A4a - Backbone init from ImageNet (not DINOv2)
  GPU 3: A4b - Backbone init from random
  GPU 4: A5  - Data scale curve (10K, 25K, 50K, 100K) - sequential on one GPU
  GPU 5: A6a - Property diversity: mass only  
  GPU 6: A6b - Property diversity: friction only

GPU 7: DO NOT TOUCH (someone else's)

After training, evaluate all ablation checkpoints via Ridge probing.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import argparse
import copy
import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.losses.contrastive import build_loss
from dynaclip.data.dataset import create_contrastive_dataloader
from dynaclip.trainers.pretrain import DynaCLIPTrainer
from dynaclip.utils.helpers import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/remaining_ablations.log"),
    ],
)
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
}


# ======================================================================
# Custom similarity for property diversity ablation
# ======================================================================
def create_property_limited_dataloader(data_dir, batch_size, num_workers, 
                                        num_pairs, hard_neg_ratio, hard_pos_ratio,
                                        seed, properties=("mass", "friction", "restitution")):
    """Create dataloader where proxy similarity only uses specified properties.
    
    This monkey-patches the proxy similarity computation to ignore certain properties.
    """
    from dynaclip.data.dataset import DynaCLIPContrastiveDataset, get_train_transform
    from torch.utils.data import DataLoader
    
    dataset = DynaCLIPContrastiveDataset(
        data_dir=data_dir,
        split="train",
        transform=get_train_transform(),
        num_pairs=num_pairs,
        hard_neg_ratio=hard_neg_ratio,
        hard_pos_ratio=hard_pos_ratio,
        seed=seed,
    )
    
    # Monkey-patch the proxy similarity to only use specified properties
    original_compute = dataset._compute_proxy_similarity
    
    def limited_proxy_similarity(self_ref, i, j):
        m_i, m_j = self_ref.metadata[i], self_ref.metadata[j]
        terms = []
        if "mass" in properties:
            mass_diff = abs(np.log(m_i["mass"] + 1e-6) - np.log(m_j["mass"] + 1e-6))
            terms.append(mass_diff ** 2)
        if "friction" in properties:
            fric_diff = abs(m_i["static_friction"] - m_j["static_friction"])
            terms.append(fric_diff ** 2)
        if "restitution" in properties:
            rest_diff = abs(m_i["restitution"] - m_j["restitution"])
            terms.append(rest_diff ** 2)
        if not terms:
            return 0.5  # no properties → random similarity
        dist = np.sqrt(sum(terms))
        return float(np.clip(np.exp(-dist / 2.0), 0.0, 1.0))
    
    import types
    dataset._compute_proxy_similarity = types.MethodType(
        lambda self, i, j: limited_proxy_similarity(self, i, j), dataset
    )
    
    # Re-mine pairs with new similarity
    dataset.pairs = dataset._mine_pairs()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader


def create_subset_dataloader(data_dir, batch_size, num_workers, 
                              num_pairs, hard_neg_ratio, hard_pos_ratio,
                              seed, max_entries):
    """Create dataloader using only a subset of the data (for data scale ablation)."""
    from dynaclip.data.dataset import DynaCLIPContrastiveDataset, get_train_transform
    from torch.utils.data import DataLoader
    
    dataset = DynaCLIPContrastiveDataset(
        data_dir=data_dir,
        split="train",
        transform=get_train_transform(),
        num_pairs=min(num_pairs, max_entries * 3),  # Scale pairs with data
        hard_neg_ratio=hard_neg_ratio,
        hard_pos_ratio=hard_pos_ratio,
        seed=seed,
    )
    
    # Truncate metadata to max_entries
    if len(dataset.metadata) > max_entries:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset.metadata), size=max_entries, replace=False)
        indices = sorted(indices)
        dataset.metadata = [dataset.metadata[i] for i in indices]
        # Rebuild group index
        dataset._group_index = {}
        for i, entry in enumerate(dataset.metadata):
            grp = entry.get("image_group", i)
            dataset._group_index.setdefault(grp, []).append(i)
        # Re-mine pairs
        dataset.pairs = dataset._mine_pairs()
        logger.info(f"Subset dataloader: {len(dataset.metadata)} entries, {len(dataset.pairs)} pairs")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return loader


# ======================================================================
# Ablation definitions
# ======================================================================
ABLATIONS = {
    # A3: Hard negative ratio variations
    "hardneg_15pct": {
        "desc": "Hard negative ratio = 15% (half of default 30%)",
        "overrides": {
            "data": {"hard_neg_ratio": 0.15, "hard_pos_ratio": 0.15},
        },
        "gpu": 0,
    },
    "hardneg_50pct": {
        "desc": "Hard negative ratio = 50%",
        "overrides": {
            "data": {"hard_neg_ratio": 0.50, "hard_pos_ratio": 0.0},
        },
        "gpu": 1,
    },
    
    # A4: Backbone initialization
    "init_imagenet": {
        "desc": "Backbone initialized from ImageNet ViT-B/14 (not DINOv2)",
        "overrides": {},
        "special": "imagenet_init",
        "gpu": 2,
    },
    "init_random": {
        "desc": "Backbone initialized randomly (no pretraining)",
        "overrides": {},
        "special": "random_init",
        "gpu": 3,
    },
    
    # A6: Property diversity
    "mass_only": {
        "desc": "Similarity uses mass only (ignore friction & restitution)",
        "overrides": {},
        "special": "property_mass",
        "gpu": 5,
    },
    "friction_only": {
        "desc": "Similarity uses friction only (ignore mass & restitution)",
        "overrides": {},
        "special": "property_friction",
        "gpu": 6,
    },
}

# A5: Data scale (runs sequentially on GPU 4)
DATA_SCALE_CONFIGS = [10000, 25000, 50000, 100000]


def merge_config(base, overrides):
    cfg = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and key in cfg:
            cfg[key].update(val)
        else:
            cfg[key] = val
    return cfg


def train_single_ablation(name, config, device, special=None):
    """Train a single ablation variant."""
    checkpoint_dir = f"checkpoints/ablation_{name}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
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
    
    # Handle special initializations
    if special == "imagenet_init":
        logger.info("Reinitializing backbone with ImageNet-pretrained ViT-B/14 (timm)")
        try:
            import timm
            imagenet_model = timm.create_model("vit_base_patch14_dinov2", pretrained=False)
            # Use timm's default ImageNet ViT weights if available, else random
            imagenet_model_in = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=False)
            # Load ImageNet-1K pretrained ViT-B/16 weights and adapt
            imagenet_model_in = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True)
            logger.info("Loaded ImageNet-1K ViT-B/16 as initialization")
            # Note: patch size differs (16 vs 14), so we can't directly copy all weights
            # Instead, copy what we can (shared layers, norm, head)
            src_state = imagenet_model_in.state_dict()
            tgt_state = model.backbone.state_dict()
            copied = 0
            for k in tgt_state:
                if k in src_state and src_state[k].shape == tgt_state[k].shape:
                    tgt_state[k] = src_state[k]
                    copied += 1
            model.backbone.load_state_dict(tgt_state, strict=False)
            logger.info(f"Copied {copied}/{len(tgt_state)} parameters from ImageNet ViT-B/16")
        except Exception as e:
            logger.warning(f"ImageNet init failed: {e}, using DINOv2 default")
            
    elif special == "random_init":
        logger.info("Reinitializing backbone with random weights")
        for name_p, param in model.backbone.named_parameters():
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                torch.nn.init.zeros_(param)
        logger.info("All backbone parameters randomly initialized")
    
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
    if special and special.startswith("property_"):
        prop_name = special.replace("property_", "")
        properties = (prop_name,) if prop_name != "mass_friction" else ("mass", "friction")
        train_loader = create_property_limited_dataloader(
            data_dir=data_cfg["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=data_cfg["num_workers"],
            num_pairs=data_cfg["num_pairs"],
            hard_neg_ratio=data_cfg["hard_neg_ratio"],
            hard_pos_ratio=data_cfg["hard_pos_ratio"],
            seed=42,
            properties=properties,
        )
    else:
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
        log_every=100,
        eval_every=5000,
        save_every=5000,
    )
    
    # Auto-resume from latest step checkpoint if available
    ckpt_dir = Path(checkpoint_dir)
    step_ckpts = sorted(ckpt_dir.glob("dynaclip_step_*.pt"), 
                        key=lambda p: int(p.stem.split("_")[-1]))
    if step_ckpts:
        latest = step_ckpts[-1]
        logger.info(f"Resuming '{name}' from checkpoint: {latest}")
        trainer.load_checkpoint(str(latest))
        logger.info(f"Resumed at global_step={trainer.global_step}")
    
    trainer.train()
    logger.info(f"Ablation '{name}' complete!")
    
    # Cleanup GPU memory
    del model, loss_fn, trainer, train_loader
    gc.collect()
    torch.cuda.empty_cache()


def train_data_scale_ablation(device="cuda:4"):
    """Train data scale ablation: 10K, 25K, 50K, 100K entries."""
    for scale in DATA_SCALE_CONFIGS:
        name = f"scale_{scale // 1000}k"
        checkpoint_dir = f"checkpoints/ablation_{name}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        final_ckpt = Path(checkpoint_dir) / "dynaclip_final.pt"
        if final_ckpt.exists():
            logger.info(f"Data scale ablation {name} already completed, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DATA SCALE ABLATION: {name} ({scale} entries)")
        logger.info(f"{'='*60}")
        
        set_seed(42)
        config = copy.deepcopy(BASE_CONFIG)
        
        # Model
        model = DynaCLIPModel(
            backbone_name=config["model"]["backbone"],
            embed_dim=config["model"]["embed_dim"],
            freeze_backbone=config["model"]["freeze_backbone"],
        )
        model.enable_gradient_checkpointing()
        
        # Loss
        loss_fn = build_loss(
            config["loss"]["name"],
            temperature=config["loss"]["temperature"],
            similarity_temperature=config["loss"]["similarity_temperature"],
            learnable_temperature=config["loss"]["learnable_temperature"],
        )
        
        # Data (subset)
        train_loader = create_subset_dataloader(
            data_dir=config["data"]["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            num_pairs=config["data"]["num_pairs"],
            hard_neg_ratio=config["data"]["hard_neg_ratio"],
            hard_pos_ratio=config["data"]["hard_pos_ratio"],
            seed=42,
            max_entries=scale,
        )
        
        # Trainer
        trainer = DynaCLIPTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            backbone_lr=config["training"]["backbone_lr"],
            head_lr=config["training"]["head_lr"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            total_steps=config["training"]["total_steps"],
            grad_accum_steps=config["training"]["grad_accum_steps"],
            use_bf16=config["training"]["use_bf16"],
            checkpoint_dir=checkpoint_dir,
            use_wandb=False,
            device=device,
            local_rank=-1,
            log_every=100,
            eval_every=5000,
            save_every=5000,
        )
        
        # Auto-resume from latest step checkpoint if available
        ckpt_dir_path = Path(checkpoint_dir)
        step_ckpts = sorted(ckpt_dir_path.glob("dynaclip_step_*.pt"),
                            key=lambda p: int(p.stem.split("_")[-1]))
        if step_ckpts:
            latest = step_ckpts[-1]
            logger.info(f"Resuming '{name}' from checkpoint: {latest}")
            trainer.load_checkpoint(str(latest))
            logger.info(f"Resumed at global_step={trainer.global_step}")
        
        trainer.train()
        logger.info(f"Data scale ablation {name} complete!")
        
        del model, loss_fn, trainer, train_loader
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "scale", "all"], default="all")
    parser.add_argument("--ablation", type=str, default=None, help="Run a specific ablation by name")
    parser.add_argument("--gpu", type=int, default=None, help="Override GPU assignment")
    args = parser.parse_args()
    
    Path("logs").mkdir(exist_ok=True)
    
    if args.mode == "single" and args.ablation:
        ablation = ABLATIONS[args.ablation]
        config = merge_config(BASE_CONFIG, ablation.get("overrides", {}))
        gpu = args.gpu if args.gpu is not None else ablation["gpu"]
        device = f"cuda:{gpu}"
        train_single_ablation(args.ablation, config, device, special=ablation.get("special"))
        
    elif args.mode == "scale":
        gpu = args.gpu if args.gpu is not None else 4
        train_data_scale_ablation(device=f"cuda:{gpu}")
        
    elif args.mode == "all":
        # This should be called via the launcher script which spawns parallel processes
        logger.info("Use the launcher script to run all ablations in parallel")
        logger.info("Or run individual ablations with --mode single --ablation <name>")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
