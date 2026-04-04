"""
DynaCLIP Pre-training Script.

Train DynaCLIP with Soft InfoNCE loss on physics-varied contrastive pairs.
Supports multi-GPU training via torchrun.

Usage:
  Single GPU:  python scripts/pretrain.py --config configs/pretrain.yaml
  Multi-GPU:   torchrun --nproc_per_node=2 scripts/pretrain.py --config configs/pretrain.yaml
"""

import argparse
import logging
import yaml

import torch

from dynaclip.models.dynaclip import DynaCLIPModel
from dynaclip.losses.contrastive import build_loss
from dynaclip.data.dataset import create_contrastive_dataloader
from dynaclip.trainers.pretrain import DynaCLIPTrainer
from dynaclip.utils.helpers import setup_logging, set_seed, setup_distributed, cleanup_distributed, count_parameters

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DynaCLIP Pre-training")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()

    # Logging
    setup_logging("INFO", "logs/pretrain.log", rank=rank)
    set_seed(args.seed + rank)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logger.info("=== DynaCLIP Pre-training ===")
    logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")

    # Device
    device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"

    # Model
    model_cfg = cfg["model"]
    model = DynaCLIPModel(
        backbone_name=model_cfg["backbone"],
        embed_dim=model_cfg["embed_dim"],
        freeze_backbone=model_cfg["freeze_backbone"],
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", -1),
    )
    # Enable gradient checkpointing to reduce GPU memory ~40%
    if model_cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
    logger.info(f"Model params: {count_parameters(model)}")

    # Loss
    loss_cfg = cfg["loss"]
    loss_fn = build_loss(
        loss_cfg["name"],
        temperature=loss_cfg.get("temperature", 0.07),
        similarity_temperature=loss_cfg.get("similarity_temperature", 0.1),
        learnable_temperature=loss_cfg.get("learnable_temperature", True),
    )

    # Data
    data_cfg = cfg["data"]
    train_loader = create_contrastive_dataloader(
        data_dir=data_cfg["data_dir"],
        batch_size=cfg["training"]["batch_size"] // max(world_size, 1),
        num_workers=data_cfg["num_workers"],
        distributed=(world_size > 1),
        num_pairs=data_cfg["num_pairs"],
        hard_neg_ratio=data_cfg["hard_neg_ratio"],
        hard_pos_ratio=data_cfg["hard_pos_ratio"],
        seed=args.seed,
    )

    # Trainer
    train_cfg = cfg["training"]
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
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        use_wandb=cfg["logging"].get("use_wandb", False),
        project_name=cfg["logging"].get("project_name", "dynaclip"),
        device=device,
        local_rank=local_rank,
        log_every=cfg["logging"]["log_every"],
        eval_every=cfg["logging"]["eval_every"],
        save_every=cfg["logging"]["save_every"],
        wiseft_alpha=train_cfg.get("wiseft_alpha", 0.0),
        use_physics_vectors=(loss_cfg["name"] in ("rnc", "pairwise_physics")),
    )

    # Resume
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")

    # Train
    trainer.train()

    # Cleanup
    cleanup_distributed()
    logger.info("Pre-training complete!")


if __name__ == "__main__":
    main()
