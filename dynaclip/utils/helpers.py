"""
DynaCLIP Utility Functions: Logging, seeding, distributed setup.
"""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    rank: int = 0,
):
    """Setup logging with optional file output."""
    handlers = []
    if rank == 0:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_file and rank == 0:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    return 0, -1, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": total / 1e6,
        "trainable_M": trainable / 1e6,
    }


def get_device(local_rank: int = -1) -> str:
    if local_rank >= 0:
        return f"cuda:{local_rank}"
    return "cuda" if torch.cuda.is_available() else "cpu"


class AverageMeter:
    """Rolling average tracker."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
