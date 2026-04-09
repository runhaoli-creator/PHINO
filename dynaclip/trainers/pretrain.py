"""
DynaCLIP Training Pipeline: Distributed pre-training with RnC / Soft InfoNCE.

Supports:
  - Multi-GPU training (DDP) via torchrun
  - bf16 mixed precision (NO GradScaler for bf16, only for fp16)
  - Cosine annealing with warmup
  - Separate LR for backbone (1e-5) and projection head (1e-3)
  - Learnable temperature (in loss function)
  - WiSE-FT feature-space regularization (prevents embedding distortion)
  - WandB logging
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            factor = self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(base_lr * factor, self.min_lr)

    def get_lr(self) -> list:
        return [pg["lr"] for pg in self.optimizer.param_groups]


class DynaCLIPTrainer:
    """Main pre-training loop for DynaCLIP."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        total_steps: int = 100_000,
        grad_accum_steps: int = 1,
        log_every: int = 100,
        eval_every: int = 5000,
        save_every: int = 10000,
        checkpoint_dir: str = "checkpoints",
        use_bf16: bool = True,
        use_wandb: bool = True,
        project_name: str = "dynaclip",
        run_name: Optional[str] = None,
        device: str = "cuda",
        local_rank: int = -1,
        # WiSE-FT feature-space regularization
        wiseft_alpha: float = 0.0,  # 0 = disabled, >0 = regularize toward frozen backbone
        use_physics_vectors: bool = False,  # Whether loss expects physics vectors
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_rank = local_rank
        self.use_bf16 = use_bf16
        self.use_wandb = use_wandb
        self.grad_accum_steps = grad_accum_steps
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.total_steps = total_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model = self.model.to(device)
        # Also move loss_fn (it may have learnable params like temperature)
        self.loss_fn = self.loss_fn.to(device)

        # DDP wrapping
        if local_rank >= 0:
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # Optimizer with separate LR for backbone and head + loss params
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        param_groups = raw_model.get_param_groups(backbone_lr=backbone_lr, head_lr=head_lr)
        # Add loss function's learnable parameters (e.g., temperature)
        loss_params = list(self.loss_fn.parameters())
        if loss_params:
            param_groups.append({"params": loss_params, "lr": head_lr})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # Mixed precision: bf16 does NOT use GradScaler (that's only for fp16)
        self.use_scaler = not use_bf16  # Only use scaler for fp16
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        self.amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # WandB
        if use_wandb and (local_rank <= 0):
            try:
                import wandb
                wandb.init(
                    project=project_name,
                    name=run_name or f"dynaclip_{time.strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "backbone_lr": backbone_lr,
                        "head_lr": head_lr,
                        "weight_decay": weight_decay,
                        "warmup_steps": warmup_steps,
                        "total_steps": total_steps,
                        "use_bf16": use_bf16,
                        "grad_accum_steps": grad_accum_steps,
                    },
                )
            except Exception:
                self.use_wandb = False

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.use_physics_vectors = use_physics_vectors
        self.wiseft_alpha = wiseft_alpha

        # WiSE-FT: store frozen copy of backbone for feature-space regularization
        if wiseft_alpha > 0:
            import copy
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            self.frozen_backbone = copy.deepcopy(raw_model.backbone)
            self.frozen_backbone.eval()
            for p in self.frozen_backbone.parameters():
                p.requires_grad = False
            self.frozen_backbone = self.frozen_backbone.to(device)
            logger.info(f"WiSE-FT enabled: alpha={wiseft_alpha}, frozen backbone stored")
        else:
            self.frozen_backbone = None

    def train(self):
        """Run full pre-training loop."""
        logger.info(f"Starting DynaCLIP pre-training for {self.total_steps} steps")
        logger.info(f"  bf16={self.use_bf16}, scaler={self.use_scaler}, "
                     f"grad_accum={self.grad_accum_steps}")
        self.model.train()
        self.loss_fn.train()

        data_iter = iter(self.train_loader)
        running_loss = 0.0
        micro_step = 0
        start_time = time.time()

        while self.global_step < self.total_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            loss_dict = self._train_step(batch, micro_step)
            micro_step += 1

            # Only count global_step on actual optimizer steps
            if micro_step % self.grad_accum_steps == 0:
                self.global_step += 1
                running_loss += loss_dict["loss"]

                if self.global_step % self.log_every == 0:
                    avg_loss = running_loss / self.log_every
                    elapsed = time.time() - start_time
                    steps_per_sec = self.log_every / max(elapsed, 1e-8)

                    lr_str = ", ".join(f"{lr:.2e}" for lr in self.scheduler.get_lr())
                    temp = loss_dict.get("temperature", 0.07)
                    logger.info(
                        f"Step {self.global_step}/{self.total_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: [{lr_str}] | "
                        f"Temp: {temp:.4f} | {steps_per_sec:.1f} steps/s"
                    )

                    if self.use_wandb and self.local_rank <= 0:
                        try:
                            import wandb
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/temperature": temp,
                                "train/lr_backbone": self.scheduler.get_lr()[0],
                                "train/lr_head": self.scheduler.get_lr()[1],
                                "train/steps_per_sec": steps_per_sec,
                                "step": self.global_step,
                            })
                        except Exception:
                            pass

                    running_loss = 0.0
                    start_time = time.time()

                # Validation
                if self.val_loader and self.global_step % self.eval_every == 0:
                    val_loss = self._validate()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                    self.model.train()
                    self.loss_fn.train()

                # Checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

        # Final save
        self._save_checkpoint("final")
        logger.info("Pre-training complete!")

    def _train_step(self, batch: dict, micro_step: int) -> dict:
        """Single training step with proper bf16/fp16 handling."""
        img_i = batch["img_i"].to(self.device, non_blocking=True)
        img_j = batch["img_j"].to(self.device, non_blocking=True)
        dyn_sim = batch["dynamics_similarity"].to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=self.amp_dtype):
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            z_i, z_j = raw_model.encode_pair(img_i, img_j)

            # Route to appropriate loss interface
            if self.use_physics_vectors:
                physics_i = batch["physics_i"].to(self.device, non_blocking=True)
                physics_j = batch["physics_j"].to(self.device, non_blocking=True)
                loss_dict = self.loss_fn(z_i, z_j, physics_i, physics_j, dyn_sim)
            else:
                loss_dict = self.loss_fn(z_i, z_j, dyn_sim)

            # WiSE-FT: feature-space regularization toward frozen DINOv2
            if self.frozen_backbone is not None and self.wiseft_alpha > 0:
                with torch.no_grad():
                    frozen_out = self.frozen_backbone.forward_features(img_i)
                    if isinstance(frozen_out, dict):
                        frozen_cls = frozen_out.get("x_norm_clstoken", None)
                        frozen_patch = frozen_out.get("x_norm_patchtokens", None)
                        if frozen_cls is None or frozen_patch is None:
                            x = frozen_out.get("x", None)
                            if x is not None:
                                frozen_cls = x[:, 0]
                                frozen_patch = x[:, 1:]
                            else:
                                raise RuntimeError(f"Unexpected frozen DINOv2 output keys: {frozen_out.keys()}")
                    else:
                        frozen_cls = frozen_out[:, 0]
                        frozen_patch = frozen_out[:, 1:]
                    frozen_feats = torch.cat([frozen_cls, frozen_patch.mean(dim=1)], dim=-1)
                    frozen_feats = frozen_feats.to(dtype=self.amp_dtype)  # Match AMP dtype

                live_feats = raw_model.extract_features(img_i)
                reg_loss = F.mse_loss(live_feats, frozen_feats)
                loss_dict["loss"] = loss_dict["loss"] + self.wiseft_alpha * reg_loss
                loss_dict["reg_loss"] = reg_loss.detach()

        loss = loss_dict["loss"] / self.grad_accum_steps

        if self.use_scaler:
            # fp16 path: use GradScaler
            self.scaler.scale(loss).backward()
            if (micro_step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if hasattr(self.loss_fn, 'parameters'):
                    nn.utils.clip_grad_norm_(self.loss_fn.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            # bf16 path: no GradScaler needed
            loss.backward()
            if (micro_step + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if hasattr(self.loss_fn, 'parameters'):
                    nn.utils.clip_grad_norm_(self.loss_fn.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.eval()
        self.loss_fn.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            img_i = batch["img_i"].to(self.device)
            img_j = batch["img_j"].to(self.device)
            dyn_sim = batch["dynamics_similarity"].to(self.device)

            with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                z_i, z_j = raw_model.encode_pair(img_i, img_j)
                loss_dict = self.loss_fn(z_i, z_j, dyn_sim)
            total_loss += loss_dict["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Validation loss: {avg_loss:.4f}")

        if self.use_wandb and self.local_rank <= 0:
            try:
                import wandb
                wandb.log({"val/loss": avg_loss, "step": self.global_step})
            except Exception:
                pass

        return avg_loss

    def _save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        if self.local_rank > 0:
            return

        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt = {
            "global_step": self.global_step,
            "model_state_dict": raw_model.state_dict(),
            "loss_state_dict": self.loss_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step_count": self.scheduler.step_count,
            "best_val_loss": self.best_val_loss,
        }
        path = self.checkpoint_dir / f"dynaclip_{tag}.pt"
        torch.save(ckpt, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if "loss_state_dict" in ckpt:
            self.loss_fn.load_state_dict(ckpt["loss_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        # Restore scheduler state
        if "scheduler_step_count" in ckpt:
            self.scheduler.step_count = ckpt["scheduler_step_count"]
        else:
            # Old checkpoint without scheduler state: fast-forward
            self.scheduler.step_count = self.global_step
        # Recompute LR from restored step
        for pg, base_lr in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
            if self.scheduler.step_count <= self.scheduler.warmup_steps:
                factor = self.scheduler.step_count / self.scheduler.warmup_steps
            else:
                progress = (self.scheduler.step_count - self.scheduler.warmup_steps) / max(
                    self.scheduler.total_steps - self.scheduler.warmup_steps, 1)
                progress = min(progress, 1.0)
                factor = 0.5 * (1 + np.cos(np.pi * progress))
            pg["lr"] = max(base_lr * factor, self.scheduler.min_lr)
        logger.info(f"Loaded checkpoint from step {self.global_step}, "
                    f"scheduler at step {self.scheduler.step_count}")
