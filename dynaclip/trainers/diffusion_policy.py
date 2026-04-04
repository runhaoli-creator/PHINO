"""
Diffusion Policy Trainer: 16-step action chunks with DDPM/DDIM.

Primary policy baseline for all downstream evaluations.
Uses frozen visual backbone + diffusion model for action generation.
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------------------------------------------------------------------------
# 1D U-Net for action denoising
# ---------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=2),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, 5, padding=2),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.blocks(x)
        # Add condition
        cond_emb = self.cond_proj(cond).unsqueeze(-1)
        h = h + cond_emb
        return h + self.residual(x)


class ConditionalUNet1D(nn.Module):
    """1D U-Net for denoising action sequences, conditioned on visual obs + timestep."""

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        obs_dim: int = 768,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        # Condition projection (obs features + timestep)
        cond_dim = obs_dim + diffusion_step_embed_dim

        # Encoder
        in_ch = action_dim
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        for dim in down_dims:
            self.encoder_blocks.append(ConditionalResBlock1D(in_ch, dim, cond_dim))
            self.downsample.append(nn.Conv1d(dim, dim, 3, stride=2, padding=1))
            in_ch = dim

        # Bottleneck
        self.mid_block = ConditionalResBlock1D(down_dims[-1], down_dims[-1], cond_dim)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, dim in enumerate(reversed(down_dims)):
            skip_dim = dim
            in_dim = down_dims[-1] if i == 0 else down_dims[len(down_dims) - i]
            self.decoder_blocks.append(ConditionalResBlock1D(in_dim + skip_dim, dim, cond_dim))
            self.upsample.append(nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1))

        # Output
        self.final = nn.Sequential(
            nn.Conv1d(down_dims[0], down_dims[0], 5, padding=2),
            nn.GroupNorm(8, down_dims[0]),
            nn.Mish(),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        noisy_action: torch.Tensor,   # (B, action_dim, action_horizon)
        timestep: torch.Tensor,        # (B,)
        obs_features: torch.Tensor,    # (B, obs_dim)
    ) -> torch.Tensor:
        # Timestep embedding
        t_emb = self.time_embed(timestep)

        # Condition = obs + timestep
        cond = torch.cat([obs_features, t_emb], dim=-1)

        # Encoder
        x = noisy_action
        skips = []
        for block, down in zip(self.encoder_blocks, self.downsample):
            x = block(x, cond)
            skips.append(x)
            x = down(x)

        # Mid
        x = self.mid_block(x, cond)

        # Decoder
        for block, up, skip in zip(self.decoder_blocks, self.upsample, reversed(skips)):
            x = up(x)
            # Handle size mismatch
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
            x = torch.cat([x, skip], dim=1)
            x = block(x, cond)

        return self.final(x)


# ---------------------------------------------------------------------------
# Diffusion Policy
# ---------------------------------------------------------------------------
class DiffusionPolicy(nn.Module):
    """Diffusion Policy with visual backbone.

    DDPM (100 steps) for training, DDIM (10 steps) for inference.
    """

    def __init__(
        self,
        visual_backbone: nn.Module,
        action_dim: int = 7,
        action_horizon: int = 16,
        obs_horizon: int = 2,
        num_train_timesteps: int = 100,
        num_inference_timesteps: int = 10,
        obs_dim: Optional[int] = None,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.visual_backbone = visual_backbone
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps

        # Freeze visual backbone
        for param in self.visual_backbone.parameters():
            param.requires_grad = False
        self.visual_backbone.eval()

        # Get obs dimension
        if obs_dim is None:
            obs_dim = getattr(visual_backbone, 'output_dim', 768)
        if hasattr(visual_backbone, '_output_dim'):
            obs_dim = visual_backbone._output_dim

        # Observation encoder (aggregate multi-step obs)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 768),
        )

        # Noise prediction network
        self.noise_pred_net = ConditionalUNet1D(
            action_dim=action_dim,
            action_horizon=action_horizon,
            obs_dim=768,
        )

        # Noise schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_train_timesteps)
        else:
            betas = linear_beta_schedule(num_train_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def encode_obs(self, images: torch.Tensor) -> torch.Tensor:
        """Encode observation images through frozen backbone.

        Args:
            images: (B, obs_horizon, 3, H, W)
        Returns:
            obs_features: (B, 768)
        """
        B, T = images.shape[:2]
        images_flat = rearrange(images, "b t c h w -> (b t) c h w")

        with torch.no_grad():
            features = self.visual_backbone(images_flat)

        features = rearrange(features, "(b t) d -> b (t d)", b=B, t=T)
        return self.obs_encoder(features)

    def forward(
        self,
        images: torch.Tensor,     # (B, obs_horizon, 3, H, W)
        actions: torch.Tensor,    # (B, action_horizon, action_dim)
    ) -> dict:
        """Training forward: add noise to actions and predict it.

        Returns dict with 'loss'.
        """
        B = actions.shape[0]
        device = actions.device

        # Encode observations
        obs_features = self.encode_obs(images)

        # Sample timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(actions)
        actions_t = rearrange(actions, "b t d -> b d t")
        noise_t = rearrange(noise, "b t d -> b d t")

        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(B, 1, 1)
        noisy_actions = sqrt_alpha * actions_t + sqrt_one_minus_alpha * noise_t

        # Predict noise
        noise_pred = self.noise_pred_net(noisy_actions, timesteps.float(), obs_features)

        loss = F.mse_loss(noise_pred, noise_t)
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(self, images: torch.Tensor) -> torch.Tensor:
        """DDIM inference: generate action sequence from observation.

        Args:
            images: (B, obs_horizon, 3, H, W)
        Returns:
            actions: (B, action_horizon, action_dim)
        """
        B = images.shape[0]
        device = images.device

        obs_features = self.encode_obs(images)

        # Start from noise
        x = torch.randn(B, self.action_dim, self.action_horizon, device=device)

        # DDIM sampling
        step_ratio = self.num_train_timesteps // self.num_inference_timesteps
        timesteps = torch.arange(0, self.num_train_timesteps, step_ratio).flip(0)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.float)
            noise_pred = self.noise_pred_net(x, t_batch, obs_features)

            alpha_t = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0)

            # DDIM update
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            pred_x0 = pred_x0.clamp(-1, 1)
            x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred

        return rearrange(x, "b d t -> b t d")


# ---------------------------------------------------------------------------
# Diffusion Policy Trainer
# ---------------------------------------------------------------------------
class DiffusionPolicyTrainer:
    """Train Diffusion Policy on demonstration data."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        num_epochs: int = 500,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints/diffusion_policy",
        use_wandb: bool = True,
    ):
        self.policy = policy.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb

        self.optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
        )

    def train(self):
        logger.info(f"Training Diffusion Policy for {self.num_epochs} epochs")
        best_loss = float("inf")

        for epoch in range(self.num_epochs):
            self.policy.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in self.train_loader:
                images = batch["images"].to(self.device)
                actions = batch["actions"].to(self.device)

                loss_dict = self.policy(images, actions)
                loss = loss_dict["loss"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            self.scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}/{self.num_epochs} | Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    self.policy.state_dict(),
                    self.checkpoint_dir / "best.pt",
                )

        logger.info(f"Diffusion Policy training complete. Best loss: {best_loss:.4f}")


from pathlib import Path
from torch.utils.data import DataLoader
