"""
Experiment 3: Downstream World Model (Dreamer-v3 style RSSM).

Train RSSM with each of 8 backbones as frozen visual encoder on ManiSkill3
push, pick-and-place, stack with physics variation (10K trajectories each).

Metrics:
  - Latent MSE at horizons t+1, t+5, t+10, t+20
  - Reconstruction SSIM and LPIPS
  - FVD over 16-frame clips
  - Object position L2 error
  - Physics violation rate
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RSSMWorldModel(nn.Module):
    """Dreamer-v3 style RSSM world model.

    - 512-dim stochastic state: 32 categoricals × 32 classes
    - 512-dim deterministic GRU state
    """

    def __init__(
        self,
        obs_dim: int = 768,
        action_dim: int = 7,
        stoch_size: int = 32,
        stoch_classes: int = 32,
        deter_size: int = 512,
        hidden_size: int = 512,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.stoch_size = stoch_size
        self.stoch_classes = stoch_classes
        self.deter_size = deter_size
        self.total_stoch = stoch_size * stoch_classes

        # Sequence model: GRU
        self.gru = nn.GRUCell(self.total_stoch + action_dim, deter_size)

        # Prior: p(s_t | h_t)
        self.prior = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.total_stoch),
        )

        # Posterior: q(s_t | h_t, o_t)
        self.posterior = nn.Sequential(
            nn.Linear(deter_size + obs_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self.total_stoch),
        )

        # Observation decoder
        self.obs_decoder = nn.Sequential(
            nn.Linear(deter_size + self.total_stoch, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, obs_dim),
        )

        # Reward decoder
        self.reward_decoder = nn.Sequential(
            nn.Linear(deter_size + self.total_stoch, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

        # Continue (termination) decoder
        self.continue_decoder = nn.Sequential(
            nn.Linear(deter_size + self.total_stoch, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device):
        return (
            torch.zeros(batch_size, self.deter_size, device=device),
            torch.zeros(batch_size, self.stoch_size, self.stoch_classes, device=device),
        )

    def observe(self, obs_embeds, actions, initial_state=None):
        """Process sequence with posterior (training).

        Args:
            obs_embeds: (B, T, obs_dim)
            actions: (B, T, action_dim)
        """
        B, T, _ = obs_embeds.shape
        device = obs_embeds.device

        if initial_state is None:
            deter, stoch = self.initial_state(B, device)
        else:
            deter, stoch = initial_state

        priors, posteriors = [], []
        deters, stochs = [], []

        for t in range(T):
            # GRU
            gru_input = torch.cat([stoch.flatten(1), actions[:, t]], dim=-1)
            deter = self.gru(gru_input, deter)

            # Prior
            prior_logits = self.prior(deter).reshape(B, self.stoch_size, self.stoch_classes)
            priors.append(prior_logits)

            # Posterior
            post_input = torch.cat([deter, obs_embeds[:, t]], dim=-1)
            post_logits = self.posterior(post_input).reshape(B, self.stoch_size, self.stoch_classes)
            posteriors.append(post_logits)

            # Sample from posterior (straight-through)
            post_dist = torch.distributions.OneHotCategorical(logits=post_logits)
            stoch = post_dist.sample() + post_dist.probs - post_dist.probs.detach()

            deters.append(deter)
            stochs.append(stoch)

        return {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
            "prior_logits": torch.stack(priors, dim=1),
            "posterior_logits": torch.stack(posteriors, dim=1),
        }

    def imagine(self, initial_state, actions):
        """Imagine future states using prior only (no observations).

        Args:
            initial_state: (deter, stoch)
            actions: (B, T, action_dim)
        """
        B, T, _ = actions.shape
        deter, stoch = initial_state
        device = deter.device

        deters, stochs = [], []
        for t in range(T):
            gru_input = torch.cat([stoch.flatten(1), actions[:, t]], dim=-1)
            deter = self.gru(gru_input, deter)

            prior_logits = self.prior(deter).reshape(B, self.stoch_size, self.stoch_classes)
            prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
            stoch = prior_dist.sample() + prior_dist.probs - prior_dist.probs.detach()

            deters.append(deter)
            stochs.append(stoch)

        return {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
        }

    def decode(self, deter, stoch):
        """Decode observations and rewards from state."""
        feat = torch.cat([deter, stoch.flatten(-2)], dim=-1)
        obs_pred = self.obs_decoder(feat)
        reward_pred = self.reward_decoder(feat)
        continue_pred = self.continue_decoder(feat)
        return obs_pred, reward_pred, continue_pred

    def compute_loss(self, obs_embeds, actions, rewards=None, continues=None):
        """Compute RSSM training loss."""
        results = self.observe(obs_embeds, actions)

        deter = results["deter"]
        stoch = results["stoch"]
        prior_logits = results["prior_logits"]
        posterior_logits = results["posterior_logits"]

        # Decode
        obs_pred, reward_pred, continue_pred = self.decode(deter, stoch)

        # Reconstruction loss
        recon_loss = F.mse_loss(obs_pred, obs_embeds)

        # KL divergence (between prior and posterior)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)
        post_dist = torch.distributions.OneHotCategorical(logits=posterior_logits)
        kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1).mean()

        # Reward loss
        if rewards is not None:
            reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards)
        else:
            reward_loss = torch.tensor(0.0, device=obs_embeds.device)

        total_loss = recon_loss + 0.1 * kl_loss + reward_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "reward_loss": reward_loss,
        }


class WorldModelEvaluator:
    """Evaluate world model prediction quality across backbones."""

    HORIZONS = [1, 5, 10, 20]

    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        test_loader,
        device: str = "cuda",
    ):
        self.backbones = backbones
        self.test_loader = test_loader
        self.device = device

    def train_world_model(
        self,
        backbone: nn.Module,
        train_loader,
        obs_dim: int,
        num_epochs: int = 50,
        lr: float = 3e-4,
    ) -> RSSMWorldModel:
        """Train RSSM world model with frozen visual backbone."""
        wm = RSSMWorldModel(obs_dim=obs_dim).to(self.device)
        optimizer = torch.optim.Adam(wm.parameters(), lr=lr)

        backbone = backbone.to(self.device).eval()

        for epoch in range(num_epochs):
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                images = batch["images"].to(self.device)  # (B, T, 3, H, W)
                actions = batch["actions"].to(self.device)  # (B, T, action_dim)

                B, T = images.shape[:2]
                with torch.no_grad():
                    imgs_flat = images.reshape(-1, *images.shape[2:])
                    obs_embeds = backbone(imgs_flat).reshape(B, T, -1)

                loss_dict = wm.compute_loss(obs_embeds, actions)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if epoch % 10 == 0:
                logger.info(f"  WM Epoch {epoch}: loss = {total_loss / max(n_batches, 1):.4f}")

        return wm

    @torch.no_grad()
    def evaluate_prediction_horizons(
        self,
        wm: RSSMWorldModel,
        backbone: nn.Module,
        test_loader,
    ) -> dict:
        """Evaluate latent prediction at multiple horizons."""
        wm.eval()
        backbone = backbone.to(self.device).eval()

        horizon_errors = {h: [] for h in self.HORIZONS}
        position_errors = {h: [] for h in self.HORIZONS}

        for batch in test_loader:
            images = batch["images"].to(self.device)
            actions = batch["actions"].to(self.device)

            B, T = images.shape[:2]
            imgs_flat = images.reshape(-1, *images.shape[2:])
            obs_embeds = backbone(imgs_flat).reshape(B, T, -1)

            # Get initial state from first few steps
            init_results = wm.observe(obs_embeds[:, :5], actions[:, :5])
            init_state = (init_results["deter"][:, -1], init_results["stoch"][:, -1])

            # Imagine future
            for h in self.HORIZONS:
                if T > 5 + h:
                    imagined = wm.imagine(init_state, actions[:, 5:5+h])
                    pred_obs, _, _ = wm.decode(
                        imagined["deter"][:, -1:],
                        imagined["stoch"][:, -1:],
                    )

                    target_obs = obs_embeds[:, 5+h-1:5+h]
                    latent_mse = F.mse_loss(pred_obs, target_obs).item()
                    horizon_errors[h].append(latent_mse)

        results = {}
        for h in self.HORIZONS:
            if horizon_errors[h]:
                results[f"latent_mse_t+{h}"] = float(np.mean(horizon_errors[h]))
            else:
                results[f"latent_mse_t+{h}"] = float("nan")

        return results

    def run(self, train_loader) -> dict:
        """Run full world model experiment."""
        logger.info("=== Experiment 3: Downstream World Model ===")
        results = {}

        for name, backbone in self.backbones.items():
            logger.info(f"--- World Model with {name} ---")
            obs_dim = getattr(backbone, 'output_dim',
                              getattr(backbone, '_output_dim', 768))

            wm = self.train_world_model(backbone, train_loader, obs_dim)
            metrics = self.evaluate_prediction_horizons(wm, backbone, self.test_loader)

            results[name] = metrics
            logger.info(f"  {name}: {metrics}")

        return results
