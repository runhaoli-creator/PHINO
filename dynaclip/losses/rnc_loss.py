"""
Rank-N-Contrast Loss for DynaCLIP v2.

Implements the RnC framework (Zha et al., NeurIPS 2023 Spotlight) adapted for
physics-grounded contrastive learning with continuous regression targets.

Instead of the ad-hoc geometric mean approximation, we use actual pairwise
physics distances to define a ranking-based contrastive objective that
provably preserves target ordering in the representation space.

Reference: arXiv:2210.01189
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RnCLoss(nn.Module):
    """Rank-N-Contrast loss for continuous physics-aware representation learning.

    For each anchor, every other sample in the batch acts as a potential positive.
    The loss ranks samples by their physics distance to the anchor and ensures
    the representation similarity follows the same ordering.

    This replaces SoftInfoNCELoss with a theoretically grounded alternative.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        label_diff: str = "l1",
        feature_sim: str = "cosine",
        learnable_temperature: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.label_diff = label_diff
        self.feature_sim = feature_sim

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.1, max=10.0)

    def _compute_label_diffs(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute pairwise label distances. labels: (N, D) -> (N, N)."""
        if self.label_diff == "l1":
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        elif self.label_diff == "l2":
            return (labels[:, None, :] - labels[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(f"Unknown label_diff: {self.label_diff}")

    def _compute_feature_sims(self, features: torch.Tensor) -> torch.Tensor:
        """Compute pairwise feature similarities. features: (N, D) -> (N, N)."""
        if self.feature_sim == "cosine":
            features = F.normalize(features, dim=-1)
            return torch.matmul(features, features.T)
        elif self.feature_sim == "l2":
            return -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(f"Unknown feature_sim: {self.feature_sim}")

    def forward(
        self,
        z_i: torch.Tensor,           # (B, D) L2-normalized embeddings view 1
        z_j: torch.Tensor,           # (B, D) L2-normalized embeddings view 2
        physics_i: torch.Tensor,     # (B, 3) [log_mass, friction, restitution] for view 1
        physics_j: torch.Tensor,     # (B, 3) [log_mass, friction, restitution] for view 2
        dynamics_sim: torch.Tensor = None,  # (B,) unused, kept for API compat
    ) -> dict:
        """Compute RnC loss over the batch.

        We concatenate both views to get 2B samples, then apply the
        ranking-based contrastive loss across all pairs.
        """
        B = z_i.shape[0]

        # Concatenate both views: (2B, D)
        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([physics_i, physics_j], dim=0)  # (2B, 3)

        n = features.shape[0]  # 2B

        # Pairwise label differences: (2B, 2B)
        label_diffs = self._compute_label_diffs(labels)

        # Pairwise feature similarities / tau: (2B, 2B)
        logits = self._compute_feature_sims(features) / self.temperature

        # Numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = logits.exp()

        # Remove diagonal (self-similarity)
        diag_mask = (1 - torch.eye(n, device=logits.device)).bool()
        logits = logits.masked_select(diag_mask).view(n, n - 1)
        exp_logits = exp_logits.masked_select(diag_mask).view(n, n - 1)
        label_diffs = label_diffs.masked_select(diag_mask).view(n, n - 1)

        # RnC loss: for each position k in the sorted order,
        # the denominator includes only samples with >= label distance
        loss = torch.tensor(0.0, device=features.device)
        for k in range(n - 1):
            pos_logits = logits[:, k]        # (2B,)
            pos_label_diffs = label_diffs[:, k]  # (2B,)

            # Negatives: all samples with label_diff >= current sample's label_diff
            neg_mask = (label_diffs >= pos_label_diffs.unsqueeze(1)).float()  # (2B, n-1)

            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1) + 1e-8)
            loss = loss - (pos_log_probs / (n * (n - 1))).sum()

        return {
            "loss": loss,
            "temperature": self.temperature.detach(),
            "logits_mean": torch.tensor(0.0),  # placeholder
        }


class PairwisePhysicsRnCLoss(nn.Module):
    """Simplified RnC loss using actual pairwise physics distances as soft targets.

    Instead of the full ranking formulation, this computes actual pairwise
    physics distances between all batch elements and uses them as soft
    contrastive weights. This is more computationally efficient than full RnC
    while still being principled (uses actual physics vectors, not geometric mean).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        similarity_temperature: float = 0.1,
        learnable_temperature: bool = True,
        physics_sigma: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.similarity_temperature = similarity_temperature
        self.physics_sigma = physics_sigma

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(
        self,
        z_i: torch.Tensor,           # (B, D) L2-normalized
        z_j: torch.Tensor,           # (B, D) L2-normalized
        physics_i: torch.Tensor,     # (B, 3)
        physics_j: torch.Tensor,     # (B, 3)
        dynamics_sim: torch.Tensor = None,
    ) -> dict:
        B = z_i.shape[0]

        # Compute logits: (B, B) cross-view similarity
        logits = torch.matmul(z_i, z_j.T) / self.temperature

        # Build actual pairwise physics similarity matrix (B, B)
        # physics_i[a] vs physics_j[b] for all pairs
        physics_diff = (physics_i.unsqueeze(1) - physics_j.unsqueeze(0))  # (B, B, 3)
        physics_dist = physics_diff.norm(dim=-1)  # (B, B)
        sim_matrix = torch.exp(-physics_dist / self.physics_sigma)  # (B, B)

        # Soft labels from actual pairwise similarities
        soft_labels = F.softmax(sim_matrix / self.similarity_temperature, dim=-1)

        # Cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(soft_labels * log_probs, dim=-1).mean()

        # Symmetric
        logits_t = logits.T
        sim_matrix_t = sim_matrix.T
        soft_labels_t = F.softmax(sim_matrix_t / self.similarity_temperature, dim=-1)
        log_probs_t = F.log_softmax(logits_t, dim=-1)
        loss_t = -torch.sum(soft_labels_t * log_probs_t, dim=-1).mean()

        total_loss = (loss + loss_t) / 2

        return {
            "loss": total_loss,
            "loss_i2j": loss,
            "loss_j2i": loss_t,
            "temperature": self.temperature.detach(),
            "logits_mean": logits.diagonal().mean().detach(),
        }
