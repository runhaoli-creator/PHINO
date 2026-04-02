"""
DynaCLIP Losses: Soft InfoNCE with continuous dynamics similarity targets.

Soft InfoNCE loss:
  Given embeddings z_i, z_j for B pairs and B×B dynamics similarity matrix S:
  - logits = z_i @ z_j^T / temperature
  - soft_labels = softmax(S / tau_sim)
  - loss = cross_entropy(soft_labels, log_softmax(logits))

Temperature is managed ONLY here (not in the model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftInfoNCELoss(nn.Module):
    """Soft InfoNCE loss with dynamics similarity as soft targets."""

    def __init__(
        self,
        temperature: float = 0.07,
        similarity_temperature: float = 0.1,
        learnable_temperature: bool = True,
    ):
        super().__init__()
        self.similarity_temperature = similarity_temperature

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_temperature", torch.log(torch.tensor(temperature)))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)

    def forward(
        self,
        z_i: torch.Tensor,       # (B, D) L2-normalized embeddings
        z_j: torch.Tensor,       # (B, D) L2-normalized embeddings
        dynamics_sim: torch.Tensor,  # (B,) pairwise dynamics similarities
    ) -> dict:
        B = z_i.shape[0]
        device = z_i.device

        # Compute logits: (B, B) similarity matrix
        logits = torch.matmul(z_i, z_j.T) / self.temperature

        # Build B x B dynamics similarity matrix (vectorized)
        sim_matrix = self._build_similarity_matrix(dynamics_sim, B, device)

        # Soft labels: softmax over similarity matrix
        soft_labels = F.softmax(sim_matrix / self.similarity_temperature, dim=-1)

        # Cross-entropy loss between soft labels and log-softmax of logits
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(soft_labels * log_probs, dim=-1).mean()

        # Symmetric loss
        logits_t = logits.T
        soft_labels_t = F.softmax(sim_matrix.T / self.similarity_temperature, dim=-1)
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

    def _build_similarity_matrix(
        self,
        pair_sim: torch.Tensor,
        B: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build B x B similarity matrix from pair similarities.

        Diagonal = pair_sim[i] (similarity between matched pair i).
        Off-diagonal: geometric mean approximation * 0.5.
        """
        pair_sim_i = pair_sim.unsqueeze(1).expand(B, B)
        pair_sim_j = pair_sim.unsqueeze(0).expand(B, B)
        sim_matrix = torch.sqrt(pair_sim_i * pair_sim_j + 1e-8) * 0.5

        # Diagonal: vectorized assignment
        sim_matrix.diagonal().copy_(pair_sim)

        return sim_matrix


# ---------------------------------------------------------------------------
# Loss registry
# ---------------------------------------------------------------------------
LOSS_REGISTRY = {
    "soft_infonce": SoftInfoNCELoss,
}


def build_loss(name: str, **kwargs) -> nn.Module:
    """Build a loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)
