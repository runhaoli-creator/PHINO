"""
DynaCLIP Model: DINOv2-ViT-B/14 backbone with dynamics contrastive projection head.

Architecture:
  - Backbone: DINOv2-ViT-B/14 (86M params), UNFROZEN during pre-training
  - Feature extraction: CLS token || mean-pooled patch tokens → 1536-dim
  - Projection head: Linear(1536,768) → LayerNorm → GELU → Linear(768,512) → L2-norm
  - Output: 512-dim unit-norm embedding
  - Projection head discarded after pre-training

NOTE: Temperature is managed by the loss function, NOT the model.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

logger = logging.getLogger(__name__)


class DynaCLIPProjectionHead(nn.Module):
    """Projection head: maps concatenated features to dynamics embedding space."""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        return F.normalize(x, dim=-1)  # L2 normalize


class DynaCLIPModel(nn.Module):
    """DynaCLIP: Physics-grounded contrastive visual encoder.

    During pre-training, the backbone is UNFROZEN and the projection head is used.
    After pre-training, the projection head is discarded and the backbone serves
    as a general-purpose physics-aware visual encoder.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitb14",
        embed_dim: int = 512,
        freeze_backbone: bool = False,
        unfreeze_last_n_blocks: int = -1,  # -1 = all unfrozen
    ):
        super().__init__()
        self.backbone_name = backbone_name

        # Load DINOv2 backbone
        logger.info(f"Loading backbone: {backbone_name}")
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            backbone_name,
            pretrained=True,
        )

        # Get backbone output dim
        self.backbone_dim = self.backbone.embed_dim  # 768 for ViT-B/14
        # CLS + mean-pooled patches = 2 * backbone_dim
        self.feature_dim = 2 * self.backbone_dim  # 1536

        # Projection head
        self.embed_dim = embed_dim
        self.projection_head = DynaCLIPProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=self.backbone_dim,
            output_dim=embed_dim,
        )

        # NO learnable temperature here — temperature belongs in the loss function only

        # Freeze control
        if freeze_backbone:
            self._freeze_backbone()
        elif unfreeze_last_n_blocks > 0:
            self._unfreeze_last_n_blocks(unfreeze_last_n_blocks)

        self._use_gradient_checkpointing = False

        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"DynaCLIP: {n_params/1e6:.1f}M params, {n_trainable/1e6:.1f}M trainable")

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to reduce memory at cost of ~30% slower training."""
        self._use_gradient_checkpointing = True
        # Also enable on backbone if it supports it
        if hasattr(self.backbone, 'set_grad_checkpointing'):
            self.backbone.set_grad_checkpointing(True)
        logger.info("Gradient checkpointing enabled")

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def _unfreeze_last_n_blocks(self, n: int):
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        logger.info(f"Unfrozen last {n} transformer blocks")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS + mean-pooled patch tokens.

        Args:
            images: (B, 3, 224, 224)
        Returns:
            features: (B, 1536) = CLS(768) || mean_patch(768)
        """
        if self._use_gradient_checkpointing and self.training:
            # Use gradient checkpointing: wrap forward_features
            output = grad_checkpoint(
                self.backbone.forward_features, images, use_reentrant=False
            )
        else:
            output = self.backbone.forward_features(images)

        if isinstance(output, dict):
            cls_token = output.get("x_norm_clstoken", None)
            patch_tokens = output.get("x_norm_patchtokens", None)
            if cls_token is None or patch_tokens is None:
                # Fallback for different DINOv2 versions
                x = output.get("x", None)
                if x is not None:
                    cls_token = x[:, 0]
                    patch_tokens = x[:, 1:]
                else:
                    raise RuntimeError(f"Unexpected DINOv2 output keys: {output.keys()}")
        else:
            cls_token = output[:, 0]
            patch_tokens = output[:, 1:]

        mean_patches = patch_tokens.mean(dim=1)
        features = torch.cat([cls_token, mean_patches], dim=-1)
        return features

    def forward(
        self,
        images: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """Forward pass through backbone + projection head.

        Args:
            images: (B, 3, 224, 224)
            return_features: if True, return raw 1536-d features before projection
        Returns:
            embeddings: (B, 512) unit-norm embeddings
        """
        features = self.extract_features(images)
        if return_features:
            return features
        return self.projection_head(features)

    def encode_pair(
        self,
        img_i: torch.Tensor,
        img_j: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a pair of images for contrastive loss."""
        z_i = self.forward(img_i)
        z_j = self.forward(img_j)
        return z_i, z_j

    def get_backbone_only(self) -> nn.Module:
        """Return backbone without projection head (for downstream use)."""
        return self.backbone

    def get_param_groups(
        self,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
    ) -> list:
        """Parameter groups with different learning rates."""
        return [
            {"params": list(self.backbone.parameters()), "lr": backbone_lr},
            {"params": list(self.projection_head.parameters()), "lr": head_lr},
        ]


# ---------------------------------------------------------------------------
# Frozen feature extractor for downstream evaluation
# ---------------------------------------------------------------------------
class DynaCLIPEncoder(nn.Module):
    """Frozen DynaCLIP encoder for downstream tasks.

    After pre-training, loads the fine-tuned backbone weights
    and produces features for downstream linear probing.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        backbone_name: str = "dinov2_vitb14",
        feature_type: str = "cls_mean",  # "cls", "mean", "cls_mean"
    ):
        super().__init__()
        self.feature_type = feature_type

        # Build full model then extract backbone
        full_model = DynaCLIPModel(backbone_name=backbone_name)
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            full_model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded DynaCLIP checkpoint: {checkpoint_path}")

        self.backbone = full_model.backbone
        self.backbone_dim = full_model.backbone_dim

        # Freeze
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Output dimension depends on feature_type
        if feature_type == "cls_mean":
            self._output_dim = 2 * self.backbone_dim  # 1536
        else:
            self._output_dim = self.backbone_dim  # 768

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.backbone.forward_features(images)

        if isinstance(output, dict):
            cls_token = output.get("x_norm_clstoken", None)
            patch_tokens = output.get("x_norm_patchtokens", None)
            if cls_token is None or patch_tokens is None:
                x = output.get("x", None)
                if x is not None:
                    cls_token = x[:, 0]
                    patch_tokens = x[:, 1:]
                else:
                    raise RuntimeError(f"Unexpected DINOv2 output keys: {output.keys()}")
        else:
            cls_token = output[:, 0]
            patch_tokens = output[:, 1:]

        if self.feature_type == "cls":
            return cls_token
        elif self.feature_type == "mean":
            return patch_tokens.mean(dim=1)
        else:  # cls_mean
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
