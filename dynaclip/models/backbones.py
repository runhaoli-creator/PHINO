"""
Backbone Registry: Unified interface for visual encoders.

Backbones:
  1. DynaCLIP (ours) - DINOv2-ViT-B/14 fine-tuned
  2. DINOv2-ViT-B/14 - pretrained frozen
  3. DINOv2-ViT-L/14 - pretrained frozen
  4. SigLIP-ViT-B/16 - google/siglip-base-patch16-224
  5. CLIP-ViT-L/14 - openai/clip-vit-large-patch14
  6. R3M - ResNet-50 (requires r3m package)
  7. VIP - ResNet-50 (requires vip package)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BackboneWrapper(nn.Module):
    """Unified wrapper for all backbone encoders.

    Always extracts CLS token from ViT backbones for consistency.
    """

    def __init__(self, name: str, model: nn.Module, output_dim: int,
                 transform=None, extract_fn=None):
        super().__init__()
        self.name = name
        self.model = model
        self._output_dim = output_dim
        self._transform = transform
        self._extract_fn = extract_fn  # Custom extraction function

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def transform(self):
        return self._transform

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self._extract_fn is not None:
                return self._extract_fn(self.model, images)
            return self.model(images)


def _dinov2_cls_extract(model, images):
    """Extract CLS token from DINOv2 model (consistent extraction)."""
    output = model.forward_features(images)
    if isinstance(output, dict):
        cls = output.get("x_norm_clstoken", None)
        if cls is None:
            x = output.get("x", None)
            cls = x[:, 0] if x is not None else model(images)
        return cls
    return output[:, 0]


# ---------------------------------------------------------------------------
# Individual backbone loaders
# ---------------------------------------------------------------------------
def load_dynaclip(checkpoint_path: Optional[str] = None) -> BackboneWrapper:
    """Load DynaCLIP (fine-tuned DINOv2-ViT-B/14)."""
    from dynaclip.models.dynaclip import DynaCLIPEncoder
    encoder = DynaCLIPEncoder(checkpoint_path=checkpoint_path, feature_type="cls")
    return BackboneWrapper("dynaclip", encoder, output_dim=encoder.output_dim)


def load_dinov2_vitb14() -> BackboneWrapper:
    """Load pretrained DINOv2-ViT-B/14."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return BackboneWrapper("dinov2_vitb14", model, output_dim=768,
                           extract_fn=_dinov2_cls_extract)


def load_dinov2_vitl14() -> BackboneWrapper:
    """Load pretrained DINOv2-ViT-L/14."""
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    return BackboneWrapper("dinov2_vitl14", model, output_dim=1024,
                           extract_fn=_dinov2_cls_extract)


def load_siglip() -> BackboneWrapper:
    """Load SigLIP-ViT-B/16 from HuggingFace."""
    try:
        from transformers import SiglipModel

        model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")

        class SigLIPVisionEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.vision_model = model.vision_model

            def forward(self, images):
                outputs = self.vision_model(pixel_values=images)
                return outputs.pooler_output

        encoder = SigLIPVisionEncoder(model)
        return BackboneWrapper("siglip_vitb16", encoder, output_dim=768)
    except Exception as e:
        raise RuntimeError(
            f"SigLIP load failed: {e}. Install transformers: pip install transformers"
        )


def load_clip_vitl14() -> BackboneWrapper:
    """Load CLIP-ViT-L/14 from OpenAI."""
    try:
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        class CLIPVisionEncoder(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.vision_model = model.vision_model

            def forward(self, images):
                outputs = self.vision_model(pixel_values=images)
                return outputs.pooler_output

        encoder = CLIPVisionEncoder(model)
        return BackboneWrapper("clip_vitl14", encoder, output_dim=768)
    except Exception as e:
        raise RuntimeError(
            f"CLIP load failed: {e}. Install transformers: pip install transformers"
        )


def load_r3m() -> BackboneWrapper:
    """Load R3M ResNet-50."""
    try:
        from r3m import load_r3m
        model = load_r3m("resnet50")
        model.eval()

        class R3MEncoder(nn.Module):
            def __init__(self, r3m_model):
                super().__init__()
                self.r3m = r3m_model

            def forward(self, images):
                return self.r3m(images * 255.0)

        encoder = R3MEncoder(model)
        return BackboneWrapper("r3m", encoder, output_dim=2048)
    except ImportError:
        raise RuntimeError(
            "R3M not installed. Install from: https://github.com/facebookresearch/r3m"
        )


def load_vip() -> BackboneWrapper:
    """Load VIP ResNet-50."""
    try:
        from vip import load_vip
        model = load_vip()
        model.eval()

        class VIPEncoder(nn.Module):
            def __init__(self, vip_model):
                super().__init__()
                self.vip = vip_model

            def forward(self, images):
                return self.vip(images * 255.0)

        encoder = VIPEncoder(model)
        return BackboneWrapper("vip", encoder, output_dim=2048)
    except ImportError:
        raise RuntimeError(
            "VIP not installed. Install from: https://github.com/facebookresearch/vip"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BACKBONE_REGISTRY = {
    "dynaclip": load_dynaclip,
    "dinov2_vitb14": load_dinov2_vitb14,
    "dinov2_vitl14": load_dinov2_vitl14,
    "siglip": load_siglip,
    "clip_vitl14": load_clip_vitl14,
    "r3m": load_r3m,
    "vip": load_vip,
}


def load_backbone(name: str, **kwargs) -> BackboneWrapper:
    """Load a visual backbone by name."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}")
    loader = BACKBONE_REGISTRY[name]
    return loader(**kwargs)


def get_all_backbone_names() -> list:
    return list(BACKBONE_REGISTRY.keys())
