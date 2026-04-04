"""
Shared backbone loading utilities for Voltron and Theia.

Must be imported BEFORE any transformers imports to apply patches.
"""
import contextlib
import torch
import torch.nn as nn

# ── Theia Patches (must run before importing transformers) ──
def apply_theia_patches():
    """Apply transformers 5.x compatibility patches for Theia."""
    import transformers.modeling_utils as mu
    import transformers.integrations.accelerate as acc_module
    
    # Patch 1: Disable meta device check (Theia's DeiT calls from_pretrained internally)
    acc_module.check_and_set_device_map = lambda x: x
    mu.check_and_set_device_map = lambda x: x
    
    # Patch 2: Disable fast init (meta device context)
    mu._fast_init_context = lambda *a, **kw: contextlib.nullcontext()
    
    # Patch 3: Fix missing all_tied_weights_keys attribute
    orig_mark = mu.PreTrainedModel.mark_tied_weights_as_initialized
    def patched_mark(self, loading_info):
        if not hasattr(self, 'all_tied_weights_keys'):
            self.all_tied_weights_keys = {}
        return orig_mark(self, loading_info)
    mu.PreTrainedModel.mark_tied_weights_as_initialized = patched_mark

# Apply patches immediately on import
apply_theia_patches()


# ── Voltron Wrapper ──
class VoltronWrapper(nn.Module):
    """Wraps Voltron v-cond model → mean-pooled features (384d)."""
    def __init__(self):
        super().__init__()
        from voltron import load as voltron_load
        self.model, self.preprocess = voltron_load("v-cond", device="cpu", freeze=True)
        self.model.eval()
        self.output_dim = 384
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Voltron expects specific preprocessing but works with standard ImageNet-normalized inputs
        # Output: [B, 216, 384] (216 = 1 CLS + 14*14 + 1 reg tokens for ViT-Small/14)
        out = self.model(x)
        if out.dim() == 3:
            return out[:, 0]  # CLS token → (B, 384)
        return out


# ── Theia Wrapper ──
class TheiaWrapper(nn.Module):
    """Wraps Theia model → mean-pooled DINOv2-large prediction (1024d)."""
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            "theaiinstitute/theia-base-patch16-224-cdiv",
            trust_remote_code=True,
        )
        self.model.eval()
        self.output_dim = 1024  # DINOv2-large prediction dimension
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.model(x)
        # out is dict: {"facebook/dinov2-large": [B,256,1024], ...}
        # Use DINOv2-large prediction with mean pooling
        dinov2_feat = out["facebook/dinov2-large"]  # [B, 256, 1024]
        return dinov2_feat.mean(dim=1)  # [B, 1024]


def load_voltron(device="cpu"):
    """Load Voltron v-cond model."""
    wrapper = VoltronWrapper()
    wrapper.eval().to(device)
    return wrapper


def load_theia(device="cpu"):
    """Load Theia base model."""
    wrapper = TheiaWrapper()
    wrapper.eval().to(device)
    return wrapper
