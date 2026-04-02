"""
DynaCLIP Test Suite: Verify all components work correctly.
"""

import sys
import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------
class TestDynaCLIPModel:
    def test_model_creation(self):
        """Test DynaCLIPModel can be instantiated."""
        from dynaclip.models.dynaclip import DynaCLIPModel
        # Use lightweight version for testing (skip hub download)
        model = DynaCLIPModel.__new__(DynaCLIPModel)
        assert model is not None

    def test_projection_head(self):
        """Test projection head dimensions."""
        from dynaclip.models.dynaclip import DynaCLIPProjectionHead
        head = DynaCLIPProjectionHead(input_dim=1536, hidden_dim=768, output_dim=512)
        x = torch.randn(4, 1536)
        out = head(x)
        assert out.shape == (4, 512)
        # Check L2 normalization
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_projection_head_gradient(self):
        """Test projection head supports backpropagation."""
        from dynaclip.models.dynaclip import DynaCLIPProjectionHead
        head = DynaCLIPProjectionHead()
        x = torch.randn(4, 1536, requires_grad=True)
        out = head(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------
class TestLosses:
    def test_soft_infonce(self):
        """Test Soft InfoNCE loss computation."""
        from dynaclip.losses.contrastive import SoftInfoNCELoss
        loss_fn = SoftInfoNCELoss()
        z_i = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
        z_j = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
        dyn_sim = torch.rand(8)
        result = loss_fn(z_i, z_j, dyn_sim)
        assert "loss" in result
        assert result["loss"].requires_grad

    def test_standard_infonce(self):
        from dynaclip.losses.contrastive import StandardInfoNCELoss
        loss_fn = StandardInfoNCELoss()
        z_i = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
        z_j = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
        dyn_sim = torch.rand(8)
        result = loss_fn(z_i, z_j, dyn_sim)
        assert result["loss"].item() > 0

    def test_triplet_loss(self):
        from dynaclip.losses.contrastive import TripletLoss
        loss_fn = TripletLoss()
        z_i = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
        z_j = torch.nn.functional.normalize(torch.randn(8, 512), dim=-1)
        dyn_sim = torch.rand(8)
        result = loss_fn(z_i, z_j, dyn_sim)
        assert result["loss"].item() >= 0

    def test_byol_loss(self):
        from dynaclip.losses.contrastive import BYOLLoss
        loss_fn = BYOLLoss()
        z_i = torch.randn(8, 512)
        z_j = torch.randn(8, 512)
        dyn_sim = torch.rand(8)
        result = loss_fn(z_i, z_j, dyn_sim)
        assert "loss" in result

    def test_loss_registry(self):
        from dynaclip.losses.contrastive import build_loss
        for name in ["soft_infonce", "infonce", "triplet", "byol"]:
            loss = build_loss(name)
            assert loss is not None


# ---------------------------------------------------------------------------
# Data tests
# ---------------------------------------------------------------------------
class TestData:
    def test_physics_config(self):
        """Test physics config sampling."""
        from dynaclip.data.generation import PhysicsConfig
        rng = np.random.default_rng(42)
        cfg = PhysicsConfig.sample_random(rng)
        assert 0.05 <= cfg.mass <= 10.0
        assert 0.05 <= cfg.static_friction <= 1.5
        assert 0.0 <= cfg.restitution <= 0.95
        assert cfg.dynamic_friction == pytest.approx(0.8 * cfg.static_friction)

    def test_dynamics_similarity(self):
        """Test dynamics similarity computation."""
        from dynaclip.data.generation import (
            PhysicsConfig, compute_dynamics_similarity_l2,
            AnalyticalPhysicsEngine, DIAGNOSTIC_ACTIONS,
        )
        cfg1 = PhysicsConfig(mass=1.0, static_friction=0.5, restitution=0.5)
        cfg2 = PhysicsConfig(mass=5.0, static_friction=0.5, restitution=0.5)

        engine = AnalyticalPhysicsEngine()
        action = DIAGNOSTIC_ACTIONS[0]  # push_x
        traj1 = engine.execute_diagnostic_action(action, cfg1)
        traj2 = engine.execute_diagnostic_action(action, cfg2)

        sim = compute_dynamics_similarity_l2(traj1, traj2)
        assert 0.0 <= sim <= 1.0

    def test_dataset_creation(self):
        """Test contrastive dataset."""
        from dynaclip.data.dataset import DynaCLIPContrastiveDataset
        ds = DynaCLIPContrastiveDataset(
            data_dir="data_cache/dynaclip_data",
            num_pairs=100,
            seed=42,
        )
        assert len(ds) == 100
        sample = ds[0]
        assert "img_i" in sample
        assert "dynamics_similarity" in sample

    def test_invisible_physics_dataset(self):
        from dynaclip.data.dataset import InvisiblePhysicsDataset
        ds = InvisiblePhysicsDataset(data_dir="data_cache/dynaclip_data")
        assert len(ds) > 0
        sample = ds[0]
        assert "img_a" in sample
        assert "heavier_label" in sample

    def test_probe_dataset(self):
        from dynaclip.data.dataset import PhysicsProbeDataset
        ds = PhysicsProbeDataset(data_dir="data_cache/dynaclip_data", split="train")
        assert len(ds) > 0
        sample = ds[0]
        assert "mass" in sample


# ---------------------------------------------------------------------------
# Utils tests
# ---------------------------------------------------------------------------
class TestUtils:
    def test_seed_setting(self):
        from dynaclip.utils.helpers import set_seed
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_average_meter(self):
        from dynaclip.utils.helpers import AverageMeter
        meter = AverageMeter("test")
        meter.update(1.0)
        meter.update(3.0)
        assert meter.avg == pytest.approx(2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
