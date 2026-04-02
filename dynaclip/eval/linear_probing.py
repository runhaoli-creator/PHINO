"""
Experiment 1: Physics Property Linear Probing.

Train a single linear layer on frozen representations from all backbones
to predict mass (R²), friction (R²), restitution (R²), and object category (accuracy).
5 seeds with 95% confidence intervals.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, accuracy_score
from scipy import stats

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Single linear layer for probing physics properties."""

    def __init__(self, input_dim: int, output_dim: int = 1, task: str = "regression"):
        super().__init__()
        self.task = task
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PhysicsLinearProbing:
    """Run linear probing experiment across all backbones.

    Properties to probe:
      - mass: regression (MSE → R²)
      - static_friction: regression (MSE → R²)
      - restitution: regression (MSE → R²)
      - category: classification (DomainNet classes → accuracy) [optional]
    """

    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_seeds: int = 5,
        num_epochs: int = 100,
        lr: float = 1e-3,
        device: str = "cuda",
        probe_category: bool = True,
    ):
        self.backbones = backbones
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_seeds = num_seeds
        self.num_epochs = num_epochs
        self.lr = lr
        self.device = device
        self.probe_category = probe_category
        self._cat_to_idx = {}

    @torch.no_grad()
    def extract_features(self, backbone: nn.Module, loader: DataLoader) -> tuple:
        """Extract frozen features for all data."""
        backbone.eval()
        all_features = []
        all_labels = {"mass": [], "static_friction": [], "restitution": []}
        if self.probe_category:
            all_labels["category"] = []

        for batch in loader:
            img = batch["image"].to(self.device)
            feat = backbone(img)
            all_features.append(feat.cpu())
            all_labels["mass"].append(batch["mass"])
            all_labels["static_friction"].append(batch["static_friction"])
            all_labels["restitution"].append(batch["restitution"])

            if self.probe_category and "category" in batch:
                cats = batch["category"]
                indices = []
                for c in cats:
                    if c not in self._cat_to_idx:
                        self._cat_to_idx[c] = len(self._cat_to_idx)
                    indices.append(self._cat_to_idx[c])
                all_labels["category"].append(torch.tensor(indices, dtype=torch.long))

        features = torch.cat(all_features, dim=0)
        labels = {k: torch.cat(v, dim=0) for k, v in all_labels.items()}
        return features, labels

    def _get_properties(self) -> dict:
        """Build PROPERTIES dict based on available data."""
        props = {
            "mass": {"task": "regression", "output_dim": 1},
            "static_friction": {"task": "regression", "output_dim": 1},
            "restitution": {"task": "regression", "output_dim": 1},
        }
        if self.probe_category and len(self._cat_to_idx) > 0:
            props["category"] = {
                "task": "classification",
                "output_dim": len(self._cat_to_idx),
            }
        return props

    def train_probe(
        self,
        features_train: torch.Tensor,
        labels_train: torch.Tensor,
        features_val: torch.Tensor,
        labels_val: torch.Tensor,
        prop_name: str,
        prop_cfg: dict,
        seed: int,
    ) -> nn.Module:
        """Train a single linear probe."""
        torch.manual_seed(seed)

        input_dim = features_train.shape[-1]
        probe = LinearProbe(input_dim, prop_cfg["output_dim"], prop_cfg["task"]).to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_state = None

        for epoch in range(self.num_epochs):
            probe.train()
            feat = features_train.to(self.device)
            target = labels_train.to(self.device)

            pred = probe(feat)

            if prop_cfg["task"] == "regression":
                loss = F.mse_loss(pred.squeeze(), target.float())
            else:
                loss = F.cross_entropy(pred, target.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation
            probe.eval()
            with torch.no_grad():
                val_pred = probe(features_val.to(self.device))
                if prop_cfg["task"] == "regression":
                    val_loss = F.mse_loss(val_pred.squeeze(), labels_val.to(self.device).float())
                else:
                    val_loss = F.cross_entropy(val_pred, labels_val.to(self.device).long())

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

        if best_state:
            probe.load_state_dict(best_state)
        return probe

    def evaluate_probe(
        self,
        probe: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        prop_name: str,
        prop_cfg: dict,
    ) -> dict:
        """Evaluate a trained probe."""
        probe.eval()

        with torch.no_grad():
            pred = probe(features.to(self.device))

        if prop_cfg["task"] == "regression":
            pred_np = pred.squeeze().cpu().numpy()
            target_np = labels.numpy()
            r2 = r2_score(target_np, pred_np)
            mse = float(np.mean((pred_np - target_np) ** 2))
            return {"r2": r2, "mse": mse}
        else:
            pred_classes = pred.argmax(dim=-1).cpu().numpy()
            target_np = labels.numpy()
            acc = accuracy_score(target_np, pred_classes)
            return {"accuracy": acc}

    def run(self) -> dict:
        """Run full linear probing experiment."""
        results = {}

        for backbone_name, backbone in self.backbones.items():
            logger.info(f"=== Linear Probing: {backbone_name} ===")
            backbone = backbone.to(self.device)

            # Extract features (category mapping built incrementally)
            self._cat_to_idx = {}  # Reset for each backbone
            train_feats, train_labels = self.extract_features(backbone, self.train_loader)
            val_feats, val_labels = self.extract_features(backbone, self.val_loader)
            test_feats, test_labels = self.extract_features(backbone, self.test_loader)

            properties = self._get_properties()

            backbone_results = {}
            for prop_name, prop_cfg in properties.items():
                seed_results = []
                for seed in range(self.num_seeds):
                    probe = self.train_probe(
                        train_feats, train_labels[prop_name],
                        val_feats, val_labels[prop_name],
                        prop_name, prop_cfg, seed,
                    )
                    metrics = self.evaluate_probe(
                        probe, test_feats, test_labels[prop_name], prop_name, prop_cfg,
                    )
                    seed_results.append(metrics)

                # Aggregate with confidence intervals
                metric_key = "r2" if prop_cfg["task"] == "regression" else "accuracy"
                values = [r[metric_key] for r in seed_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                if len(values) > 1 and std_val > 0:
                    ci_95 = stats.t.interval(
                        0.95, len(values) - 1, loc=mean_val,
                        scale=std_val / np.sqrt(len(values))
                    )
                else:
                    ci_95 = (mean_val, mean_val)

                backbone_results[prop_name] = {
                    "metric": metric_key,
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "ci_lower": float(ci_95[0]) if not np.isnan(ci_95[0]) else float(mean_val - 2 * std_val),
                    "ci_upper": float(ci_95[1]) if not np.isnan(ci_95[1]) else float(mean_val + 2 * std_val),
                    "per_seed": values,
                }

                logger.info(f"  {prop_name}: {metric_key} = {mean_val:.4f} ± {std_val:.4f}")

            results[backbone_name] = backbone_results

        return results
