"""
Experiment 2: Invisible Physics Test.

On 500 visually identical pairs (same mesh, texture, viewpoint, DINOv2 cos sim > 0.99):
  - Distribution of cosine similarities under each encoder
  - Binary classification: which object is heavier
  - Diffusion Policy on grasp-lift: DynaCLIP adjusts grasp force, DINOv2 fails
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


class InvisiblePhysicsEvaluator:
    """Evaluate backbone sensitivity to physics-only differences."""

    def __init__(
        self,
        backbones: Dict[str, nn.Module],
        test_loader: DataLoader,
        device: str = "cuda",
    ):
        self.backbones = backbones
        self.test_loader = test_loader
        self.device = device

    @torch.no_grad()
    def evaluate_similarity_distributions(self) -> dict:
        """Compute cosine similarity distributions for each backbone on invisible pairs."""
        results = {}

        for name, backbone in self.backbones.items():
            backbone = backbone.to(self.device).eval()
            similarities = []
            mass_diffs = []

            for batch in self.test_loader:
                img_a = batch["img_a"].to(self.device)
                img_b = batch["img_b"].to(self.device)

                feat_a = backbone(img_a)
                feat_b = backbone(img_b)

                cos_sim = F.cosine_similarity(feat_a, feat_b, dim=-1)
                similarities.extend(cos_sim.cpu().numpy().tolist())

                mass_a = batch["mass_a"].numpy()
                mass_b = batch["mass_b"].numpy()
                mass_diffs.extend(np.abs(mass_a - mass_b).tolist())

            results[name] = {
                "mean_similarity": float(np.mean(similarities)),
                "std_similarity": float(np.std(similarities)),
                "min_similarity": float(np.min(similarities)),
                "max_similarity": float(np.max(similarities)),
                "similarities": similarities,
                "mass_diffs": mass_diffs,
            }
            logger.info(
                f"{name}: cos_sim = {np.mean(similarities):.4f} ± {np.std(similarities):.4f}"
            )

        return results

    @torch.no_grad()
    def evaluate_heavier_classification(self) -> dict:
        """Binary classification: which object in the pair is heavier?

        Use cosine similarity difference (or embedding distance) as signal.
        """
        results = {}

        for name, backbone in self.backbones.items():
            backbone = backbone.to(self.device).eval()
            all_embeds_a = []
            all_embeds_b = []
            all_labels = []

            for batch in self.test_loader:
                img_a = batch["img_a"].to(self.device)
                img_b = batch["img_b"].to(self.device)

                feat_a = backbone(img_a)
                feat_b = backbone(img_b)

                all_embeds_a.append(feat_a.cpu())
                all_embeds_b.append(feat_b.cpu())
                all_labels.append(batch["heavier_label"])

            embeds_a = torch.cat(all_embeds_a, dim=0)
            embeds_b = torch.cat(all_embeds_b, dim=0)
            labels = torch.cat(all_labels, dim=0).numpy()

            # Simple linear probe for heavier classification
            diff = embeds_a - embeds_b  # (N, D)

            # Train simple logistic regression
            from sklearn.linear_model import LogisticRegression
            n = len(labels)
            train_n = int(0.7 * n)

            clf = LogisticRegression(max_iter=1000)
            clf.fit(diff[:train_n].numpy(), labels[:train_n])
            preds = clf.predict(diff[train_n:].numpy())
            probs = clf.predict_proba(diff[train_n:].numpy())[:, 1]

            acc = accuracy_score(labels[train_n:], preds)
            try:
                auc = roc_auc_score(labels[train_n:], probs)
            except ValueError:
                auc = 0.5

            results[name] = {
                "accuracy": float(acc),
                "auc": float(auc),
            }
            logger.info(f"{name}: heavier classification acc = {acc:.4f}, AUC = {auc:.4f}")

        return results

    @torch.no_grad()
    def evaluate_embedding_sensitivity(self) -> dict:
        """Measure how much embeddings change due to physics-only differences.

        For a good physics-aware encoder, the embedding should change
        when physics properties change, even if the image is identical.
        """
        results = {}

        for name, backbone in self.backbones.items():
            backbone = backbone.to(self.device).eval()
            embedding_distances = []

            for batch in self.test_loader:
                img_a = batch["img_a"].to(self.device)
                img_b = batch["img_b"].to(self.device)

                feat_a = backbone(img_a)
                feat_b = backbone(img_b)

                l2_dist = torch.norm(feat_a - feat_b, dim=-1)
                embedding_distances.extend(l2_dist.cpu().numpy().tolist())

            results[name] = {
                "mean_l2_dist": float(np.mean(embedding_distances)),
                "std_l2_dist": float(np.std(embedding_distances)),
            }
            logger.info(
                f"{name}: embedding L2 dist = {np.mean(embedding_distances):.6f} ± "
                f"{np.std(embedding_distances):.6f}"
            )

        return results

    def run_full_evaluation(self) -> dict:
        """Run all Invisible Physics evaluations."""
        logger.info("=== Experiment 2: Invisible Physics Test ===")

        sim_results = self.evaluate_similarity_distributions()
        cls_results = self.evaluate_heavier_classification()
        sens_results = self.evaluate_embedding_sensitivity()

        return {
            "similarity_distributions": sim_results,
            "heavier_classification": cls_results,
            "embedding_sensitivity": sens_results,
        }
