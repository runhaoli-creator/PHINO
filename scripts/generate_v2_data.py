#!/usr/bin/env python
"""
DynaCLIP v2: Regenerate dataset with multi-material categories.

This breaks the deterministic category->material->physics confound (P0-1 fix)
by sampling materials probabilistically for each category.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from dynaclip.data.generation import (
    DynaCLIPDataGenerator, PhysicsConfig,
    sample_material_for_category, MATERIAL_PRIORS,
)


class MultiMaterialDataGenerator(DynaCLIPDataGenerator):
    """Data generator that always uses multi-material sampling."""

    def generate_all(self, compute_similarity=True, max_sim_pairs=500000):
        """Override to use multi_material=True in all PhysicsConfig sampling."""
        # Temporarily patch sample_for_category to always use multi_material
        original = PhysicsConfig.sample_for_category

        @staticmethod
        def patched_sample(category, rng, multi_material=True):
            material = sample_material_for_category(category, rng)
            prior = MATERIAL_PRIORS[material]
            mass = rng.uniform(*prior["mass"])
            sf = rng.uniform(*prior["static_friction"])
            rest = rng.uniform(*prior["restitution"])
            return PhysicsConfig(
                mass=float(mass), static_friction=float(sf),
                restitution=float(rest), material=material,
            )

        PhysicsConfig.sample_for_category = patched_sample
        try:
            super().generate_all(
                compute_similarity=compute_similarity,
                max_sim_pairs=max_sim_pairs,
            )
        finally:
            PhysicsConfig.sample_for_category = original


def main():
    rng = np.random.default_rng(42)

    # Verify multi-material works
    materials_seen = set()
    for _ in range(100):
        pc = PhysicsConfig.sample_for_category("cup", rng, multi_material=True)
        materials_seen.add(pc.material)
    print(f"Materials for 'cup' across 100 samples: {materials_seen}")
    assert len(materials_seen) > 1, "Multi-material mapping failed!"

    # Generate v2 dataset with multi-material
    generator = MultiMaterialDataGenerator(
        output_dir="data_cache/dynaclip_v2_data",
        dataset_root="/home/kztrgg/datasets",
        num_physics_per_image=5,
        max_images=20000,
        seed=42,
        similarity_metric="dtw",
    )
    generator.generate_all(compute_similarity=True, max_sim_pairs=500000)

    # Verify the generated data has multi-material
    import json
    with open("data_cache/dynaclip_v2_data/metadata.json") as f:
        metadata = json.load(f)
    from collections import Counter
    cup_mats = Counter(
        e["material"] for e in metadata if e["category"] == "cup"
    )
    print(f"\nVerification - 'cup' materials in generated data: {dict(cup_mats)}")
    print(f"Total entries: {len(metadata)}")
    print("v2 data generation complete!")


if __name__ == "__main__":
    main()
