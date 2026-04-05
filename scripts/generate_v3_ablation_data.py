"""
Generate data for V3 ablation experiments.

Variants:
  - full:        All 3 physics dimensions vary (mass, friction, restitution)
  - random:      Physics params sampled randomly (no category correlation)
  - mass_only:   Only mass varies; friction & restitution fixed to material mean
  - fric_only:   Only friction varies; mass & restitution fixed to material mean
  - rest_only:   Only restitution varies; mass & friction fixed to material mean

Each variant generates its own data directory with trajectories and similarity
matrix. The analytical physics engine computes trajectories from the sampled
physics, and pairwise L2 trajectory similarity is used as the contrastive label.
"""

import logging
import time
import sys
import argparse
from copy import deepcopy

import numpy as np

sys.path.insert(0, ".")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

from dynaclip.data.generation import (
    DynaCLIPDataGenerator,
    PhysicsConfig,
    MATERIAL_PRIORS,
    get_material_for_category,
    sample_material_for_category,
)

logger = logging.getLogger(__name__)


class AblationPhysicsConfig:
    """Generate PhysicsConfig variants for ablation experiments."""

    @staticmethod
    def sample_mass_only(category: str, rng: np.random.Generator) -> PhysicsConfig:
        """Only mass varies. Friction and restitution fixed to material midpoint."""
        material = sample_material_for_category(category, rng)
        prior = MATERIAL_PRIORS[material]
        mass = rng.uniform(*prior["mass"])
        sf_mid = (prior["static_friction"][0] + prior["static_friction"][1]) / 2
        rest_mid = (prior["restitution"][0] + prior["restitution"][1]) / 2
        return PhysicsConfig(mass=float(mass), static_friction=float(sf_mid),
                             restitution=float(rest_mid), material=material)

    @staticmethod
    def sample_friction_only(category: str, rng: np.random.Generator) -> PhysicsConfig:
        """Only friction varies. Mass and restitution fixed to material midpoint."""
        material = sample_material_for_category(category, rng)
        prior = MATERIAL_PRIORS[material]
        mass_mid = (prior["mass"][0] + prior["mass"][1]) / 2
        sf = rng.uniform(*prior["static_friction"])
        rest_mid = (prior["restitution"][0] + prior["restitution"][1]) / 2
        return PhysicsConfig(mass=float(mass_mid), static_friction=float(sf),
                             restitution=float(rest_mid), material=material)

    @staticmethod
    def sample_restitution_only(category: str, rng: np.random.Generator) -> PhysicsConfig:
        """Only restitution varies. Mass and friction fixed to material midpoint."""
        material = sample_material_for_category(category, rng)
        prior = MATERIAL_PRIORS[material]
        mass_mid = (prior["mass"][0] + prior["mass"][1]) / 2
        sf_mid = (prior["static_friction"][0] + prior["static_friction"][1]) / 2
        rest = rng.uniform(*prior["restitution"])
        return PhysicsConfig(mass=float(mass_mid), static_friction=float(sf_mid),
                             restitution=float(rest), material=material)


def main():
    parser = argparse.ArgumentParser(description="Generate V3 ablation data")
    parser.add_argument("--variant", type=str, required=True,
                        choices=["full", "random", "mass_only", "fric_only", "rest_only"],
                        help="Ablation variant to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data_cache/dynaclip_v3_{variant})")
    parser.add_argument("--dataset_root", type=str, default="/home/kztrgg/datasets")
    parser.add_argument("--max_images", type=int, default=20000)
    parser.add_argument("--num_physics", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir or f"data_cache/dynaclip_v3_{args.variant}"

    t0 = time.time()
    logger.info(f"Generating V3 ablation data: variant={args.variant}, output={output_dir}")

    # Monkey-patch PhysicsConfig.sample_for_category for ablation variants
    original_sample = PhysicsConfig.sample_for_category

    if args.variant == "full":
        pass  # Use default sample_for_category (multi-material)
    elif args.variant == "random":
        PhysicsConfig.sample_for_category = staticmethod(
            lambda category, rng, multi_material=True: PhysicsConfig.sample_random(rng)
        )
    elif args.variant == "mass_only":
        PhysicsConfig.sample_for_category = staticmethod(
            lambda category, rng, multi_material=True: AblationPhysicsConfig.sample_mass_only(category, rng)
        )
    elif args.variant == "fric_only":
        PhysicsConfig.sample_for_category = staticmethod(
            lambda category, rng, multi_material=True: AblationPhysicsConfig.sample_friction_only(category, rng)
        )
    elif args.variant == "rest_only":
        PhysicsConfig.sample_for_category = staticmethod(
            lambda category, rng, multi_material=True: AblationPhysicsConfig.sample_restitution_only(category, rng)
        )

    gen = DynaCLIPDataGenerator(
        output_dir=output_dir,
        dataset_root=args.dataset_root,
        num_physics_per_image=args.num_physics,
        max_images=args.max_images,
        seed=args.seed,
        similarity_metric="l2",
    )
    gen.generate_all(compute_similarity=True, max_sim_pairs=500000)

    # Restore original
    PhysicsConfig.sample_for_category = original_sample

    elapsed = (time.time() - t0) / 60
    logger.info(f"Done: {args.variant} in {elapsed:.1f} min → {output_dir}")


if __name__ == "__main__":
    main()
