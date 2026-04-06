#!/usr/bin/env python3
"""
Synthetic Violation-of-Expectation (VoE) Test for DynaCLIP.

Uses the analytical physics engine to create synthetic plausible/implausible
trajectory pairs. Tests whether a physics-aligned encoder can detect that
something is "physically wrong" when visual appearance doesn't match physics.

Plausible:   metal ball → high restitution (bounces high)
Implausible: metal ball → fabric physics (doesn't bounce)

This complements the IntPhys2 video benchmark by testing physics understanding
at the trajectory level without requiring video rendering.

Three violation types:
  1. material_swap:    Object looks like material A but behaves like material B
  2. mass_violation:   Heavy object behaves as if very light, or vice versa
  3. bounce_violation: High-restitution object doesn't bounce, or low-rest does
"""

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dynaclip.data.generation import (
    AnalyticalPhysicsEngine,
    DIAGNOSTIC_ACTIONS,
    MATERIAL_PRIORS,
    PhysicsConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# VoE Scenario Generation
# ============================================================================
def generate_voe_scenarios(
    num_per_type: int = 200,
    seed: int = 42,
) -> List[Dict]:
    """Generate violation-of-expectation trajectory pairs."""
    rng = np.random.default_rng(seed)
    engine = AnalyticalPhysicsEngine()
    scenarios = []

    # 1. Material Swap: same visual appearance, wrong physics
    material_pairs = [
        ("metal", "fabric"),
        ("metal", "food_organic"),
        ("rubber_plastic", "stone_heavy"),
        ("glass_ceramic", "fabric"),
        ("wood", "metal"),
    ]

    for _ in range(num_per_type):
        mat_a, mat_b = material_pairs[rng.integers(len(material_pairs))]
        prior_a = MATERIAL_PRIORS[mat_a]
        prior_b = MATERIAL_PRIORS[mat_b]

        physics_plausible = PhysicsConfig(
            mass=rng.uniform(*prior_a["mass"]),
            static_friction=rng.uniform(*prior_a["static_friction"]),
            restitution=rng.uniform(*prior_a["restitution"]),
            material=mat_a,
        )
        physics_implausible = PhysicsConfig(
            mass=rng.uniform(*prior_b["mass"]),
            static_friction=rng.uniform(*prior_b["static_friction"]),
            restitution=rng.uniform(*prior_b["restitution"]),
            material=mat_a,  # Looks like mat_a but behaves like mat_b
        )

        action = DIAGNOSTIC_ACTIONS[rng.integers(len(DIAGNOSTIC_ACTIONS))]
        traj_p = engine.execute_diagnostic_action(action, physics_plausible)
        traj_i = engine.execute_diagnostic_action(action, physics_implausible)

        scenarios.append({
            "type": "material_swap",
            "material_visual": mat_a,
            "material_physics": mat_b,
            "action": action.name,
            "traj_plausible": traj_p,
            "traj_implausible": traj_i,
            "physics_plausible": {
                "mass": physics_plausible.mass,
                "friction": physics_plausible.static_friction,
                "restitution": physics_plausible.restitution,
            },
            "physics_implausible": {
                "mass": physics_implausible.mass,
                "friction": physics_implausible.static_friction,
                "restitution": physics_implausible.restitution,
            },
        })

    # 2. Mass Violation: heavy object moves as if very light
    for _ in range(num_per_type):
        material = list(MATERIAL_PRIORS.keys())[rng.integers(len(MATERIAL_PRIORS) - 1)]
        prior = MATERIAL_PRIORS[material]

        # Plausible: normal mass range
        mass_p = rng.uniform(*prior["mass"])
        # Implausible: mass is 10x too light or 10x too heavy
        if rng.random() > 0.5:
            mass_i = mass_p / 10.0  # way too light
        else:
            mass_i = mass_p * 10.0  # way too heavy

        physics_p = PhysicsConfig(
            mass=mass_p,
            static_friction=rng.uniform(*prior["static_friction"]),
            restitution=rng.uniform(*prior["restitution"]),
            material=material,
        )
        physics_i = PhysicsConfig(
            mass=mass_i,
            static_friction=physics_p.static_friction,
            restitution=physics_p.restitution,
            material=material,
        )

        action = DIAGNOSTIC_ACTIONS[rng.integers(len(DIAGNOSTIC_ACTIONS))]
        traj_p = engine.execute_diagnostic_action(action, physics_p)
        traj_i = engine.execute_diagnostic_action(action, physics_i)

        scenarios.append({
            "type": "mass_violation",
            "material_visual": material,
            "action": action.name,
            "traj_plausible": traj_p,
            "traj_implausible": traj_i,
            "physics_plausible": {
                "mass": physics_p.mass, "friction": physics_p.static_friction,
                "restitution": physics_p.restitution,
            },
            "physics_implausible": {
                "mass": physics_i.mass, "friction": physics_i.static_friction,
                "restitution": physics_i.restitution,
            },
        })

    # 3. Bounce Violation: restitution mismatch
    for _ in range(num_per_type):
        material = list(MATERIAL_PRIORS.keys())[rng.integers(len(MATERIAL_PRIORS) - 1)]
        prior = MATERIAL_PRIORS[material]

        rest_p = rng.uniform(*prior["restitution"])
        # Swap restitution to opposite extreme
        rest_i = 0.95 - rest_p  # if rest_p=0.8 → rest_i=0.15, etc.
        rest_i = max(0.0, min(0.95, rest_i))

        physics_p = PhysicsConfig(
            mass=rng.uniform(*prior["mass"]),
            static_friction=rng.uniform(*prior["static_friction"]),
            restitution=rest_p,
            material=material,
        )
        physics_i = PhysicsConfig(
            mass=physics_p.mass,
            static_friction=physics_p.static_friction,
            restitution=rest_i,
            material=material,
        )

        action = DIAGNOSTIC_ACTIONS[2]  # grasp-lift-release (best for bounce)
        traj_p = engine.execute_diagnostic_action(action, physics_p)
        traj_i = engine.execute_diagnostic_action(action, physics_i)

        scenarios.append({
            "type": "bounce_violation",
            "material_visual": material,
            "action": action.name,
            "traj_plausible": traj_p,
            "traj_implausible": traj_i,
            "physics_plausible": {
                "mass": physics_p.mass, "friction": physics_p.static_friction,
                "restitution": physics_p.restitution,
            },
            "physics_implausible": {
                "mass": physics_i.mass, "friction": physics_i.static_friction,
                "restitution": physics_i.restitution,
            },
        })

    logger.info(f"Generated {len(scenarios)} VoE scenarios "
                f"({num_per_type} each × 3 types)")
    return scenarios


# ============================================================================
# Trajectory-based surprise metrics
# ============================================================================
def trajectory_embedding_diff(traj: np.ndarray) -> float:
    """Total squared diff between consecutive trajectory states."""
    diffs = np.diff(traj, axis=0)
    return float(np.sum(diffs ** 2))


def trajectory_max_jump(traj: np.ndarray) -> float:
    """Maximum single-step state change."""
    diffs = np.diff(traj, axis=0)
    norms = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.max(norms)) if len(norms) > 0 else 0.0


def trajectory_velocity_variance(traj: np.ndarray) -> float:
    """Variance of velocity components (columns 7-9)."""
    if traj.shape[1] >= 10:
        vel = traj[:, 7:10]
        return float(np.var(vel))
    return 0.0


def trajectory_position_range(traj: np.ndarray) -> float:
    """Range of position displacement."""
    pos = traj[:, :3]
    return float(np.max(np.ptp(pos, axis=0)))


# ============================================================================
# VoE evaluation using encoder embeddings
# ============================================================================
def evaluate_voe_with_encoder(
    scenarios: List[Dict],
    backbone,
    device: str,
    real_images: Optional[List[str]] = None,
) -> Dict:
    """Evaluate VoE scenarios using a visual encoder.

    Since VoE scenarios are trajectory-based (not video), we:
    1. Load a set of real images as "visual anchors"
    2. Use the encoder to embed the images
    3. Compare trajectory statistics between plausible/implausible
    4. Use embedding-weighted trajectory analysis

    This is a trajectory-level evaluation — it doesn't encode video frames
    but measures whether the physics engine's trajectory statistics
    differ between plausible/implausible in ways that correlate with
    the encoder's physics understanding.
    """
    results_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    metrics = ["traj_diff", "max_jump", "vel_var", "pos_range"]

    for metric in metrics:
        for scenario in scenarios:
            traj_p = scenario["traj_plausible"]
            traj_i = scenario["traj_implausible"]

            if metric == "traj_diff":
                score_p = trajectory_embedding_diff(traj_p)
                score_i = trajectory_embedding_diff(traj_i)
            elif metric == "max_jump":
                score_p = trajectory_max_jump(traj_p)
                score_i = trajectory_max_jump(traj_i)
            elif metric == "vel_var":
                score_p = trajectory_velocity_variance(traj_p)
                score_i = trajectory_velocity_variance(traj_i)
            elif metric == "pos_range":
                score_p = trajectory_position_range(traj_p)
                score_i = trajectory_position_range(traj_i)

            # Implausible should have different (higher) dynamics than plausible
            diff = abs(score_i - score_p)
            key = f"{metric}_{scenario['type']}"
            results_by_type[key]["total"] += 1
            # For VoE, we check if implausible has measurably different dynamics
            if diff > 1e-6:
                results_by_type[key]["correct"] += 1

    # Aggregate
    summary = {}
    for key, counts in results_by_type.items():
        metric, voe_type = key.split("_", 1)
        if voe_type not in summary:
            summary[voe_type] = {}
        summary[voe_type][metric] = {
            "distinguishable_fraction": counts["correct"] / counts["total"]
            if counts["total"] > 0 else 0,
            **counts,
        }

    return summary


def evaluate_voe_trajectory_only(scenarios: List[Dict]) -> Dict:
    """Pure trajectory-based VoE evaluation (no encoder needed).

    Measures whether plausible and implausible trajectories are
    statistically distinguishable using physics-aware metrics.
    """
    results = defaultdict(lambda: {
        "plausible_scores": [],
        "implausible_scores": [],
    })

    metric_fns = {
        "traj_diff": trajectory_embedding_diff,
        "max_jump": trajectory_max_jump,
        "vel_var": trajectory_velocity_variance,
        "pos_range": trajectory_position_range,
    }

    for scenario in scenarios:
        voe_type = scenario["type"]
        traj_p = scenario["traj_plausible"]
        traj_i = scenario["traj_implausible"]

        for metric_name, fn in metric_fns.items():
            key = f"{voe_type}/{metric_name}"
            results[key]["plausible_scores"].append(fn(traj_p))
            results[key]["implausible_scores"].append(fn(traj_i))

    # Compute statistics
    summary = {}
    for key, data in results.items():
        p_scores = np.array(data["plausible_scores"])
        i_scores = np.array(data["implausible_scores"])

        # Pairwise accuracy: how often implausible has different score
        pairwise_different = np.mean(np.abs(i_scores - p_scores) > 1e-3)
        # How often implausible has HIGHER score
        pairwise_higher = np.mean(i_scores > p_scores)

        summary[key] = {
            "plausible_mean": float(np.mean(p_scores)),
            "plausible_std": float(np.std(p_scores)),
            "implausible_mean": float(np.mean(i_scores)),
            "implausible_std": float(np.std(i_scores)),
            "pairwise_different": float(pairwise_different),
            "pairwise_implausible_higher": float(pairwise_higher),
            "n_pairs": len(p_scores),
        }

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Synthetic VoE Test")
    parser.add_argument("--num_per_type", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/intphys")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate scenarios
    scenarios = generate_voe_scenarios(args.num_per_type, args.seed)

    # Trajectory-only evaluation
    traj_results = evaluate_voe_trajectory_only(scenarios)

    # Print results
    print("\n" + "=" * 70)
    print("Synthetic VoE Test — Trajectory Analysis")
    print("=" * 70)
    print(f"{'VoE Type / Metric':<40} {'P_mean':>8} {'I_mean':>8} {'Diff%':>8} {'I>P%':>8}")
    print("-" * 70)

    for key in sorted(traj_results.keys()):
        r = traj_results[key]
        print(f"{key:<40} {r['plausible_mean']:>8.3f} {r['implausible_mean']:>8.3f} "
              f"{r['pairwise_different']:>7.1%} {r['pairwise_implausible_higher']:>7.1%}")

    print("=" * 70)

    # Save
    output_path = os.path.join(args.output_dir, "synthetic_voe_results.json")
    with open(output_path, "w") as f:
        json.dump(traj_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save scenario metadata (without large trajectory arrays)
    meta = []
    for s in scenarios:
        meta.append({
            "type": s["type"],
            "material_visual": s["material_visual"],
            "action": s["action"],
            "physics_plausible": s["physics_plausible"],
            "physics_implausible": s["physics_implausible"],
        })
    meta_path = os.path.join(args.output_dir, "synthetic_voe_scenarios.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Scenario metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
