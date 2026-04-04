"""
Downstream Policy Evaluation (Future Work).

This module is a placeholder for downstream robotic manipulation evaluation.
Integration with simulation benchmarks (LIBERO, CALVIN, ManiSkill3, etc.)
requires installing their respective environments and training manipulation
policies on top of frozen DynaCLIP features.

This is explicitly listed as future work in the paper. The current evaluation
suite focuses on representation quality via linear probing, k-NN inference,
and material clustering (see evaluate_full.py).

Planned benchmarks:
  1. LIBERO-10/LIBERO-Long: requires libero package + MuJoCo
  2. CALVIN ABC→D: requires CALVIN environment
  3. ManiSkill3: requires maniskill3 package
  4. Physics-Varying: custom environment with variable physics

To evaluate downstream, one would:
  1. Install the appropriate simulator
  2. Freeze DynaCLIP features as the visual backbone
  3. Train a policy head (e.g., Diffusion Policy, ACT)
  4. Evaluate success rate over episodes
"""

import logging

logger = logging.getLogger(__name__)


def check_downstream_availability():
    """Check which downstream simulators are available."""
    available = {}

    try:
        import libero
        available["libero"] = True
    except ImportError:
        available["libero"] = False

    try:
        import mani_skill
        available["maniskill"] = True
    except ImportError:
        available["maniskill"] = False

    for env, status in available.items():
        if status:
            logger.info(f"  {env}: available")
        else:
            logger.info(f"  {env}: not installed (downstream eval not possible)")

    return available
