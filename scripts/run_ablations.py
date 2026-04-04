"""
DynaCLIP Ablation Script: Run all 8 ablation studies.

Usage: python scripts/run_ablations.py --config configs/pretrain.yaml
"""

import argparse
import logging

import yaml

from dynaclip.eval.ablations import AblationStudy
from dynaclip.utils.helpers import setup_logging, set_seed

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="DynaCLIP Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--output_dir", type=str, default="results/ablations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    setup_logging("INFO", "logs/ablations.log")
    set_seed(args.seed)

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    logger.info("=== DynaCLIP Ablation Studies ===")

    ablation = AblationStudy(
        base_config=base_config,
        output_dir=args.output_dir,
        device=args.device,
    )

    results = ablation.run_all()

    logger.info(f"All 8 ablation studies complete. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
