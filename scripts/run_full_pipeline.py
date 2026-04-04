#!/usr/bin/env python3
"""
DynaCLIP Full Pipeline: Data Generation → Training → Evaluation.

This script orchestrates the complete DynaCLIP pipeline:
  Phase 1: ManiSkill3 data generation (GPU 0)
  Phase 2: Model pretraining 100K steps (GPUs 1-5, DDP)
  Phase 3: All 8 ablation trainings (GPUs across 0-6)
  Phase 4: Full evaluation suite (all 5 experiments, all backbones)
  Phase 5: Paper results update

Usage:
  python scripts/run_full_pipeline.py --phase all
  python scripts/run_full_pipeline.py --phase data
  python scripts/run_full_pipeline.py --phase train
  python scripts/run_full_pipeline.py --phase eval
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache" / "maniskill3_data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_cmd(cmd: str, env=None, timeout=None):
    """Run a shell command and log output."""
    logger.info(f"Running: {cmd}")
    proc = subprocess.run(cmd, shell=True, env=env or os.environ.copy(),
                         capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        logger.error(f"Command failed (rc={proc.returncode}):\n{proc.stderr[:2000]}")
    else:
        logger.info(f"Command succeeded:\n{proc.stdout[-500:]}")
    return proc


# ============================================================================
# Phase 1: Data Generation
# ============================================================================
def phase_data_generation():
    """Generate full dataset using ManiSkill3 + DomainNet."""
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA GENERATION")
    logger.info("=" * 60)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use the generation module directly for more control
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from dynaclip.data.generation_maniskill3 import ManiSkill3DataGenerator
    
    generator = ManiSkill3DataGenerator(
        output_dir=str(DATA_DIR),
        num_configs=20000,
        num_physics_per_object=5,
        gpu_id=0,
        seed=42,
    )
    generator.generate_all(max_sim_pairs=500000)
    
    logger.info("Phase 1 complete!")


# ============================================================================
# Phase 2: Training (100K steps)
# ============================================================================
def phase_training(resume_from: str = None):
    """Train DynaCLIP for 100K steps on 5 GPUs."""
    logger.info("=" * 60)
    logger.info("PHASE 2: PRETRAINING (100K steps)")
    logger.info("=" * 60)
    
    # Update config for 100K steps
    config = {
        "model": {
            "backbone": "dinov2_vitb14",
            "embed_dim": 512,
            "freeze_backbone": False,
            "unfreeze_last_n_blocks": -1,
            "gradient_checkpointing": True,
        },
        "loss": {
            "name": "soft_infonce",
            "temperature": 0.07,
            "similarity_temperature": 0.1,
            "learnable_temperature": True,
        },
        "training": {
            "backbone_lr": 1e-5,
            "head_lr": 1e-3,
            "weight_decay": 0.05,
            "warmup_steps": 1000,
            "total_steps": 100000,
            "batch_size": 640,
            "grad_accum_steps": 2,
            "use_bf16": True,
        },
        "data": {
            "data_dir": str(DATA_DIR),
            "num_pairs": 500000,
            "hard_neg_ratio": 0.3,
            "hard_pos_ratio": 0.3,
            "num_workers": 4,
        },
        "logging": {
            "log_every": 100,
            "eval_every": 5000,
            "save_every": 5000,
            "use_wandb": False,
            "project_name": "dynaclip",
        },
        "output": {
            "checkpoint_dir": str(CHECKPOINT_DIR / "pretrain_v3"),
            "log_dir": str(PROJECT_ROOT / "logs" / "pretrain_v3"),
        },
    }
    
    config_path = PROJECT_ROOT / "configs" / "pretrain_v3.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Check if we should use existing data or the new ManiSkill3 data
    data_dir = config["data"]["data_dir"]
    if not Path(data_dir).exists() or not (Path(data_dir) / "metadata.json").exists():
        # Fall back to existing data
        data_dir = str(PROJECT_ROOT / "data_cache" / "dynaclip_data")
        config["data"]["data_dir"] = data_dir
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Using existing data at {data_dir}")
    
    # Launch DDP training on GPUs 1-5 (avoid GPU 7 which is in use)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
    
    cmd = (
        f"cd {PROJECT_ROOT} && "
        f"torchrun --nproc_per_node=5 --master_port=29500 "
        f"scripts/pretrain.py --config {config_path}"
    )
    if resume_from:
        cmd += f" --resume {resume_from}"
    
    run_cmd(cmd, env=env)
    logger.info("Phase 2 complete!")


# ============================================================================
# Phase 3: Ablation Training
# ============================================================================
def phase_ablations():
    """Run all 8 ablation trainings in parallel across GPUs."""
    logger.info("=" * 60)
    logger.info("PHASE 3: ABLATION TRAININGS")
    logger.info("=" * 60)
    
    import yaml
    
    # Check which data directory to use
    data_dir = str(DATA_DIR)
    if not Path(data_dir).exists() or not (Path(data_dir) / "metadata.json").exists():
        data_dir = str(PROJECT_ROOT / "data_cache" / "dynaclip_data")
    
    ablation_configs = [
        # (1) Frozen backbone
        {"name": "frozen_backbone", "gpu": 0, "steps": 10000,
         "overrides": {"model.freeze_backbone": True}},
        # (2) Standard InfoNCE
        {"name": "infonce", "gpu": 1, "steps": 10000,
         "overrides": {"loss.name": "infonce"}},
        # (3) Triplet loss
        {"name": "triplet", "gpu": 2, "steps": 10000,
         "overrides": {"loss.name": "triplet"}},
        # (4) BYOL loss
        {"name": "byol", "gpu": 3, "steps": 10000,
         "overrides": {"loss.name": "byol"}},
        # (5) No hard negative mining
        {"name": "no_hard_neg", "gpu": 4, "steps": 10000,
         "overrides": {"data.hard_neg_ratio": 0.0, "data.hard_pos_ratio": 0.0}},
        # (6) Random physics (no material correlation)
        {"name": "random_physics", "gpu": 5, "steps": 10000,
         "overrides": {"random_physics": True}},
        # (7) Unfreeze last 2 blocks only
        {"name": "last2_blocks", "gpu": 6, "steps": 10000,
         "overrides": {"model.unfreeze_last_n_blocks": 2}},
        # (8) Unfreeze last 4 blocks
        {"name": "last4_blocks", "gpu": 0, "steps": 10000,
         "overrides": {"model.unfreeze_last_n_blocks": 4}},
    ]
    
    processes = []
    for ablation in ablation_configs:
        name = ablation["name"]
        gpu = ablation["gpu"]
        steps = ablation["steps"]
        
        # Create ablation config
        config = {
            "model": {
                "backbone": "dinov2_vitb14",
                "embed_dim": 512,
                "freeze_backbone": False,
                "unfreeze_last_n_blocks": -1,
                "gradient_checkpointing": True,
            },
            "loss": {
                "name": "soft_infonce",
                "temperature": 0.07,
                "similarity_temperature": 0.1,
                "learnable_temperature": True,
            },
            "training": {
                "backbone_lr": 1e-5,
                "head_lr": 1e-3,
                "weight_decay": 0.05,
                "warmup_steps": 200,
                "total_steps": steps,
                "batch_size": 128,
                "grad_accum_steps": 2,
                "use_bf16": True,
            },
            "data": {
                "data_dir": data_dir,
                "num_pairs": 200000,
                "hard_neg_ratio": 0.3,
                "hard_pos_ratio": 0.3,
                "num_workers": 4,
            },
            "logging": {
                "log_every": 100,
                "eval_every": 2000,
                "save_every": 5000,
                "use_wandb": False,
            },
            "output": {
                "checkpoint_dir": str(CHECKPOINT_DIR / f"ablation_{name}"),
                "log_dir": str(PROJECT_ROOT / "logs" / f"ablation_{name}"),
            },
        }
        
        # Apply overrides
        for key, val in ablation.get("overrides", {}).items():
            parts = key.split(".")
            d = config
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = val
        
        config_path = PROJECT_ROOT / "configs" / f"ablation_{name}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        cmd = (
            f"cd {PROJECT_ROOT} && "
            f"python scripts/pretrain.py --config {config_path}"
        )
        
        logger.info(f"Launching ablation '{name}' on GPU {gpu}")
        proc = subprocess.Popen(cmd, shell=True, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((name, proc))
    
    # Wait for all ablations
    for name, proc in processes:
        proc.wait()
        logger.info(f"Ablation '{name}' completed with rc={proc.returncode}")
    
    logger.info("Phase 3 complete!")


# ============================================================================
# Phase 4: Full Evaluation
# ============================================================================
def phase_evaluation():
    """Run all 5 experiments with all backbones."""
    logger.info("=" * 60)
    logger.info("PHASE 4: FULL EVALUATION")
    logger.info("=" * 60)
    
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Determine checkpoint path
    checkpoint_v3 = CHECKPOINT_DIR / "pretrain_v3" / "dynaclip_final.pt"
    checkpoint_v1 = CHECKPOINT_DIR / "pretrain" / "dynaclip_final.pt"
    checkpoint = str(checkpoint_v3) if checkpoint_v3.exists() else str(checkpoint_v1)
    
    # Determine data directory
    data_dir = str(DATA_DIR)
    if not Path(data_dir).exists() or not (Path(data_dir) / "metadata.json").exists():
        data_dir = str(PROJECT_ROOT / "data_cache" / "dynaclip_data")
    
    device = "cuda:0"
    results = {}
    
    # === Experiment 1: Physics Linear Probing ===
    logger.info("--- Exp 1: Physics Linear Probing ---")
    results["linear_probing"] = run_linear_probing(checkpoint, data_dir, device)
    
    # === Experiment 2: Invisible Physics Test ===
    logger.info("--- Exp 2: Invisible Physics Test ---")
    results["invisible_physics"] = run_invisible_physics(checkpoint, data_dir, device)
    
    # === Experiment 3: k-NN Physics Inference ===
    logger.info("--- Exp 3: k-NN Physics Inference ---")
    results["knn"] = run_knn_evaluation(checkpoint, data_dir, device)
    
    # === Experiment 4: Material Clustering ===
    logger.info("--- Exp 4: Material Clustering ---")
    results["clustering"] = run_clustering(checkpoint, data_dir, device)
    
    # === Experiment 5: Cross-Domain Transfer ===
    logger.info("--- Exp 5: Cross-Domain Transfer ---")
    results["cross_domain"] = run_cross_domain_transfer(checkpoint, data_dir, device)
    
    # Save all results
    results_dir = RESULTS_DIR / "final_v3"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All results saved to {results_dir}")
    
    # === Ablation evaluation ===
    logger.info("--- Ablation Evaluation ---")
    ablation_results = run_ablation_evaluation(data_dir, device)
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
    
    logger.info("Phase 4 complete!")


def run_linear_probing(checkpoint: str, data_dir: str, device: str) -> dict:
    """Experiment 1: Linear probing for physics property prediction.
    
    Tests ALL backbones: DynaCLIP, DINOv2-B/14, DINOv2-L/14, CLIP, SigLIP, R3M, VIP
    """
    from dynaclip.eval.linear_probing import PhysicsLinearProbe
    from dynaclip.models.backbones import BACKBONE_REGISTRY
    
    # All backbones to evaluate
    backbone_configs = [
        ("dynaclip", {"checkpoint_path": checkpoint}),
        ("dynaclip_backbone", {"checkpoint_path": checkpoint}),  # backbone features
        ("dinov2_vitb14", {}),
        ("dinov2_vitl14", {}),
        ("clip_vitl14", {}),
        ("siglip", {}),
    ]
    
    # Try adding R3M and VIP
    try:
        from r3m import load_r3m
        backbone_configs.append(("r3m", {}))
    except ImportError:
        logger.warning("R3M not available, skipping")
    
    try:
        from vip import load_vip
        backbone_configs.append(("vip", {}))
    except ImportError:
        logger.warning("VIP not available, skipping")
    
    results = {}
    properties = ["mass", "static_friction", "restitution"]
    
    for backbone_name, kwargs in backbone_configs:
        logger.info(f"  Linear probing with {backbone_name}")
        try:
            probe = PhysicsLinearProbe(
                backbone_name=backbone_name,
                data_dir=data_dir,
                device=device,
                **kwargs,
            )
            r2_scores = probe.train_and_evaluate(properties=properties, n_seeds=5)
            results[backbone_name] = r2_scores
            logger.info(f"    {backbone_name}: {r2_scores}")
        except Exception as e:
            logger.warning(f"    {backbone_name} failed: {e}")
            results[backbone_name] = {"error": str(e)}
    
    return results


def run_invisible_physics(checkpoint: str, data_dir: str, device: str) -> dict:
    """Experiment 2: Invisible physics discrimination."""
    try:
        from dynaclip.eval.invisible_physics import InvisiblePhysicsEvaluator
        
        evaluator = InvisiblePhysicsEvaluator(
            checkpoint_path=checkpoint,
            data_dir=data_dir,
            device=device,
        )
        results = evaluator.evaluate()
        logger.info(f"  Invisible physics: {results}")
        return results
    except Exception as e:
        logger.warning(f"  Invisible physics failed: {e}")
        return {"error": str(e)}


def run_knn_evaluation(checkpoint: str, data_dir: str, device: str) -> dict:
    """Experiment 3: k-NN physics property prediction."""
    from dynaclip.models.backbones import BACKBONE_REGISTRY
    
    backbone_configs = [
        ("dynaclip", {"checkpoint_path": checkpoint}),
        ("dinov2_vitb14", {}),
        ("clip_vitl14", {}),
        ("siglip", {}),
    ]
    
    try:
        from r3m import load_r3m
        backbone_configs.append(("r3m", {}))
    except ImportError:
        pass
    try:
        from vip import load_vip
        backbone_configs.append(("vip", {}))
    except ImportError:
        pass
    
    results = {}
    for backbone_name, kwargs in backbone_configs:
        try:
            from dynaclip.eval.zero_shot import KNNPhysicsEvaluator
            evaluator = KNNPhysicsEvaluator(
                backbone_name=backbone_name,
                data_dir=data_dir,
                device=device,
                **kwargs,
            )
            knn_results = evaluator.evaluate(k_values=[1, 3, 5, 10])
            results[backbone_name] = knn_results
            logger.info(f"  k-NN {backbone_name}: {knn_results}")
        except Exception as e:
            logger.warning(f"  k-NN {backbone_name} failed: {e}")
            results[backbone_name] = {"error": str(e)}
    
    return results


def run_clustering(checkpoint: str, data_dir: str, device: str) -> dict:
    """Experiment 4: Material-based clustering."""
    from dynaclip.models.backbones import BACKBONE_REGISTRY
    
    backbone_configs = [
        ("dynaclip", {"checkpoint_path": checkpoint}),
        ("dinov2_vitb14", {}),
        ("clip_vitl14", {}),
        ("siglip", {}),
    ]
    
    try:
        from r3m import load_r3m
        backbone_configs.append(("r3m", {}))
    except ImportError:
        pass
    try:
        from vip import load_vip
        backbone_configs.append(("vip", {}))
    except ImportError:
        pass
    
    results = {}
    for backbone_name, kwargs in backbone_configs:
        try:
            from dynaclip.eval.linear_probing import MaterialClusteringEvaluator
            evaluator = MaterialClusteringEvaluator(
                backbone_name=backbone_name,
                data_dir=data_dir,
                device=device,
                **kwargs,
            )
            cluster_results = evaluator.evaluate()
            results[backbone_name] = cluster_results
            logger.info(f"  Clustering {backbone_name}: {cluster_results}")
        except Exception as e:
            logger.warning(f"  Clustering {backbone_name} failed: {e}")
            results[backbone_name] = {"error": str(e)}
    
    return results


def run_cross_domain_transfer(checkpoint: str, data_dir: str, device: str) -> dict:
    """Experiment 5: Cross-domain transfer evaluation.
    
    Use DomainNet domains (sketch, clipart, painting) as target domains.
    Train linear probe on 'real' domain, test on other domains.
    """
    logger.info("  Cross-domain transfer: real → {sketch, clipart, painting}")
    
    results = {}
    domains = ["sketch", "clipart", "painting"]
    
    for target_domain in domains:
        try:
            from dynaclip.eval.cross_domain import CrossDomainEvaluator
            evaluator = CrossDomainEvaluator(
                checkpoint_path=checkpoint,
                source_domain="real",
                target_domain=target_domain,
                data_dir=data_dir,
                device=device,
            )
            domain_results = evaluator.evaluate()
            results[target_domain] = domain_results
            logger.info(f"    real→{target_domain}: {domain_results}")
        except Exception as e:
            logger.warning(f"    Cross-domain {target_domain} failed: {e}")
            results[target_domain] = {"error": str(e)}
    
    return results


def run_ablation_evaluation(data_dir: str, device: str) -> dict:
    """Evaluate all ablation checkpoints."""
    ablation_names = [
        "frozen_backbone", "infonce", "triplet", "byol",
        "no_hard_neg", "random_physics", "last2_blocks", "last4_blocks",
    ]
    
    results = {}
    properties = ["mass", "static_friction", "restitution"]
    
    for name in ablation_names:
        ckpt_dir = CHECKPOINT_DIR / f"ablation_{name}"
        ckpt = ckpt_dir / "dynaclip_final.pt"
        if not ckpt.exists():
            # Try step checkpoints
            ckpts = sorted(ckpt_dir.glob("dynaclip_step_*.pt"))
            if ckpts:
                ckpt = ckpts[-1]
            else:
                logger.warning(f"No checkpoint found for ablation '{name}'")
                results[name] = {"error": "no checkpoint"}
                continue
        
        try:
            from dynaclip.eval.linear_probing import PhysicsLinearProbe
            probe = PhysicsLinearProbe(
                backbone_name="dynaclip",
                data_dir=data_dir,
                device=device,
                checkpoint_path=str(ckpt),
            )
            r2 = probe.train_and_evaluate(properties=properties, n_seeds=3)
            results[name] = r2
            logger.info(f"  Ablation '{name}': {r2}")
        except Exception as e:
            logger.warning(f"  Ablation '{name}' eval failed: {e}")
            results[name] = {"error": str(e)}
    
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="DynaCLIP Full Pipeline")
    parser.add_argument("--phase", default="all",
                       choices=["all", "data", "train", "ablations", "eval"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    if args.phase in ("all", "data"):
        phase_data_generation()
    
    if args.phase in ("all", "train"):
        phase_training(resume_from=args.resume)
    
    if args.phase in ("all", "ablations"):
        phase_ablations()
    
    if args.phase in ("all", "eval"):
        phase_evaluation()
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
