#!/usr/bin/env python3
"""
Run LIBERO-10 and CALVIN evaluations in parallel across multiple GPUs.

Usage:
  python scripts/run_libero_calvin_eval.py --gpus 0,1,2,3,4,5,6,7
"""

import os
import sys
import json
import subprocess
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONDA_PREFIX = "/home/kztrgg/miniconda3/bin/conda"


def run_cmd(cmd, gpu_id=0, log_file=None, env_extra=None):
    """Run a command with specific GPU and env vars."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MUJOCO_GL"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    if env_extra:
        env.update(env_extra)

    full_cmd = f"{CONDA_PREFIX} run -n dynaclip --no-capture-output {cmd}"

    if log_file:
        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                full_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT,
                env=env, cwd=str(PROJECT_ROOT)
            )
    else:
        proc = subprocess.Popen(
            full_cmd, shell=True, env=env, cwd=str(PROJECT_ROOT)
        )
    return proc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--skip_libero", action="store_true")
    parser.add_argument("--skip_calvin", action="store_true")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    log_dir = PROJECT_ROOT / "logs" / "eval_downstream"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []

    # LIBERO-10: Split tasks across GPUs
    if not args.skip_libero:
        # Split 10 tasks across available GPUs
        tasks_per_gpu = 10 // len(gpus)
        remainder = 10 % len(gpus)

        for i, gpu_id in enumerate(gpus):
            start = i * tasks_per_gpu + min(i, remainder)
            n_tasks = tasks_per_gpu + (1 if i < remainder else 0)
            if n_tasks == 0:
                continue
            task_ids = ",".join(str(t) for t in range(start, start + n_tasks))

            cmd = (
                f"python scripts/evaluate_libero.py "
                f"--gpu 0 --tasks {task_ids} "
                f"--n_seeds {args.n_seeds} --n_episodes {args.n_episodes} "
                f"--n_epochs {args.n_epochs} "
                f"--output results/libero10/gpu{gpu_id}"
            )

            log_file = log_dir / f"libero_gpu{gpu_id}.log"
            log.info(f"GPU {gpu_id}: LIBERO tasks {task_ids}")
            proc = run_cmd(cmd, gpu_id=gpu_id, log_file=str(log_file))
            procs.append(("libero", gpu_id, proc, log_file))

    # CALVIN: Run on a separate GPU
    if not args.skip_calvin:
        calvin_gpu = gpus[-1] if len(gpus) > 1 else gpus[0]
        cmd = (
            f"python scripts/evaluate_calvin.py "
            f"--gpu 0 "
            f"--n_seeds {args.n_seeds} --n_epochs {args.n_epochs}"
        )
        log_file = log_dir / f"calvin_gpu{calvin_gpu}.log"
        log.info(f"GPU {calvin_gpu}: CALVIN")
        proc = run_cmd(cmd, gpu_id=calvin_gpu, log_file=str(log_file))
        procs.append(("calvin", calvin_gpu, proc, log_file))

    # Wait for all processes
    log.info(f"\nWaiting for {len(procs)} evaluation jobs...")
    for name, gpu_id, proc, log_file in procs:
        proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        log.info(f"  {name} GPU {gpu_id}: {status} (log: {log_file})")

    # Merge LIBERO results
    if not args.skip_libero:
        merge_libero_results(PROJECT_ROOT / "results" / "libero10")

    log.info("\nAll evaluations complete!")


def merge_libero_results(results_dir: Path):
    """Merge per-GPU LIBERO results into a single file."""
    merged = {}

    for gpu_dir in sorted(results_dir.glob("gpu*")):
        result_file = gpu_dir / "libero10_results.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            for backbone, res in data.items():
                if backbone not in merged:
                    merged[backbone] = {"per_task": {}, "feature_dim": res.get("feature_dim", 768)}
                merged[backbone]["per_task"].update(res.get("per_task", {}))

    # Compute overall averages
    for backbone in merged:
        task_means = [v["mean"] for v in merged[backbone]["per_task"].values()]
        merged[backbone]["avg_success_rate"] = float(np.mean(task_means)) if task_means else 0.0
        merged[backbone]["std_success_rate"] = float(np.std(task_means)) if task_means else 0.0

    output_file = results_dir / "libero10_results_merged.json"
    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)
    log.info(f"Merged LIBERO results → {output_file}")

    # Print table
    import numpy as np
    log.info("\nLIBERO-10 Merged Results:")
    log.info(f"{'Backbone':<20} {'Avg SR':>10} {'n_tasks':>10}")
    log.info("-" * 40)
    for name, res in sorted(merged.items(), key=lambda x: -x[1].get("avg_success_rate", 0)):
        n = len(res["per_task"])
        log.info(f"{name:<20} {res['avg_success_rate']:>10.1%} {n:>10}")


if __name__ == "__main__":
    import numpy as np
    main()
