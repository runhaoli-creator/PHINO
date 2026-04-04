#!/usr/bin/env python3
"""Merge parallel LIBERO-10 v4 v2 results from multiple GPU runs into one file."""
import json
import glob
import numpy as np
from pathlib import Path

results_dir = Path("results/libero10_v4_v2")

# Collect all DynaCLIP result files
files = sorted(results_dir.glob("libero10_v4_DynaCLIP*.json"))
if not files:
    print("No result files found yet.")
    exit(1)

merged_tasks = {}
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    if "DynaCLIP" in data:
        for task_name, task_res in data["DynaCLIP"]["per_task"].items():
            merged_tasks[task_name] = task_res

print(f"Merged {len(merged_tasks)} tasks from {len(files)} files")

if merged_tasks:
    all_means = [v["mean"] for v in merged_tasks.values()]
    merged = {
        "DynaCLIP_v2": {
            "per_task": merged_tasks,
            "avg_success_rate": float(np.mean(all_means)),
            "std_across_tasks": float(np.std(all_means)),
            "n_tasks": len(merged_tasks),
            "version": "v4_v2",
        }
    }
    
    out_path = results_dir / "libero10_v4_DynaCLIP_v2_merged.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    
    print(f"\nDynaCLIP v2 LIBERO-10 Results ({len(merged_tasks)}/10 tasks):")
    print(f"{'='*60}")
    for task_name, tr in sorted(merged_tasks.items(), key=lambda x: x[1]["task_id"]):
        print(f"  Task {tr['task_id']:2d}: {tr['mean']:.1%} ± {tr['std']:.1%}  {task_name[:55]}")
    print(f"{'='*60}")
    print(f"  Average: {np.mean(all_means):.1%} ± {np.std(all_means):.1%}")
    print(f"\nSaved to: {out_path}")
