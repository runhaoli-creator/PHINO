#!/usr/bin/env python3
"""
Extract training loss curves from DynaCLIP V2 and V3 training logs
and generate comparison plots.

Outputs:
  - results/intphys/loss_curves_v2_v3.json  (raw data)
  - paper/figures/loss_curves_v2_v3.pdf      (plot)
  - paper/figures/loss_curves_v2_v3.png      (plot)
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "intphys"
FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"


def parse_training_log(log_path: str) -> dict:
    """Parse a DynaCLIP training log to extract loss, LR, temperature curves."""
    steps, losses, lrs_backbone, lrs_head, temps = [], [], [], [], []

    pattern = re.compile(
        r"Step (\d+)/\d+ \| Loss: ([\d.]+) \| LR: \[([\d.e+-]+),\s*([\d.e+-]+)"
        r"(?:,\s*([\d.e+-]+))?\] \| Temp: ([\d.]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                lr_b = float(m.group(3))
                lr_h = float(m.group(4))
                temp = float(m.group(6))
                steps.append(step)
                losses.append(loss)
                lrs_backbone.append(lr_b)
                lrs_head.append(lr_h)
                temps.append(temp)

    return {
        "steps": steps,
        "losses": losses,
        "lrs_backbone": lrs_backbone,
        "lrs_head": lrs_head,
        "temperatures": temps,
    }


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def plot_loss_curves(v2_data: dict, v3_data: dict, output_dir: str, fig_dir: str):
    """Generate comprehensive loss curve comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel 1: Loss curves ---
    ax = axes[0]
    ax.plot(v2_data["steps"], v2_data["losses"], alpha=0.3, color="tab:blue", linewidth=0.8)
    ax.plot(v3_data["steps"], v3_data["losses"], alpha=0.3, color="tab:orange", linewidth=0.8)
    # Smoothed
    w = 10
    if len(v2_data["losses"]) > w:
        s2 = smooth(v2_data["losses"], w)
        ax.plot(v2_data["steps"][w-1:], s2, color="tab:blue", linewidth=2, label="V2 (param. dist.)")
    if len(v3_data["losses"]) > w:
        s3 = smooth(v3_data["losses"], w)
        ax.plot(v3_data["steps"][w-1:], s3, color="tab:orange", linewidth=2, label="V3 (traj. sim.)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Temperature ---
    ax = axes[1]
    ax.plot(v2_data["steps"], v2_data["temperatures"], color="tab:blue", linewidth=1.5, label="V2")
    ax.plot(v3_data["steps"], v3_data["temperatures"], color="tab:orange", linewidth=1.5, label="V3")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Temperature (τ)")
    ax.set_title("Learned Temperature")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Learning Rate ---
    ax = axes[2]
    ax.plot(v2_data["steps"], v2_data["lrs_backbone"], color="tab:blue", linewidth=1.5, label="V2 backbone")
    ax.plot(v3_data["steps"], v3_data["lrs_backbone"], color="tab:orange", linewidth=1.5, label="V3 backbone")
    ax.plot(v2_data["steps"], v2_data["lrs_head"], color="tab:blue", linewidth=1.5, linestyle="--", label="V2 head", alpha=0.6)
    ax.plot(v3_data["steps"], v3_data["lrs_head"], color="tab:orange", linewidth=1.5, linestyle="--", label="V3 head", alpha=0.6)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend(fontsize=8, ncol=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(fig_dir, f"loss_curves_v2_v3.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()


def main():
    v2_log = LOG_DIR / "pretrain_v2.log"
    v3_log = LOG_DIR / "pretrain_v3.log"

    if not v2_log.exists():
        # Try train_v2.log as alternative
        v2_log = LOG_DIR / "train_v2.log"
    if not v3_log.exists():
        print(f"V3 log not found at {v3_log}")
        sys.exit(1)

    print(f"Parsing V2 log: {v2_log}")
    v2_data = parse_training_log(str(v2_log))
    print(f"  Found {len(v2_data['steps'])} data points, "
          f"steps {v2_data['steps'][0]}–{v2_data['steps'][-1]}")
    print(f"  Loss: {v2_data['losses'][0]:.4f} → {v2_data['losses'][-1]:.4f}")

    print(f"Parsing V3 log: {v3_log}")
    v3_data = parse_training_log(str(v3_log))
    print(f"  Found {len(v3_data['steps'])} data points, "
          f"steps {v3_data['steps'][0]}–{v3_data['steps'][-1]}")
    print(f"  Loss: {v3_data['losses'][0]:.4f} → {v3_data['losses'][-1]:.4f}")

    # Save raw data
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    raw_path = OUTPUT_DIR / "loss_curves_v2_v3.json"
    with open(raw_path, "w") as f:
        json.dump({"v2": v2_data, "v3": v3_data}, f, indent=2)
    print(f"Raw data saved to {raw_path}")

    # Plot
    plot_loss_curves(v2_data, v3_data, str(OUTPUT_DIR), str(FIG_DIR))
    print("Done!")


if __name__ == "__main__":
    main()
