#!/usr/bin/env python3
"""
Compile all IntPhys2 benchmark results into comprehensive JSON and generate figures.

Reads:
  - results/intphys/intphys_results.json (main eval, stride=2)
  - results/intphys_stride{1,4,8}/intphys_results.json (stride sensitivity)
  - results/intphys/loss_curves_v2_v3.json
  - results/intphys/synthetic_voe_results.json

Produces:
  - results/intphys/comprehensive_results.json
  - paper/figures/intphys_accuracy_bar.pdf
  - paper/figures/intphys_stride_sensitivity.pdf
  - paper/figures/intphys_condition_heatmap.pdf
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "intphys"
FIGURES_DIR = ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def compile_results():
    """Load all results into a single dict."""
    compiled = {}

    # Main results (stride=2)
    main_path = RESULTS_DIR / "intphys_results.json"
    if main_path.exists():
        compiled["main_eval"] = load_json(main_path)
        compiled["main_eval"]["_meta"] = {"split": "Main", "frame_stride": 2, "max_frames": 100}

    # Stride sensitivity
    stride_results = {}
    for stride in [1, 2, 4, 8]:
        if stride == 2:
            path = RESULTS_DIR / "intphys_results.json"
        else:
            path = ROOT / "results" / f"intphys_stride{stride}" / "intphys_results.json"
        if path.exists():
            data = load_json(path)
            stride_results[stride] = {}
            for name, result in data.items():
                if "error" not in result:
                    stride_results[stride][name] = {
                        "best_metric": result["best_metric"],
                        "best_accuracy": result["best_accuracy"],
                        "all_accuracies": {
                            m: result["all_metrics"][m]["overall"]["accuracy"]
                            for m in result["all_metrics"]
                        },
                    }
    compiled["stride_sensitivity"] = stride_results

    # Loss curves
    loss_path = RESULTS_DIR / "loss_curves_v2_v3.json"
    if loss_path.exists():
        compiled["loss_curves"] = load_json(loss_path)

    # Synthetic VoE
    voe_path = RESULTS_DIR / "synthetic_voe_results.json"
    if voe_path.exists():
        compiled["synthetic_voe"] = load_json(voe_path)

    return compiled


def plot_accuracy_bar(compiled):
    """Bar chart of overall accuracy for all backbones (prediction_error metric)."""
    main = compiled["main_eval"]
    names = []
    accs = []
    colors = []

    # Define order + colors
    backbone_order = ["DynaCLIP-V2", "DynaCLIP-V3", "dinov2_vitb14", "dinov2_vitl14",
                      "clip_vitl14", "siglip_vitb16"]
    color_map = {
        "DynaCLIP-V2": "#2196F3",
        "DynaCLIP-V3": "#1565C0",
        "dinov2_vitb14": "#4CAF50",
        "dinov2_vitl14": "#388E3C",
        "clip_vitl14": "#FF9800",
        "siglip_vitb16": "#F44336",
    }

    for name in backbone_order:
        if name in main and "error" not in main[name]:
            names.append(name)
            accs.append(main[name]["best_accuracy"] * 100)
            colors.append(color_map.get(name, "#9E9E9E"))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    # Add value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=10)
    ax.set_ylabel("Pairwise Accuracy (%)", fontsize=12)
    ax.set_title("IntPhys2 Violation-of-Expectation Detection\n(Main Split, prediction_error metric)", fontsize=13)
    ax.set_ylim(40, 75)
    ax.legend(loc="upper left", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "intphys_accuracy_bar.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "intphys_accuracy_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved accuracy bar chart to {FIGURES_DIR / 'intphys_accuracy_bar.pdf'}")


def plot_stride_sensitivity(compiled):
    """Line plot showing accuracy vs stride for DynaCLIP-V2 and DINOv2."""
    stride_data = compiled.get("stride_sensitivity", {})
    if not stride_data:
        print("No stride sensitivity data found")
        return

    strides = sorted([int(s) for s in stride_data.keys()])
    models = ["DynaCLIP-V2", "dinov2_vitb14"]
    model_labels = {"DynaCLIP-V2": "DynaCLIP-V2", "dinov2_vitb14": "DINOv2-ViT-B/14"}
    model_colors = {"DynaCLIP-V2": "#2196F3", "dinov2_vitb14": "#4CAF50"}
    model_markers = {"DynaCLIP-V2": "o", "dinov2_vitb14": "s"}

    fig, ax = plt.subplots(figsize=(7, 5))

    for model in models:
        accs = []
        valid_strides = []
        for s in strides:
            if model in stride_data.get(s, {}):
                accs.append(stride_data[s][model]["best_accuracy"] * 100)
                valid_strides.append(s)

        if accs:
            ax.plot(valid_strides, accs, marker=model_markers[model], linewidth=2,
                    markersize=8, label=model_labels[model], color=model_colors[model])
            for s, acc in zip(valid_strides, accs):
                ax.annotate(f"{acc:.1f}%", (s, acc), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=9)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random chance")
    ax.set_xlabel("Frame Stride", fontsize=12)
    ax.set_ylabel("Pairwise Accuracy (%)", fontsize=12)
    ax.set_title("IntPhys2: Stride Sensitivity Analysis", fontsize=13)
    ax.set_xticks(strides)
    ax.set_ylim(45, 70)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "intphys_stride_sensitivity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "intphys_stride_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stride sensitivity plot to {FIGURES_DIR / 'intphys_stride_sensitivity.pdf'}")


def plot_condition_heatmap(compiled):
    """Heatmap showing per-condition accuracy for all backbones."""
    main = compiled["main_eval"]

    backbone_order = ["DynaCLIP-V2", "DynaCLIP-V3", "dinov2_vitb14", "dinov2_vitl14",
                      "clip_vitl14", "siglip_vitb16"]
    conditions = ["continuity", "immutability", "permanence", "solidity"]
    cond_labels = ["Continuity", "Immutability", "Permanence", "Solidity"]

    # Build matrix
    matrix = np.zeros((len(backbone_order), len(conditions)))
    valid_backbones = []
    for i, name in enumerate(backbone_order):
        if name in main and "error" not in main[name]:
            valid_backbones.append(name)
            best = main[name]["best_metric"]
            by_cond = main[name]["all_metrics"][best].get("by_condition", {})
            for j, cond in enumerate(conditions):
                if cond in by_cond:
                    matrix[i, j] = by_cond[cond]["accuracy"] * 100
                else:
                    matrix[i, j] = 50.0  # default

    matrix = matrix[:len(valid_backbones)]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=45, vmax=75, aspect="auto")

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(cond_labels, fontsize=11)
    ax.set_yticks(np.arange(len(valid_backbones)))
    ax.set_yticklabels([n.replace("_", "\n") for n in valid_backbones], fontsize=10)

    # Add text annotations
    for i in range(len(valid_backbones)):
        for j in range(len(conditions)):
            color = "white" if matrix[i, j] < 55 or matrix[i, j] > 68 else "black"
            ax.text(j, i, f"{matrix[i, j]:.1f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_title("IntPhys2: Per-Condition Accuracy (prediction_error)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "intphys_condition_heatmap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "intphys_condition_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved condition heatmap to {FIGURES_DIR / 'intphys_condition_heatmap.pdf'}")


def plot_metric_comparison(compiled):
    """Grouped bar chart comparing 3 surprise metrics across backbones."""
    main = compiled["main_eval"]

    backbone_order = ["DynaCLIP-V2", "DynaCLIP-V3", "dinov2_vitb14", "dinov2_vitl14",
                      "clip_vitl14", "siglip_vitb16"]
    metrics = ["embedding_diff", "max_jump", "prediction_error"]
    metric_labels = ["Embedding Diff", "Max Jump", "Prediction Error"]
    metric_colors = ["#42A5F5", "#66BB6A", "#EF5350"]

    valid = [(n, main[n]) for n in backbone_order if n in main and "error" not in main[n]]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(valid))
    width = 0.25

    for k, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        accs = []
        for name, result in valid:
            acc = result["all_metrics"][metric]["overall"]["accuracy"] * 100
            accs.append(acc)
        offset = (k - 1) * width
        bars = ax.bar(x + offset, accs, width, label=label, color=color, alpha=0.85)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n, _ in valid], fontsize=10)
    ax.set_ylabel("Pairwise Accuracy (%)", fontsize=12)
    ax.set_title("IntPhys2: Comparison of Surprise Metrics", fontsize=13)
    ax.set_ylim(40, 75)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "intphys_metric_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "intphys_metric_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metric comparison to {FIGURES_DIR / 'intphys_metric_comparison.pdf'}")


def print_summary(compiled):
    """Print a comprehensive text summary."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE IntPhys2 BENCHMARK RESULTS")
    print("=" * 80)

    # Main results
    main = compiled.get("main_eval", {})
    print("\n--- Main Evaluation (stride=2, max_frames=100) ---")
    for name in ["DynaCLIP-V2", "DynaCLIP-V3", "dinov2_vitb14", "dinov2_vitl14",
                 "clip_vitl14", "siglip_vitb16"]:
        if name in main and "error" not in main[name]:
            r = main[name]
            print(f"  {name:<20} best={r['best_metric']:<18} acc={r['best_accuracy']:.1%}")
            print(f"    {'':20} emb_diff={r['all_metrics']['embedding_diff']['overall']['accuracy']:.1%}"
                  f"  max_jump={r['all_metrics']['max_jump']['overall']['accuracy']:.1%}"
                  f"  pred_err={r['all_metrics']['prediction_error']['overall']['accuracy']:.1%}")

    # Stride sensitivity
    stride_data = compiled.get("stride_sensitivity", {})
    if stride_data:
        print("\n--- Stride Sensitivity ---")
        print(f"  {'Stride':<10} {'DynaCLIP-V2':>15} {'dinov2_vitb14':>15}")
        for s in sorted([int(k) for k in stride_data.keys()]):
            row = f"  {s:<10}"
            for model in ["DynaCLIP-V2", "dinov2_vitb14"]:
                if model in stride_data.get(s, {}):
                    row += f" {stride_data[s][model]['best_accuracy']:>14.1%}"
                else:
                    row += f" {'N/A':>15}"
            print(row)

    # Loss curves
    loss = compiled.get("loss_curves", {})
    if loss:
        print("\n--- Training Loss Curves ---")
        for version in ["v2", "v3"]:
            if version in loss:
                steps = loss[version]["steps"]
                losses = loss[version]["losses"]
                print(f"  {version.upper()}: {losses[0]:.4f} → {losses[-1]:.4f} "
                      f"({len(steps)} points, steps {steps[0]}-{steps[-1]})")

    # Synthetic VoE
    voe = compiled.get("synthetic_voe", {})
    if voe:
        print("\n--- Synthetic VoE Test ---")
        for vtype, data in voe.items():
            if isinstance(data, dict) and "pct_more_detectable" in data:
                print(f"  {vtype}: {data['pct_more_detectable']:.1f}% distinguishable")

    print("\n" + "=" * 80)


def main():
    compiled = compile_results()

    # Save comprehensive results
    out_path = RESULTS_DIR / "comprehensive_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(compiled, f, indent=2, default=convert)
    print(f"Comprehensive results saved to {out_path}")

    # Generate figures
    plot_accuracy_bar(compiled)
    plot_stride_sensitivity(compiled)
    plot_condition_heatmap(compiled)
    plot_metric_comparison(compiled)

    # Print summary
    print_summary(compiled)


if __name__ == "__main__":
    main()
