#!/usr/bin/env python
"""
Update paper files with real experimental results from JSON outputs.

Reads results from results/ directory and fills in the --- placeholders
in both paper/dynaclip_paper.tex and paper/dynaclip_paper.txt.
"""

import json
import re
import sys
from pathlib import Path


def load_results():
    """Load all result JSON files."""
    results = {}

    # Find the latest step results
    results_dir = Path("results")
    step_dirs = sorted(results_dir.glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
    if step_dirs:
        latest = step_dirs[-1]
        for f in latest.glob("*.json"):
            results[f.stem] = json.loads(f.read_text())

    # Ablation results
    ablation_file = results_dir / "ablations" / "ablation_results.json"
    if ablation_file.exists():
        results["ablation_results"] = json.loads(ablation_file.read_text())

    return results


def fmt(val, digits=4):
    """Format a float value."""
    if val is None or val == 0:
        return "---"
    return f"{val:.{digits}f}"


def fmt_ci(mean, ci=None, digits=4):
    """Format mean ± CI."""
    if mean is None or mean == 0:
        return "---"
    if ci is not None and ci > 0:
        return f"{mean:.{digits}f}$\\pm${ci:.3f}"
    return f"{mean:.{digits}f}"


def update_tex(results, tex_path):
    """Update LaTeX file with real results."""
    tex = Path(tex_path).read_text()

    lp = results.get("linear_probing", {})
    knn = results.get("knn_results", {})
    cluster = results.get("clustering_results", {})
    ablation = results.get("ablation_results", {})

    # Table 1: Linear probing
    backbone_order = ["DINOv2-B/14", "DINOv2-L/14", "CLIP-L/14", "SigLIP-B/16",
                      "DynaCLIP-backbone", "DynaCLIP"]
    for bname in backbone_order:
        if bname not in lp:
            continue
        bres = lp[bname]
        mass = bres.get("mass", {}).get("mean", None)
        fric = bres.get("static_friction", {}).get("mean", None)
        rest = bres.get("restitution", {}).get("mean", None)
        cat = bres.get("category", {}).get("mean", None)

        # Find the row in the table and replace --- values
        bname_tex = bname.replace("/", "/")
        if "DynaCLIP" in bname and bname != "DynaCLIP-backbone":
            pattern = rf"(\\textbf\{{\\methodname\{{\}}\}}\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---"
            replacement = rf"\g<1>\\textbf{{{fmt(mass)}}}\2\\textbf{{{fmt(fric)}}}\3\\textbf{{{fmt(rest)}}}\4\\textbf{{{fmt(cat)}}}"
        elif bname == "DynaCLIP-backbone":
            pattern = rf"(\\methodname\{{\}}-backbone\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---"
            replacement = rf"\g<1>{fmt(mass)}\2{fmt(fric)}\3{fmt(rest)}\4{fmt(cat)}"
        else:
            escaped = bname_tex.replace("-", r"\-?").replace("/", r"[/]")
            pattern = rf"({escaped}\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---"
            replacement = rf"\g<1>{fmt(mass)}\2{fmt(fric)}\3{fmt(rest)}\4{fmt(cat)}"

        tex = re.sub(pattern, replacement, tex, count=1)

    # Table 2: k-NN (simplified update)
    for bname in ["DINOv2-B/14", "CLIP-L/14", "DynaCLIP"]:
        if bname not in knn:
            continue
        k5 = knn[bname].get("k=5", {})
        k10 = knn[bname].get("k=10", {})

        if "DynaCLIP" in bname and bname != "DINOv2-B/14":
            pattern = rf"(\\methodname\{{\}}\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---"
        else:
            escaped = bname.replace("/", r"[/]").replace("-", r"\-?")
            pattern = rf"({escaped}\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---"

        replacement = (
            rf"\g<1>{fmt(k5.get('mass_r2'))}\2{fmt(k5.get('friction_r2'))}"
            rf"\3{fmt(k5.get('restitution_r2'))}\4{fmt(k10.get('mass_r2'))}"
            rf"\5{fmt(k10.get('friction_r2'))}\6{fmt(k10.get('restitution_r2'))}"
        )
        tex = re.sub(pattern, replacement, tex, count=1)

    # Table 3: Clustering
    for bname in ["DINOv2-B/14", "DINOv2-L/14", "CLIP-L/14", "SigLIP-B/16", "DynaCLIP"]:
        if bname not in cluster:
            continue
        nmi = cluster[bname].get("nmi")
        ari = cluster[bname].get("ari")

        if "DynaCLIP" in bname:
            pattern = rf"(\\textbf\{{\\methodname\{{\}}\}}\s*&\s*)---(\s*&\s*)---"
        else:
            escaped = bname.replace("/", r"[/]").replace("-", r"\-?")
            pattern = rf"({escaped}\s*&\s*)---(\s*&\s*)---"

        replacement = rf"\g<1>{fmt(nmi)}\2{fmt(ari)}"
        tex = re.sub(pattern, replacement, tex, count=1)

    # Table 4: Ablations
    ablation_names = {
        "DynaCLIP (full)": r"\\methodname\{\} \(full\)",
        "Frozen backbone": r"Frozen backbone",
        "No hard mining": r"No hard mining",
        "Random physics": r"Random physics assignment",
        "Standard InfoNCE": r"Standard InfoNCE",
    }
    for vname, tex_pattern in ablation_names.items():
        if vname not in ablation:
            continue
        vres = ablation[vname]
        mass = vres.get("mass", {}).get("mean")
        fric = vres.get("static_friction", {}).get("mean")
        rest = vres.get("restitution", {}).get("mean")
        cat = vres.get("category", {}).get("mean")

        pattern = rf"({tex_pattern}\s*&\s*)---(\s*&\s*)---(\s*&\s*)---(\s*&\s*)---"
        replacement = rf"\g<1>{fmt(mass)}\2{fmt(fric)}\3{fmt(rest)}\4{fmt(cat)}"
        tex = re.sub(pattern, replacement, tex, count=1)

    Path(tex_path).write_text(tex)
    print(f"Updated {tex_path}")


def update_txt(results, txt_path):
    """Update plain-text file with real results."""
    txt = Path(txt_path).read_text()

    lp = results.get("linear_probing", {})
    knn = results.get("knn_results", {})
    cluster = results.get("clustering_results", {})
    ablation = results.get("ablation_results", {})

    # Replace --- in linear probing table
    for bname in lp:
        bres = lp[bname]
        mass = fmt(bres.get("mass", {}).get("mean"), 4)
        fric = fmt(bres.get("static_friction", {}).get("mean"), 4)
        rest = fmt(bres.get("restitution", {}).get("mean"), 4)
        cat = fmt(bres.get("category", {}).get("mean"), 4)

        old_line_pattern = rf"({re.escape(bname)}\s+)---\s+---\s+---\s+---"
        new_line = rf"\g<1>{mass:<10} {fric:<12} {rest:<15} {cat}"
        txt = re.sub(old_line_pattern, new_line, txt, count=1)

    # Replace --- in k-NN table
    for bname in knn:
        k5 = knn[bname].get("k=5", {})
        k10 = knn[bname].get("k=10", {})
        old_pattern = rf"({re.escape(bname)}\s+)---\s+---\s+---\s+---\s+---\s+---"
        new_vals = (
            f"{fmt(k5.get('mass_r2'))}   {fmt(k5.get('friction_r2'))}     {fmt(k5.get('restitution_r2'))}      "
            f"{fmt(k10.get('mass_r2'))}   {fmt(k10.get('friction_r2'))}     {fmt(k10.get('restitution_r2'))}"
        )
        new_line = rf"\g<1>{new_vals}"
        txt = re.sub(old_pattern, new_line, txt, count=1)

    # Replace --- in clustering table
    for bname in cluster:
        nmi = fmt(cluster[bname].get("nmi"), 4)
        ari = fmt(cluster[bname].get("ari"), 4)
        old_pattern = rf"({re.escape(bname)}\s+)---\s+---"
        new_line = rf"\g<1>{nmi}    {ari}"
        txt = re.sub(old_pattern, new_line, txt, count=1)

    # Replace --- in ablation table
    for vname in ablation:
        vres = ablation[vname]
        mass = fmt(vres.get("mass", {}).get("mean"), 4)
        fric = fmt(vres.get("static_friction", {}).get("mean"), 4)
        rest = fmt(vres.get("restitution", {}).get("mean"), 4)
        cat = fmt(vres.get("category", {}).get("mean"), 4)
        old_pattern = rf"({re.escape(vname)}\s+)---\s+---\s+---\s+---"
        new_line = rf"\g<1>{mass:<9} {fric:<12} {rest:<15} {cat}"
        txt = re.sub(old_pattern, new_line, txt, count=1)

    Path(txt_path).write_text(txt)
    print(f"Updated {txt_path}")


def main():
    results = load_results()
    if not results:
        print("No results found in results/ directory. Run evaluation first.")
        sys.exit(1)

    print(f"Found result files: {list(results.keys())}")

    tex_path = "paper/dynaclip_paper.tex"
    txt_path = "paper/dynaclip_paper.txt"

    if Path(tex_path).exists():
        update_tex(results, tex_path)
    if Path(txt_path).exists():
        update_txt(results, txt_path)

    print("\nDone! Paper files updated with experimental results.")


if __name__ == "__main__":
    main()
