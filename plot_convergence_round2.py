#!/usr/bin/env python3
"""Plot per-epoch convergence for the 10 ResNet18 SGLD round-2 runs.

Reads convergence_*_seed70[0-9].csv from the save dir and writes
  convergence_round2.png

2x3 grid: rows = {nonpacked, packed}, cols = {train_loss, val_loss, val_acc}.
Each line = one config, colored consistently across rows. Phase boundaries
shown as vertical dashed lines.
"""

import csv
import os
import matplotlib.pyplot as plt

SAVE_DIR = "experiment_results/cifar10_resnet18_no_SAM"

RUNS = [
    (700, "T0_sanity",    False),
    (701, "T0_sanity",    True),
    (702, "base_burn180", False),
    (703, "base_burn180", True),
    (704, "warm_burn180", False),
    (705, "warm_burn180", True),
    (706, "base_burn150", False),
    (707, "base_burn150", True),
    (708, "warm_burn150", False),
    (709, "warm_burn150", True),
]

COLORS = {
    "T0_sanity":    "#444",
    "base_burn180": "#1f77b4",
    "warm_burn180": "#d62728",
    "base_burn150": "#17becf",
    "warm_burn150": "#ff7f0e",
}


def convergence_path(seed, packed):
    stem = "resnet18_cifar10_SGLD" + ("_packed" if packed else "")
    return os.path.join(SAVE_DIR, f"convergence_{stem}_seed{seed}.csv")


def load(path):
    rows = {"epoch": [], "train_loss": [], "val_loss": [],
            "val_accuracy": [], "phase": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows["epoch"].append(int(r["epoch"]))
            rows["train_loss"].append(float(r["train_loss"]))
            rows["val_loss"].append(float(r["val_loss"]))
            rows["val_accuracy"].append(float(r["val_accuracy"]))
            rows["phase"].append(r["phase"])
    return rows


def phase_boundaries(phases):
    boundaries = {}
    prev = None
    for i, p in enumerate(phases):
        if p != prev:
            boundaries[p] = i
            prev = p
    return boundaries


def main():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    metric_cols = ["train_loss", "val_loss", "val_accuracy"]
    titles = ["Train loss", "Val loss", "Val accuracy"]

    for row_idx, is_packed in enumerate([False, True]):
        variant_name = "packed" if is_packed else "nonpacked"

        boundary_source = None
        for seed, config, packed in RUNS:
            if packed != is_packed:
                continue
            path = convergence_path(seed, packed)
            if not os.path.exists(path):
                continue
            data = load(path)
            if boundary_source is None or "180" in config:
                boundary_source = data["phase"]

            for col_idx, m in enumerate(metric_cols):
                ax = axes[row_idx, col_idx]
                ax.plot(data["epoch"], data[m],
                        color=COLORS[config],
                        linewidth=1.2,
                        alpha=0.85,
                        label=config if col_idx == 0 else None)

        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if boundary_source is not None:
                bounds = phase_boundaries(boundary_source)
                for phase, x in bounds.items():
                    if phase in ("burn_in", "cooldown", "sampling"):
                        ax.axvline(x, color="gray", linestyle="--",
                                   linewidth=0.7, alpha=0.5)
                        ax.text(x, ax.get_ylim()[1] * 0.98, f" {phase}",
                                fontsize=7, color="gray",
                                verticalalignment="top")
            ax.set_title(f"{variant_name} — {titles[col_idx]}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if col_idx == 2:
                ax.set_ylim(0.5, 1.0)
            elif metric_cols[col_idx] == "train_loss":
                ax.set_yscale("log")

        axes[row_idx, 0].legend(fontsize=8, loc="upper right")

    for col_idx in range(3):
        axes[1, col_idx].set_xlabel("epoch")

    fig.suptitle(
        "ResNet18 CIFAR-10 SGLD round-2 — per-epoch convergence\n"
        "burn_in=180 (10 samples from 180–198) vs burn_in=150 (25 samples from 150–198)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "convergence_round2.png"
    fig.savefig(out, dpi=140)
    print(f"[plot] wrote {out}")


if __name__ == "__main__":
    main()
