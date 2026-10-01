#!/usr/bin/env python3
"""Plot sweep_sgld_results.csv — bar charts of every metric per variant, plus
a (temperature × sampling_lr) heatmap of NLL per variant.

Usage:
    python plot_sweep.py                     # sweep_sgld_results.csv → sweep_plots/
    python plot_sweep.py my_results.csv --outdir figs
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    ("clean_accuracy", "clean accuracy", True),
    ("ECE",            "ECE",             False),
    ("nll",            "NLL",             False),
    ("OOD AUROC",      "OOD AUROC",       True),
    ("SHIFT ACCURACY", "shift accuracy",  True),
    ("SHIFT ECE",      "shift ECE",       False),
]


def _load(path):
    df = pd.read_csv(path)
    for m, *_ in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    for h in ("lr", "temperature", "sampling_lr", "val_loss"):
        if h in df.columns:
            df[h] = pd.to_numeric(df[h], errors="coerce")
    return df


def plot_metric_bars(df, variant, outpath):
    sub = df[df["variant"] == variant].copy()
    if sub.empty:
        return
    # order configs by NLL asc so best-first
    sub = sub.sort_values("nll", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"SGLD sweep — {variant}", fontsize=14, fontweight="bold")

    for ax, (col, label, higher_is_better) in zip(axes.flat, METRICS):
        vals = sub[col].values
        configs = sub["config"].values
        best_idx = int(np.argmax(vals) if higher_is_better else np.argmin(vals))
        colors = ["#3b82f6"] * len(vals)
        colors[best_idx] = "#22c55e"

        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
        arrow = "↑" if higher_is_better else "↓"
        ax.set_title(f"{label} {arrow}")
        ax.grid(axis="y", alpha=0.3)

        # annotate values
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3g}", ha="center", va="bottom", fontsize=7)

        # ylim so annotations don't clip
        finite = vals[np.isfinite(vals)]
        if len(finite):
            lo, hi = finite.min(), finite.max()
            margin = (hi - lo) * 0.15 or abs(hi) * 0.05 or 1.0
            ax.set_ylim(lo - margin * 0.3, hi + margin)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outpath}")


def plot_heatmap(df, variant, metric, outpath, higher_is_better=False):
    sub = df[df["variant"] == variant].copy()
    if sub.empty or sub[metric].isna().all():
        return
    pivot = sub.pivot_table(index="temperature", columns="sampling_lr",
                            values=metric, aggfunc="mean")
    pivot = pivot.sort_index(ascending=False).sort_index(axis=1, ascending=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = "viridis_r" if higher_is_better else "viridis"
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index])
    ax.set_xlabel("sampling_lr")
    ax.set_ylabel("temperature")
    arrow = "↑" if higher_is_better else "↓"
    ax.set_title(f"{variant} — {metric} {arrow}")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3g}", ha="center", va="center",
                        color="white", fontsize=9)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outpath}")


def plot_variant_compare(df, outpath):
    """Side-by-side bar chart comparing nonpacked vs packed on each metric."""
    variants = sorted(df["variant"].unique())
    if len(variants) < 2:
        return
    configs = sorted(df["config"].unique(),
                     key=lambda c: df.loc[df["config"] == c, "nll"].mean())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("nonpacked vs packed", fontsize=14, fontweight="bold")

    width = 0.4
    x = np.arange(len(configs))
    palette = {"nonpacked": "#3b82f6", "packed": "#f97316"}

    for ax, (col, label, higher_is_better) in zip(axes.flat, METRICS):
        for i, v in enumerate(variants):
            sub = df[df["variant"] == v].set_index("config").reindex(configs)
            vals = sub[col].values
            offset = (i - (len(variants) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=v,
                   color=palette.get(v, None), edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
        arrow = "↑" if higher_is_better else "↓"
        ax.set_title(f"{label} {arrow}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="sweep_sgld_results.csv")
    ap.add_argument("--outdir", default="sweep_plots")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"[error] not found: {args.csv}")

    df = _load(args.csv)
    if df.empty:
        sys.exit(f"[error] empty: {args.csv}")

    os.makedirs(args.outdir, exist_ok=True)
    print(f"[source] {args.csv}   ({len(df)} rows)")

    for variant in sorted(df["variant"].unique()):
        plot_metric_bars(df, variant,
                         os.path.join(args.outdir, f"bars_{variant}.png"))
        for col, label, higher_is_better in [("nll", "NLL", False),
                                             ("clean_accuracy", "acc", True),
                                             ("ECE", "ECE", False)]:
            plot_heatmap(df, variant, col,
                         os.path.join(args.outdir, f"heatmap_{variant}_{col.replace(' ', '_')}.png"),
                         higher_is_better=higher_is_better)

    plot_variant_compare(df, os.path.join(args.outdir, "compare_variants.png"))
    print(f"\nDone. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()
