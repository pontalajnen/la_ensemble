#!/usr/bin/env python3
"""Headline plot: SGLD temperature vs ECE and accuracy (packed ResNet18, CIFAR-10).

Reads combined_resnet18_results.csv and plots the T-sweep for packed runs,
with SGD-packed and the round-2 best as horizontal reference lines.
"""

import csv
import matplotlib.pyplot as plt

CSV_PATH = "combined_resnet18_results.csv"
OUT_PNG  = "T_sweep_packed.png"


def load():
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


def main():
    rows = load()
    packed = [r for r in rows if r["variant"] == "packed"]

    # SGLD temperature sweep from R2 + R3 packed runs (skip baselines).
    sgld = [r for r in packed if r["round"] in ("R2", "R3") and r["temperature"]]
    points = sorted(((float(r["temperature"]), float(r["acc"]),
                      float(r["ECE"]), r["config"], r["round"])
                     for r in sgld))

    # SGD-packed reference line (no-mom).
    sgd = next((r for r in packed
                if r["round"] == "baseline" and r["config"] == "sgd_nomom"), None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- ECE panel ---
    ax = axes[0]
    for T, acc, ece, cfg, rnd in points:
        T_plot = 1e-4 if T == 0 else T
        ax.scatter(T_plot, ece, s=70, alpha=0.85,
                   color="#1f77b4" if rnd == "R2" else "#d62728",
                   edgecolor="black", linewidth=0.5,
                   label=f"{rnd} {cfg}")
        ax.annotate(cfg, (T_plot, ece), xytext=(5, 5),
                    textcoords="offset points", fontsize=7, color="#333")
    if sgd:
        ax.axhline(float(sgd["ECE"]), color="gray", linestyle="--",
                   linewidth=1.2, alpha=0.8, label="SGD-packed baseline")
    ax.set_xscale("log")
    ax.set_xlabel("SGLD temperature  T  (T=0 shown at 1e-4)")
    ax.set_ylabel("ECE (%)")
    ax.set_title("Calibration vs temperature — packed ResNet18")
    ax.grid(True, alpha=0.3)
    ax.axvspan(2e-2, 7e-2, color="green", alpha=0.07)
    ax.text(3.5e-2, ax.get_ylim()[1] * 0.95, "sweet spot",
            fontsize=8, color="green", ha="center")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper right",
              ncol=2, framealpha=0.9)

    # --- Accuracy panel ---
    ax = axes[1]
    for T, acc, ece, cfg, rnd in points:
        T_plot = 1e-4 if T == 0 else T
        ax.scatter(T_plot, acc, s=70, alpha=0.85,
                   color="#1f77b4" if rnd == "R2" else "#d62728",
                   edgecolor="black", linewidth=0.5,
                   label=f"{rnd} {cfg}")
        ax.annotate(cfg, (T_plot, acc), xytext=(5, 5),
                    textcoords="offset points", fontsize=7, color="#333")
    if sgd:
        ax.axhline(float(sgd["acc"]), color="gray", linestyle="--",
                   linewidth=1.2, alpha=0.8, label="SGD-packed baseline")
    ax.set_xscale("log")
    ax.set_xlabel("SGLD temperature  T  (T=0 shown at 1e-4)")
    ax.set_ylabel("Clean accuracy")
    ax.set_title("Accuracy vs temperature — packed ResNet18")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "SGLD packed ResNet18 / CIFAR-10 — ECE & accuracy vs temperature\n"
        "Round-2 (blue) and Round-3 (red).  Sweet spot: T ≈ 3e-2 (ECE 0.48%, "
        "−80% vs SGD baseline, −0.7pp acc).",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[plot] wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
