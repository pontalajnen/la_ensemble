#!/usr/bin/env python3
"""Merge SGD baselines + round-2 + round-3 eval rows into one CSV.

Writes combined_resnet18_results.csv with a flat schema:
  round, config, variant, temperature, n_samples, acc, ECE, nll, OOD_AUROC
"""

import csv
import json
import os

METRICS_DIR = "experiment_results/table_metrics"
OUT_CSV     = "combined_resnet18_results.csv"
FIELDS = ["round", "config", "variant", "temperature", "n_samples",
          "acc", "ECE", "nll", "OOD_AUROC"]


def load_metrics(path):
    if not os.path.exists(path):
        return None
    return next(iter(json.load(open(path)).values()))


def sgd_rows():
    rows = []
    for config, variant, jname in [
        ("sgd_mom",          "nonpacked", "resnet18_cifar10_sgd_mom.json"),
        ("sgd_mom",          "packed",    "resnet18_cifar10_sgd_mom_packed.json"),
        ("sgd_nomom",        "nonpacked", "resnet18_cifar10_sgd_nomom.json"),
        ("sgd_nomom",        "packed",    "resnet18_cifar10_sgd_nomom_packed.json"),
        ("sgld_baseline_R1", "nonpacked", "resnet18_cifar10_sgld_baseline.json"),
        ("sgld_baseline_R1", "packed",    "resnet18_cifar10_sgld_baseline_packed.json"),
    ]:
        m = load_metrics(os.path.join(METRICS_DIR, jname))
        if m is None:
            continue
        rows.append({
            "round":       "baseline",
            "config":      config,
            "variant":     variant,
            "temperature": "",
            "n_samples":   "",
            "acc":         f"{m['clean_accuracy']:.4f}",
            "ECE":         f"{m['ECE']:.4f}",
            "nll":         f"{m['nll']:.4f}",
            "OOD_AUROC":   f"{m['OOD AUROC']:.4f}",
        })
    return rows


def sweep_rows(round_tag, csv_path, cols):
    """cols: dict mapping output field -> CSV column name (or None if missing)."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    for r in csv.DictReader(open(csv_path)):
        if not r.get("clean_accuracy"):
            continue
        rows.append({
            "round":       round_tag,
            "config":      r["config"],
            "variant":     r["variant"],
            "temperature": r.get("temperature", ""),
            "n_samples":   r.get("n_samples", ""),
            "acc":         f"{float(r['clean_accuracy']):.4f}",
            "ECE":         f"{float(r['ECE']):.4f}",
            "nll":         f"{float(r['nll']):.4f}",
            "OOD_AUROC":   f"{float(r['OOD AUROC']):.4f}",
        })
    return rows


def main():
    all_rows = []
    all_rows += sgd_rows()
    all_rows += sweep_rows("R2", "sweep_sgld_resnet18_round2_results.csv", None)
    all_rows += sweep_rows("R3", "sweep_sgld_resnet18_round3_results.csv", None)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"[combined] wrote {len(all_rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
