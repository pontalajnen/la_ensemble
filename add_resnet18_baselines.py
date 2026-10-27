#!/usr/bin/env python3
"""Append ResNet18 SGD + SGLD-baseline rows to sweep_sgld_resnet18_results.csv.

Idempotent — skips any (config, variant) already present.
"""

import csv
import json
import os
import sys

CSV_PATH = "sweep_sgld_resnet18_results.csv"
METRICS_DIR = "experiment_results/table_metrics"

# config name, variant tag, json file
BASELINES = [
    ("sgd_mom",         "nonpacked", "resnet18_cifar10_sgd_mom.json"),
    ("sgd_mom_packed",  "packed",    "resnet18_cifar10_sgd_mom_packed.json"),
    ("sgd_nomom",       "nonpacked", "resnet18_cifar10_sgd_nomom.json"),
    ("sgd_nomom_packed","packed",    "resnet18_cifar10_sgd_nomom_packed.json"),
    ("sgld_baseline",   "nonpacked", "resnet18_cifar10_sgld_baseline.json"),
    ("sgld_baseline_packed", "packed", "resnet18_cifar10_sgld_baseline_packed.json"),
]

EVAL_METRICS = ["clean_accuracy", "ECE", "nll",
                "OOD AUROC", "SHIFT ACCURACY", "SHIFT ECE"]


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return next(iter(data.values()))


def main():
    if not os.path.exists(CSV_PATH):
        print(f"[info] creating {CSV_PATH}")
        fieldnames = (["trial", "config", "variant", "seed",
                       "lr", "temperature", "sampling_lr", "val_loss"]
                      + EVAL_METRICS
                      + ["samples_file"])
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        existing = set()
    else:
        with open(CSV_PATH) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            sys.exit("[error] CSV has header but no rows and no fieldnames")
        fieldnames = list(rows[0].keys())
        existing = {(r["config"], r["variant"]) for r in rows}

    new_rows = []
    for config, variant, json_name in BASELINES:
        if (config, variant) in existing:
            print(f"[skip] already present: {config} / {variant}")
            continue
        json_path = os.path.join(METRICS_DIR, json_name)
        if not os.path.exists(json_path):
            print(f"[warn] missing: {json_path}")
            continue
        m = load_metrics(json_path)
        row = {k: "" for k in fieldnames}
        row.update({"config": config, "variant": variant})
        for metric in EVAL_METRICS:
            v = m.get(metric)
            row[metric] = f"{v:.4f}" if isinstance(v, (int, float)) else ""
        new_rows.append(row)
        acc = m.get("clean_accuracy") or 0
        ece = m.get("ECE") or 0
        nll = m.get("nll") or 0
        print(f"[add]  {config:22s} / {variant:9s}  "
              f"acc={acc:.4f}  ECE={ece:.3f}  NLL={nll:.4f}")

    if not new_rows:
        print("Nothing to add.")
        return

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for r in new_rows:
            writer.writerow(r)

    print(f"\nAppended {len(new_rows)} row(s) to {CSV_PATH}.")


if __name__ == "__main__":
    main()
