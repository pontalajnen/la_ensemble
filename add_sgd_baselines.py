#!/usr/bin/env python3
"""Append SGD baseline rows to sweep_sgld_results.csv.

Reads three evaluation JSONs produced by run_sgd_eval.sh and appends them
as rows with empty hyperparameter columns. Idempotent — skips any
(config, variant) pair that already appears in the CSV.
"""

import csv
import json
import os
import sys

CSV_PATH = "sweep_sgld_results.csv"
METRICS_DIR = "experiment_results/table_metrics"

# config name, variant tag, json file
BASELINES = [
    ("sgd_single",   "nonpacked", "resnet20_cifar10_sgd.json"),
    ("sgd_ensemble", "nonpacked", "resnet20_cifar10_sgd_ensemble.json"),
    ("sgd_packed",   "packed",    "resnet20_cifar10_sgd_packed.json"),
]

EVAL_METRICS = ["clean_accuracy", "ECE", "nll",
                "OOD AUROC", "SHIFT ACCURACY", "SHIFT ECE"]


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return next(iter(data.values()))


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"[error] CSV not found: {CSV_PATH}")

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    if not fieldnames:
        sys.exit("[error] CSV is empty / has no header")

    existing = {(r["config"], r["variant"]) for r in rows}

    new_rows = []
    for config, variant, json_name in BASELINES:
        if (config, variant) in existing:
            print(f"[skip] already present: {config} / {variant}")
            continue
        json_path = os.path.join(METRICS_DIR, json_name)
        if not os.path.exists(json_path):
            print(f"[warn] missing eval JSON: {json_path} — did you run run_sgd_eval.sh?")
            continue
        m = load_metrics(json_path)
        row = {k: "" for k in fieldnames}
        row.update({
            "trial":        "",
            "config":       config,
            "variant":      variant,
            "seed":         "",
            "lr":           "",
            "temperature":  "",
            "sampling_lr":  "",
            "val_loss":     "",
            "samples_file": "",
        })
        for metric in EVAL_METRICS:
            v = m.get(metric)
            row[metric] = f"{v:.4f}" if isinstance(v, (int, float)) else ""
        new_rows.append(row)
        print(f"[add]  {config} / {variant}  "
              f"acc={m.get('clean_accuracy'):.4f}  "
              f"ECE={m.get('ECE'):.3f}  NLL={m.get('nll'):.4f}")

    if not new_rows:
        print("Nothing to add.")
        return

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for r in new_rows:
            writer.writerow(r)

    print(f"\nAppended {len(new_rows)} baseline row(s) to {CSV_PATH}.")


if __name__ == "__main__":
    main()
