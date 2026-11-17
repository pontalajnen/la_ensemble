#!/usr/bin/env python3
"""Evaluate the 8 ResNet18 SGLD round-3 runs (seeds 800-807).

Produces:
  - experiment_results/table_metrics/resnet18_round3_<name>.json per run
  - sweep_sgld_resnet18_round3_results.csv

Idempotent: skips any run whose JSON already exists.
"""

import csv
import json
import os
import shutil
import subprocess

DATASET      = "cifar10"
MODEL        = "resnet18"
RESULTS_DIR  = f"experiment_results/{DATASET}_{MODEL}_no_SAM"
METRICS_DIR  = "experiment_results/table_metrics"
PATH_DIR     = "eval_path_files"
CSV_PATH     = "sweep_sgld_resnet18_round3_results.csv"
EVAL_METRICS = ["clean_accuracy", "ECE", "nll",
                "OOD AUROC", "SHIFT ACCURACY", "SHIFT ECE"]

# seed, config, variant (packed?), T, sample_interval, max_samples
RUNS = [
    (800, "hot_1e-1",  False, 1e-1, 2, 50),
    (801, "hot_1e-1",  True,  1e-1, 2, 50),
    (802, "hot_3e-2",  False, 3e-2, 2, 50),
    (803, "hot_3e-2",  True,  3e-2, 2, 50),
    (804, "warm_more", False, 1e-2, 2, 50),
    (805, "warm_more", True,  1e-2, 2, 50),
    (806, "warm_fine", False, 1e-2, 4, 25),
    (807, "warm_fine", True,  1e-2, 4, 25),
]


def model_tag(packed): return "packed" if packed else "nonpacked"


def samples_src(packed, seed):
    stem = f"{MODEL}_{DATASET}_SGLD" + ("_packed" if packed else "")
    return os.path.join(RESULTS_DIR, f"sgld_samples_{stem}_seed{seed}.txt")


def evaluate(seed, config, packed, max_samples):
    tag = f"round3_{config}_{model_tag(packed)}_seed{seed}"
    path_file = f"resnet18_{tag}.txt"
    save_file = f"resnet18_{tag}.json"

    src = samples_src(packed, seed)
    if not os.path.exists(src):
        print(f"[warn] missing samples file for seed {seed}: {src}")
        return None

    os.makedirs(PATH_DIR, exist_ok=True)
    shutil.copy(src, os.path.join(PATH_DIR, path_file))

    result_path = os.path.join(METRICS_DIR, save_file)
    if os.path.exists(result_path):
        print(f"[skip] result already exists: {save_file}")
    else:
        cmd = [
            "python", "evaluate.py",
            "--save_file_name", save_file,
            "--model_path_file", path_file,
            "--model_type", MODEL + ("_packed" if packed else ""),
            "--dataset", DATASET,
            "--batch_size", "128",
            "--sgld_ensemble",
            "--max_sgld_samples", str(max_samples),
            "--no-eval_train",
        ]
        print(f"\n{'='*60}\n[eval] seed={seed} {config} {model_tag(packed)} "
              f"max_samples={max_samples}\n{'='*60}")
        proc = subprocess.run(cmd)
        if proc.returncode != 0 or not os.path.exists(result_path):
            print(f"[error] eval failed for seed {seed}")
            return None

    with open(result_path) as f:
        data = json.load(f)
    metrics = next(iter(data.values()))
    return {k: metrics.get(k) for k in EVAL_METRICS}


def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    fieldnames = (["seed", "config", "variant", "temperature",
                   "sample_interval", "n_samples"]
                  + EVAL_METRICS)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed, config, packed, T, interval, max_samples in RUNS:
            metrics = evaluate(seed, config, packed, max_samples) or {}
            row = {
                "seed":            seed,
                "config":          config,
                "variant":         model_tag(packed),
                "temperature":     T,
                "sample_interval": interval,
                "n_samples":       max_samples,
            }
            for k in EVAL_METRICS:
                v = metrics.get(k)
                row[k] = f"{v:.4f}" if isinstance(v, (int, float)) else ""
            writer.writerow(row)
            f.flush()

    print(f"\nDone. CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
