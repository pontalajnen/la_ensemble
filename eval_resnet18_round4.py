#!/usr/bin/env python3
"""Evaluate the 5 ResNet18 SGLD round-4 verification runs (seeds 900-904).

Produces:
  - experiment_results/table_metrics/resnet18_round4_<name>.json per run
  - sweep_sgld_resnet18_round4_results.csv

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
CSV_PATH     = "sweep_sgld_resnet18_round4_results.csv"
EVAL_METRICS = ["clean_accuracy", "ECE", "nll",
                "OOD AUROC", "SHIFT ACCURACY", "SHIFT ECE"]

# seed, config, T
RUNS = [
    (900, "T3e-2_repro1",   3e-2),
    (901, "T3e-2_repro2",   3e-2),
    (902, "T3e-2_repro3",   3e-2),
    (903, "T5e-2_bracket1", 5e-2),
    (904, "T5e-2_bracket2", 5e-2),
]

MAX_SAMPLES = 50


def samples_src(seed):
    return os.path.join(
        RESULTS_DIR,
        f"sgld_samples_{MODEL}_{DATASET}_SGLD_packed_seed{seed}.txt",
    )


def evaluate(seed, config):
    tag = f"round4_{config}_packed_seed{seed}"
    path_file = f"resnet18_{tag}.txt"
    save_file = f"resnet18_{tag}.json"

    src = samples_src(seed)
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
            "--model_type", f"{MODEL}_packed",
            "--dataset", DATASET,
            "--batch_size", "128",
            "--sgld_ensemble",
            "--max_sgld_samples", str(MAX_SAMPLES),
            "--no-eval_train",
        ]
        print(f"\n{'='*60}\n[eval] seed={seed} {config} packed "
              f"max_samples={MAX_SAMPLES}\n{'='*60}")
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
    fieldnames = (["seed", "config", "variant", "temperature", "n_samples"]
                  + EVAL_METRICS)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed, config, T in RUNS:
            metrics = evaluate(seed, config) or {}
            row = {
                "seed":        seed,
                "config":      config,
                "variant":     "packed",
                "temperature": T,
                "n_samples":   MAX_SAMPLES,
            }
            for k in EVAL_METRICS:
                v = metrics.get(k)
                row[k] = f"{v:.4f}" if isinstance(v, (int, float)) else ""
            writer.writerow(row)
            f.flush()

    # Quick summary printed at the end
    rows = list(csv.DictReader(open(CSV_PATH)))
    print("\n=== Round 4 summary ===")
    print(f"{'config':20s} {'T':>6s} {'acc':>7s} {'ECE%':>7s} "
          f"{'NLL':>7s} {'OOD':>7s}")
    for r in rows:
        if r["clean_accuracy"]:
            print(f"{r['config']:20s} {r['temperature']:>6s} "
                  f"{float(r['clean_accuracy']):7.4f} "
                  f"{float(r['ECE']):7.3f} "
                  f"{float(r['nll']):7.3f} "
                  f"{float(r['OOD AUROC']):7.3f}")

    for T, label in [("0.03", "T=3e-2"), ("0.05", "T=5e-2")]:
        eces = [float(r["ECE"]) for r in rows
                if r["temperature"] == T and r["clean_accuracy"]]
        accs = [float(r["clean_accuracy"]) for r in rows
                if r["temperature"] == T and r["clean_accuracy"]]
        if eces:
            mean_ece = sum(eces) / len(eces)
            spread_ece = max(eces) - min(eces)
            mean_acc = sum(accs) / len(accs)
            print(f"\n  {label} packed (n={len(eces)}): "
                  f"ECE mean={mean_ece:.3f}% range={spread_ece:.3f}%  "
                  f"acc mean={mean_acc:.4f}")

    print(f"\nDone. CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
