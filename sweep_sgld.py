#!/usr/bin/env python3
"""
Curated grid search for SGLD training over both ResNet20 and
ResNet20 packed. 12 total configs, ~2-4h each = fits in ~48h.

Design principle: hold most things fixed at known-good values
(epochs=200, warmup=5, batch=1024, burn_in=100, sample_interval=5,
num_sgld_samples=20) and vary the four axes that actually control
posterior sampling behavior:

    learning_rate      burn-in LR
    sgld_gamma         SGHMC friction (0 = plain SGLD)
    sgld_temperature   posterior temperature
    sgld_sampling_lr   fixed LR during sampling phase

The grid deliberately probes higher-noise regimes than the previous
random sweep — with T=1e-3, sampling_lr=1e-4, N=45000 the effective
noise is ~2e-6, i.e. no sampling was happening.

Results append to sweep_sgld_results.csv row-by-row (interrupt-safe).
Runs are skipped if the sgld_samples_*.txt file already exists.
"""

import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys

DATASET = "cifar10"
MODEL   = "resnet20"

FIXED_ARGS = {
    "--model":              MODEL,
    "--dataset":            DATASET,
    "--val_split":          "0.1",
    "--base_optimizer":     "SGLD",
    "--epochs":             "200",
    "--warmup_epochs":      "5",
    "--batch_size":         "1024",
    "--burn_in_epochs":     "100",
    "--sgld_sample_interval": "5",
    "--num_sgld_samples":   "20",
    "--weight_decay":       "5e-4",
}

# Six configurations, applied to both non-packed and packed ⇒ 12 runs.
# Each row: (name, lr, gamma, temperature, sampling_lr)
CONFIGS = [
    ("baseline",       0.5, 0.0,  1e-3, 1e-4),   # matches run_all_models.sh
    ("hot_lowsamp",    0.5, 0.0,  1e-2, 1e-4),   # more noise via higher T
    ("hot_hisamp",     0.5, 0.0,  1e-2, 1e-3),   # +larger sampling lr
    ("sghmc_weak",     0.5, 0.1,  1e-2, 1e-3),   # weak momentum
    ("sghmc_strong",   0.5, 0.02, 1e-2, 1e-3),   # strong momentum
    ("sghmc_hot",      0.5, 0.02, 1e-1, 1e-3),   # strong momentum + hot posterior
]

VARIANTS = [
    ("nonpacked", []),
    ("packed",    ["--packed"]),
]

RESULTS_DIR    = f"experiment_results/{DATASET}_{MODEL}_no_SAM"
EVAL_DIR       = f"experiment_results/table_metrics"
PATH_FILES_DIR = "eval_path_files"
CSV_PATH       = "sweep_sgld_results.csv"

EVAL_METRICS = ["clean_accuracy", "ECE", "nll", "OOD AUROC",
                "SHIFT ACCURACY", "SHIFT ECE"]

SEED_BASE = 200


def model_name(packed):
    base = f"{MODEL}_{DATASET}_SGLD"
    return base + ("_packed" if packed else "")


def samples_path(packed, seed):
    return os.path.join(RESULTS_DIR,
                        f"sgld_samples_{model_name(packed)}_seed{seed}.txt")


def val_loss_path(packed, seed):
    return os.path.join(RESULTS_DIR,
                        f"best_val_loss_{model_name(packed)}_seed{seed}.txt")


def read_val_loss(packed, seed):
    p = val_loss_path(packed, seed)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        try:
            return float(f.readline().strip())
        except ValueError:
            return None


def train(trial_idx, cfg_name, lr, gamma, temp, samp_lr, packed, seed):
    if os.path.exists(samples_path(packed, seed)):
        print(f"[skip] samples already exist for {cfg_name} "
              f"{'packed' if packed else 'nonpacked'} seed={seed}")
        return True

    args = ["python", "train.py"]
    for k, v in {**FIXED_ARGS, "--seed": str(seed)}.items():
        args += [k, v]
    args += ["--learning_rate",    f"{lr:.6g}"]
    args += ["--sgld_gamma",       f"{gamma:.6g}"]
    args += ["--sgld_temperature", f"{temp:.6g}"]
    args += ["--sgld_sampling_lr", f"{samp_lr:.6g}"]
    args += VARIANTS[1][1] if packed else VARIANTS[0][1]

    print(f"\n{'='*70}")
    print(f"[trial {trial_idx}] {cfg_name} | "
          f"{'packed' if packed else 'nonpacked'} | seed={seed}")
    print(f"  lr={lr}  gamma={gamma}  T={temp}  sampling_lr={samp_lr}")
    print(f"{'='*70}")
    proc = subprocess.run(args)
    return proc.returncode == 0 and os.path.exists(samples_path(packed, seed))


def evaluate(trial_idx, packed, seed):
    src = samples_path(packed, seed)
    tag = f"sweep_sgld_trial{trial_idx:02d}_{'packed' if packed else 'nonpacked'}"
    path_file = f"{tag}.txt"
    save_file = f"{tag}.json"

    os.makedirs(PATH_FILES_DIR, exist_ok=True)
    shutil.copy(src, os.path.join(PATH_FILES_DIR, path_file))

    cmd = [
        "python", "evaluate.py",
        "--save_file_name", save_file,
        "--model_path_file", path_file,
        "--model_type",     MODEL + ("_packed" if packed else ""),
        "--dataset",        DATASET,
        "--batch_size",     "128",
        "--sgld_ensemble",
        "--max_sgld_samples", "10",
        "--no-eval_train",
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        return {}

    result_path = os.path.join(EVAL_DIR, save_file)
    if not os.path.exists(result_path):
        return {}
    with open(result_path) as f:
        data = json.load(f)
    metrics = next(iter(data.values()))
    return {k: metrics.get(k) for k in EVAL_METRICS}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR,    exist_ok=True)
    os.makedirs(PATH_FILES_DIR, exist_ok=True)

    fieldnames = (["trial", "config", "variant", "seed",
                   "lr", "gamma", "temperature", "sampling_lr",
                   "val_loss"]
                  + EVAL_METRICS
                  + ["samples_file"])
    write_header = not os.path.exists(CSV_PATH)

    trial_idx = 0
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        for variant_name, _ in VARIANTS:
            packed = (variant_name == "packed")
            for cfg_idx, (cfg_name, lr, gamma, temp, samp_lr) in enumerate(CONFIGS):
                trial_idx += 1
                seed = SEED_BASE + trial_idx

                ok = train(trial_idx, cfg_name, lr, gamma, temp, samp_lr,
                           packed, seed)
                if not ok:
                    print(f"[trial {trial_idx}] training FAILED — logging row anyway")
                    metrics = {}
                else:
                    print(f"[trial {trial_idx}] evaluating ensemble …")
                    metrics = evaluate(trial_idx, packed, seed)

                row = {
                    "trial":        trial_idx,
                    "config":       cfg_name,
                    "variant":      variant_name,
                    "seed":         seed,
                    "lr":           lr,
                    "gamma":        gamma,
                    "temperature":  temp,
                    "sampling_lr":  samp_lr,
                    "val_loss":     read_val_loss(packed, seed),
                    "samples_file": samples_path(packed, seed) if ok else "",
                }
                row.update({k: (f"{v:.4f}" if isinstance(v, (int, float)) else "")
                            for k, v in metrics.items()})
                writer.writerow(row)
                f.flush()

    print(f"\nAll trials done. Results in {CSV_PATH}")


if __name__ == "__main__":
    main()
