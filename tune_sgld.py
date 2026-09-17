#!/usr/bin/env python3
"""
Random hyperparameter search for SGLD training.

Each trial trains, then immediately evaluates on the test set.
Results are appended to sgld_sweep_results.csv after each trial —
safe to interrupt and resume.
"""

import csv
import glob
import json
import math
import os
import random
import re
import shutil
import subprocess

# ── Settings ─────────────────────────────────────────────────────────────────
MODEL = "resnet20"
DATASET = "cifar10"
N_TRIALS = 20
SEED_START = 100   # trial i uses seed SEED_START + i

FIXED = {
    "--model":           MODEL,
    "--dataset":         DATASET,
    "--val_split":       "0.1",
    "--base_optimizer":  "SGLD",
    "--epochs":          "50",
    "--batch_size":      "1024",
}

# (min, max, scale)  scale: "log" | "linear" | "int"
SEARCH_SPACE = {
    "--learning_rate":     (0.1,  2.0,  "log"),
    "--warmup_epochs":     (3,    15,   "int"),
    "--sgld_sampling_lr":  (1e-6, 1e-3, "log"),
    "--sgld_temperature":  (1e-5, 1e-1, "log"),
    "--burn_in_epochs":    (20,   40,   "int"),
}

RESULTS_DIR    = f"experiment_results/{DATASET}_{MODEL}_no_SAM"
EVAL_DIR       = f"experiment_results/table_metrics"
PATH_FILES_DIR = "eval_path_files"
CSV_PATH       = "sgld_sweep_results.csv"

EVAL_METRICS = ["clean_accuracy", "ECE", "nll", "OOD AUROC", "SHIFT ACCURACY", "SHIFT ECE"]
# ─────────────────────────────────────────────────────────────────────────────


def sample(space):
    lo, hi, scale = space
    if scale == "log":
        return math.exp(random.uniform(math.log(lo), math.log(hi)))
    elif scale == "int":
        return random.randint(int(lo), int(hi))
    else:
        return random.uniform(lo, hi)


def parse_val_loss(path):
    m = re.search(r"val_loss=([0-9.]+)", path)
    return float(m.group(1)) if m else float("inf")


def train(trial_id, params):
    seed = SEED_START + trial_id
    samples_file = os.path.join(
        RESULTS_DIR, f"sgld_samples_{MODEL}_{DATASET}_SGLD_seed{seed}.txt"
    )

    if os.path.exists(samples_file):
        print(f"[skip]: samples already exist for seed {seed}")
        return None, samples_file

    cmd = ["python", "train.py"]
    for k, v in {**FIXED, "--seed": seed}.items():
        cmd += [k, str(v)]
    for k, v in params.items():
        cmd += [k, f"{v:.6g}"]

    before = set(glob.glob(os.path.join(RESULTS_DIR, f"mn={MODEL}_{DATASET}_SGLD-*.pth")))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        return None, None

    new = set(glob.glob(os.path.join(RESULTS_DIR, f"mn={MODEL}_{DATASET}_SGLD-*.pth"))) - before
    val_loss = min((parse_val_loss(p) for p in new), default=None)
    return val_loss, samples_file if os.path.exists(samples_file) else None


def evaluate(trial_id, samples_file):
    path_file = f"run_sgld_trial{trial_id + 1}.txt"
    save_file = f"sgld_trial{trial_id + 1}.json"

    shutil.copy(samples_file, os.path.join(PATH_FILES_DIR, path_file))

    cmd = [
        "python", "evaluate.py",
        "--save_file_name", save_file,
        "--model_path_file", path_file,
        "--model_type",     MODEL,
        "--dataset",        DATASET,
        "--batch_size",     "128",
        "--sgld_ensemble",
        "--max_sgld_samples", "5",
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
    random.seed(0)

    fieldnames = (
        ["trial", "seed", "val_loss"]
        + sorted(SEARCH_SPACE)
        + EVAL_METRICS
        + ["samples_file"]
    )
    write_header = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        best_nll, best_trial, best_params = float("inf"), None, None

        for i in range(N_TRIALS):
            params = {k: sample(v) for k, v in SEARCH_SPACE.items()}
            seed = SEED_START + i

            print(f"\n{'='*60}")
            print(f"Trial {i + 1}/{N_TRIALS}  (seed={seed})")
            for k, v in params.items():
                print(f"  {k:25s} {v:.4g}")
            print("=" * 60)

            val_loss, samples_file = train(i, params)
            if samples_file is None:
                print(f"[trial {i + 1}] training FAILED — skipping eval")
                metrics = {}
            else:
                print(f"[trial {i + 1}] evaluating...")
                metrics = evaluate(i, samples_file)

            row = {
                "trial": i + 1, "seed": seed, "val_loss": val_loss,
                "samples_file": samples_file,
            }
            row.update({k: f"{v:.6g}" for k, v in params.items()})
            row.update({k: f"{v:.4f}" if v is not None else "" for k, v in metrics.items()})
            writer.writerow(row)
            f.flush()

            nll = metrics.get("nll")
            if nll is not None and nll < best_nll:
                best_nll, best_trial, best_params = nll, i + 1, params

    print(f"\n{'='*60}")
    if best_params is not None:
        print(f"Best trial: {best_trial}  (nll={best_nll:.4f})")
        print("Best params:")
        for k, v in best_params.items():
            print(f"  {k:25s} {v:.4g}")
    else:
        print("No trials completed successfully.")
    print(f"\nFull results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
