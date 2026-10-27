#!/usr/bin/env python3
"""
SGLD hyperparameter sweep for ResNet18 on CIFAR-10 (both variants).

ResNet18 uses BatchNorm, which interacts awkwardly with SGLD (running stats
aren't resampled per posterior sample). So the ResNet20_FRN sweep winners
may not transfer. This grid stays close to the ResNet20 winners but adds a
few neighbouring cells.

Each row: (name, lr, temperature, sampling_lr)
Base config: lr=0.05 (paper's SGD lr for packed), batch_size=128,
warmup=5, burn_in=100, sample_interval=5, num_sgld_samples=20.

Appends to sweep_sgld_resnet18_results.csv row-by-row; interrupt-safe.
"""

import csv
import json
import os
import shutil
import subprocess

DATASET = "cifar10"
MODEL = "resnet18"

FIXED_ARGS = {
    "--model":              MODEL,
    "--dataset":            DATASET,
    "--val_split":          "0.1",
    "--base_optimizer":     "SGLD",
    "--epochs":             "100",
    "--warmup_epochs":      "3",
    "--batch_size":         "128",
    "--burn_in_epochs":     "50",
    "--sgld_sample_interval": "5",
    "--num_sgld_samples":   "10",
    "--weight_decay":       "5e-4",
    "--learning_rate":      "0.05",
}

# Each row: (name, temperature, sampling_lr)
# Four configs anchored on the ResNet20 winners (vcold packed, warm nonpacked)
# plus one point further each direction (base, hot) so we see the gradient.
#
#                    sampling_lr
#            1e-5    1e-4    1e-3
#   T=1e-3  vcold   base      —
#   T=1e-2    —     warm     hot
CONFIGS = [
    ("vcold",   1e-3, 1e-5),
    ("base",    1e-3, 1e-4),
    ("warm",    1e-2, 1e-4),
    ("hot",     1e-2, 1e-3),
]

VARIANTS = [
    ("nonpacked", []),
    ("packed",    ["--packed"]),
]

RESULTS_DIR    = f"experiment_results/{DATASET}_{MODEL}_no_SAM"
EVAL_DIR       = f"experiment_results/table_metrics"
PATH_FILES_DIR = "eval_path_files"
CSV_PATH       = "sweep_sgld_resnet18_results.csv"
EVAL_METRICS = ["clean_accuracy", "ECE", "nll", "OOD AUROC",
                "SHIFT ACCURACY", "SHIFT ECE"]

SEED_BASE = 600


def model_name(packed):
    return f"{MODEL}_{DATASET}_SGLD" + ("_packed" if packed else "")


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
    try:
        return float(open(p).readline().strip())
    except (ValueError, OSError):
        return None


def train(trial_idx, cfg_name, temp, samp_lr, packed, seed):
    if os.path.exists(samples_path(packed, seed)):
        print(f"[skip] samples already exist for {cfg_name} "
              f"{'packed' if packed else 'nonpacked'} seed={seed}")
        return True

    args = ["python", "train.py"]
    for k, v in {**FIXED_ARGS, "--seed": str(seed)}.items():
        args += [k, v]
    args += ["--sgld_temperature", f"{temp:.6g}"]
    args += ["--sgld_sampling_lr", f"{samp_lr:.6g}"]
    args += VARIANTS[1][1] if packed else VARIANTS[0][1]

    print(f"\n{'='*70}")
    print(f"[trial {trial_idx}] {cfg_name} | "
          f"{'packed' if packed else 'nonpacked'} | seed={seed}")
    print(f"  T={temp}  sampling_lr={samp_lr}")
    print(f"{'='*70}")
    proc = subprocess.run(args)
    return proc.returncode == 0 and os.path.exists(samples_path(packed, seed))


def evaluate(trial_idx, packed, seed):
    src = samples_path(packed, seed)
    tag = f"sweep_sgld_r18_trial{trial_idx:02d}_{'packed' if packed else 'nonpacked'}"
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
                   "lr", "temperature", "sampling_lr", "val_loss"]
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
            for cfg_name, temp, samp_lr in CONFIGS:
                trial_idx += 1
                seed = SEED_BASE + trial_idx
                ok = train(trial_idx, cfg_name, temp, samp_lr, packed, seed)
                metrics = evaluate(trial_idx, packed, seed) if ok else {}

                row = {
                    "trial":        trial_idx,
                    "config":       cfg_name,
                    "variant":      variant_name,
                    "seed":         seed,
                    "lr":           FIXED_ARGS["--learning_rate"],
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
