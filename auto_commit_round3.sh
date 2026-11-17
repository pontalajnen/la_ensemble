#!/bin/bash
# Auto-commit & push every hour while round-3 training runs overnight.
#
# What it commits:
#   - convergence_*_seed80[0-7].csv        (per-epoch logs, ~10 KB each)
#   - sgld_samples_*_seed80[0-7].txt       (sample path lists, ~2 KB)
#   - best_val_loss_*_seed80[0-7].txt
#   - any tracked file you've edited
#
# What it does NOT commit:
#   - sgld_sample_*.pth model weights (too large for GitHub)
#   - anything else under experiment_results/
#
# Usage:
#   nohup bash auto_commit_round3.sh > auto_commit.log 2>&1 &
#   echo $! > auto_commit.pid     # save PID so you can kill it later
#
# To stop it:
#   kill $(cat auto_commit.pid)

set -u

cd "$(dirname "$0")"
BRANCH=$(git rev-parse --abbrev-ref HEAD)
INTERVAL=$((60 * 60))        # 1h in seconds
RESULTS_DIR="experiment_results/cifar10_resnet18_no_SAM"

echo "[auto_commit] branch=$BRANCH  interval=${INTERVAL}s  started $(date)"

while true; do
    # Force-add round-3 training artifacts (they live under gitignored path).
    for seed in 800 801 802 803 804 805 806 807; do
        for pattern in \
            "convergence_resnet18_cifar10_SGLD_seed${seed}.csv" \
            "convergence_resnet18_cifar10_SGLD_packed_seed${seed}.csv" \
            "sgld_samples_resnet18_cifar10_SGLD_seed${seed}.txt" \
            "sgld_samples_resnet18_cifar10_SGLD_packed_seed${seed}.txt" \
            "best_val_loss_resnet18_cifar10_SGLD_seed${seed}.txt" \
            "best_val_loss_resnet18_cifar10_SGLD_packed_seed${seed}.txt"
        do
            f="$RESULTS_DIR/$pattern"
            [ -f "$f" ] && git add -f "$f"
        done
    done

    # Also pick up any edits to tracked files (scripts, etc.).
    git add -u

    # Skip the commit if there's nothing staged.
    if git diff --cached --quiet; then
        echo "[auto_commit] $(date): nothing to commit"
    else
        msg="auto: round-3 checkpoint $(date -Iseconds)"
        git commit -m "$msg" && git push origin "$BRANCH" && \
            echo "[auto_commit] $(date): pushed" || \
            echo "[auto_commit] $(date): commit/push FAILED (will retry next cycle)"
    fi

    sleep "$INTERVAL"
done
