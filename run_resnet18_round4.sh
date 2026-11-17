#!/bin/bash
# ResNet18 CIFAR-10 SGLD, round 4: verification of the T=3e-2 packed winner.
#
# Seed 803 (T=3e-2 packed) got ECE=0.48% — single-draw, unverified.
# This round reproduces it with 3 more seeds and brackets with T=5e-2.
#
# All runs: packed only, burn_in=100, lr=0.1, sampling_lr=1e-4,
# interval=2, n_samples=50, epochs=200, warmup=5 (identical to seed 803).
#
# 5 runs total, seeds 900-904:
#   900,901,902: T=3e-2 packed  — reproduce the ECE=0.48% result
#   903,904:     T=5e-2 packed  — bracket between 3e-2 and 1e-1

set -e

MODEL=resnet18
DATASET=cifar10
EPOCHS=200
BATCH=128
LR=0.1
WARMUP=5
BURN_IN=100
SAMPLING_LR=1e-4
INTERVAL=2
NUM_SAMPLES=50

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_DIR="$SCRIPT_DIR/experiment_results/${DATASET}_${MODEL}_no_SAM"

COMMON="--model $MODEL --dataset $DATASET --val_split 0.1 \
    --epochs $EPOCHS --batch_size $BATCH \
    --base_optimizer SGLD --learning_rate $LR \
    --warmup_epochs $WARMUP \
    --burn_in_epochs $BURN_IN \
    --sgld_sampling_lr $SAMPLING_LR \
    --sgld_sample_interval $INTERVAL \
    --num_sgld_samples $NUM_SAMPLES \
    --packed"

already_done() {
    local seed=$1
    [ -f "$SAVE_DIR/sgld_samples_resnet18_cifar10_SGLD_packed_seed${seed}.txt" ]
}

run() {
    local label=$1
    local seed=$2
    local temperature=$3
    echo ""
    echo "========================================"
    echo "[running] $label  seed=$seed  T=$temperature"
    echo "========================================"
    if already_done "$seed"; then
        echo "[skip]: already have samples for seed=$seed"
        return
    fi
    python train.py $COMMON --seed $seed --sgld_temperature $temperature
}

# Reproduce T=3e-2 packed (seed 803 result)
run "T3e-2 packed repro #1"  900 3e-2
run "T3e-2 packed repro #2"  901 3e-2
run "T3e-2 packed repro #3"  902 3e-2

# Bracket between 3e-2 (winner) and 1e-1 (too hot)
run "T5e-2 packed bracket #1" 903 5e-2
run "T5e-2 packed bracket #2" 904 5e-2

echo ""
echo "Round 4 finished. Convergence logs in $SAVE_DIR/convergence_*.csv"
