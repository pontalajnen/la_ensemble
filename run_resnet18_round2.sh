#!/bin/bash
# ResNet18 CIFAR-10 SGLD, round 2:
#   - longer burn-in (150 and 180 of 200), sample every 2 epochs
#   - lr=0.1 (matches no-mom SGD)
#   - per-epoch train_loss/val_loss/val_acc/lr/temp written to
#     convergence_<model_name>_seed<seed>.csv
#
# 10 runs total:
#   seed 700/701: T=0 sanity, burn-in 180 (both variants)
#   seed 702-705: base & warm, burn-in 180 (both variants)
#   seed 706-709: base & warm, burn-in 150 (both variants)

set -e

MODEL=resnet18
DATASET=cifar10
EPOCHS=200
BATCH=128
LR=0.1
SAMPLE_INTERVAL=2
NUM_SAMPLES=25           # 150-burn runs collect 25; 180-burn naturally stops at 10
WARMUP=5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_DIR="$SCRIPT_DIR/experiment_results/${DATASET}_${MODEL}_no_SAM"

COMMON="--model $MODEL --dataset $DATASET --val_split 0.1 \
    --epochs $EPOCHS --batch_size $BATCH \
    --base_optimizer SGLD --learning_rate $LR \
    --warmup_epochs $WARMUP \
    --sgld_sample_interval $SAMPLE_INTERVAL --num_sgld_samples $NUM_SAMPLES"

already_done() {
    local name=$1
    local seed=$2
    [ -f "$SAVE_DIR/sgld_samples_${name}_seed${seed}.txt" ]
}

run() {
    local label=$1
    local name=$2
    local seed=$3
    local burn_in=$4
    local temperature=$5
    local sampling_lr=$6
    shift 6
    echo ""
    echo "========================================"
    echo "[running] $label  seed=$seed  burn_in=$burn_in  T=$temperature  s_lr=$sampling_lr"
    echo "========================================"
    if already_done "$name" "$seed"; then
        echo "[skip]: already have samples for $name seed=$seed"
        return
    fi
    python train.py $COMMON --seed $seed \
        --burn_in_epochs $burn_in \
        --sgld_temperature $temperature \
        --sgld_sampling_lr $sampling_lr "$@"
}

NAME="${MODEL}_${DATASET}_SGLD"
NAME_P="${MODEL}_${DATASET}_SGLD_packed"

# T=0 sanity, burn-in 180
run "T0 sanity nonpacked"  "$NAME"   700 180 0    1e-4
run "T0 sanity packed"     "$NAME_P" 701 180 0    1e-4 --packed

# base + warm, burn-in 180
run "base burn180 nonpacked" "$NAME"   702 180 1e-3 1e-4
run "base burn180 packed"    "$NAME_P" 703 180 1e-3 1e-4 --packed
run "warm burn180 nonpacked" "$NAME"   704 180 1e-2 1e-4
run "warm burn180 packed"    "$NAME_P" 705 180 1e-2 1e-4 --packed

# base + warm, burn-in 150
run "base burn150 nonpacked" "$NAME"   706 150 1e-3 1e-4
run "base burn150 packed"    "$NAME_P" 707 150 1e-3 1e-4 --packed
run "warm burn150 nonpacked" "$NAME"   708 150 1e-2 1e-4
run "warm burn150 packed"    "$NAME_P" 709 150 1e-2 1e-4 --packed

echo ""
echo "Round 2 finished. Convergence logs in $SAVE_DIR/convergence_*.csv"
