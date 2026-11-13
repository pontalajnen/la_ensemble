#!/bin/bash
# ResNet18 CIFAR-10 SGLD, round 3:
#   - probe the upper-T boundary (T=3e-2, T=1e-1)
#   - compare more samples (50) vs thinner chain (25 @ interval=4)
#   - keep warm (T=1e-2) as the known-best anchor
#   - packed AND nonpacked (nonpacked for comparison)
#
# 8 runs total, seeds 800-807:
#   800/801: hot_1e-1   nonpacked/packed   T=1e-1  interval=2  n=50
#   802/803: hot_3e-2   nonpacked/packed   T=3e-2  interval=2  n=50
#   804/805: warm_more  nonpacked/packed   T=1e-2  interval=2  n=50
#   806/807: warm_fine  nonpacked/packed   T=1e-2  interval=4  n=25
#
# Per-epoch train_loss/val_loss/val_acc/lr/temp go to
# convergence_<model_name>_seed<seed>.csv

set -e

MODEL=resnet18
DATASET=cifar10
EPOCHS=200
BATCH=128
LR=0.1
WARMUP=5
BURN_IN=100
SAMPLING_LR=1e-4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_DIR="$SCRIPT_DIR/experiment_results/${DATASET}_${MODEL}_no_SAM"

COMMON="--model $MODEL --dataset $DATASET --val_split 0.1 \
    --epochs $EPOCHS --batch_size $BATCH \
    --base_optimizer SGLD --learning_rate $LR \
    --warmup_epochs $WARMUP \
    --burn_in_epochs $BURN_IN \
    --sgld_sampling_lr $SAMPLING_LR"

already_done() {
    local name=$1
    local seed=$2
    [ -f "$SAVE_DIR/sgld_samples_${name}_seed${seed}.txt" ]
}

run() {
    local label=$1
    local name=$2
    local seed=$3
    local temperature=$4
    local sample_interval=$5
    local num_samples=$6
    shift 6
    echo ""
    echo "========================================"
    echo "[running] $label  seed=$seed  T=$temperature  interval=$sample_interval  n=$num_samples"
    echo "========================================"
    if already_done "$name" "$seed"; then
        echo "[skip]: already have samples for $name seed=$seed"
        return
    fi
    python train.py $COMMON --seed $seed \
        --sgld_temperature $temperature \
        --sgld_sample_interval $sample_interval \
        --num_sgld_samples $num_samples "$@"
}

NAME="${MODEL}_${DATASET}_SGLD"
NAME_P="${MODEL}_${DATASET}_SGLD_packed"

# hot_1e-1: T=1e-1, dense sampling (interval=2, n=50)
run "hot_1e-1 nonpacked"   "$NAME"   800 1e-1 2 50
run "hot_1e-1 packed"      "$NAME_P" 801 1e-1 2 50 --packed

# hot_3e-2: T=3e-2, dense sampling
run "hot_3e-2 nonpacked"   "$NAME"   802 3e-2 2 50
run "hot_3e-2 packed"      "$NAME_P" 803 3e-2 2 50 --packed

# warm_more: T=1e-2, dense sampling (double the winner's samples)
run "warm_more nonpacked"  "$NAME"   804 1e-2 2 50
run "warm_more packed"     "$NAME_P" 805 1e-2 2 50 --packed

# warm_fine: T=1e-2, thinned chain (interval=4, n=25) — autocorrelation probe
run "warm_fine nonpacked"  "$NAME"   806 1e-2 4 25
run "warm_fine packed"     "$NAME_P" 807 1e-2 4 25 --packed

echo ""
echo "Round 3 finished. Convergence logs in $SAVE_DIR/convergence_*.csv"
