#!/bin/bash
# Six ResNet18 training runs on CIFAR-10:
#   1) ResNet18         SGD    (momentum=0.9, paper setup — sanity check)
#   2) ResNet18 packed  SGD    (momentum=0.9, paper setup — sanity check)
#   3) ResNet18         SGD    (no momentum)
#   4) ResNet18 packed  SGD    (no momentum)
#   5) ResNet18         SGLD
#   6) ResNet18 packed  SGLD
#
# SGD-no-momentum uses lr=0.1 because the paper's lr=0.05 was with momentum=0.9;
# without momentum we need ~2× to converge in 200 epochs.
#
# SGLD uses the ResNet20 sweep winners per variant:
#   nonpacked: T=1e-2  sampling_lr=1e-4   ("warm")
#   packed:    T=1e-3  sampling_lr=1e-5   ("verycold")

set -e

MODEL=resnet18
DATASET=cifar10
EPOCHS=200
BATCH_SIZE=128

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_DIR="$SCRIPT_DIR/experiment_results/${DATASET}_${MODEL}_no_SAM"

COMMON="--model $MODEL --dataset $DATASET --val_split 0.1 \
    --epochs $EPOCHS --batch_size $BATCH_SIZE"

already_trained_sgd() {
    local model_name=$1
    ls "$SAVE_DIR"/mn="${model_name}"-*.pth 2>/dev/null | grep -q .
}

already_trained_sgld() {
    local model_name=$1
    local seed=$2
    [ -f "$SAVE_DIR/sgld_samples_${model_name}_seed${seed}.txt" ]
}

run_sgd_mom() {
    local name=$1
    local model_name=$2
    shift 2
    echo ""
    echo "======================================"
    echo "[running SGD mom=0.9]: $name"
    echo "======================================"
    if already_trained_sgd "$model_name"; then
        echo "[skip]: checkpoint already exists for $model_name"
        return
    fi
    python train.py $COMMON --base_optimizer SGD --momentum 0.9 \
        --learning_rate 0.05 --warmup_epochs 5 "$@"
}

run_sgd_nomom() {
    local name=$1
    local model_name=$2
    shift 2
    echo ""
    echo "======================================"
    echo "[running SGD no-mom]: $name"
    echo "======================================"
    if already_trained_sgd "$model_name"; then
        echo "[skip]: checkpoint already exists for $model_name"
        return
    fi
    python train.py $COMMON --base_optimizer SGD --momentum 0.0 \
        --learning_rate 0.1 --warmup_epochs 5 "$@"
}

run_sgld() {
    local name=$1
    local model_name=$2
    local seed=$3
    local temperature=$4
    local sampling_lr=$5
    shift 5
    echo ""
    echo "======================================"
    echo "[running SGLD]: $name"
    echo "======================================"
    if already_trained_sgld "$model_name" "$seed"; then
        echo "[skip]: SGLD samples already exist for $model_name seed=$seed"
        return
    fi
    python train.py $COMMON --base_optimizer SGLD --seed $seed \
        --learning_rate 0.05 --warmup_epochs 5 \
        --sgld_temperature $temperature --sgld_sampling_lr $sampling_lr \
        --burn_in_epochs 100 --sgld_sample_interval 5 --num_sgld_samples 20 \
        "$@"
}

run_sgd_mom   "SGD (mom=0.9)"         "${MODEL}_${DATASET}_SGD"
run_sgd_mom   "SGD packed (mom=0.9)"  "${MODEL}_${DATASET}_SGD_packed"        --packed
run_sgd_nomom "SGD (no mom)"          "${MODEL}_${DATASET}_SGD_nomom"
run_sgd_nomom "SGD packed (no mom)"   "${MODEL}_${DATASET}_SGD_nomom_packed"  --packed
run_sgld      "SGLD"                  "${MODEL}_${DATASET}_SGLD"              500 1e-2 1e-4
run_sgld      "SGLD packed"           "${MODEL}_${DATASET}_SGLD_packed"       501 1e-3 1e-5 --packed

echo ""
echo "All ResNet18 runs finished."
