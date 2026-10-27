#!/bin/bash
# Train the three SGD baselines WITHOUT momentum, to isolate what role
# momentum plays versus the existing momentum=0.9 SGD baselines.
#
# Checkpoints get a "_nomom" suffix in the model_name so they don't
# overwrite the existing SGD-with-momentum .pth files.

set -e

MODEL=${1:-resnet20}
DATASET=${2:-cifar10}
EPOCHS=${3:-200}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/experiment_results"
SAVE_DIR="$RESULTS_DIR/${DATASET}_${MODEL}_no_SAM"

BASE_ARGS="--model $MODEL --dataset $DATASET --val_split 0.1 --epochs $EPOCHS \
    --base_optimizer SGD --momentum 0.0"

already_trained() {
    local model_name=$1
    if ls "$SAVE_DIR"/mn="${model_name}"-*.pth 2>/dev/null | grep -q .; then
        return 0
    fi
    return 1
}

run() {
    local name=$1
    local model_name=$2
    shift 2
    echo ""
    echo "======================================"
    echo "[running]: $name"
    echo "======================================"
    if already_trained "$model_name"; then
        echo "[skip]: checkpoint already exists for $model_name"
        return
    fi
    python train.py $BASE_ARGS "$@"
}

run "SGD (no momentum)"          "${MODEL}_${DATASET}_SGD_nomom"
run "SGD ensemble (no momentum)" "${MODEL}_${DATASET}_SGD_nomom_ensemble" --ensemble
run "SGD packed (no momentum)"   "${MODEL}_${DATASET}_SGD_nomom_packed"   --packed

echo ""
echo "All SGD no-momentum models finished."
