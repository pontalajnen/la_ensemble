#!/bin/bash
set -e

MODEL=${1:-resnet20}
DATASET=${2:-cifar10}
EPOCHS=${3:-200}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/experiment_results"

BASE_ARGS="--model $MODEL --dataset $DATASET --val_split 0.1 --epochs $EPOCHS"

# Returns 0 (true) if a completed checkpoint already exists for the given model_name and save_dir
already_trained() {
    local save_dir=$1
    local model_name=$2
    # After training, best checkpoint is renamed to mn=<model_name>-*.pth
    # For SGLD, a samples list file is created instead
    if ls "$save_dir"/mn="${model_name}"-*.pth 2>/dev/null | grep -q .; then
        return 0
    fi
    if [ -f "$save_dir/sgld_samples_${model_name}_seed1.txt" ]; then
        return 0
    fi
    return 1
}

run() {
    local name=$1
    local model_name=$2
    local save_dir=$3
    shift 3
    echo ""
    echo "======================================"
    echo "[running]: $name"
    echo "======================================"
    if already_trained "$save_dir" "$model_name"; then
        echo "[skip]: checkpoint already exists"
        return
    fi
    python train.py $BASE_ARGS "$@"
}

NO_SAM_DIR="$RESULTS_DIR/${DATASET}_${MODEL}_no_SAM"
SAM_DIR="$RESULTS_DIR/${DATASET}_${MODEL}__SAM"

run "Normal (SGD)" "${MODEL}_${DATASET}_SGD"          "$NO_SAM_DIR"
run "Ensemble"     "${MODEL}_${DATASET}_SGD_ensemble"  "$NO_SAM_DIR" --ensemble
run "Packed"       "${MODEL}_${DATASET}_SGD_packed"    "$NO_SAM_DIR" --packed
run "SAM"          "${MODEL}_${DATASET}_SGD"           "$SAM_DIR"    --SAM
run "SGLD+SAM"     "${MODEL}_${DATASET}_SGLD_SAM"      "$SAM_DIR"    --base_optimizer SGLD --SAM \
    --learning_rate 0.5 --warmup_epochs 5 --epochs 200 --batch_size 1024 \
    --sgld_sampling_lr 1e-4 --sgld_temperature 1e-3
run "SGLD"         "${MODEL}_${DATASET}_SGLD"          "$NO_SAM_DIR" --base_optimizer SGLD \
    --learning_rate 0.5 --warmup_epochs 5 --epochs 200 --batch_size 1024 \
    --sgld_sampling_lr 1e-04 --sgld_temperature 1e-3 \
    --burn_in_epochs 40

echo ""
echo "All models finished."
