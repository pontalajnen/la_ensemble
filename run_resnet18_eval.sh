#!/bin/bash
# Evaluate the 6 ResNet18 CIFAR-10 training runs:
#   SGD (mom=0.9) non-packed and packed  — paper-regime sanity check
#   SGD (no mom)  non-packed and packed  — momentum-free baseline
#   SGLD baseline (seed 500 / 501) non-packed and packed — single-point SGLD runs
#
# Results go to experiment_results/table_metrics/ as JSON.

set -e

sgd_eval() {
    local save=$1
    local pathfile=$2
    local model_type=$3
    python evaluate.py \
        --save_file_name "$save" \
        --model_path_file "$pathfile" \
        --model_type "$model_type" \
        --dataset cifar10 \
        --batch_size 128 \
        --no-eval_train
}

sgld_eval() {
    local save=$1
    local pathfile=$2
    local model_type=$3
    python evaluate.py \
        --save_file_name "$save" \
        --model_path_file "$pathfile" \
        --model_type "$model_type" \
        --dataset cifar10 \
        --batch_size 128 \
        --sgld_ensemble \
        --max_sgld_samples 10 \
        --no-eval_train
}

sgd_eval  resnet18_cifar10_sgd_mom.json          resnet18_cifar10_sgd_mom.txt          resnet18
sgd_eval  resnet18_cifar10_sgd_mom_packed.json   resnet18_cifar10_sgd_mom_packed.txt   resnet18_packed
sgd_eval  resnet18_cifar10_sgd_nomom.json        resnet18_cifar10_sgd_nomom.txt        resnet18
sgd_eval  resnet18_cifar10_sgd_nomom_packed.json resnet18_cifar10_sgd_nomom_packed.txt resnet18_packed

sgld_eval resnet18_cifar10_sgld_baseline.json         resnet18_cifar10_sgld_baseline.txt         resnet18
sgld_eval resnet18_cifar10_sgld_baseline_packed.json  resnet18_cifar10_sgld_baseline_packed.txt  resnet18_packed

echo ""
echo "Done. Results in experiment_results/table_metrics/"
