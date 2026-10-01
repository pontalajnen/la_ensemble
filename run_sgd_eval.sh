#!/bin/bash
# Evaluate the three SGD baselines on CIFAR-10:
#   plain SGD    (single model)
#   SGD ensemble (deep ensemble, 4 nets)
#   SGD packed   (packed ensemble, 4 heads)
#
# Writes JSON metrics to experiment_results/table_metrics/.

set -e

python evaluate.py \
    --save_file_name resnet20_cifar10_sgd.json \
    --model_path_file resnet20_cifar10_sgd.txt \
    --model_type resnet20 \
    --dataset cifar10 \
    --batch_size 128 \
    --no-eval_train

python evaluate.py \
    --save_file_name resnet20_cifar10_sgd_ensemble.json \
    --model_path_file resnet20_cifar10_sgd_ensemble.txt \
    --model_type resnet20_ensemble \
    --dataset cifar10 \
    --batch_size 128 \
    --no-eval_train

python evaluate.py \
    --save_file_name resnet20_cifar10_sgd_packed.json \
    --model_path_file resnet20_cifar10_sgd_packed.txt \
    --model_type resnet20_packed \
    --dataset cifar10 \
    --batch_size 128 \
    --no-eval_train

echo ""
echo "Done. Results in experiment_results/table_metrics/"
