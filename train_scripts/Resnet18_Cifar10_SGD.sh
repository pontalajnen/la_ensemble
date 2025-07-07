#!/bin/bash
# cd "//////laplace_approx_SAM"

echo "--- Start training ---"
/proj/berzelius-aiics-real/users/$USER/la_ensamble train.py \
    --model ResNet18 \
    --dataset CIFAR10 \
    --batch_size 128 \
    --val_split 0.1 \
    --ensemble

echo "--- Training done ---"