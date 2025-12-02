#!/bin/bash
torchrun \
    --nproc_per_node=1 \
    train.py \
    --distributed \
    --model resnet20 \
    --dataset cifar100 \
    --val_split 0.1 \
    --SAM \
    --epochs 5 \
    "$@"

    # --packed \