#!/bin/bash
torchrun \
    --nproc_per_node=1 \
    train.py \
    --distributed \
    --model resnet18 \
    --dataset cifar10 \
    --val_split 0.1 \
    --epochs 5 \
    --packed \
    "$@"
