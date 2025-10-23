#!/bin/bash
torchrun \
    --nproc_per_node=1 \
    train.py \
    --distributed \
    --val_split 0.1 \
    --epochs 10 \
    --SAM \
    "$@"
