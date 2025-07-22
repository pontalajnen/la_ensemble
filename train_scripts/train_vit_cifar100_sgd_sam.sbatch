#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem 8GB
#SBATCH --cpus-per-task 4
#SBATCH --time 48:00:00
#SBATCH --output /Midgard/home/%u/thesis/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/thesis/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

PYTHON=/Midgard/home/kadec/miniforge3/envs/venv/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/kadec/miniforge3

$PYTHON train.py --model ViT --dataset CIFAR100 --batch_size 8 --val_split 0.1 --seed 10 --seeds_per_job 1 --base_optimizer AdamW --learning_rate 2.344308749167926e-05 --weight_decay 0.01 --epochs 30 --SAM --rho 0.11700293773517456
