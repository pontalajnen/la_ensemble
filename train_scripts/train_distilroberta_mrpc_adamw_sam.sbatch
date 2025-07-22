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

$PYTHON train_nlp.py --model ROBERTA --NLP_model distilbert/distilroberta-base --dataset MRPC --batch_size 16 --seed 0 --seeds_per_job 5 --base_optimizer AdamW --learning_rate 2e-05 --num_warmup_steps 0.1  --lr_scheduler linear --weight_decay 0.01 --epochs 3 --SAM true --adaptive true --rho 0.010831860566663788