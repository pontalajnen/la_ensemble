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
export PYTHONPATH=/Midgard/home/kadec/thesis/laplace_approx_SAM

$PYTHON utils/analyze_sharpness.py --save_file_name ViT_Cifar10_SGD_lanczos.txt --model_path_file evaluate_vit_cifar10_sgd.txt --model_type ViT --dataset CIFAR10 --batch_size 4 --shift_severity 3 --mode lanczos --num_steps 2000 --num_eigenthings 10 --full_dataset