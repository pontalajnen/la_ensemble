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

$PYTHON experiment_results/cifar10H/eval_cifar10h_per_img.py --save_file_name Resnet18_SGD_SAM__H.txt --model_1_path_file /experiment_results/cifar10H/resnet_sgd.txt --model_2_path_file /experiment_results/cifar10H/resnet_sam.txt --number_model_1 5 --number_model_2 5