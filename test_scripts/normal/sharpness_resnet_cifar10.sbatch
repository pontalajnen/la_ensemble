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

$PYTHON utils/analyze_sharpness.py --save_file_name Resnet18_Cifar100_SGD_pi_10_full_dataset.txt --model_path_file evaluate_resnet18_cifar100_sgd.txt --model_type ResNet18 --dataset CIFAR100 --batch_size 512 --shift_severity 3 --mode power_iter --num_steps 70 --num_eigenthings 10 --tol 1e-2 --full_dataset