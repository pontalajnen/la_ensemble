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

$PYTHON evaluate.py --save_file_name Resnet18_Cifar10_SGD_la_ll_NN_5000.txt --model_path_file evaluate_resnet18_cifar10_sgd.txt --model_type ResNet18 --dataset CIFAR10 --batch_size 1 --shift_severity 3 --laplace --hessian_approx diag --subset_of_weights all --pred_type nn --approx_link mc --mc_samples 5000 --backend AsdlGGN