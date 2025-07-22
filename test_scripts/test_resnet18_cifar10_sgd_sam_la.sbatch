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

$PYTHON evaluate.py --save_file_name Resnet18_Cifar10_SGD_SAM_la_ll_kron_mc_opt_asdl.txt --model_path_file evaluate_resnet18_cifar10_sgd_sam.txt --model_type ResNet18 --dataset CIFAR10 --batch_size 128 --shift_severity 3 --laplace --hessian_approx kron --pred_type glm  --approx_link mc --mc_samples 5000 --subset_of_weights last_layer --backend AsdlGGN --optimize_prior_precision marglik