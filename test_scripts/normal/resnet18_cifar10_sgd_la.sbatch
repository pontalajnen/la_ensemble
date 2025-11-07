#!/bin/bash
#SBATCH --output=/proj/berzelius-aiics-real/users/x_ponhe/la_ensamble/logs/%A_%a_slurm.out
#SBATCH --error=/proj/berzelius-aiics-real/users/x_ponhe/la_ensamble/logs/%A_%a_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="ponhe215@student.liu.se"
#SBATCH --gres gpu:1
#SBATCH --time 48:00:00
#SBATCH -C "thin"

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

# Activate conda
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba activate /home/x_ponhe/.conda/envs/pytorch-2.5.1-python-3.10


python3 evaluate.py \
    --save_file_name resnet18_cifar10_sgd_la.json \
    --model_path_file resnet18_cifar10_sgd.txt \
    --model_type resnet18 \
    --dataset cifar10 \
    --batch_size 4 \
    --laplace \
    --hessian_approx kron \

    # --shift_severity 3 \
    # --num_data 100 \
    # --subset_of_weights all \
    # --backend AsdlGGN