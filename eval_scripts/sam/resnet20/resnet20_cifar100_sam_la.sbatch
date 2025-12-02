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
    --save_file_name resnet20_cifar100_sam_la.json \
    --model_path_file resnet20_cifar100_sam.txt \
    --model_type resnet20 \
    --dataset cifar100 \
    --batch_size 32 \
    --laplace \
    --hessian_approx kron \
    --subset_of_weights last_layer \
    --no-eval_shift
