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

# Load gcc for compilation
#module load gcc

# Extract files
# mkdir /scratch/local/cifar10
# tar -xf /proj/berzelius-aiics-real/users/x_ponhe/la_ensamble/data/cifar10-python.tar.gz -C /scratch/local/cifar10


# Train and save the exit code of the python script
torchrun --nproc_per_node=1 train.py \
    --distributed \
    --model resnet20 \
    --dataset cifar10 \
    --val_split 0.1 \
    --ensemble
