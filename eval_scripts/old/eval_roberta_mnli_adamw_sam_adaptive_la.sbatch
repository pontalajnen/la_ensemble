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

$PYTHON evaluate.py --save_file_name ROBERTA_MNLI_AdamW_SAM_adaptive_rho01_ll_full_mc_asdl.txt --model_path_file evaluate_roberta_mnli_adamw_sam_adaptive_rho01.txt --model_type ROBERTA --NLP_model distilbert/distilroberta-base --dataset MNLI --batch_size 8 --no-eval_train --laplace --hessian_approx full --subset_of_weights last_layer --approx_link mc --mc_samples 5000 --backend AsdlGGN