#!/usr/bin/env bash
#SBATCH --gres gpu:1
#SBATCH --mem 8GB
#SBATCH --cpus-per-task 4
#SBATCH --time 48:00:00
#SBATCH --output /proj/berzelius-aiics-real/users/$USER/la_ensamble/logs/%J_slurm.out
#SBATCH --error  /proj/berzelius-aiics-real/users/$USER/la_ensamble/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

PYTHON=/usr/bin/python3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/software/sse/modules/Miniforge3

$PYTHON evaluate.py \
    --save_file_name BERT_MNLI_AdamW_ll_full_mc_asdl_ood_acc.txt \
    --model_path_file evaluate_bert_mnli_adamw.txt \
    --model_type BERT \
    --NLP_model bert-base-cased \
    --dataset MNLI --batch_size 8 --no-eval_train --laplace \
    --hessian_approx full --subset_of_weights last_layer --approx_link mc --mc_samples 5000 --backend AsdlGGN