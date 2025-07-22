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

$PYTHON evaluate.py --save_file_name BERT_MNLI_AdamW_SAM_adaptive_ll_full_probit_asdl_ood_acc.txt --model_path_file evaluate_bert_mnli_adamw_sam_adaptive.txt --model_type BERT --NLP_model bert-base-cased --dataset MNLI --batch_size 8 --no-eval_train  --laplace --hessian_approx full --subset_of_weights last_layer --approx_link probit --backend AsdlGGN