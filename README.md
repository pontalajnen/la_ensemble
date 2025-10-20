# Laplace Approximation With SAM and Ensemble Models

Master thesis: Combining Laplace Approximation with Sharpness Aware Minimization

## Datasets

The following datasets are available via torch_uncertainty DataModules:

| Dataset   | Shift | OOD | Alternative Test |
| --------- | -----  | ---  | ---------------- |
| CIFAR-10  | CIFAR-10-C    | SVHN    | CIFAR-10-H |
| CIFAR-100 | CIFAR-100-C   | SVHN    |            |

Datasets for NLP tasks use HuggingFace. MRPC and MNLI are supported.

## Models

The code supports ResNet18, ViTs, BERT, DistilRoBERTa (and would with minor adjustments support similar HuggingFace models and further ResNet and WRN models).

## 🛠️ Setup

Before running the code, make sure to:

- Install the required packages by running ```pip install -r requirements.txt
  > **Note**: Follow this [instruction](https://github.com/noahgolmant/pytorch-hessian-eigenthings) to install hessian_eigenthings (no pip install available)
- Create a folder named `eval_path_files`  
  This is where you store `.txt` files containing model paths to be evaluated.
- Create a `data/` folder  
  This folder is used to store vision datasets.  
  > **Note**: HuggingFace datasets don't require this folder — it is only used by `DataModules` from [torch-uncertainty](https://github.com/ENSTA-U2IS/torch-uncertainty).

---

## 🏋️‍♂️ Training

There are currently three training scripts:

- `train.py`  
  Original script used for training **vision models**.

- `train_nlp.py`  
  Modified script that supports training **language models**.  
  It *should* also work with vision models, but this hasn't been fully tested.

- `train_lang_OLD.py`  
  ⚠️ **Deprecated** — do not use.  
  This was an older version with a different approach to managing Weights & Biases sweeps.  
  It will be removed in a future version.

> 📝 **Example bash scripts** for training can be found in the `train_scripts/` directory.

---

## 📊 Evaluation

- Use the `evaluate.py` file for evaluation. `--model_path_file` should point to a `.txt` file in the `eval_path_files` folder. The default evaluation does not use Laplace Approximation.
- 📝 **Example bash scripts** for evaluation can be found in the `test_scripts/` directory.
- If you run an evaluation of the same model and use the same `--save_file_name` the result will not be overwritten but evaluation will be skipped.
- You can use the flags `--no-eval_train`, `--no-eval_shift`and `--no-eval_ood`if you don't want to or can't (e.g. no ood dataset available) evaluate them
- To run **CIFAR-10H** experiments, use the `--cifar10h` flag when evaluating models on CIFAR-10.
- Create a `.txt` file in the `cifar10H` folder that contains the paths to the `.npy` files
- Run the `eval_cifar10h_per_image.py` script to evaluate model predictions on CIFAR-10H.

> **Note**: This script calculates the image-wise mean over n=X models and does a t-test with n= 10000. `eval_cifar10h_OLD_average_over_images.py` calculates the mean over all images and the corresponding t-test has a sample size of n=X.

### 🔧 Reliability Diagrams

- Use the `--rel_plot` flag to generate reliability diagrams during evaluation.

## Private Notes

## Login to Berzelius

```bash
ssh <username>@berzelius1.nsc.liu.se  # Node 1
ssh <username>@berzelius2.nsc.liu.se  # Node 2
ssh <username>@berzelius.nsc.liu.se   # Auto assignment
```

## Timeline

- SAM with ResNet18 on Cifar10
- Ensemble x 5 (naive)
- Read the paper

## TODO

- [x] SAM with ResNet18 on Cifar10
- [x] Ensemble x 5 (naive)
- [ ] Laplace
- [ ] Packed Ensembles (AdamW and SAM)
- [ ] Packed Ensembles + Laplace
- [ ] Cifar100
- [ ] BatchEnsembles
- [ ] Transformers (ViT-x)
- [ ] Laplace on Batch

interactive -C "fat"

resnet18 på cifar10 och 100 (laplace)
sam (laplace)
ensemble (subset i laplace biblioteket)
packed ensemble

## Questions For Erik

- Hyperparametrar, hur ska vi gå tillväga, eller ska vi kopiera??

## Berzelius Documentation

**Project storage directories available to you:**
/proj/berzelius-aiics-real

**Documentation and getting help:**
[Getting Started](https://www.nsc.liu.se/support/systems/berzelius-getting-started/)
[Support](https://www.nsc.liu.se/support)

**Useful commands**
To see your active projects and CPU time usage: projinfo
To see available disk storage and usage: nscquota
To see your last jobs: lastjobs
Login to compute node to check running job: jobsh

To tweak job priorities, extend time limits and reserve nodes: see
[Job Priorities and Time Limits](https://www.nsc.liu.se/support/batch-jobs/boost-tools/)

(Run "nsc-mute-login" to not show this information)

## Setting up Berzelius Environment

Create a mamba environment with the correct python and torch version ([documentation](https://www.nsc.liu.se/support/systems/berzelius-software/berzelius-conda-mamba/)):

```bash
module load Miniforge3/24.7.1-2-hpc1-bdist
mamba create --name pytorch-2.5.1-python-3.10 python=3.10
mamba activate pytorch-2.5.1-python-3.100
CONDA_OVERRIDE_CUDA=11.8 mamba install pytorch==2.5.1=cuda* torchvision=*=cuda* torchaudio=*=cuda*
```

For convenience, add a command to activate the environment in your .bashrc file, as it has a complicated name:

```bash
vim ~/.bashrc

# Add the following to the bottom of the file (G, o, ctrl+shift+v, esc, :wq)
```bash
alias <name of choice>="mamba activate pytorch-2.5.1-python-3.10"
```

## Starting the Environment

To run files, first mamba has to be activated:

```bash
startmamba
```

## Running With Multiple GPUs

```bash
interactive --gpus=x

torchrun --nproc_per_node=2 train.py --distributed [other args]
```

## Verify ResNet implementation

- Check parameters
- Forward pass med samma init

## Tests to run

- eval_resnet18_cifar10_sgd_la_GLM.sbatch
- eval_resnet18_cifar10_sgd_la_NN.sbatch
- eval_bert_mnli_adamw_la.sbatch
- eval_bert_mnli_adamw_sam_adaptive_la.sbatch
- eval_bert_mnli_adamw_sam_adaptive.sbatch
- eval_bert_mnli_adamw_sam_la.sbatch
- eval_bert_mnli_adamw_sam.sbatch
- eval_bert_mnli_adamw.sbatch
- eval_bert_mrpc_adamw_la.sbatch
- eval_bert_mrpc_adamw_sam_adaptive_la.sbatch
- eval_bert_mrpc_adamw_sam_adaptive.sbatch
- eval_bert_mrpc_adamw_sam_la.sbatch
- eval_bert_mrpc_adamw_sam.sbatch
- eval_bert_mrpc_adamw.sbatch
- eval_cifar10H.sbatch
- eval_roberta_mnli_adamw_la.sbatch
- eval_roberta_mnli_adamw_sam_adaptive_la.sbatch
- eval_roberta_mnli_adamw_sam_adaptive.sbatch
- eval_roberta_mnli_adamw_sam_la.sbatch
- eval_roberta_mnli_adamw_sam.sbatch
- eval_roberta_mnli_adamw.sbatch
- eval_roberta_mrpc_adamw_la.sbatch
- eval_roberta_mrpc_adamw_sam_adaptive_la.sbatch
- eval_roberta_mrpc_adamw_sam_adaptive.sbatch
- eval_roberta_mrpc_adamw_sam_la.sbatch
- eval_roberta_mrpc_adamw_sam.sbatch
- eval_roberta_mrpc_adamw.sbatch
- eval_vit_cifar100_sgd_la.sbatch
- eval_vit_cifar100_sgd_sam_la.sbatch
- eval_vit_cifar100_sgd_sam.sbatch
- eval_vit_cifar100_sgd.sbatch
- eval_vit_cifar10_sgd_la.sbatch
- eval_vit_cifar10_sgd_sam_la.sbatch
- eval_vit_cifar10_sgd_sam.sbatch
- eval_vit_cifar10_sgd.sbatch
- resnet18_cifar100_sgd_la.sbatch
- resnet18_cifar100_sgd_sam_la.sbatch
- sharpness_resnet_cifar10_lanczos.sbatch
- sharpness_resnet_cifar10.sbatch
- sharpness_vit_cifar10.sbatch
- sharpness_vit_lanzcos.sbatch
- test_resnet18_cifar100_sgd_sam.sbatch
- test_resnet18_cifar100_sgd.sbatch
- test_resnet18_cifar10_sgd_H.sbatch
- test_resnet18_cifar10_sgd_la.sbatch
- Test_Resnet18_Cifar10_SGD_LA.sh
- test_resnet18_cifar10_sgd_sam_la.sbatch
- Test_Resnet18_Cifar10_SGD_SAM_LA.sh
- test_resnet18_cifar10_sgd_sam.sbatch
- Test_Resnet18_Cifar10_SGD_SAM.sh
- test_resnet18_cifar10_sgd.sbatch
- Test_Resnet18_Cifar10_SGD.sh
- test_vit_cifar10H.sbatch
