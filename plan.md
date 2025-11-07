# Internal Project Structure

## TODO

- [x] SAM with ResNet18 on Cifar10
- [x] Ensemble x 5 (naive)
- [x] Laplace
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

prior precision optimization (laplace uncertainty)

## Models to Train

- Ensemble (normal)
  - Cifar10
  - Cifar100

- Ensemble (packed)
  - Cifar10
  - Cifar100

- Ensemble (batched)
  - Cifar10
  - Cifar100

- Transformers

## Models to Test

- Normal
  - Cifar100

- SAM
  - Cifar10
  - Cifar100

- Ensemble (normal)
  - Cifar10
  - Cifar100

- Ensemble (packed)
  - Cifar10
  - Cifar100

- Ensemble (batched)
  - Cifar10
  - Cifar100

- Transformers

## Questions For Erik

- Omstrukturera/tips på testning. Blir många filer och mycket att hålla koll på
- Körtider för full hessian laplace
