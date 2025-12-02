# Internal Project Structure

## TODO

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

- Cifar packed (resnet20)
- cifar packed la
- imagenet transformers
- imagenet transformers la
- imagenet transformers packed (la)
- compare with FRA with tensorflow representation

## Models to Train

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

## Models to Test

- Normal
  - Cifar10
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

Frågor:

- Mer minne för större dataset?

| Option                         | Pros                           | Cons

| hessian_structure="diag"       | Works with any layer           | Less accurate than KFAC

```python
model = Laplace(
    model,
    likelihood="classification",
    subset_of_weights="last_layer",  # or "all"
    hessian_structure="diag",        # instead of "kron"
    backend=BACKENDS[args.la_backend],
)
model.fit(train_loader, progress_bar=True)
```

| subset_of_weights="last_layer" | Fast, robust, often sufficient | Only calibrates last layer

```python
model = Laplace(
    model,
    likelihood="classification",
    subset_of_weights="last_layer",  # only the final nn.Linear
    hessian_structure="kron",        # kron works for Linear
    backend=BACKENDS[args.la_backend],
)
model.fit(train_loader, progress_bar=True)
```

| Freeze FRN/TLU params          | Keeps KFAC for Conv/Linear     | Ignores uncertainty in FRN/TLU

```python
for name, module in model.named_modules():
    if isinstance(module, (FRN, TLU)):
        for param in module.parameters():
            param.requires_grad = False

model = Laplace(
    model,
    likelihood="classification",
    subset_of_weights="all",
    hessian_structure="kron",
    backend=BACKENDS[args.la_backend],
)
model.fit(train_loader, progress_bar=True)
```

| Replace FRN → BatchNorm        | Full KFAC support              | Requires retraining
