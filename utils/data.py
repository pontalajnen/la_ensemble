# from torchvision import datasets, transforms
# import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
# import os
from torch_uncertainty.datamodules import CIFAR10DataModule, ImageNetDataModule, CIFAR100DataModule, MNISTDataModule
from transformers import ViTImageProcessor
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import numpy as np
from functools import partial


def load_data_module(args, root):
    dataset = args.dataset
    batch_size = args.batch_size
    num_workers = args.num_workers if hasattr(args, "num_workers") else 1
    val_split = args.val_split if hasattr(args, "val_split") else 0.0
    test_alt = args.test_alt if hasattr(args, "test_alt") else None
    eval_ood = args.eval_ood if hasattr(args, "eval_ood") else False
    eval_shift = args.eval_shift if hasattr(args, "eval_shift") else False
    shift_severity = args.shift_severity if hasattr(args, "shift_severity") else 1
    basic_augment = args.basic_augment if hasattr(args, "basic_augment") else True
    ood_ds = args.ood_ds if hasattr(args, "ood_ds") else "openimage-o"

    if dataset == "cifar10":
        num_classes = 10
        dm = CIFAR10DataModule(root=root, batch_size=batch_size, num_workers=num_workers, val_split=val_split,
                               test_alt=test_alt, eval_ood=eval_ood, eval_shift=eval_shift,
                               shift_severity=shift_severity, basic_augment=basic_augment)
    elif dataset == "cifar100":
        num_classes = 100
        dm = CIFAR100DataModule(root=root, batch_size=batch_size, num_workers=num_workers, val_split=val_split,
                                eval_ood=eval_ood, eval_shift=eval_shift, shift_severity=shift_severity,
                                basic_augment=basic_augment)
    elif dataset == "imagenet":
        num_classes = 1000
        dm = ImageNetDataModule(root=root, batch_size=batch_size, num_workers=num_workers, val_split=val_split,
                                test_alt=test_alt, ood_ds=ood_ds, eval_ood=eval_ood, eval_shift=eval_shift,
                                basic_augment=basic_augment)
    elif dataset == "mnist":
        num_classes = 10
        if ood_ds == "openimage-o":
            ood_ds = "fashion"
        dm = MNISTDataModule(root=root, batch_size=batch_size, eval_ood=eval_ood, eval_shift=eval_shift,
                             num_workers=num_workers, ood_ds=ood_ds, val_split=val_split, basic_augment=basic_augment)
    else:
        raise Exception("Dataset not supported!")
    return dm, num_classes


def load_hf_dataset(NLP_model, dataset_name, eval_ood, eval_shift, batch_size):
    is_nlp_task = True
    tokenizer = AutoTokenizer.from_pretrained(NLP_model)

    # Load raw dataset
    if dataset_name == "MNLI":
        num_classes = 3
        raw = load_dataset("glue", "mnli")
        raw_val_set = raw["validation_matched"]
        raw_train_set = raw["train"]
        split_required = True
        encode_fn = encode_mnli
    elif dataset_name == "MRPC":
        eval_ood = False
        eval_shift = False
        num_classes = 2
        raw = load_dataset("glue", "mrpc")
        raw_val_set = raw["validation"]
        raw_train_set = raw["train"]
        raw_test_set = raw["test"]
        split_required = False
        encode_fn = encode_mrpc
    elif dataset_name == "RTE":
        eval_ood = False
        eval_shift = False
        num_classes = 2
        raw = load_dataset("glue", "rte")
        raw_val_set = raw["validation"]
        raw_train_set = raw["train"]
        split_required = True
        encode_fn = encode_mrpc

    if split_required:
        # Stratified deterministic split of validation set
        val_labels = raw_val_set["label"]
        val_idx, test_idx = train_test_split(
            np.arange(len(val_labels)),
            test_size=0.5,
            stratify=val_labels,
            random_state=42
        )
        raw_val_split = raw_val_set.select(val_idx)
        raw_test_split = raw_val_set.select(test_idx)
        print(f"[INFO] Validation set size: {len(raw_val_split)}")
        print(f"[INFO] Test set size (from split): {len(raw_test_split)}")
    else:
        raw_val_split = raw_val_set
        raw_test_split = raw_test_set
        print(f"[INFO] Validation set size: {len(raw_val_split)}")
        print(f"[INFO] Test set size (official): {len(raw_test_split)}")

    print(f"[INFO] Train set size: {len(raw_train_set)}")

    # Encode all splits
    tokenize = partial(encode_fn, tokenizer=tokenizer)
    train_dataset = raw_train_set.map(tokenize, batched=True)
    val_dataset = raw_val_split.map(tokenize, batched=True)
    test_dataset = raw_test_split.map(tokenize, batched=True)

    # Add 'labels' field
    train_dataset = train_dataset.map(lambda x: {'labels': x['label']}, batched=True)
    val_dataset = val_dataset.map(lambda x: {'labels': x['label']}, batched=True)
    test_dataset = test_dataset.map(lambda x: {'labels': x['label']}, batched=True)

    # Format for PyTorch
    columns = ['input_ids', 'attention_mask', 'labels']
    if 'token_type_ids' in train_dataset.column_names:
        columns.append('token_type_ids')

    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.set_format(type='torch', columns=columns)

    # Dataloaders
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    shift_loader = None
    ood_loader = None

    # create shift and ood loaders if applicable
    if eval_shift and dataset_name == "MNLI":
        print("Creating Shift dataloader from MNLI mismatched")
        shift_data = raw["validation_mismatched"]
        shift_dataset = shift_data.map(tokenize, batched=True)
        shift_dataset = shift_dataset.map(lambda x: {'labels': x['label']}, batched=True)
        shift_dataset.set_format(type='torch', columns=columns)
        shift_loader = DataLoader(shift_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    if eval_ood and dataset_name == "MNLI":
        print("Creating OOD dataloader from SNLI")
        snli = load_dataset("stanfordnlp/snli")
        snli = snli.filter(lambda x: x["label"] != -1)
        snli = snli.map(lambda batch: encode_mnli(batch, tokenizer), batched=True)
        snli = snli.map(lambda x: {'labels': x['label']}, batched=True)
        columns = ['input_ids', 'attention_mask', 'labels']
        if 'token_type_ids' in snli.column_names:
            columns.append('token_type_ids')
        snli.set_format(type='torch', columns=columns)
        ood_loader = DataLoader(snli["test"], batch_size=batch_size, shuffle=False, collate_fn=collator)

    return is_nlp_task, train_loader, val_loader, test_loader, shift_loader, ood_loader, num_classes


def encode_mrpc(examples, tokenizer):
    return tokenizer(examples['sentence1'], examples['sentence2'],
                     truncation=True, padding='max_length', max_length=256)  # padding='max_length')


def encode_mnli(batch, tokenizer):
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding="max_length", max_length=256)


def load_vision_dataset(args, data_path):
    dataset = args.dataset
    eval_ood = args.eval_ood
    eval_shift = args.eval_shift
    normalize_pretrained_dataset = args.normalize_pretrained_dataset
    dm, num_classes = load_data_module(args, data_path)
    if args.model_type == "ViT":

        if normalize_pretrained_dataset:
            model_name = args.ViT_model
            processor = ViTImageProcessor.from_pretrained(model_name)
            image_mean, image_std = processor.image_mean, processor.image_std
            # size = processor.size["height"]
            normalize = v2.Normalize(mean=image_mean, std=image_std)
        else:
            if dataset == "cifar10":
                normalize = v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
            elif dataset == "cifar100":
                normalize = v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

        dm.train_transform = v2.Compose([
            v2.Resize(256),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

        dm.test_transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
        ])

    dm.prepare_data()
    dm.setup("test")
    test_loader = dm.test_dataloader()[0]

    if eval_shift:  # DataModule does not correctly apply test_transform to shift data
        shift_loader = dm.test_dataloader()[2]
        images, labels = next(iter(shift_loader))
        if images.shape[2] == 32:
            dm.shift = TransformWrapper(dm.shift, dm.test_transform)
            shift_loader = dm.test_dataloader()[2]
            images, labels = next(iter(shift_loader))
    else:
        shift_loader = None

    if eval_ood:
        ood_loader = dm.test_dataloader()[1]
        images, labels = next(iter(ood_loader))

    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    return False, dm, num_classes, train_loader, val_loader, test_loader, shift_loader, ood_loader


# this is needed to evaluate the VIT on shift data
# not the best way to do this but DataModules did something unexpected
# (for some reason the shift images were not resized correctly otherwise)
class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        return self.transform(x), y
