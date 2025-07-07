from argparse import ArgumentParser, BooleanOptionalAction
import torch
from torchvision.transforms import v2
import os
import wandb
import timm
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils.data import load_data_module
from utils.sam import SAM, enable_running_stats, disable_running_stats
from utils.eval import evaluate_model
from models.resnet import torch_resnet18
import torch.optim as optim
from utils.paths import LOCAL_STORAGE, DATA_DIR, MODEL_PATH_LOCAL
from transformers import ViTImageProcessor  # , ViTForImageClassification
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from helpers import common_arguments
from models.ensemble_model import EnsembleModel


def init_transformer(args, data_module):
    # Resize the images so that it is compatible with a ViT pretrained on ImageNet-21k
    model_name = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_name)
    image_mean, image_std = processor.image_mean, processor.image_std
    # size = processor.size["height"]
    if args.normalize_pretrained_dataset:
        normalize = v2.Normalize(mean=image_mean, std=image_std)
    else:
        if args.dataset == "CIFAR10":
            normalize = v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
        elif args.dataset == "CIFAR100":
            normalize = v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    # resize_transform = v2.Resize((224, 224))

    data_module.train_transform = v2.Compose([
        v2.Resize(256),
        v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    data_module.test_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    return data_module


def init_model(args, device, num_classes):
    if args.model == "ResNet18":
        if args.ensemble:
            models = EnsembleModel(
                num_models=args.num_ensemble_models,
                num_classes=num_classes,
                model=torch_resnet18
            )
        else:
            models = torch_resnet18(num_classes=num_classes)

        print("----- Model loaded -----")
        models = models.to(device)

    elif args.model == "ViT":
        models = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True, num_classes=num_classes)
        models = models.to(device)
        print("----- Model ready and on device -----")
    else:
        raise Exception("Requested model does not exist! Has to be one of 'ResNet18', 'ViT'")
    if args.distributed:
        models = DDP(models, device_ids=[args.local_rank])

    return models


def init_optimizer(args, model):
    if args.base_optimizer == "SGD":
        base_optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.base_optimizer == "AdamW":
        base_optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise Exception("Requested optimizer does not exist! Optimizer has to be one of 'SGD', 'AdamW'")

    # Determine the base optimizer and SAM optimizer setup
    if args.SAM:
        print("----- Using SAM optimizer -----")
        if args.adaptive:
            print("----- Using Adaptive SAM optimizer -----")
        # Set up arguments for both SAM and the base optimizer
        optimizer_args = {
            'params': model.parameters(),
            'base_optimizer': type(base_optimizer),
            'rho': args.rho,
            'adaptive': args.adaptive,
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
        }
        if isinstance(base_optimizer, optim.SGD):
            optimizer_args['momentum'] = args.momentum

        # Create the SAM optimizer
        opt = SAM(**optimizer_args)

        # Create the learning rate scheduler for SAM
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, args.epochs)
    else:
        # Use the base optimizer without SAM
        opt = base_optimizer
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    return opt, scheduler


def train(args):
    # Set device
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        print(args.local_rank, "local rank")
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else  # TODO: Not sure if ":0" is needed
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

    print("Device:", device)

    # AVOID WANDB TIMEOUT
    os.environ['WANDB_INIT_TIMEOUT'] = '800'

    # Set path to datasets
    DATA_PATH = LOCAL_STORAGE + DATA_DIR
    # os.makedirs(DATA_PATH, exist_ok=True)

    # Load the dataset
    dm, num_classes = load_data_module(
        args.dataset,
        DATA_PATH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        basic_augment=args.basic_augment
    )

    if args.model == "ViT":
        dm = init_transformer(args, dm)

    dm.prepare_data()
    dm.setup("fit")

    if args.distributed:
        temp_train_loader = dm.train_dataloader()
        temp_val_loader = dm.val_dataloader()

        train_dataset = temp_train_loader.dataset
        val_dataset = temp_val_loader.dataset

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=False
        )
    else:
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

    images, labels = next(iter(train_loader))

    print("Loaded the dataset!")
    wandb.login()

    seed = args.seed
    model_name = args.model + "_" + args.dataset + "_" + args.base_optimizer
    torch_seed = torch.Generator()
    torch_seed.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    project = f"LA_SAM_{args.dataset}_{args.model}_SAM{args.SAM}_adaptive{args.adaptive}"
    # Initialize W&B run and log hyperparameters
    run = wandb.init(project=project, name=model_name, config={
        "base_optimizer": args.base_optimizer,
        "rho": args.rho,
        "adaptive": args.adaptive,
        "lr": args.learning_rate,
        "lr_scheduler": args.lr_scheduler,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "seed": seed,
        "SAM": args.SAM,
        "momentum": args.momentum,
        "epochs": args.epochs,
        "dataset": args.dataset,
        "model": args.model
    })

    # Prepare for saving model checkpoints locally and log them to W&B
    save_dir = MODEL_PATH_LOCAL + f"{args.dataset}_{args.model}_{'' if args.SAM else 'no'}_SAM/"
    os.makedirs(save_dir, exist_ok=True)

    artifact = wandb.Artifact("model_checkpoints", type="model")

    model = init_model(args, device, num_classes)

    # Optimizer
    opt, scheduler = init_optimizer(args, model)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = os.path.join(save_dir, f"model_{args.model}_seed{seed}_best.pth")
    print("----- Start training loop -----")
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_sampler.set_epoch(epoch) if args.distributed else None
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if args.ensemble:
                if args.SAM:
                    enable_running_stats(model)
                    loss = sum([criterion(y_pred, y) for y_pred in model(x)])
                    loss.mean().backward()
                    opt.first_step(zero_grad=True)

                    disable_running_stats(model)
                    loss = sum([criterion(y_pred, y) for y_pred in model(x)])
                    loss.mean().backward()
                    opt.second_step(zero_grad=True)
                else:
                    loss = sum([criterion(y_pred, y) for y_pred in model(x)])
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
            else:
                if args.SAM:
                    enable_running_stats(model)
                    loss = criterion(model(x), y).mean()
                    loss.backward()
                    opt.first_step(zero_grad=True)

                    disable_running_stats(model)
                    criterion(model(x), y).mean().backward()
                    opt.second_step(zero_grad=True)
                else:
                    loss = criterion(model(x), y)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

        # Validation loop
        val_accuracy, val_loss = evaluate_model(model, val_loader, device, criterion)

        if args.distributed:
            # Gather validation results from all processes
            val_accuracy = torch.tensor(val_accuracy, device=device)
            val_loss = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_accuracy /= dist.get_world_size()
            val_loss /= dist.get_world_size()
        # Log validation accuracy
        wandb.log({"epoch": epoch, "val_accuracy": val_accuracy,
                   "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

        # Save and track the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint_path)  # Overwrites temp file

        scheduler.step()

    print("Finished training loop!")
    # Rename the best checkpoint with metadata
    final_checkpoint_path = os.path.join(
        save_dir,
        (f"seed={seed}-epoch={best_epoch:02d}-val_loss={best_val_loss:.4f}-model={args.model}-"
         f"optimizer={args.base_optimizer}-rho={args.rho}-adaptive={args.adaptive}-model_name={model_name}.pth")
    )
    os.rename(best_checkpoint_path, final_checkpoint_path)  # Rename the best model file

    last_epoch_checkpoint_path = os.path.join(
        save_dir,
        (f"seed={seed}-epoch={args.epochs}-val_loss={val_loss:.4f}-model={args.model}-"
         f"optimizer={args.base_optimizer}-rho={args.rho}-adaptive={args.adaptive}-model_name={model_name}.pth")
    )
    # Store model after last epoch
    torch.save(model.state_dict(), last_epoch_checkpoint_path)

    artifact.add_file(final_checkpoint_path)
    wandb.log_artifact(artifact)
    run.finish()


def main():
    parser = ArgumentParser()
    # SEED
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds_per_job", type=int, default=1)

    # Model and Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True,
                        help="Basic augmentations (horizontal flip, random crop with padding).")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Split the training set into train and validation set.")
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--depth", default=18, type=int,
                        help="Number of layers.")
    parser.add_argument("--width_factor", default=8, type=int,
                        help="How many times wider compared to normal ResNet.")
    parser.add_argument("--model_name", type=str, default="Unknown")
    parser.add_argument("--ViT_model", type=str, default='google/vit-base-patch16-224-in21k',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--normalize_pretrained_dataset", action=BooleanOptionalAction, default=False,
                        help="Finetune the dataset using the normalization values of the pretrained dataset (VIT)")

    # Training
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Total number of epochs.")
    parser.add_argument("--dropout", default=0.0, type=float,
                        help="Dropout rate.")
    parser.add_argument("--SAM", default=False, action=BooleanOptionalAction,
                        help="Enable SAM optimizer.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        help="Learning rate scheduler.")
    parser.add_argument("--base_optimizer", type=str, default="SGD",
                        help="Base optimizer.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        help="L2 weight decay.")
    parser.add_argument("--threads", default=8, type=int,
                        help="Number of CPU threads for dataloaders.")

    # SAM hyperparameters
    parser.add_argument("--adaptive", default=False, action=BooleanOptionalAction,
                        help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--rho", default=0.05, type=float,
                        help="Rho parameter for SAM.")
    # parser.add_argument("--rho_min", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=float,
                        help="Rho parameter for SAM.")
    parser.add_argument("--eta", default=0.1, type=float,
                        help="Eta parameter for ASAM.")

    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Use 0.0 for no label smoothing.")
    parser.add_argument("--distributed", action="store_true", help="Use distributed data parallel")
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int, help="Local rank for distributed training")

    # Ensemble arguments
    parser.add_argument("--ensemble", action=BooleanOptionalAction, default=False)
    parser.add_argument("--num_ensemble_models", type=int, default=5)

    parser = common_arguments(parser)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
