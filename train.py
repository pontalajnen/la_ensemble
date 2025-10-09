from argparse import ArgumentParser, BooleanOptionalAction
import torch
import os
import wandb
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils.data import load_data_module
from utils.sam import enable_running_stats, disable_running_stats
from utils.eval import evaluate_model
from utils.paths import LOCAL_STORAGE, DATA_DIR, MODEL_PATH_LOCAL
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils.train_helpers import init_model, init_transformer, init_optimizer
from utils.arguments import arguments


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
            'cuda:0' if torch.cuda.is_available() else
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
    packed = "ResNet_packed" in [type(m).__name__ for _, m in model.named_modules()]
    num_estimators = model.module.num_estimators if model.module else model.num_estimators
    print("----- Start training loop -----")
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_sampler.set_epoch(epoch) if args.distributed else None
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if args.ensemble:
                if args.SAM:
                    enable_running_stats(model)
                    loss = sum([criterion(
                        y_pred,
                        y if not packed else y.repeat(num_estimators)
                    ) for y_pred in model(x)])
                    loss.mean().backward()
                    opt.first_step(zero_grad=True)

                    disable_running_stats(model)
                    loss = sum([criterion(
                        y_pred,
                        y if not packed else y.repeat(num_estimators)
                    ) for y_pred in model(x)])
                    loss.mean().backward()
                    opt.second_step(zero_grad=True)
                else:
                    loss = sum([criterion(
                        y_pred,
                        y if not packed else y.repeat(num_estimators)
                    ) for y_pred in model(x)])
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
            else:
                if args.SAM:
                    enable_running_stats(model)
                    loss = criterion(
                        model(x),
                        y if not packed else y.repeat(num_estimators)
                    ).mean()
                    loss.backward()
                    opt.first_step(zero_grad=True)

                    disable_running_stats(model)
                    criterion(
                        model(x),
                        y if not packed else y.repeat(num_estimators)
                    ).mean().backward()
                    opt.second_step(zero_grad=True)
                else:
                    loss = criterion(
                        model(x),
                        y if not packed else y.repeat(num_estimators)
                    )
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
    args = train_args()
    train(args)


if __name__ == "__main__":
    main()
