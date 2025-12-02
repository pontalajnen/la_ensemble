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
from utils.arguments import train_args
from utils.helpers import torch_device
from pathlib import Path


def model_path(args, save_dir, epoch, val_loss, model_name):
    return os.path.join(
        save_dir,
        (f"model={args.model}-epoch={epoch:02d}-val_loss={val_loss:.4f}-"
         f"optimizer={args.base_optimizer}-rho={args.rho}-adaptive={args.adaptive}-model_name={model_name}.pth")
    )


def save_model(model, path):
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, path)


def init_wandb(args, model_name):
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
        "seed": args.seed,
        "SAM": args.SAM,
        "momentum": args.momentum,
        "epochs": args.epochs,
        "dataset": args.dataset,
        "model": args.model
    })

    artifact = wandb.Artifact("model_checkpoints", type="model")

    return run, artifact


def train(args):
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        print(f"[local rank]: {args.local_rank}")
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch_device()

    print("[device]:", device)

    os.environ['WANDB_INIT_TIMEOUT'] = '800'

    DATA_PATH = LOCAL_STORAGE + DATA_DIR

    dm, num_classes = load_data_module(args, DATA_PATH)

    if args.model == "vit":
        dm = init_transformer(args, dm)

    dm.prepare_data()
    dm.setup("fit")

    if args.distributed:
        train_dataset = dm.train_dataloader().dataset
        val_dataset = dm.val_dataloader().dataset

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

    print(f"[dataset]: {args.dataset}")
    wandb.login()

    seed = args.seed
    model_name = args.model + "_" + args.dataset + "_" + args.base_optimizer
    model_name += "_ensemble" if args.ensemble else ""
    model_name += "_packed" if args.packed else ""
    torch_seed = torch.Generator()
    torch_seed.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    run, artifact = init_wandb(args, model_name)

    save_dir = MODEL_PATH_LOCAL + f"{args.dataset}_{args.model}_{'' if args.SAM else 'no'}_SAM/"
    os.makedirs(save_dir, exist_ok=True)

    model = init_model(args, device, num_classes)

    opt, scheduler = init_optimizer(args, model)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = os.path.join(save_dir, f"model_{args.model}_seed{seed}_best.pth")
    packed = "ResNet_packed" in [type(m).__name__ for _, m in model.named_modules()]
    if packed:
        num_estimators = model.module.num_estimators if model.module else model.num_estimators
    else:
        num_estimators = 1

    print("[training loop]: starting")
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_sampler.set_epoch(epoch) if args.distributed else None
        model.train()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y = y.repeat(num_estimators) if packed else y
            if args.SAM:
                enable_running_stats(model)
                loss = sum([criterion(pred, y) for pred in model(x)]) if args.ensemble else criterion(model(x), y)
                loss.mean().backward()
                opt.first_step(zero_grad=True)

                disable_running_stats(model)
                loss = sum([criterion(pred, y) for pred in model(x)]) if args.ensemble else criterion(model(x), y)
                loss.mean().backward()
                opt.second_step(zero_grad=True)
            else:
                loss = sum([criterion(pred, y) for pred in model(x)]) if args.ensemble else criterion(model(x), y)
                loss.backward()
                opt.step()
                opt.zero_grad()

        # Validation loop
        val_accuracy, val_loss = evaluate_model(model, val_loader, device, criterion)

        if args.distributed:
            val_accuracy = torch.tensor(val_accuracy, device=device)
            val_loss = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_accuracy /= dist.get_world_size()
            val_loss /= dist.get_world_size()

        wandb.log({"epoch": epoch, "val_accuracy": val_accuracy,
                   "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(model, best_checkpoint_path)

        scheduler.step()

    print("[training loop]: finished")

    # Rename the best checkpoint with metadata
    final_checkpoint_path = model_path(args, save_dir, best_epoch, best_val_loss, model_name)
    os.rename(best_checkpoint_path, final_checkpoint_path)
    last_epoch_checkpoint_path = model_path(args, save_dir, args.epochs, val_loss, model_name)

    save_model(model, last_epoch_checkpoint_path)

    artifact.add_file(final_checkpoint_path)
    wandb.log_artifact(artifact)
    run.finish()


def main():
    args = train_args()

    save_dir = MODEL_PATH_LOCAL + f"{args.dataset}_{args.model}_{'' if args.SAM else 'no'}_SAM/"
    if Path(save_dir).exists() and any(Path(save_dir).iterdir()):  # TODO: Make more robust
        print(f"[warning]: model already trained at {save_dir}, skipping...")

    train(args)


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()
