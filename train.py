from xml.parsers.expat import model

import math
import torch
import os
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
from utils.sgld import SGLD


def model_path(args, save_dir, epoch, val_loss, model_name):
    return os.path.join(
        save_dir,
        (f"mn={model_name}-epoch={epoch:02d}-val_loss={val_loss:.3f}-"
         f"rho={args.rho}-adaptive={args.adaptive}.pth")
    )


def packed_logic(model):
    packed = "ResNet_packed" in [type(m).__name__ for _, m in model.named_modules()]
    if packed:
        num_estimators = model.module.num_estimators if hasattr(model, "module") else model.num_estimators
    else:
        num_estimators = 1
    return packed, num_estimators


def save_model(model, path, args):
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, path)


def create_model_name(args):
    model_name = args.model + "_" + args.dataset + "_" + args.base_optimizer
    if args.base_optimizer == "SGLD" and args.SAM:
        model_name += "_SAM"
    model_name += "_ensemble" if args.ensemble else ""
    model_name += "_packed" if args.packed else ""
    return model_name


def init_dataloaders(dm):
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    train_sampler = None

    return train_loader, val_loader, train_sampler


def train(args):
    device = torch_device()
    print("[device]:", device)

    DATA_PATH = LOCAL_STORAGE + DATA_DIR

    dm, num_classes = load_data_module(args, DATA_PATH)

    if args.model == "vit":
        dm = init_transformer(args, dm)

    dm.prepare_data()
    dm.setup("fit")

    train_loader, val_loader, train_sampler = init_dataloaders(dm)

    print(f"[dataset]: {args.dataset}")

    model_name = create_model_name(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = MODEL_PATH_LOCAL + f"{args.dataset}_{args.model}_{'' if args.SAM else 'no'}_SAM/"
    os.makedirs(save_dir, exist_ok=True)

    model = init_model(args, device, num_classes)
    optimizer, scheduler = init_optimizer(args, model)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = os.path.join(save_dir, f"model_{model_name}_seed{args.seed}_best.pth")
    packed, num_estimators = packed_logic(model)

    using_sgld = isinstance(optimizer, SGLD)
    burn_in = args.burn_in_epochs if args.burn_in_epochs is not None else args.epochs // 2
    num_train_samples = len(train_loader.dataset) if using_sgld else None
    sgld_samples = []
    sgld_sample_paths_file = os.path.join(save_dir, f"sgld_samples_{model_name}_seed{args.seed}.txt")
    sgld_val_loss_file = os.path.join(save_dir, f"best_val_loss_{model_name}_seed{args.seed}.txt")
    # Cool-down window: cosine from end-of-burn-in LR to sgld_sampling_lr.
    cooldown_epochs = min(5, max(0, args.epochs - burn_in - 1)) if using_sgld and args.sgld_sampling_lr is not None else 0
    burn_in_end_lr = None

    print("[training loop]: starting")
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()

        in_sampling_phase = using_sgld and epoch >= burn_in

        if using_sgld:
            # SGLD's step() always injects noise, so burn-in (pure SGD) is
            # emulated by zeroing the temperature until sampling starts.
            for g in optimizer.param_groups:
                g['temperature'] = args.sgld_temperature if in_sampling_phase else 0.0

        if using_sgld and epoch == burn_in and args.sgld_sampling_lr is not None:
            burn_in_end_lr = optimizer.param_groups[0]['lr']

        if in_sampling_phase and args.sgld_sampling_lr is not None:
            # Cosine cool-down from burn_in_end_lr → sgld_sampling_lr over cooldown_epochs.
            offset = epoch - burn_in
            if offset < cooldown_epochs and burn_in_end_lr is not None:
                t = (offset + 1) / max(1, cooldown_epochs)
                cos = 0.5 * (1 + math.cos(math.pi * t))
                new_lr = args.sgld_sampling_lr + (burn_in_end_lr - args.sgld_sampling_lr) * cos
            else:
                new_lr = args.sgld_sampling_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y = y.repeat(num_estimators) if packed else y
            if using_sgld and args.SAM:
                # SAM perturbation + SGLD noisy update
                enable_running_stats(model)
                loss = criterion(model(x), y)
                loss.backward()
                grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                perturbations = {}
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        eps = p.grad * (args.rho / (grad_norm + 1e-12))
                        perturbations[name] = eps
                        p.data.add_(eps)
                optimizer.zero_grad()
                disable_running_stats(model)
                loss = criterion(model(x), y)
                loss.backward()
                for name, p in model.named_parameters():
                    if name in perturbations:
                        p.data.sub_(perturbations[name])
                optimizer.step(num_samples=num_train_samples)
                optimizer.zero_grad()
            elif args.SAM:
                enable_running_stats(model)
                loss = sum([criterion(pred, y) for pred in model(x)]) if args.ensemble else criterion(model(x), y)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)

                disable_running_stats(model)
                loss = sum([criterion(pred, y) for pred in model(x)]) if args.ensemble else criterion(model(x), y)
                loss.mean().backward()
                optimizer.second_step(zero_grad=True)
            elif using_sgld:
                loss = sum([criterion(pred, y) for pred in model(x)]) if args.ensemble else criterion(model(x), y)
                loss.mean().backward() if args.ensemble else loss.backward()
                optimizer.step(num_samples=num_train_samples)
                optimizer.zero_grad()
            else:
                loss = sum([criterion(pred, y) for pred in model(x)]) if args.ensemble else criterion(model(x), y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # Collect SGLD posterior sample
        if (in_sampling_phase
                and (epoch - burn_in) % args.sgld_sample_interval == 0
                and len(sgld_samples) < args.num_sgld_samples):
            sample_path = os.path.join(
                save_dir, f"sgld_sample_{len(sgld_samples):03d}_epoch{epoch:03d}_seed{args.seed}.pth"
            )
            save_model(model, sample_path, args)
            sgld_samples.append(sample_path)
            print(f"[SGLD]: saved sample {len(sgld_samples)} → {sample_path}")
            with open(sgld_sample_paths_file, "w") as f:
                f.write("\n".join(sgld_samples))

        # Validation loop
        val_accuracy, val_loss = evaluate_model(model, val_loader, device, criterion)

        if args.distributed:
            val_accuracy = torch.tensor(val_accuracy, device=device)
            val_loss = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_accuracy, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
            val_accuracy /= dist.get_world_size()
            val_loss /= dist.get_world_size()

        log_dict = {
            "epoch": epoch, "val_accuracy": val_accuracy,
            "val_loss": val_loss, "lr": optimizer.param_groups[0]['lr'],
            "sgld_samples_collected": len(sgld_samples),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(model, best_checkpoint_path, args)
            if using_sgld:
                with open(sgld_val_loss_file, "w") as f:
                    f.write(f"{float(best_val_loss):.6f}\n{best_epoch}\n")

        if not in_sampling_phase:
            scheduler.step()

    print("[training loop]: finished")

    if using_sgld:
        print(f"[SGLD]: collected {len(sgld_samples)} posterior samples → {sgld_sample_paths_file}")
    else:
        final_checkpoint_path = model_path(args, save_dir, best_epoch, best_val_loss, model_name)
        os.rename(best_checkpoint_path, final_checkpoint_path)
        last_epoch_checkpoint_path = model_path(args, save_dir, args.epochs, val_loss, model_name)
        save_model(model, last_epoch_checkpoint_path, args)
        print(f"[saving model]: {last_epoch_checkpoint_path}")


if __name__ == "__main__":
    args = train_args()
    train(args)
    if dist.is_initialized():
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()
