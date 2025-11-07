from argparse import ArgumentParser, BooleanOptionalAction, Action, ArgumentTypeError
import torch
import torch.nn as nn
import os
from utils.data import load_hf_dataset, load_vision_dataset
from utils.sam import SAM
from utils.sam import enable_running_stats, disable_running_stats
from utils.eval import evaluate_model_lang
from models.resnet import torch_resnet18
from utils.paths import LOCAL_STORAGE, DATA_DIR, MODEL_PATH_LOCAL
import wandb
import torch.optim as optim
from tqdm import tqdm
import timm
# from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoModelForSequenceClassification  # , AutoTokenizer,  DataCollatorWithPadding
from transformers import get_scheduler
# from datasets import load_dataset
import numpy as np


def train(args):
    # Set device
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print("Device:", device)
    print("-------------------SAM IS ", args.SAM, "-----------------------")

    # AVOID WANDB TIMEOUT
    os.environ['WANDB_INIT_TIMEOUT'] = '800'

    # Set path to datasets
    DATA_PATH = LOCAL_STORAGE + DATA_DIR
    # --------------------------------------------------------------------

    # Load Dataset
    # --------------------------------------------------------------------

    print("Loading dataset: ", args.dataset)
    if args.dataset in ("CIFAR10", "CIFAR100", "MNIST", "ImageNet"):
        nlp, dm, num_classes, train_loader, val_loader, _, _, _ = load_vision_dataset(
            dataset=args.dataset,
            model_type=args.model,
            ViT_model=args.ViT_model,
            DATA_PATH=DATA_PATH,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            test_alt=None,
            eval_ood=False,
            eval_shift=False,
            shift_severity=1,
            basic_augment=args.basic_augment,
            ood_ds="openimage-o",
            normalize_pretrained_dataset=args.normalize_pretrained_dataset
        )
    elif args.dataset in ("MNLI", "RTE", "MRPC"):
        nlp, train_loader, val_loader, _, _, _, num_classes = load_hf_dataset(
            NLP_model=args.NLP_model,
            dataset_name=args.dataset,
            eval_ood=False,
            eval_shift=False,
            batch_size=args.batch_size
        )
    else:
        raise Exception("Oops, requested dataset does not exist!")
    print("Loading done!")
    print("Successfully loaded the dataset!")

    # --------------------------------------------------------------------
    # WandB setup
    # --------------------------------------------------------------------

    wandb.login()

    for i in range(args.seeds_per_job):
        # set seed for reproducibility
        seed = args.seed + i
        torch_seed = torch.Generator()
        torch_seed.manual_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        project = f"LA_SAM_{args.dataset}_{args.model}_SAM{args.SAM}_adaptive{args.adaptive}"
        model_name = args.model+"_"+args.dataset+"_seed"+str(seed)+"_"+args.base_optimizer
        # Initialize W&B run and log hyperparameters
        run = wandb.init(project=project, name=model_name, config=args)
        # wandb.config["seed"] = seed
        wandb.config.update({"seed": seed}, allow_val_change=True)

        model_type = wandb.config["model"]
        dataset = wandb.config["dataset"]
        base_optimizer = wandb.config["base_optimizer"]
        use_SAM = wandb.config["SAM"]
        adaptive = wandb.config["adaptive"]
        print("SAM, adaptive", type(use_SAM), use_SAM, type(adaptive), adaptive)
        if isinstance(use_SAM, str):
            use_SAM = str2bool(use_SAM)
        if isinstance(adaptive, str):
            adaptive = str2bool(adaptive)
        print("SAM, adaptive", type(use_SAM), use_SAM, type(adaptive), adaptive)
        rho = wandb.config["rho"]
        normalize_pretrained_dataset = wandb.config["normalize_pretrained_dataset"]
        if isinstance(normalize_pretrained_dataset, str):
            normalize_pretrained_dataset = str2bool(normalize_pretrained_dataset)

        ViT_model = wandb.config["ViT_model"]
        NLP_model = wandb.config["NLP_model"]
        learning_rate = wandb.config["learning_rate"]
        weight_decay = wandb.config["weight_decay"]
        momentum = wandb.config["momentum"]
        lr_scheduler = wandb.config["lr_scheduler"]
        epochs = wandb.config["epochs"]
        num_warmup_steps = wandb.config["num_warmup_steps"]
        batch_size = wandb.config["batch_size"]
        seeds_per_job = wandb.config["seeds_per_job"]
        store_last_ckpt = wandb.config["store_last_ckpt"]
        if isinstance(store_last_ckpt, str):
            store_last_ckpt = str2bool(store_last_ckpt)

        # Prepare for saving model checkpoints locally and log them to W&B
        save_dir = MODEL_PATH_LOCAL + f"{dataset}_{model_type}_{'' if use_SAM == True else 'no'}_SAM/"
        os.makedirs(save_dir, exist_ok=True)

        artifact = wandb.Artifact("model_checkpoints", type="model")
        print("Successfully initialized W&B run!")

        # --------------------------------------------------------------------
        # Load Model and Optimizer
        # --------------------------------------------------------------------

        # Model
        if model_type == "ResNet18":
            model = torch_resnet18(num_classes=num_classes)
            model = model.to(device)
        elif model_type == "ViT":
            model = timm.create_model(ViT_model, pretrained=True, num_classes=num_classes)
            model = model.to(device)
            print("Model ready and on device!")
        elif model_type == "BERT" or model_type == "ROBERTA":
            model = AutoModelForSequenceClassification.from_pretrained(NLP_model, num_labels=num_classes)
            model.to(device)
        else:
            raise Exception("Requested model doesn't exist! Has to be one of 'ResNet18', 'ViT', 'BERT', 'ROBERTA'")

        # Optimizer
        if base_optimizer == "SGD":
            base_opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        elif base_optimizer == "AdamW":
            base_opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif base_optimizer == "Adam":
            base_opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception("Oops, requested optimizer does not exist! Optimizer has to be one of 'SGD'")
        print("Optimizer ready!")

        # Determine the base optimizer and SAM optimizer setup
        if use_SAM:
            print("Using SAM optimizer!")
            if adaptive:
                print("--------------------------------------------------------------------------------")
                print("--------------------------------------------------------------------------------")
                print("----------------------Using Adaptive SAM optimizer!-----------------------------")
                print("--------------------------------------------------------------------------------")
                print("--------------------------------------------------------------------------------")

            # Set up arguments for both SAM and the base optimizer
            optimizer_args = {
                'params': model.parameters(),
                'base_optimizer': type(base_opt),
                'rho': rho,
                'adaptive': adaptive,
                'lr': learning_rate,
                'weight_decay': weight_decay,
            }

            if isinstance(base_opt, optim.SGD):
                optimizer_args['momentum'] = momentum

            # Create the SAM optimizer
            opt = SAM(**optimizer_args)

            # Create the learning rate scheduler for SAM
            if lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, T_max=epochs)
            elif lr_scheduler == "linear":
                num_training_steps = len(train_loader) * epochs
                num_warmup_steps = int(num_warmup_steps * num_training_steps)
                scheduler = get_scheduler(
                    name="linear",
                    optimizer=opt.base_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
        else:
            # Base optimizer without SAM
            opt = base_opt

            if lr_scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            elif lr_scheduler == "linear":
                num_training_steps = len(train_loader) * epochs
                num_warmup_steps = int(num_warmup_steps * num_training_steps)
                scheduler = get_scheduler(
                    name="linear",
                    optimizer=opt,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )

        # --------------------------------------------------------------------
        # Training
        # --------------------------------------------------------------------

        # Loss function
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float("inf")
        best_epoch = 0
        best_checkpoint_path = os.path.join(
            save_dir,
            f"model_{model_type}_seed{seed}_adaptive{adaptive}_SAM{use_SAM}_best.pth"
        )
        print("Initialized model, optimizer and loss function!")
        print("----- Start training loop -----")
        print("Printing all values: ", model_type, NLP_model, base_optimizer, learning_rate, weight_decay,
              epochs, dataset, batch_size, seed, seeds_per_job, use_SAM, adaptive, lr_scheduler, num_warmup_steps, rho)
        for epoch in tqdm(range(epochs), desc="Epochs"):
            # Train as usual
            model.train()

            for batch_idx, batch in enumerate(train_loader):
                # Extract inputs and targets
                if nlp:
                    x = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                    y = batch['labels'].to(device)
                else:
                    x, y = batch
                    x, y = x.to(device), y.to(device)

                if use_SAM:
                    if batch_idx == 1:
                        print("---------------- Using SAM ------------------")
                    # ---- SAM Step 1 ----
                    enable_running_stats(model)
                    y_pred, loss = forward_and_loss(model, x, y, criterion, nlp)
                    loss.mean().backward()
                    opt.first_step(zero_grad=True)

                    # ---- SAM Step 2 ----
                    disable_running_stats(model)
                    y_pred, loss = forward_and_loss(model, x, y, criterion, nlp)
                    loss.mean().backward()
                    opt.second_step(zero_grad=True)
                else:
                    y_pred, loss = forward_and_loss(model, x, y, criterion, nlp)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                if lr_scheduler == "linear":
                    scheduler.step()

            # Validation loop
            val_accuracy, val_loss = evaluate_model_lang(model, val_loader, device, criterion, nlp)
            wandb.log({"epoch": epoch, "val_accuracy": val_accuracy,
                       "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

            # Save and track the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), best_checkpoint_path)  # Overwrites temp file

            if lr_scheduler == "cosine":
                scheduler.step()
            print("Finished epoch", epoch)

        print("Finished training loop!")

        # --------------------------------------------------------------------
        # Store Checkpoints
        # --------------------------------------------------------------------

        # **Rename the best checkpoint with metadata**
        final_checkpoint_path = os.path.join(
            save_dir,
            (f"seed={seed}-epoch={best_epoch:02d}-val_loss={best_val_loss:.4f}-model={model_type}-"
             f"optimizer={base_optimizer}-rho={rho}-adaptive={adaptive}-model_name={model_name}.pth")
        )
        os.rename(best_checkpoint_path, final_checkpoint_path)  # Rename the best model file

        last_epoch_checkpoint_path = os.path.join(
            save_dir,
            (f"seed={seed}-epoch={epochs}-val_loss={val_loss:.4f}-model={model_type}-"
             f"optimizer={base_optimizer}-rho={rho}-adaptive={adaptive}-model_name={model_name}.pth")
        )
        # Store model after last epoch
        if store_last_ckpt:
            torch.save(model.state_dict(), last_epoch_checkpoint_path)

        artifact.add_file(final_checkpoint_path)
        wandb.log_artifact(artifact)
        run.finish()


def forward_and_loss_short(model, x, y, criterion, nlp):
    if nlp:
        output = model(**x)
        return output.logits, output.loss
    else:
        y_pred = model(x)
        return y_pred, criterion(y_pred, y)


def forward_and_loss(model, x, y, criterion, nlp):
    if nlp:  # NLP
        output = model(**x)
        y_pred = output.logits
        loss = output.loss
        if loss is None:
            loss = criterion(y_pred, y)
    else:  # Vision
        y_pred = model(x)
        loss = criterion(y_pred, y)
    return y_pred, loss


def encode_mrpc(examples, tokenizer):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')


def encode_mnli(batch, tokenizer):
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding="max_length")


class BooleanAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower() in ('yes', 'true', 't', '1'):
            setattr(namespace, self.dest, True)
        elif values.lower() in ('no', 'false', 'f', '0'):
            setattr(namespace, self.dest, False)
        else:
            raise ArgumentTypeError(f"Unsupported boolean value: {values}")


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', '0'):
        return False
    raise ValueError(f"Unsupported boolean value: {val}")


def main():

    parser = ArgumentParser()
    # SEED
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds_per_job", type=int, default=1)

    # Model and Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True,
                        help="True if you want to use basic augmentations (horizontal flip, random crop with padding).")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Split the training set into train and validation set.")
    parser.add_argument("--model", type=str, default="ResNet18",
                        help="Supported models are ResNet18, ViT, BERT, ROBERTA, DistilGPT2")
    parser.add_argument("--model_name", type=str, default="Unknown")
    parser.add_argument("--ViT_model", type=str, default='vit_base_patch16_224.orig_in21k',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--NLP_model", type=str, default='bert-base-uncased',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--normalize_pretrained_dataset", action=BooleanOptionalAction, default=False,
                        help="Finetune the dataset using the normalization values of the pretrained dataset (VIT)")
    parser.add_argument("--store_last_ckpt", action=BooleanOptionalAction, default=False,
                        help="Store last model checkpoint on top of the corresponding with lowest validation accuracy.")

    # Training
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size used in the training and validation loop.")
    parser.add_argument("--epochs", default=200, type=int,
                        help="Total number of epochs.")
    parser.add_argument("--SAM", dest='SAM', action=BooleanAction, type=str,
                        choices=['yes', 'no', 'true', 'false', 't', 'f', '1', '0'],
                        help="True if you want to use the SAM optimizer.")
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="Base learning rate at the start of the training.")
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        help="Learning rate scheduler. Has to be one of 'cosine' or 'linear'.")
    parser.add_argument("--num_warmup_steps", type=float, default=0.1,
                        help="Ratio of warmup steps for linear optimizer")
    parser.add_argument("--base_optimizer", type=str, default="SGD",
                        help="Base optimizer.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="SGD Momentum.")
    parser.add_argument("--weight_decay", default=0.0005, type=float,
                        help="L2 weight decay.")

    # SAM hyperparameters
    parser.add_argument('--adaptive', dest='adaptive', action=BooleanAction, type=str,
                        choices=['yes', 'no', 'true', 'false', 't', 'f', '1', '0'],
                        help='True if you want to use Adaptive SAM')
    parser.add_argument("--rho", default=0.05, type=float,
                        help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=float,
                        help="Rho parameter for SAM.")
    parser.add_argument("--eta", default=0.1, type=float,
                        help="Eta parameter for ASAM.")

    # parser = common_arguments(parser)

    # args = parser.parse_args()

    # train(args)


if __name__ == "__main__":
    main()
