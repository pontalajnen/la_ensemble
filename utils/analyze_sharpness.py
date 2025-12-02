import os
import torch
import json
# import numpy as np
from argparse import ArgumentParser, BooleanOptionalAction
from utils.eval import load_model
from utils.data import load_hf_dataset, load_vision_dataset
from utils.paths import ROOT, LOCAL_STORAGE, DATA_DIR, RESULT_DIR
from torch.backends.cuda import sdp_kernel
from hessian_eigenthings import compute_hessian_eigenthings


def main():
    parser = ArgumentParser()
    parser.add_argument("--save_file_name", type=str, default="",
                        help="The name of the file to save the results to.")
    parser.add_argument("--model_path_file", type=str, default="",
                        help="A file with the path(s) to the model instance(s) to evaluate.")
    parser.add_argument("--model_type", type=str, default="",
                        help="The type of model to evaluate (e.g. ResNet18, ResNet50, etc.)")
    parser.add_argument("--NLP_model", type=str, default='bert-base-uncased',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--ViT_model", type=str, default='vit_base_patch16_224.orig_in21k',
                        help="Path to checkpoint for fine-tuning")

    parser.add_argument("--dataset", type=str, default="",
                        help="The dataset to evaluate on (e.g. CIFAR10, ImageNet etc.).")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True,
                        help="True if you want to use basic augmentations (horizontal flip, random crop with padding).")
    parser.add_argument("--val_split", type=int, default=0.0,
                        help="Split the training set into train and validation set.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for the dataloader.")
    parser.add_argument("--eval_ood", action=BooleanOptionalAction, default=True,
                        help="Whether to evaluate on OOD data.")
    parser.add_argument("--eval_shift", action=BooleanOptionalAction, default=True,
                        help="Whether to evaluate on shifted data.")
    parser.add_argument("--eval_train", action=BooleanOptionalAction, default=True,
                        help="Whether to evaluate on training data (gives nll).")
    parser.add_argument("--shift_severity", type=int, default=1,
                        help="The severity of the shift to evaluate on (1-5).")
    parser.add_argument("--ood_ds", type=str, default="openimage-o",
                        help="The OOD dataset to use (e.g. openimage-o, fashion).")
    parser.add_argument("--test_alt", type=str, default=None,
                        help="The alternative test set to use (e.g. CIFAR10, CIFAR100).")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Batch size for evaluation.")
    parser.add_argument("--normalize_pretrained_dataset", action=BooleanOptionalAction, default=False,
                        help="Finetune the dataset using the normalization values of the pretrained dataset (VIT)")

    parser.add_argument("--num_eigenthings", default=10, type=int,
                        help="number of eigenvalues/vectors to compute")
    parser.add_argument("--num_steps", default=50, type=int,
                        help="number of power iter steps")
    parser.add_argument("--max_samples", default=2048, type=int)
    parser.add_argument("--cuda", action="store_true",
                        help="if true, use CUDA/GPUs")
    parser.add_argument("--full_dataset", action="store_true",
                        help="if true, loop over all batches in set for each gradient step")
    parser.add_argument("--fname", default="", type=str)
    parser.add_argument("--mode", type=str, choices=["power_iter", "lanczos"])
    parser.add_argument("--tol", type=float, default=1e-4)

    args = parser.parse_args()

    print("Starting Sharpness Evaluation!")

    if args.save_file_name == "":
        raise Exception("Oops you did not provide a save_file_name!")
    if args.model_path_file == "":
        raise Exception("Oops you did not provide a model_name_file!")
    # root_dir = ROOT + "/"
    # batch_size = 512  # choose batch size for evaluation (should not change the evaluation results)

    model_paths = open(ROOT+"/eval_path_files/"+args.model_path_file, "r")

    # Set device
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print("Using device: ", device)

    # Set Path to Datasets
    DATA_PATH = LOCAL_STORAGE + DATA_DIR
    RESULT_PATH = ROOT + RESULT_DIR + "sharpness_values/"

    os.makedirs(RESULT_PATH, exist_ok=True)
    print(RESULT_PATH+args.save_file_name)

    # Get Dataset
    print("Loading dataset: ", args.dataset)
    if args.dataset in ("CIFAR10", "CIFAR100", "MNIST", "ImageNet"):
        nlp, dm, num_classes, train_loader, val_loader, test_loader, shift_loader, ood_loader = load_vision_dataset(
            dataset=args.dataset,
            model_type=args.model_type,
            ViT_model=args.ViT_model,
            data_path=DATA_PATH,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            test_alt=args.test_alt,
            eval_ood=args.eval_ood,
            eval_shift=args.eval_shift,
            shift_severity=args.shift_severity,
            basic_augment=args.basic_augment,
            ood_ds=args.ood_ds,
            normalize_pretrained_dataset=args.normalize_pretrained_dataset
        )
    elif args.dataset in ("MNLI", "RTE", "MRPC"):
        nlp, train_loader, val_loader, test_loader, shift_loader, ood_loader, num_classes = load_hf_dataset(
            NLP_model=args.NLP_model,
            dataset_name=args.dataset,
            eval_ood=args.eval_ood,
            eval_shift=args.eval_shift,
            batch_size=args.batch_size
        )
    else:
        raise Exception("Oops, requested dataset does not exist!")
    print("Loading done!")

    # more than one model can be in the model path file
    # in that case, mean and standard error of the mean is calculated
    # for each metric
    # counter is needed to keep track of the number of models
    num_models = 0

    # prepare reliability diagram plot
    results = {}

    for model_path in model_paths.read().splitlines():
        model_path = model_path.strip()
        model_name = model_path.split("model_name=")[1].replace(".ckpt", "")

        print("-"*30)
        print("Loading model: ", model_name)
        feature_reduction, model = load_model(
            name=args.model_type,
            vit=args.ViT_model,
            nlp=args.NLP_model,
            path=model_path,
            device=device,
            num_classes=num_classes
        )

        print("-"*30)
        print("Evaluate model!")
        model.eval()
        model = model.to(device)

        criterion = torch.nn.functional.cross_entropy

        print("Using full dataset?", args.full_dataset)

        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            if args.mode == "power_iter":
                eigenvalues, eigenvectors = compute_hessian_eigenthings(
                    model,
                    train_loader,
                    criterion,
                    args.num_eigenthings,
                    mode=args.mode,
                    power_iter_steps=args.num_steps,
                    max_possible_gpu_samples=args.max_samples,
                    power_iter_err_threshold=args.tol,
                    # momentum=args.momentum,
                    full_dataset=args.full_dataset,
                    use_gpu=torch.cuda.is_available()  # args.cuda,
                )
            elif args.mode == "lanczos":
                print(args.num_steps)
                eigenvalues, eigenvectors = compute_hessian_eigenthings(
                    model,
                    train_loader,
                    criterion,
                    args.num_eigenthings,
                    mode=args.mode,
                    max_steps=args.num_steps,
                    max_possible_gpu_samples=args.max_samples,
                    # momentum=args.momentum,
                    full_dataset=args.full_dataset,
                    use_gpu=torch.cuda.is_available()  # args.cuda,
                )

        results[num_models] = eigenvalues.tolist()
        num_models += 1

    # Save as JSON
    with open(RESULT_PATH+args.save_file_name, 'w') as output:
        print("saving results")
        json.dump(results, output, indent=4)


if __name__ == "__main__":
    main()
    print("All models evaluated!")
