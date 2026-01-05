from argparse import ArgumentParser, BooleanOptionalAction


def common_args():
    parser = ArgumentParser()
    parser.add_argument("--val_split", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_norm", action=BooleanOptionalAction, default=False)
    return parser


def eval_args():
    parser = common_args()
    parser.add_argument("--save_file_name", type=str, required=True)
    parser.add_argument("--model_path_file", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--NLP_model", type=str, default='bert-base-uncased',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--ViT_model", type=str, default='vit_base_patch16_224.orig_in21k',
                        help="Path to checkpoint for fine-tuning")

    # --------------------------------------------

    parser.add_argument("--dataset", type=str, default="",  # Add check to only eval on correct dataset
                        help="The dataset to evaluate on (e.g. CIFAR10, ImageNet etc.).")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True,
                        help="True if you want to use basic augmentations (horizontal flip, random crop with padding).")
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

    parser.add_argument("--laplace", action=BooleanOptionalAction, default=False, type=bool,
                        help="Whether to use Laplace approximation.")
    parser.add_argument("--approx_link", default="mc", type=str, choices=["mc", "probit", "bridge", "bridge_norm"])

    parser.add_argument("--hessian_approx", default="full", type=str,
                        choices=["full", "diag", "kron"])
    parser.add_argument("--subset_of_weights", default="last_layer", type=str,
                        choices=["last_layer", "all", "subnetwork"])

    parser.add_argument("--plot", action=BooleanOptionalAction, default=False)
    parser.add_argument('--optimize_prior_precision', default=None, choices=['marglik', 'gridsearch'])
    parser.add_argument('--backend', default=None,
                        choices=['CurvlinopsGGN', 'CurvlinopsEF', 'AsdlGGN', 'AsdlEF', 'BackpackGGN', 'BackpackEF'],
                        help='The backend used for Hessian approximations')
    parser.add_argument('--mc_samples', default=10, type=int)
    parser.add_argument('--num_data', default=1000, type=int,
                        help='The number of data points for Subset-of-Data (SOD) approximate GP inference.')
    parser.add_argument('--pred_type', default="nn", type=str,  # TODO: nn important
                        choices=["nn", "glm"])

    parser.add_argument('--rel_plot', action=BooleanOptionalAction, default=False,
                        help="Whether to reliability diagrams (both shift and id)")

    parser.add_argument("--redo", action=BooleanOptionalAction, default=False)
    parser.add_argument("--freeze_frn", action=BooleanOptionalAction, default=False)

    args = parser.parse_args()
    return args


def train_args():
    parser = common_args()
    # SEED
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seeds_per_job", type=int, default=1)

    # Model and Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True)
    parser.add_argument("--model", type=str, default="ResNet18")
    parser.add_argument("--depth", default=18, type=int)
    parser.add_argument("--model_name", type=str, default="Unknown")
    parser.add_argument("--ViT_model", type=str, default='google/vit-base-patch16-224-in21k',
                        help="Path to checkpoint for fine-tuning")
    parser.add_argument("--normalize_pretrained_dataset", action=BooleanOptionalAction, default=False,
                        help="Finetune the dataset using the normalization values of the pretrained dataset (VIT)")

    # Training
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--SAM", default=False, action=BooleanOptionalAction,
                        help="Enable SAM optimizer.")
    parser.add_argument("--learning_rate", default=0.1, type=float)
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
    parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    parser.add_argument("--packed", action=BooleanOptionalAction, default=False)

    # Ensemble arguments
    parser.add_argument("--ensemble", action=BooleanOptionalAction, default=False)
    parser.add_argument("--num_ensemble_models", type=int, default=4)

    parser.add_argument("--no_save", action=BooleanOptionalAction, default=False)

    args = parser.parse_args()
    return args
