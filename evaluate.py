# Code partly taken from https://github.com/MJordahn/Decoupled-Layers-for-Calibrated-NNs/blob/main/src/experiments/01_eval_models.py  # noqa
import os
import json
import numpy as np
import torch
from argparse import ArgumentParser, BooleanOptionalAction
from utils.eval import load_model, eval_train_data, plot_multi_model_reliability, eval_ood_data, eval_data
from utils.data import load_hf_dataset, load_vision_dataset
from laplace import Laplace
from utils.paths import ROOT, LOCAL_STORAGE, DATA_DIR, RESULT_DIR
# from laplace.curvature.asdfghjkl import AsdfghjklHessian
from laplace.curvature.asdl import AsdlGGN, AsdlEF
from laplace.curvature.backpack import BackPackGGN, BackPackEF
# from laplace.curvature.curvature import CurvatureInterface
from laplace.curvature.curvlinops import CurvlinopsEF, CurvlinopsGGN
from helpers import common_arguments


def eval(args):
    print("--- Starting evaluation ---")

    if args.save_file_name == "":
        raise Exception("Provide a save_file_name!")
    if args.model_path_file == "":
        raise Exception("Provide a model_name_file!")
    # root_dir = ROOT + "/"

    model_paths = open(ROOT + "/eval_path_files/" + args.model_path_file, "r")

    # Set device
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    if args.use_cpu:
        device = torch.device('cpu')
    print("Using device: ", device)

    # Set Path to Datasets
    DATA_PATH = LOCAL_STORAGE + DATA_DIR
    RESULT_PATH = ROOT + RESULT_DIR

    os.makedirs(RESULT_PATH, exist_ok=True)

    print(RESULT_PATH + args.save_file_name)
    if os.path.isfile(RESULT_PATH + args.save_file_name):
        f = open(RESULT_PATH + args.save_file_name, 'r')
        results = json.load(f)
    else:
        results = {}

    # --------------------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------------------

    # Get Dataset
    print("Loading dataset: ", args.dataset)
    if args.dataset in ("CIFAR10", "CIFAR100", "MNIST", "ImageNet"):
        nlp, dm, num_classes, train_loader, val_loader, test_loader, shift_loader, ood_loader = load_vision_dataset(
            dataset=args.dataset,
            model_type=args.model_type,
            ViT_model=args.ViT_model,
            DATA_PATH=DATA_PATH,
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
        raise Exception("Requested dataset does not exist!")
    print("--- Loading done ---")

    # --------------------------------------------------------------------
    #  Models
    # --------------------------------------------------------------------

    # more than one model can be in the model path file
    # in that case, mean and standard error of the mean is calculated
    # for each metric
    # counter is needed to keep track of the number of models
    num_models = 0

    # prepare reliability diagram plot
    model_results_id = []
    model_results_shift = []

    for model_path in model_paths.read().splitlines():
        model_path = model_path.strip()
        model_name = model_path.split("model_name=")[1].replace(".ckpt", "")

        if model_name not in results.keys():
            ood_done = False
            in_done = False
            shift_done = False
            train_done = False
            results[model_name] = {}
        else:
            ood_done = True
            in_done = True
            shift_done = True
            train_done = True
            if 'clean_accuracy' not in results[model_name].keys():
                in_done = False
            if 'SHIFT ECE' not in results[model_name].keys():
                shift_done = False
            if 'OOD AUROC' not in results[model_name].keys():
                ood_done = False
            if 'Train nll' not in results[model_name].keys():
                train_done = False
            if ood_done and in_done and shift_done and train_done:
                print("SKIPPING")
                print(model_name)
                num_models += 1
                continue

        print("Loading model: ", model_name)
        feature_reduction, model = load_model(name=args.model_type, vit=args.ViT_model, nlp=args.NLP_model,
                                              path=model_path, device=device, num_classes=num_classes)
        print(args.model_type, model_path)
        print("Evaluate model!")
        model.eval()
        model = model.to(device)

        # --------------------------------------------------------------------
        # Laplace Approximation
        # --------------------------------------------------------------------
        pred_type = args.pred_type
        # Laplace approximation
        if args.laplace:
            print("... Laplace approximation ...")
            if args.backend == "BackpackGGN":
                backend = BackPackGGN
            elif args.backend == "BackpackEF":
                backend = BackPackEF
            elif args.backend == "AsdlGGN":
                backend = AsdlGGN
            elif args.backend == "AsdlEF":
                backend = AsdlEF
            elif args.backend == "CurvlinopsGGN":
                backend = CurvlinopsGGN
            elif args.backend == "CurvlinopsEF":
                backend = CurvlinopsEF
            elif args.backend is None:
                backend = CurvlinopsGGN
            else:
                raise Exception("This backend is not supported!")
            pred_type = args.pred_type
            if args.hessian_approx == "gp":
                model = Laplace(model, "classification", hessian_structure=args.hessian_approx,
                                subset_of_weights=args.subset_of_weights, independent_outputs=True,
                                n_subset=args.num_data, backend=backend)
                print("Consider reducing batch size or GP inference!")
                pred_type = "gp"
            if feature_reduction is not None:
                model = Laplace(model, "classification", hessian_structure=args.hessian_approx,
                                subset_of_weights=args.subset_of_weights, backend=backend,
                                feature_reduction=feature_reduction)
            else:
                model = Laplace(model, "classification", hessian_structure=args.hessian_approx,
                                subset_of_weights=args.subset_of_weights, backend=backend)
            model.fit(train_loader, progress_bar=True)
            if args.optimize_prior_precision is not None:
                model.optimize_prior_precision(pred_type=pred_type, method=args.optimize_prior_precision,
                                               link_approx=args.approx_link, val_loader=val_loader)

        # --------------------------------------------------------------------
        # Start Evaluation
        # --------------------------------------------------------------------
        if not train_done and args.eval_train:
            print("... training data ...")
            nll_value = eval_train_data(model, train_loader, laplace=args.laplace, device=device, link=args.approx_link,
                                        mc_samples=args.mc_samples, pred_type=pred_type)
            results[model_name]['Train nll'] = nll_value

        if not in_done:
            print("... test data ...")
            if args.rel_plot is True:
                rel_plot = "ID"
            else:
                rel_plot = None
            ece_calc, mce_calc, aece_calc, acc, nll_value, brier_score, f1, OOD_y_preds_logits, OOD_labels, y_pred_id, y_target_id = eval_data(model, test_loader, device=device, num_classes=num_classes, laplace=args.laplace, link=args.approx_link, nll=True, mc_samples=args.mc_samples, pred_type=pred_type, cifar10H=args.cifar10H, model_name=args.save_file_name, num_models=num_models, rel_plot=rel_plot)
            results[model_name]['clean_accuracy'] = acc.to("cpu").numpy().tolist()
            results[model_name]['f1'] = f1.to("cpu").numpy().tolist()
            results[model_name]['ECE'] = ece_calc.to("cpu").numpy().tolist()*100
            results[model_name]['MCE'] = mce_calc.to("cpu").numpy().tolist()*100
            results[model_name]['aECE'] = aece_calc.to("cpu").numpy().tolist()*100
            results[model_name]['nll'] = nll_value
            results[model_name]['brier'] = brier_score
            model_results_id.append({"y_probs": y_pred_id,
                                    "y_true": y_target_id})

        if not shift_done and args.eval_shift and shift_loader is not None:
            print("... shift data ...")
            if rel_plot == "ID":
                rel_plot = "SHIFT"
            else:
                rel_plot = None
            ece_calc, mce_calc, aece_calc, acc, nll_value, brier_score, f1, _, _, y_pred_shift, y_target_shift = eval_data(model, shift_loader, device=device, num_classes=num_classes, laplace=args.laplace, link=args.approx_link, mc_samples=args.mc_samples, pred_type=pred_type, model_name=args.save_file_name, num_models=num_models, rel_plot=rel_plot)
            results[model_name]['SHIFT ECE'] = ece_calc.to("cpu").numpy().tolist()*100
            results[model_name]['SHIFT MCE'] = mce_calc.to("cpu").numpy().tolist()*100
            results[model_name]['SHIFT aECE'] = aece_calc.to("cpu").numpy().tolist()*100
            results[model_name]['SHIFT ACCURACY'] = acc.to("cpu").numpy().tolist()
            results[model_name]['SHIFT f1'] = f1.to("cpu").numpy().tolist()
            model_results_shift.append({"y_probs": y_pred_shift,
                                        "y_true": y_target_shift})

        if not ood_done and args.eval_ood and ood_loader is not None:
            print("... ood data ...")
            auroc_calc, fpr_at_95_tpr_calc, ood_acc = eval_ood_data(model, ood_loader, device=device,
                                                                    num_classes=num_classes,
                                                                    OOD_y_preds_logits=OOD_y_preds_logits,
                                                                    OOD_labels=OOD_labels, laplace=args.laplace,
                                                                    link=args.approx_link, mc_samples=args.mc_samples,
                                                                    pred_type=pred_type)
            results[model_name]['OOD AUROC'] = auroc_calc
            results[model_name]['OOD FPR95'] = fpr_at_95_tpr_calc
            results[model_name]['OOD Accuracy'] = ood_acc.to("cpu").numpy().tolist()

        with open(RESULT_PATH+args.save_file_name, 'w') as fp:
            json.dump(results, fp)

        num_models += 1
        print("Model evaluated!")

    print("All models evaluated!")
    print("Saving results to file: ", args.save_file_name)
    # --------------------------------------------------------------------
    # Calculate average over evaluated models and store it in JSON file
    # --------------------------------------------------------------------
    if num_models > 1:
        output_file = args.save_file_name.replace('.', '_summary.')
        model_results = open(RESULT_PATH+args.save_file_name, 'r')

        metrics_data = {}
        for line_data in model_results.read().splitlines():
            data = json.loads(line_data)

            # Loop over each entry in the JSON structure
            for key, metrics in data.items():
                for metric, value in metrics.items():
                    # If the value is a dictionary (for nested metrics like "SHIFT Intensity")
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            full_metric = f"{full_metric}_{sub_metric}"  # noqa
                            metrics_data.setdefault(full_metric, []).append(sub_value)
                    elif isinstance(value, list):
                        # If the metric is a list (like "OOD AUROC"), compute the average of the list
                        metrics_data.setdefault(metric, []).append(np.mean(value))
                    else:
                        metrics_data.setdefault(metric, []).append(value)

        # Compute mean and standard deviation for each metric
        metrics_summary = {}
        for metric, values in metrics_data.items():
            metrics_summary[metric] = {
                "average": np.mean(values),
                "SE": np.std(values)/np.sqrt(num_models)
            }

        with open(RESULT_PATH+output_file, 'w') as output:
            json.dump(metrics_summary, output, indent=4)

        print("Metrics summary saved to ", {RESULT_PATH+output_file})

        # --------------------------------------------------------------------
        # Plot Reliability Diagram
        # --------------------------------------------------------------------
        if args.rel_plot:
            PLOT_PATH = RESULT_PATH + "/rel_diag_probs/"
            os.makedirs(PLOT_PATH, exist_ok=True)
            print(len(model_results_id))
            print(len(model_results_shift))
            torch.save(model_results_id, PLOT_PATH + args.save_file_name[:-4]+"_"+str(num_models)+"_ID_values.pt")
            torch.save(model_results_shift, PLOT_PATH + args.save_file_name[:-4]+"_"+str(num_models)+"_SHIFT_values.pt")
            plot_multi_model_reliability(model_results_id, n_bins=10, error_type='se',
                                         color="rgba(81, 127, 252, 0.92)",
                                         model_name=args.save_file_name[:-4]+"_"+str(num_models)+"_ID_SE")
            plot_multi_model_reliability(model_results_shift, n_bins=10, error_type='se',
                                         color="rgba(252, 127, 81, 0.92)",
                                         model_name=args.save_file_name[:-4]+"_"+str(num_models)+"_SHIFT_SE")


def encode_mrpc(examples, tokenizer):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')


def encode_mnli(batch, tokenizer):
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding="max_length")


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
    parser.add_argument("--use_cpu", action=BooleanOptionalAction, default=False,
                        help="Whether to use CPU for evaluation.")

    # --------------------------------------------

    parser.add_argument("--dataset", type=str, default="",  # Add check to only eval on correct dataset
                        help="The dataset to evaluate on (e.g. CIFAR10, ImageNet etc.).")
    parser.add_argument("--basic_augment", action=BooleanOptionalAction, default=True,
                        help="True if you want to use basic augmentations (horizontal flip, random crop with padding).")
    parser.add_argument("--val_split", type=int, default=0.0,
                        help="Split the training set into train and validation set.")
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
    parser.add_argument("--approx_link", default="mc", type=str,
                        help="The approximation link to use (e.g. mc, probit, bridge, bridge_norm).")
    parser.add_argument("--hessian_approx", default="full", type=str,
                        help="The Hessian approximation to use (e.g. full, diag, kron).")
    parser.add_argument("--subset_of_weights", default="last_layer", type=str,
                        help="The subset of weights to use (e.g. last_layer, all).")
    parser.add_argument("--plot", action=BooleanOptionalAction, default=False,
                        help="Whether to plot reliability diagrams.")
    parser.add_argument('--optimize_prior_precision', default=None, choices=['marglik', 'nll'],
                        help='optimize prior precision according to specified method')
    parser.add_argument('--backend', default=None,
                        choices=['CurvlinopsGGN', 'CurvlinopsEF', 'AsdlGGN', 'AsdlEF', 'BackpackGGN', 'BackpackEF'],
                        help='The backend used for Hessian approximations')
    parser.add_argument('--mc_samples', default=10, type=int,
                        help='The number of samples used for Monte Carlo approximation of predictive distribution.')
    parser.add_argument('--num_data', default=1000, type=int,
                        help='The number of data points for Subset-of-Data (SOD) approximate GP inference.')
    parser.add_argument('--pred_type', default="glm", type=str,
                        help='Prediction type has to be one of "nn" or "glm".')

    parser.add_argument('--cifar10H', action=BooleanOptionalAction, default=False, type=bool,
                        help="Whether to evaluate models trained on CIFAR10 on CIFAR-10-H")
    parser.add_argument('--rel_plot', action=BooleanOptionalAction, default=False,
                        help="Whether to reliability diagrams (both shift and id)")

    parser = common_arguments(parser)

    args = parser.parse_args()

    eval(args)


if __name__ == "__main__":
    main()
    print("All models evaluated!")
