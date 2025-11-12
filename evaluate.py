# Code partly taken from https://github.com/MJordahn/Decoupled-Layers-for-Calibrated-NNs/blob/main/src/experiments/01_eval_models.py  # noqa
import os
import json
import numpy as np
import torch
from utils.eval import (
    load_model,
    eval_train_data,
    plot_multi_model_reliability,
    eval_ood_data,
    eval_data,
    BACKENDS
)
from utils.data import load_hf_dataset, load_vision_dataset
from laplace import Laplace
from utils.paths import ROOT, LOCAL_STORAGE, DATA_DIR, RESULT_DIR
# from laplace.curvature.asdfghjkl import AsdfghjklHessian
# from laplace.curvature.curvature import CurvatureInterface
from utils.arguments import eval_args
from pathlib import Path
from utils.helpers import torch_device

# Only for one model at a time, with possible ensemble but different paths


def eval(args):
    print("[eval]: starting")
    model_paths = open(ROOT + "/eval_path_files/" + args.model_path_file, "r")
    device = torch_device()
    print(f"[device]: {device}")

    data_path = Path(LOCAL_STORAGE) / DATA_DIR
    result_path = Path(ROOT) / RESULT_DIR

    os.makedirs(result_path, exist_ok=True)

    if os.path.isfile(result_path / args.save_file_name):
        f = open(result_path / args.save_file_name, 'r')
        results = json.load(f)
    else:
        results = {}

    print(f"[dataset]: loading {args.dataset}")
    if args.dataset in ("cifar10", "cifar100", "mnist", "imagenet"):
        nlp, dm, num_classes, train_loader, val_loader, test_loader, shift_loader, ood_loader = load_vision_dataset(
            args=args,
            data_path=data_path
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
    print("[dataset]: loading done")

    num_models = 0

    # prepare reliability diagram plot
    model_results_id = []
    model_results_shift = []

    for model_path in model_paths.read().splitlines():
        model_path = model_path.strip()
        model_name = model_path.split("model_name=")[1].replace(".ckpt", "")

        if model_name not in results.keys():
            ood_done = in_done = False
            shift_done = train_done = False
            results[model_name] = {}
        else:
            ood_done = in_done = True
            shift_done = train_done = True
            if 'clean_accuracy' not in results[model_name].keys():
                in_done = False
            if 'SHIFT ECE' not in results[model_name].keys():
                shift_done = False
            if 'OOD AUROC' not in results[model_name].keys():
                ood_done = False
            if 'Train nll' not in results[model_name].keys():
                train_done = False
            if ood_done and in_done and shift_done and train_done:
                print(f"[eval]: skipping {model_name}, already done")
                num_models += 1
                continue

        feature_reduction, model = load_model(args, path=model_path, device=device, num_classes=num_classes)
        print(f"[eval]: loaded {model_name}")
        print("[eval]: starting")
        model = model.to(device)
        model.eval()

        # --------------------------------------------------------------------
        # Laplace Approximation
        # --------------------------------------------------------------------
        pred_type = args.pred_type
        if args.laplace:
            print("[laplace]: approximation")
            backend = BACKENDS[args.backend]

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
            if device.type == "cuda":
                torch.cuda.set_device(device)
                _ = torch.tensor(0., device=device)
                torch.cuda.synchronize(device)
            with torch.cuda.device(device):
                print("[laplace]: fitting")
                model.fit(train_loader, progress_bar=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            print("[laplace]: done fitting")
            if args.optimize_prior_precision is not None:
                model.optimize_prior_precision(pred_type=pred_type, method=args.optimize_prior_precision,
                                               link_approx=args.approx_link, val_loader=val_loader)

        # --------------------------------------------------------------------
        # Start Evaluation
        # --------------------------------------------------------------------
        if not train_done and args.eval_train:
            print("[eval]: training data")
            nll_value = eval_train_data(model, train_loader, laplace=args.laplace, device=device, link=args.approx_link,
                                        mc_samples=args.mc_samples, pred_type=pred_type)
            results[model_name]['Train nll'] = nll_value

        if not in_done:
            print("[eval]: test data")
            if args.rel_plot is True:
                rel_plot = "ID"
            else:
                rel_plot = None
            ece_calc, mce_calc, aece_calc, acc, nll_value, brier_score, f1, OOD_y_preds_logits, OOD_labels, y_pred_id, y_target_id = eval_data(  # noqa
                model, test_loader, device=device, num_classes=num_classes, laplace=args.laplace,
                link=args.approx_link, nll=True, mc_samples=args.mc_samples, pred_type=pred_type,
                cifar10H=args.cifar10H, model_name=args.save_file_name, num_models=num_models, rel_plot=rel_plot)
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
            print("[eval]: shift data")
            if rel_plot == "ID":
                rel_plot = "SHIFT"
            else:
                rel_plot = None
            ece_calc, mce_calc, aece_calc, acc, nll_value, brier_score, f1, _, _, y_pred_shift, y_target_shift = eval_data(  # noqa
                model, shift_loader, device=device, num_classes=num_classes, laplace=args.laplace,
                link=args.approx_link, mc_samples=args.mc_samples, pred_type=pred_type, model_name=args.save_file_name,
                num_models=num_models, rel_plot=rel_plot)
            results[model_name]['SHIFT ECE'] = ece_calc.to("cpu").numpy().tolist()*100
            results[model_name]['SHIFT MCE'] = mce_calc.to("cpu").numpy().tolist()*100
            results[model_name]['SHIFT aECE'] = aece_calc.to("cpu").numpy().tolist()*100
            results[model_name]['SHIFT ACCURACY'] = acc.to("cpu").numpy().tolist()
            results[model_name]['SHIFT f1'] = f1.to("cpu").numpy().tolist()
            model_results_shift.append({"y_probs": y_pred_shift,
                                        "y_true": y_target_shift})

        if not ood_done and args.eval_ood and ood_loader is not None:
            print("[eval]: ood data")
            auroc_calc, fpr_at_95_tpr_calc, ood_acc = eval_ood_data(model, ood_loader, device=device,
                                                                    num_classes=num_classes,
                                                                    OOD_y_preds_logits=OOD_y_preds_logits,
                                                                    OOD_labels=OOD_labels, laplace=args.laplace,
                                                                    link=args.approx_link, mc_samples=args.mc_samples,
                                                                    pred_type=pred_type)
            results[model_name]['OOD AUROC'] = auroc_calc
            results[model_name]['OOD FPR95'] = fpr_at_95_tpr_calc
            results[model_name]['OOD Accuracy'] = ood_acc.to("cpu").numpy().tolist()

        with open(result_path/args.save_file_name, 'w') as fp:
            json.dump(results, fp, indent=4)

        num_models += 1
        print(f"[eval]: {model_name} done")

    print("[eval]: all models done")
    print(f"[saving]: filename {args.save_file_name}")
    # --------------------------------------------------------------------
    # Calculate average over evaluated models and store it in JSON file
    # --------------------------------------------------------------------
    if num_models > 1:
        output_file = args.save_file_name.replace('.', '_summary.')
        model_results = open(result_path/args.save_file_name, 'r')

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

        with open(result_path/output_file, 'w') as output:
            json.dump(metrics_summary, output, indent=4)

        print("Metrics summary saved to ", {result_path/output_file})

        # --------------------------------------------------------------------
        # Plot Reliability Diagram
        # --------------------------------------------------------------------
        if args.rel_plot:
            PLOT_PATH = result_path / "/rel_diag_probs/"
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
    args = eval_args()

    result_path = Path(ROOT) / RESULT_DIR
    save_file = args.save_file_name.split(".")[0]
    results_dir = [results.name.split("_savefile")[0] for results in result_path.iterdir()]
    results_dir = [res.split(".")[0] for res in results_dir]
    if save_file in results_dir and not args.redo:
        print(f"[main]: {args.save_file_name} already exists, skipping...")
        return

    eval(args)


if __name__ == "__main__":
    main()
    print("[main]: all models evaluated")
