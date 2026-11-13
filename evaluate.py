# Code partly taken from https://github.com/MJordahn/Decoupled-Layers-for-Calibrated-NNs/blob/main/src/experiments/01_eval_models.py  # noqa
import os
from copy import deepcopy
import json
import numpy as np
import torch
from utils.eval import (
    load_model,
    eval_train_data,
    plot_multi_model_reliability,
    eval_ood_data,
    eval_data,
)
from utils.data import load_hf_dataset, load_vision_dataset
from utils.paths import ROOT, LOCAL_STORAGE, DATA_DIR, RESULT_DIR
from utils.arguments import eval_args
from pathlib import Path
from utils.helpers import torch_device
from models.sgld_ensemble import SGLDEnsemble


def estimator_indices(model):
    from torch.nn.utils import parameters_to_vector
    num_estimators = model.num_estimators
    all_params = parameters_to_vector(model.parameters())
    total_params = all_params.numel()

    last_layer = model.linear
    last_layer_params = sum(p.numel() for p in last_layer.parameters())

    params_per_est = last_layer_params // num_estimators

    last_layer_start = total_params - last_layer_params

    estimator_indices_list = []
    for est_idx in range(num_estimators):
        est_start = last_layer_start + (est_idx * params_per_est)
        est_end = est_start + params_per_est
        estimator_indices_list.append(torch.arange(est_start, est_end))

    return estimator_indices_list


def eval_sgld_ensemble(args, device, data_path, result_path):
    """Evaluate all model paths as a single SGLD posterior ensemble."""
    model_paths = open(ROOT + "/eval_path_files/" + args.model_path_file, "r").read().splitlines()
    model_paths = [p.strip() for p in model_paths if p.strip()]
    if args.max_sgld_samples and len(model_paths) > args.max_sgld_samples:
        indices = [round(i * (len(model_paths) - 1) / (args.max_sgld_samples - 1))
                   for i in range(args.max_sgld_samples)]
        model_paths = [model_paths[i] for i in indices]

    if args.dataset in ("cifar10", "cifar100", "mnist", "imagenet"):
        _, dm, num_classes, train_loader, val_loader, test_loader, shift_loader, ood_loader = load_vision_dataset(
            args=args, data_path=data_path
        )
    elif args.dataset in ("MNLI", "RTE", "MRPC"):
        _, train_loader, val_loader, test_loader, shift_loader, ood_loader, num_classes = load_hf_dataset(
            NLP_model=args.NLP_model,
            dataset_name=args.dataset,
            eval_ood=args.eval_ood,
            eval_shift=args.eval_shift,
            batch_size=args.batch_size
        )
    else:
        raise Exception("Requested dataset does not exist!")

    def model_factory():
        _, m = load_model(args, path=model_paths[0], device=device, num_classes=num_classes)
        return m

    print(f"[SGLD ensemble]: loading {len(model_paths)} samples")
    models = []
    for path in model_paths:
        _, m = load_model(args, path=path, device=device, num_classes=num_classes)
        models.append(m)

    ensemble = SGLDEnsemble(models).to(device)
    ensemble.eval()

    ensemble_name = f"sgld_ensemble_{len(model_paths)}samples"
    results = {ensemble_name: {}}

    if args.eval_train:
        nll_value = eval_train_data(ensemble, train_loader, device=device)
        results[ensemble_name]['Train nll'] = nll_value

    ece, mce, aece, acc, nll_value, brier_score, f1, y_ood_logits, y_ood, y_pred_id, y_target_id = eval_data(
        ensemble, test_loader, device=device, num_classes=num_classes, nll=True,
        model_name=args.save_file_name, num_models=1, data_type="test data"
    )
    results[ensemble_name]['clean_accuracy'] = acc.to("cpu").numpy().tolist()
    results[ensemble_name]['f1'] = f1.to("cpu").numpy().tolist()
    results[ensemble_name]['ECE'] = ece.to("cpu").numpy().tolist() * 100
    results[ensemble_name]['MCE'] = mce.to("cpu").numpy().tolist() * 100
    results[ensemble_name]['aECE'] = aece.to("cpu").numpy().tolist() * 100
    results[ensemble_name]['nll'] = nll_value
    results[ensemble_name]['brier'] = brier_score

    if args.eval_shift and shift_loader is not None:
        ece, mce, aece, acc, nll_value, brier_score, f1, _, _, _, _ = eval_data(
            ensemble, shift_loader, device=device, num_classes=num_classes,
            model_name=args.save_file_name, num_models=1, data_type="shift data"
        )
        results[ensemble_name]['SHIFT ECE'] = ece.to("cpu").numpy().tolist() * 100
        results[ensemble_name]['SHIFT MCE'] = mce.to("cpu").numpy().tolist() * 100
        results[ensemble_name]['SHIFT aECE'] = aece.to("cpu").numpy().tolist() * 100
        results[ensemble_name]['SHIFT ACCURACY'] = acc.to("cpu").numpy().tolist()
        results[ensemble_name]['SHIFT f1'] = f1.to("cpu").numpy().tolist()

    if args.eval_ood and ood_loader is not None:
        auroc_calc, fpr95_ood, ood_acc = eval_ood_data(
            ensemble, ood_loader, device=device, num_classes=num_classes,
            y_ood_logits=y_ood_logits, OOD_labels=y_ood,
        )
        results[ensemble_name]['OOD AUROC'] = auroc_calc
        results[ensemble_name]['OOD FPR95'] = fpr95_ood
        results[ensemble_name]['OOD Accuracy'] = ood_acc.to("cpu").numpy().tolist()

    with open(result_path / args.save_file_name, 'w') as fp:
        json.dump(results, fp, indent=4)

    print(f"[SGLD ensemble]: done → {result_path / args.save_file_name}")


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

    model_results_id = []
    model_results_shift = []

    for model_path in model_paths.read().splitlines():
        model_path = model_path.strip()
        try:
            model_name = model_path.split("model_name=")[1].replace(".ckpt", "")
        except IndexError:
            try:
                model_name = model_path.split("mn=")[1].split("-")[0]
            except IndexError:
                model_name = os.path.splitext(os.path.basename(model_path))[0]

        if model_name not in results.keys():
            ood_done = in_done = False
            shift_done = train_done = False
            results[model_name] = {}
        else:
            in_done = 'clean_accuracy' in results[model_name].keys()
            ood_done = 'OOD AUROC' in results[model_name].keys()
            shift_done = 'SHIFT ECE' in results[model_name].keys()
            train_done = 'Train nll' in results[model_name].keys()
            if ood_done and in_done and shift_done and train_done:
                print(f"[eval]: skipping {model_name}, already done")
                num_models += 1
                continue

        feature_reduction, model = load_model(args, path=model_path, device=device, num_classes=num_classes)
        print(f"[eval]: loaded {model_name}")
        model = model.to(device)
        model.eval()

        rel_plot = None

        if not train_done and args.eval_train:
            nll_value = eval_train_data(model, train_loader, device=device)
            results[model_name]['Train nll'] = nll_value

        if not in_done:
            if args.rel_plot is True:
                rel_plot = "ID"
            ece, mce, aece, acc, nll_value, brier_score, f1, y_ood_logits, y_ood, y_pred_id, y_target_id = eval_data(
                model, test_loader, device=device, num_classes=num_classes, nll=True,
                model_name=args.save_file_name, num_models=num_models, rel_plot=rel_plot, data_type="test data")
            results[model_name]['clean_accuracy'] = acc.to("cpu").numpy().tolist()
            results[model_name]['f1'] = f1.to("cpu").numpy().tolist()
            results[model_name]['ECE'] = ece.to("cpu").numpy().tolist() * 100
            results[model_name]['MCE'] = mce.to("cpu").numpy().tolist() * 100
            results[model_name]['aECE'] = aece.to("cpu").numpy().tolist() * 100
            results[model_name]['nll'] = nll_value
            results[model_name]['brier'] = brier_score
            model_results_id.append({"y_probs": y_pred_id,
                                    "y_true": y_target_id})

        if not shift_done and args.eval_shift and shift_loader is not None:
            if rel_plot == "ID":
                rel_plot = "SHIFT"
            ece, mce, aece, acc, nll_value, brier_score, f1, _, _, y_pred_shift, y_target_shift = eval_data(
                model, shift_loader, device=device, num_classes=num_classes,
                model_name=args.save_file_name,
                num_models=num_models, rel_plot=rel_plot, data_type="shift data")
            results[model_name]['SHIFT ECE'] = ece.to("cpu").numpy().tolist() * 100
            results[model_name]['SHIFT MCE'] = mce.to("cpu").numpy().tolist() * 100
            results[model_name]['SHIFT aECE'] = aece.to("cpu").numpy().tolist() * 100
            results[model_name]['SHIFT ACCURACY'] = acc.to("cpu").numpy().tolist()
            results[model_name]['SHIFT f1'] = f1.to("cpu").numpy().tolist()
            model_results_shift.append({"y_probs": y_pred_shift,
                                        "y_true": y_target_shift})

        if not ood_done and args.eval_ood and ood_loader is not None:
            auroc_calc, fpr95_ood, ood_acc = eval_ood_data(
                model, ood_loader, device=device, num_classes=num_classes,
                y_ood_logits=y_ood_logits, OOD_labels=y_ood,
            )
            results[model_name]['OOD AUROC'] = auroc_calc
            results[model_name]['OOD FPR95'] = fpr95_ood
            results[model_name]['OOD Accuracy'] = ood_acc.to("cpu").numpy().tolist()

        with open(result_path / args.save_file_name, 'w') as fp:
            json.dump(results, fp, indent=4)

        num_models += 1
        print(f"[eval]: {model_name} done")

    print("[eval]: all models done")
    print(f"[saving]: filename {args.save_file_name}")

    if num_models > 1:
        output_file = args.save_file_name.replace('.', '_summary.')
        model_results = open(result_path / args.save_file_name, 'r')

        metrics_data = {}
        for line_data in model_results.read().splitlines():
            data = json.loads(line_data)

            for key, metrics in data.items():
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            full_metric = f"{full_metric}_{sub_metric}"  # noqa
                            metrics_data.setdefault(full_metric, []).append(sub_value)
                    elif isinstance(value, list):
                        metrics_data.setdefault(metric, []).append(np.mean(value))
                    else:
                        metrics_data.setdefault(metric, []).append(value)

        metrics_summary = {}
        for metric, values in metrics_data.items():
            metrics_summary[metric] = {
                "average": np.mean(values),
                "SE": np.std(values) / np.sqrt(num_models)
            }

        with open(result_path / output_file, 'w') as output:
            json.dump(metrics_summary, output, indent=4)

        print("Metrics summary saved to ", {result_path / output_file})

        if args.rel_plot:
            PLOT_PATH = result_path / "/rel_diag_probs/"
            os.makedirs(PLOT_PATH, exist_ok=True)
            print(len(model_results_id))
            print(len(model_results_shift))
            torch.save(
                model_results_id,
                PLOT_PATH / (args.save_file_name[:-4] + "_" + str(num_models) + "_ID_values.pt")
            )
            torch.save(
                model_results_shift,
                PLOT_PATH / (args.save_file_name[:-4] + "_" + str(num_models) + "_SHIFT_values.pt")
            )
            plot_multi_model_reliability(model_results_id, n_bins=10, error_type='se',
                                         color="rgba(81, 127, 252, 0.92)",
                                         model_name=args.save_file_name[:-4] + "_" + str(num_models) + "_ID_SE")
            plot_multi_model_reliability(model_results_shift, n_bins=10, error_type='se',
                                         color="rgba(252, 127, 81, 0.92)",
                                         model_name=args.save_file_name[:-4] + "_" + str(num_models) + "_SHIFT_SE")


def encode_mrpc(examples, tokenizer):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')


def encode_mnli(batch, tokenizer):
    return tokenizer(batch["premise"], batch["hypothesis"], truncation=True, padding="max_length")


def main():
    args = eval_args()

    result_path = Path(ROOT) / RESULT_DIR
    os.makedirs(result_path, exist_ok=True)

    save_file = args.save_file_name.split(".")[0]
    results_dir = [results.name.split("_savefile")[0] for results in result_path.iterdir()]
    results_dir = [res.split(".")[0] for res in results_dir]
    if save_file in results_dir and not args.redo:
        print(f"[main]: {args.save_file_name} already exists, skipping...")
        return

    data_path = Path(LOCAL_STORAGE) / DATA_DIR

    if args.sgld_ensemble:
        eval_sgld_ensemble(args, torch_device(), data_path, result_path)
    else:
        eval(args)


if __name__ == "__main__":
    main()
    print("[main]: all models evaluated")
