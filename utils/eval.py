# Code based on https://github.com/JohannaDK/DD2412-Final-Project/blob/main/src/utils/eval_utils.py
import torch
import torch.nn as nn
from models.resnet import torch_resnet56, ResNet18, torch_resnet18
from models.ensemble_model import EnsembleModel
from sklearn import metrics
from torchmetrics.classification import Accuracy, F1Score  # , MulticlassCalibrationError
from torch_uncertainty.metrics.classification import (
    CalibrationError,
    AdaptiveCalibrationError,
    AURC, FPR95,
    # BrierScore
)
import torch.distributions as dists
import numpy as np
import os
from utils.paths import ROOT, MODEL_PATH
import timm
from models.bert import BERT
from models.hf import HF
import plotly.graph_objects as go
from laplace.curvature.asdl import AsdlGGN, AsdlEF
from laplace.curvature.backpack import BackPackGGN, BackPackEF
from laplace.curvature.curvlinops import CurvlinopsEF, CurvlinopsGGN


BACKENDS = {
    "BackpackGGN": BackPackGGN,
    "BackpackEF": BackPackEF,
    "AsdlGGN": AsdlGGN,
    "AsdlEF": AsdlEF,
    "CurvlinopsGGN": CurvlinopsGGN,
    "CurvlinopsEF": CurvlinopsEF,
    None: CurvlinopsGGN
}


def load_model(name, vit, nlp, path, device, num_classes):
    print(f"[num_classes]: {num_classes}")
    print(f"[model]: loading {name}")
    # Load the model from the specified path
    feature_reduction = None
    if name == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif name == 'resnet18_ensemble':
        model = EnsembleModel(model=torch_resnet18, num_models=5, num_classes=num_classes)
    elif name == 'resnet56':
        model = torch_resnet56(num_classes=num_classes)
    elif name == "vit":
        model = timm.create_model(vit, pretrained=False, num_classes=num_classes)
    elif name == "bert":
        model = BERT(device=device, model=nlp, num_labels=num_classes)
        # as hugging face models are loaded, this would not strictly be necessary
        feature_reduction = "pick_first"
    elif name == "roberta":
        model = HF(device=device, model=nlp, num_labels=num_classes)
        feature_reduction = "pick_first"
    else:
        raise Exception("Oops, requested model is not supported!")
    if name == "roberta":
        model.model.load_state_dict(torch.load(path, map_location=device, weight_only=True))
    else:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    return feature_reduction, model


def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Loss
            val_loss += criterion(outputs, labels).item() * labels.size(0)
            total_samples += labels.size(0)

    val_loss /= total_samples
    accuracy = correct / total

    return accuracy, val_loss


@torch.no_grad()
def evaluate_model_lang(model, dataloader, device, criterion, nlp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in dataloader:
        if nlp:
            x = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            y = batch['labels'].to(device)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)

        output = model(**x) if nlp else model(x)
        logits = output.logits if hasattr(output, "logits") else output
        loss = output.loss
        if loss is None:
            loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return accuracy, avg_loss


def brier(y_pred, y_true):
    try:
        y_pred, y_true = y_pred.detach().cpu().numpy(), y_true.cpu().numpy()
    finally:
        def one_hot(targets, nb_classes):
            res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape)+[nb_classes])

        return metrics.mean_squared_error(y_pred, one_hot(y_true, y_pred.shape[-1]))


def eval_train_data(model, dataloader, device, laplace, link, mc_samples, pred_type):
    for batch in dataloader:
        if isinstance(batch, list):
            x = batch[0].to(device)
            y = batch[1].to(device)
        else:
            x, y = batch, batch['labels'].to(device)

        if isinstance(model, nn.Module):
            probs = (torch.softmax(model(x), dim=-1))
        else:
            probs = (model(x, link_approx=link, n_samples=mc_samples, pred_type=pred_type))
    nll = -dists.Categorical(probs).log_prob(y).mean()
    return nll.item()


def eval_data(model, dataloader, device, num_classes, laplace=False, link=None, nll=False,
              mc_samples=10, pred_type="glm", cifar10H=False, model_name=None, num_models=0, rel_plot=None):
    # This function is called by both shift and ID evaluation
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    ece_metric = CalibrationError("multiclass", num_classes=num_classes, num_bins=15, norm='l1')
    mce_metric = CalibrationError("multiclass", num_classes=num_classes, num_bins=15, norm='max')
    aece_metric = AdaptiveCalibrationError("multiclass", num_classes=num_classes, num_bins=15, norm='l1')

    y_preds = []
    y_targets = []
    OOD_labels = []
    OOD_y_preds_logits = []

    with torch.no_grad():
        # for images, targets in dataloader:
        #    images, targets = images.to(device), targets.to(device)
        for batch in dataloader:
            if isinstance(batch, list):
                x, y = batch
                x = x.to(device)
                labels = y.to(device)
            else:
                # batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                # x = batch
                # targets = batch['labels'].to(device)
                x, labels = batch, batch['labels'].to(device)

            if isinstance(model, nn.Module):
                probs = (torch.softmax(model(x), dim=-1))
            else:
                probs = (model(x, link_approx=link, n_samples=mc_samples, pred_type=pred_type))
            ece_metric.update(probs, labels)
            mce_metric.update(probs, labels)
            aece_metric.update(probs, labels)
            accuracy.update(probs, labels)
            f1_score.update(probs, labels)

            y_preds.append(probs)
            y_targets.append(labels)

            # OOD detection using max softmax probability
            OOD_y_preds_logits.append(probs.detach().to(device))
            OOD_labels.append(torch.ones(len(labels), device=device))  # ID samples are labeled as 1

    y_preds = torch.cat(y_preds, dim=0)
    y_targets = torch.cat(y_targets, dim=0)

    # calculate final metrics
    ece = ece_metric.compute()
    mce = mce_metric.compute()
    aece = aece_metric.compute()
    acc = accuracy.compute()
    f1 = f1_score.compute()

    # Calculate Brier score
    brier_score = brier(y_preds, y_targets)

    if nll:  # only for test data (not shift or ood)
        nll = -dists.Categorical(y_preds).log_prob(y_targets).mean().item()
    else:
        nll = None

    # Store the probabilities for the CIFAR10-H analysis
    if cifar10H:
        probabilities = torch.cat(OOD_y_preds_logits, dim=0)
        probabilities = np.array([t.cpu().numpy() for t in probabilities])

        CIFAR10H_PATH = ROOT + MODEL_PATH + "cifar10H/probs/"
        os.makedirs(CIFAR10H_PATH, exist_ok=True)

        np.save(CIFAR10H_PATH+str(model_name[:-4])+"_"+str(num_models)+'.npy', probabilities)
        print("Saving to ", CIFAR10H_PATH+str(model_name[:-4])+"_"+str(num_models)+'.npy')

    return ece, mce, aece, acc, nll, brier_score, f1, OOD_y_preds_logits, OOD_labels, y_preds, y_targets


def eval_ood_data(model, ood_dataloader, device, num_classes, OOD_y_preds_logits,
                  OOD_labels, laplace, link, mc_samples, pred_type="glm"):
    accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for batch in ood_dataloader:
            if isinstance(batch, list):
                x, y = batch
                x = x.to(device)
                labels = y.to(device)
            else:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                x = batch
                labels = batch['labels'].to(device)
            if laplace:
                probs = (model(x, link_approx=link, n_samples=mc_samples, pred_type=pred_type))
            else:
                probs = (torch.softmax(model(x), dim=-1))
            OOD_y_preds_logits.append(probs.detach().to(device))
            OOD_labels.append(torch.zeros(len(labels), device=device))  # label OOD data with 0
            accuracy.update(probs, labels)
    OOD_labels = torch.cat(OOD_labels).to(device)
    OOD_y_preds_logits = torch.cat(OOD_y_preds_logits).to(device)

    auroc_metric = AURC().to(device)
    auroc_metric.update(OOD_y_preds_logits, OOD_labels)
    auroc_calc = auroc_metric.compute().item()

    confidences = OOD_y_preds_logits.max(dim=1).values
    fpr95_metric = FPR95(pos_label=0).to(device)
    fpr95_metric.update(confidences, OOD_labels)
    fpr95_calc = fpr95_metric.compute().item()

    acc = accuracy.compute()

    return auroc_calc, fpr95_calc, acc


def get_reliability_bin_data(y_probs, y_true, n_bins=10):
    """
    Calculates and returns reliability diagram bin data for a single model.
    This function processes the raw predictions/labels and returns the binned
    accuracies and their corresponding counts.

    y_probs - a single torch tensor (size N x num_classes)
              with the y_probs (logits or pre-softmax outputs) from the final linear layer.
    y_true - a single torch tensor (size N) with the labels
    n_bins - number of bins for the reliability diagram

    Returns:
        bin_centers (torch.Tensor): Centers of the confidence bins.
        bin_accuracies (torch.Tensor): Mean accuracy in each bin.
        bin_counts (torch.Tensor): Number of samples in each bin.
    """

    # Ensure y_probs and y_true are single tensors (as per your requirement for this function)
    if not isinstance(y_probs, torch.Tensor):
        raise TypeError("get_reliability_bin_data expects y_probs to be a single torch.Tensor.")
    if not isinstance(y_true, torch.Tensor):
        raise TypeError("get_reliability_bin_data expects y_true to be a single torch.Tensor.")

    # Apply softmax to convert logits to probabilities
    # y_probs_softmax = torch.softmax(y_probs, dim=1)

    # Get confidences and predictions
    confidences, predictions = y_probs.max(1)
    accuracies = predictions.eq(y_true)

    # Define bins
    bins = torch.linspace(0, 1, n_bins + 1, device=y_probs.device)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Initialize lists to store bin data
    # bin_accuracies = []
    bin_counts = []

    """
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]

        # Select data points within the current bin
        if i == n_bins - 1:
            bin_mask = (confidences >= bin_lower) & (confidences <= bin_upper) # Use <= for the last bin
        else:
            bin_mask = (confidences >= bin_lower) & (confidences < bin_upper)

        if bin_mask.any():
            bin_accuracies.append(accuracies[bin_mask].float().mean().item())
            bin_counts.append(bin_mask.sum().item())
        else:
            bin_accuracies.append(0.0) # No samples, so accuracy is 0.0
            bin_counts.append(0)
    return bin_centers, torch.tensor(bin_accuracies), torch.tensor(bin_counts)
    """

    bin_centers = (bins[:-1] + bins[1:]) / 2  # Torch equivalent of bin centers
    bin_indices_1 = [
        (confidences >= bin_lower) & (confidences < bin_upper)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]
    print(bin_indices_1[:10], bin_indices_1[-10:])

    # Calculate bin_corrects and bin_scores
    bin_corrects = torch.tensor([
        accuracies[bin_index].float().mean() if bin_index.any() else 0.0
        for bin_index in bin_indices_1
    ], device=y_probs.device)

    bin_counts = torch.tensor([
        bin_index.sum().item()
        for bin_index in bin_indices_1
    ], device=y_probs.device)

    return bin_centers, bin_corrects, bin_counts


# Your original save_reliability_plot, now calling the helper
def save_reliability_plot(y_probs, y_true, model_name, color, n_bins=10, error_type='se'):
    """
    Creates and saves a reliability diagram for a single model.
    This function is intended for single model evaluation, using the
    get_reliability_bin_data helper.

    y_probs - a torch tensor (size N x num_classes) with the y_probs (logits or pre-softmax outputs).
    y_true - a torch tensor (size N) with the labels.
    model_name - name of the model for plot title and filename.
    color - color for the bars in the plot.
    n_bins - number of bins for the reliability diagram.
    error_type - 'sd' for standard deviation or 'se' for standard error.
    """

    # Use the helper function to get bin data
    bin_centers, bin_accuracies_tensor, bin_counts_tensor = get_reliability_bin_data(y_probs, y_true, n_bins)

    # Plot reliability diagram with Plotly
    fig = go.Figure()

    width = 1.0 / n_bins  # Define width here for plotting

    # Add bars for accuracies with error bars
    fig.add_trace(go.Bar(
        x=bin_centers.tolist(),
        y=bin_accuracies_tensor.tolist(),
        name="Accuracy",
        marker=dict(color=color),
        width=width,
        error_y=dict(
            type='data',
            array=bin_errors,  # noqa # type: ignore
            visible=True,
            color='black',
            thickness=1.5,
            width=3
        )
    ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='gray')
    ))

    # Update layout
    fig.update_layout(
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        barmode='overlay',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=5, r=5, t=5, b=5),
    )

    # Save and show plot
    fig.write_image(f"experiment_results/plots/Reliability_Plot_{model_name}.pdf")


def plot_multi_model_reliability(model_results, n_bins=10, error_type='se', color="blue", model_name=""):
    """
    Generates a reliability diagram comparing multiple models, showing the mean
    accuracy and inter-model standard error (or standard deviation) for each bin.

    Args:
        model_results (list of dict): A list, where each dict represents a model's results.
                                      Each dict should have keys:
                                      'y_probs': torch.Tensor (N x num_classes, logits or pre-softmax)
                                      'y_true': torch.Tensor (N, labels)
                                      'name': str (Model name for legend)
                                      'color': str (Plotly color for the model's line)
        n_bins (int): Number of bins for the reliability diagram.
        error_type (str): 'sd' for standard deviation across models, or 'se' for standard error across models.
        plot_title (str): Title for the overall plot.
    """

    if not model_results:
        print("No model results provided to plot.")
        return

    # Collect bin accuracies for each model
    all_models_bin_accuracies = []  # List of tensors, each tensor is bin_accuracies for one model

    # Use the first model's results to determine bin_centers and width, assuming consistent binning
    first_model_y_probs = model_results[0]['y_probs']
    first_model_y_true = model_results[0]['y_true']

    # Ensure first_model_y_probs and first_model_y_true are tensors for get_reliability_bin_data
    if isinstance(first_model_y_probs, list):
        first_model_y_probs = torch.cat(first_model_y_probs, dim=0)
    if isinstance(first_model_y_true, list):
        first_model_y_true = torch.cat(first_model_y_true, dim=0)

    bin_centers, _, _ = get_reliability_bin_data(first_model_y_probs, first_model_y_true, n_bins)
    width = 1.0 / n_bins
    print(
        f"The bin centers are {bin_centers} with a width of {width}. (this should be 0.1 as number of bins is {n_bins})"
    )

    print("How many models do we have? ", len(model_results))
    model_ctr = 0
    for model_data in model_results:
        y_probs = model_data['y_probs']
        y_true = model_data['y_true']
        print("y_probs: ", y_probs)
        print("y_true: ", y_true)

        _, bin_accuracies_for_model, bin_counts_for_model = get_reliability_bin_data(y_probs, y_true, n_bins)
        all_models_bin_accuracies.append(bin_accuracies_for_model)

        print(f"  Model '{model_ctr}' bin accuracies: {bin_accuracies_for_model.tolist()}")
        print(f"  Model '{model_ctr}' bin counts: {bin_counts_for_model.tolist()}")
        model_ctr += 1

    print("\nAll models bin accuracies collected:")
    for i, acc_tensor in enumerate(all_models_bin_accuracies):
        print(f"  Model {i+1}: {acc_tensor.tolist()}")

    # Stack the bin accuracies to easily calculate mean and SE across models
    # This will be a tensor of shape (num_models, n_bins)
    all_models_bin_accuracies_stacked = torch.stack(all_models_bin_accuracies)
    print(f"\nShape of all_models_bin_accuracies_stacked: {all_models_bin_accuracies_stacked.shape}")

    # Calculate mean accuracy across models for each bin
    mean_bin_accuracies_across_models = all_models_bin_accuracies_stacked.mean(dim=0)
    print(f"Mean bin accuracies across models: {mean_bin_accuracies_across_models.tolist()}")

    # Calculate error (SD or SE) across models for each bin
    inter_model_errors = []
    if len(model_results) > 1:  # Need at least 2 models to calculate variance across models
        for i in range(n_bins):
            # Accuracies for bin 'i' across all models
            bin_accuracies_for_this_bin_across_models = all_models_bin_accuracies_stacked[:, i]

            if error_type == 'sd':
                inter_model_errors.append(bin_accuracies_for_this_bin_across_models.std().item())
            elif error_type == 'se':
                # SE = SD / sqrt(N_models)
                se_val = bin_accuracies_for_this_bin_across_models.std().item() / np.sqrt(len(model_results))
                inter_model_errors.append(se_val)
            else:
                inter_model_errors.append(0.0)
    else:  # If only one model, no inter-model variability, so errors are 0
        inter_model_errors = [0.0] * n_bins

    print(f"Inter-model errors ({error_type}): {inter_model_errors}")

    # --- Plotting ---
    fig = go.Figure()

    # Add bars for mean accuracies across models with inter-model error bars
    fig.add_trace(go.Bar(
        x=bin_centers.tolist(),
        y=mean_bin_accuracies_across_models.tolist(),
        name="Mean Accuracy (across models)",
        marker=dict(color=color),  # A neutral color for the aggregated plot
        width=width,
        error_y=dict(
            type='data',
            array=inter_model_errors,
            visible=True,
            color='black',
            thickness=1.5,
            width=3
        )
    ))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(dash='dash', color='black')
    ))

    # Update layout
    fig.update_layout(
        # title=plot_title,
        xaxis_title="Confidence",
        yaxis_title="Accuracy",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        barmode='overlay',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=5, r=5, t=30, b=5),
    )

    # Save plot
    fig.write_image(f"experiment_results/plots/Reliability_Plot_{model_name}.pdf")
