"""
Model diagnostic functions for TTOA-NN model

Note:
The model returns utility scores, not probabilities. Choice probabilities are
therefore obtained with softmax and evaluated with log-likelihood. The
model diagnostic functions below replicate the transformations in
ttoann.model.TTOA_NN.forward so that reported diagnostics describe the
model that was actually trained.
"""

import torch
import torch.nn.functional as F

from .utils import evaluate_model, forward_model, move_batch_to_device

#-------------------------------------------------
# Functions to replicate the model's forward pass
#-------------------------------------------------

def _latent_ttoa_effect(model, batch):
    """Return the individual sensitivity tau_n for one full batch"""
    if not model.z_model:
        return F.softplus(model.ttoa).expand(batch["x_taboo"].shape[0], 1)

    z_out = model.z_layers(batch["z"])

    if not model.text_model:
        return F.softplus(z_out)

    text_out = model.text_layers(batch["text"])

    latent = torch.cat(
        [F.layer_norm(z_out, z_out.shape[1:]),
         F.layer_norm(text_out, text_out.shape[1:])],
        dim=-1,
    )

    return F.softplus(model.ttoa_layers(latent))

def _model_components(model, batch):
    """Calculate tau, taboo signal, penalty, and baseline utility"""
    taboo_signal = -torch.sigmoid(model.taboo_layers(batch["x_taboo"]))
    tau = _latent_ttoa_effect(model, batch)
    penalty = tau * taboo_signal

    baseline_utility = (
        model.asu_layers(batch["x_noncost"]["alt1"])
        + model.cost_layers(batch["x_cost"]["alt1"])
    )

    return {
        "tau_n": tau,
        "h_s": taboo_signal,
        "penalty": penalty,
        "V_base": baseline_utility,
    }

#---------------------------------
# Functions for model diagnostics
#---------------------------------

def diagnose_ttoa(model, dataloader, device=None):
    """
    Derive TTOA components and summary statistics
    
    Returns:
    * tau_n: latent TTOA sensitivity for each observation
    * h_s: taboo signal for each observation
    * penalty: TTOA penalty for each observation
    * V_base: baseline utility for each observation
    
    """
    if not model.ttoa_model:
        raise ValueError("TTOA diagnostics require a model with ttoa_model=True")

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    components = {name: [] for name in ("tau_n", "h_s", "penalty", "V_base")}

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            batch_components = _model_components(model, batch)

            for name, values in batch_components.items():
                components[name].append(values.detach().cpu())

    components = {
        name: torch.cat(values, dim=0)
        for name, values in components.items()
    }

    penalty_magnitude = components["penalty"].abs().mean().item()
    baseline_magnitude = components["V_base"].abs().mean().item()

    components["summary"] = {
        "tau_mean": components["tau_n"].mean().item(),
        "tau_std": components["tau_n"].std().item(),
        "h_s_mean": components["h_s"].mean().item(),
        "h_s_std": components["h_s"].std().item(),
        "h_s_negative_fraction": components["h_s"].lt(0).float().mean().item(),
        "penalty_abs_mean": penalty_magnitude,
        "baseline_abs_mean": baseline_magnitude,
        "penalty_to_baseline": penalty_magnitude / (baseline_magnitude + 1e-8),
    }

    return components

def moral_stance_metrics(model, dataloader, device=None):
    """Return masked MSE, RMSE, and R2 for moral-stance predictions"""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    predictions = []
    targets = []
    masks = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            if "y_2" not in batch or "text_mask" not in batch:
                raise ValueError("Moral metrics require y_2 and text_mask in each batch")
            
            outputs = forward_model(model, batch, "choice-ttoa-full")
            predictions.append(outputs["moral"].cpu())
            targets.append(batch["y_2"].cpu())
            masks.append(batch["text_mask"].cpu().bool().squeeze(1))

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    mask = torch.cat(masks)
    predictions = predictions[mask]
    targets = targets[mask]

    if not len(targets):
        raise ValueError("No informative observations were available")

    residual_sum = (predictions - targets).square().sum()
    total_sum = (targets - targets.mean()).square().sum()
    mse = residual_sum / targets.numel()

    return {
        "mse": mse.item(),
        "rmse": mse.sqrt().item(),
        "r2": (1 - residual_sum / total_sum).item(),
        "n_observations": int(mask.sum()),
    }

#-------------------------------------
# Wrapper class for model diagnostics
#-------------------------------------

class measures:
    """
    Wrapper class for model diagnostics
    """

    def __init__(self, model, num_alts, criterion_fn=None, ttoa=False, model_type=None):
        if criterion_fn not in (None, "cross-entropy"):
            raise ValueError("Only 'cross-entropy' is supported")
        
        self.model = model
        self.num_alts = num_alts
        self.ttoa = ttoa
        self.model_type = model_type or "choice"

    def model_fit(self, dataloader, device=None):
        """Evaluate choice predictions and print the resulting metrics"""
        metrics = evaluate_model(
            self.model,
            dataloader,
            device=device,
            model_pred=self.model_type,
            num_alts=self.num_alts,
        )

        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: {value}")

        return metrics

    def diagnose_ttoa(self, dataloader, device=None):
        """Return model-consistent TTOA diagnostics"""
        return diagnose_ttoa(self.model, dataloader, device=device)
