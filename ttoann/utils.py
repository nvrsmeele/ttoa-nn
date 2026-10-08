"""
Helper functions for training and evaluating the TTOA-NN model
"""

from contextlib import nullcontext
import copy
import random

import numpy as np
import torch
from tqdm import tqdm


model_modes = {
    "choice": (False, False, False),
    "choice-ttoa-z": (True, True, False),
    "choice-ttoa-full": (True, True, True),
}


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch for a repeatable experiment"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Select CUDA, Apple Silicon MPS, or CPU in that order"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch, device):
    """Move tensors in a dataset batch, including nested alternative inputs"""
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved_batch[key] = {
                name: tensor.to(device) for name, tensor in value.items()
            }
        else:
            moved_batch[key] = value.to(device)
    return moved_batch


def forward_model(model, batch, model_pred):
    """Run the model using the input fields required by model_pred"""
    if model_pred not in model_modes:
        raise ValueError(f"Unknown model_pred: {model_pred}")

    inputs = [batch["x_noncost"], batch["x_cost"]]
    if model_pred != "choice":
        inputs.extend([batch["x_taboo"], batch["z"]])
    if model_pred == "choice-ttoa-full":
        inputs.extend([batch["text"], batch["text_mask"]])
    return model(*inputs)


def _moral_loss(outputs, batch, alpha):
    """Compute masked moral-stance MSE and its weighted contribution"""
    if alpha <= 0 or "moral" not in outputs or "y_2" not in batch:
        return outputs["choice"].new_zeros(()), outputs["choice"].new_zeros(())

    squared_error = (outputs["moral"] - batch["y_2"]) ** 2
    mask = batch["text_mask"].eq(1).expand_as(squared_error)
    masked_error = squared_error * mask
    moral_mse = masked_error.sum() / mask.sum().clamp_min(1)
    return moral_mse, alpha * moral_mse


def _choice_counts(dataloader, num_alts, device):
    """Count observed choices for the empirical-share null model"""
    counts = torch.zeros(num_alts, device=device)
    for batch in dataloader:
        choices = batch["y"].to(device)
        counts.scatter_add_(0, choices, torch.ones_like(choices, dtype=torch.float))
    return counts


def evaluate_model(model, dataloader, device=None, model_pred="choice-ttoa-full", num_alts=2):
    """Evaluate log-likelihood, McFadden rho-squared, and moral MSE

    The null model uses empirical choice shares from dataloader. Moral
    stance error is calculated only for observations marked informative by the
    dataset's text_mask
    """
    device = device or get_device()
    model.eval()
    choice_counts = _choice_counts(dataloader, num_alts, device)
    shares = choice_counts / choice_counts.sum().clamp_min(1)

    log_likelihood = 0.0
    null_log_likelihood = 0.0
    moral_squared_error = 0.0
    moral_count = 0
    n_observations = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = forward_model(model, batch, model_pred)
            choices = batch["y"]
            log_probs = torch.log_softmax(outputs["choice"], dim=1)
            log_likelihood += log_probs.gather(1, choices.unsqueeze(1)).sum().item()
            null_log_likelihood += torch.log(shares[choices].clamp_min(1e-15)).sum().item()
            n_observations += choices.numel()

            if "moral" in outputs and "y_2" in batch:
                mask = batch["text_mask"].eq(1).expand_as(batch["y_2"])
                moral_squared_error += (
                    (outputs["moral"] - batch["y_2"]).square() * mask
                ).sum().item()
                moral_count += mask.sum().item()

    metrics = {
        "ll": log_likelihood,
        "null_ll": null_log_likelihood,
        "rho_squared": 1 - log_likelihood / null_log_likelihood,
        "n_observations": n_observations,
    }
    if moral_count:
        metrics["moral_mse"] = moral_squared_error / moral_count
    return metrics


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    device=None,
    model_pred="choice-ttoa-full",
    num_epochs=100,
    alpha=0.0,
    patience=15,
    gradient_clip=5.0,
    scheduler=None,
    checkpoint_path=None,
    min_epochs=1,
    use_amp=True,
    verbose=True,
):
    """
    Train one model with validation-based early stopping
    """

    device = device or get_device()
    model.to(device)

    amp_enabled = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    criterion = torch.nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_metrics = None
    best_ll = float("-inf")
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        choice_loss_total = 0.0
        moral_loss_total = 0.0
        batches = 0
        train_iterator = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}/{num_epochs}",
            disable=not verbose,
        )

        for batch in train_iterator:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_context = torch.cuda.amp.autocast() if amp_enabled else nullcontext()

            with autocast_context:
                outputs = forward_model(model, batch, model_pred)
                choice_loss = criterion(outputs["choice"], batch["y"])
                moral_mse, weighted_moral_loss = _moral_loss(outputs, batch, alpha)
                loss = choice_loss + weighted_moral_loss

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            batches += 1
            total_loss += loss.item()
            choice_loss_total += choice_loss.item()
            moral_loss_total += moral_mse.item()

        metrics = evaluate_model(
            model,
            val_dataloader,
            device=device,
            model_pred=model_pred,
            num_alts=2,
        )

        if scheduler is not None:
            scheduler.step(metrics["ll"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "train_choice_loss": choice_loss_total / max(batches, 1),
            "train_moral_mse": moral_loss_total / max(batches, 1),
            **metrics,
        }

        history.append(epoch_record)

        if verbose:
            print(
                f"epoch {epoch:03d} | loss {epoch_record['train_loss']:.4f} | "
                f"val LL {metrics['ll']:.2f} | rho^2 {metrics['rho_squared']:.4f}"
            )

        if metrics["ll"] > best_ll:
            best_ll = metrics["ll"]
            best_metrics = metrics.copy()
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience and epoch >= min_epochs:
            if verbose:
                print(f"Early stopping after epoch {epoch}.")
            break

    model.load_state_dict(best_state)

    return {"history": history, "best_metrics": best_metrics, "model": model}
