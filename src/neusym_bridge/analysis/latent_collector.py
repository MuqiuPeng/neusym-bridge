"""Collect latent representations and layer activations from trained models.

Provides hooks-based collection for layer-wise CKA analysis (Task 1.7).
"""

from __future__ import annotations

import torch
import numpy as np
from collections import defaultdict

from ..models.baseline_mlp import HeatWorldModel


def collect_latents(
    model: HeatWorldModel,
    inputs: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """Collect latent vectors for a batch of inputs.

    Args:
        model: Trained HeatWorldModel.
        inputs: Temperature fields, shape (N, grid_size, grid_size).
        device: Device.

    Returns:
        Latent matrix, shape (N, latent_dim).
    """
    model = model.to(device).eval()
    with torch.no_grad():
        z = model.encode(inputs.to(device))
    return z.cpu().numpy()


# Named layers in HeatWorldModel for hook-based collection
LAYER_HOOKS = {
    "conv1": lambda m: m.encoder.conv1,
    "conv2": lambda m: m.encoder.conv2,
    "pool": lambda m: m.encoder.pool,
    "fc_latent": lambda m: m.encoder.fc,
    "pred_hidden1": lambda m: m.predictor.hidden1,
    "pred_hidden2": lambda m: m.predictor.hidden2,
    "pred_output": lambda m: m.predictor.output,
}


def collect_layer_activations(
    model: HeatWorldModel,
    inputs: torch.Tensor,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Collect activations from every named layer via forward hooks.

    Args:
        model: Trained HeatWorldModel.
        inputs: Temperature fields, shape (N, grid_size, grid_size).
        device: Device.

    Returns:
        Dict mapping layer name → activation matrix (N, flattened_dim).
    """
    model = model.to(device).eval()
    activations: dict[str, list] = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name].append(output.detach().cpu())
        return hook_fn

    # Register hooks
    for name, get_layer in LAYER_HOOKS.items():
        h = get_layer(model).register_forward_hook(make_hook(name))
        hooks.append(h)

    # Forward pass (we need to run the predictor too, so use forward)
    with torch.no_grad():
        x = inputs.to(device)
        # Run encoder + predictor to trigger all hooks
        z = model.encode(x)
        model.predictor(z)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Flatten spatial dimensions and concatenate batches
    result = {}
    for name, act_list in activations.items():
        act = torch.cat(act_list, dim=0)
        result[name] = act.reshape(act.shape[0], -1).numpy()

    return result
