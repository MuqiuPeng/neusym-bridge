"""Control experiments for Phase 1 (Task 1.8).

Three controls to rule out confounders:
  A. Random (untrained) models — CKA should be ~0
  B. Cross-task (noise-trained) models — CKA should be low
  C. Overfitted models — CKA should drop vs normal training
"""

from __future__ import annotations

import torch
import numpy as np

from ..models.baseline_mlp import HeatWorldModel, SEED_CONFIGS
from ..models.trainer import train_model
from .latent_collector import collect_latents
from .representation import linear_cka, cka_matrix


def control_random_models(
    test_inputs: torch.Tensor,
    latent_dim: int = 32,
) -> dict:
    """Control A: CKA between untrained random models.

    Expected: avg off-diagonal CKA < 0.1.
    """
    Z = {}
    for name, seed in SEED_CONFIGS.items():
        torch.manual_seed(seed)
        model = HeatWorldModel(latent_dim=latent_dim)
        Z[name] = collect_latents(model, test_inputs)

    M, names = cka_matrix(Z)
    n = len(names)
    off_diag = (M.sum() - np.trace(M)) / (n * n - n)
    return {"cka_matrix": M, "avg_cka": float(off_diag), "names": names}


def control_noise_task(
    test_inputs: torch.Tensor,
    Z_normal: dict[str, np.ndarray],
    n_epochs: int = 50,
    latent_dim: int = 32,
) -> dict:
    """Control B: Model trained on pure noise vs models trained on heat eq.

    Expected: cross-task CKA < 0.3.
    """
    # Generate random noise "trajectories" (no physics)
    noise_traj = torch.randn(200, 101, 32, 32)

    torch.manual_seed(42)
    model_noise = HeatWorldModel(latent_dim=latent_dim)
    train_model(model_noise, noise_traj, n_epochs=n_epochs)

    Z_noise = collect_latents(model_noise, test_inputs)

    cross_cka = {}
    for name, Z_n in Z_normal.items():
        cross_cka[name] = float(linear_cka(Z_n, Z_noise))

    return {"cross_cka": cross_cka, "avg_cross_cka": float(np.mean(list(cross_cka.values())))}


def control_overfit(
    trajectories: torch.Tensor,
    test_inputs: torch.Tensor,
    Z_normal: dict[str, np.ndarray],
    n_epochs_overfit: int = 500,
    latent_dim: int = 32,
) -> dict:
    """Control C: Overfitted model (many epochs) vs normal models.

    Expected: CKA drops compared to normal training.
    """
    # Train one model for many more epochs
    torch.manual_seed(42)
    model_overfit = HeatWorldModel(latent_dim=latent_dim)
    history = train_model(model_overfit, trajectories, n_epochs=n_epochs_overfit)

    Z_overfit = collect_latents(model_overfit, test_inputs)

    overfit_cka = {}
    for name, Z_n in Z_normal.items():
        overfit_cka[name] = float(linear_cka(Z_n, Z_overfit))

    return {
        "overfit_cka": overfit_cka,
        "avg_overfit_cka": float(np.mean(list(overfit_cka.values()))),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
    }
