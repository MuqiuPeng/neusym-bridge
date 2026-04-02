"""Training loop for HeatWorldModel with reconstruction + prediction loss."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from .baseline_mlp import HeatWorldModel


def make_pairs(trajectories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert trajectories into (T_t, T_{t+1}) pairs.

    Args:
        trajectories: Shape (n_traj, n_steps+1, grid_size, grid_size).

    Returns:
        inputs: Shape (N, grid_size, grid_size).
        targets: Shape (N, grid_size, grid_size).
    """
    inputs = trajectories[:, :-1].reshape(-1, *trajectories.shape[2:])
    targets = trajectories[:, 1:].reshape(-1, *trajectories.shape[2:])
    return inputs, targets


def train_model(
    model: HeatWorldModel,
    trajectories: torch.Tensor,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    recon_weight: float = 1.0,
    pred_weight: float = 1.0,
    device: str = "cpu",
) -> dict:
    """Train a HeatWorldModel with reconstruction + prediction loss.

    Loss = recon_weight * MSE(T, T_recon) + pred_weight * MSE(z_pred, z_t1)

    Reconstruction loss prevents representation collapse by forcing the
    latent space to encode enough information to reconstruct the input.
    """
    model = model.to(device)
    inputs, targets = make_pairs(trajectories)

    # Train/val split
    n_val = int(len(inputs) * val_split)
    n_train = len(inputs) - n_val
    perm = torch.randperm(len(inputs))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_loader = DataLoader(
        TensorDataset(inputs[train_idx], targets[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(inputs[val_idx], targets[val_idx]),
        batch_size=batch_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "pred_loss": []}

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        recon_loss_sum = 0.0
        pred_loss_sum = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            z_pred, z_t1, x_recon, y_recon = model(x, y)

            # Prediction loss in latent space
            loss_pred = ((z_pred - z_t1) ** 2).mean()

            # Reconstruction loss (both frames)
            loss_recon = ((x - x_recon) ** 2).mean() + ((y - y_recon) ** 2).mean()

            loss = pred_weight * loss_pred + recon_weight * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(x)
            recon_loss_sum += loss_recon.item() * len(x)
            pred_loss_sum += loss_pred.item() * len(x)

        train_loss = train_loss_sum / n_train
        recon_loss = recon_loss_sum / n_train
        pred_loss = pred_loss_sum / n_train

        # Validate (prediction loss only for clean comparison)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                z_pred, z_t1, _, _ = model(x, y)
                val_loss_sum += ((z_pred - z_t1) ** 2).mean().item() * len(x)
        val_loss = val_loss_sum / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["recon_loss"].append(recon_loss)
        history["pred_loss"].append(pred_loss)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}/{n_epochs} | total={train_loss:.6f} "
                  f"recon={recon_loss:.6f} pred={pred_loss:.6f} val={val_loss:.6f}",
                  flush=True)

    return history


def save_model(model: HeatWorldModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, latent_dim: int = 32) -> HeatWorldModel:
    model = HeatWorldModel(latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
