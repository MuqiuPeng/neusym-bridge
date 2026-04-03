"""Training script for LeWM on tentacle data.

Loss = lambda_pred * MSE(z_pred, z_t1) + lambda_recon * MSE(s_recon, s_t)

Reconstruction loss prevents representation collapse (Phase 1 lesson).
Uses cosine annealing LR schedule and gradient clipping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.data.generate_tentacle_data import load_tentacle_dataset


def train(
    data_path: str | Path = "phase4/data/tentacle_data.h5",
    latent_dim: int = 64,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_recon: float = 1.0,
    lambda_pred: float = 1.0,
    val_split: float = 0.1,
    checkpoint_dir: str | Path = "phase4/checkpoints",
    device: str | None = None,
) -> tuple[LeWMTentacle, dict]:
    """Train LeWM on tentacle trajectory data.

    Args:
        data_path: Path to HDF5 dataset.
        latent_dim: Latent space dimensionality.
        n_epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Initial learning rate.
        lambda_recon: Weight for reconstruction loss.
        lambda_pred: Weight for prediction loss.
        val_split: Fraction of data for validation.
        checkpoint_dir: Directory for model checkpoints.
        device: Device string ("cuda"/"cpu"), auto-detected if None.

    Returns:
        Tuple of (trained model, training history dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")

    # Load data
    print(f"Loading data from {data_path}...")
    s_t_np, a_t_np, s_t1_np = load_tentacle_dataset(data_path)
    s_t = torch.tensor(s_t_np, dtype=torch.float32)
    a_t = torch.tensor(a_t_np, dtype=torch.float32)
    s_t1 = torch.tensor(s_t1_np, dtype=torch.float32)
    del s_t_np, a_t_np, s_t1_np  # free numpy copies

    print(f"Data: {len(s_t)} transitions")

    # Train/val split
    n_val = int(len(s_t) * val_split)
    n_train = len(s_t) - n_val
    perm = torch.randperm(len(s_t))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_loader = DataLoader(
        TensorDataset(s_t[train_idx], a_t[train_idx], s_t1[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )
    val_loader = DataLoader(
        TensorDataset(s_t[val_idx], a_t[val_idx], s_t1[val_idx]),
        batch_size=batch_size,
    )

    # Model
    model = LeWMTentacle(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "pred_loss": [],
        "recon_loss": [],
    }

    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        pred_loss_sum = 0.0
        recon_loss_sum = 0.0

        for batch_s, batch_a, batch_s1 in train_loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_s1 = batch_s1.to(device)

            z_pred, z_t1, s_recon, s_orig = model(batch_s, batch_a, batch_s1)

            loss_pred = ((z_pred - z_t1) ** 2).mean()
            loss_recon = ((s_recon - s_orig) ** 2).mean()
            loss = lambda_pred * loss_pred + lambda_recon * loss_recon

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_s)
            pred_loss_sum += loss_pred.item() * len(batch_s)
            recon_loss_sum += loss_recon.item() * len(batch_s)

        scheduler.step()

        train_loss = train_loss_sum / n_train
        pred_loss = pred_loss_sum / n_train
        recon_loss = recon_loss_sum / n_train

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_s, batch_a, batch_s1 in val_loader:
                batch_s = batch_s.to(device)
                batch_a = batch_a.to(device)
                batch_s1 = batch_s1.to(device)
                z_pred, z_t1, _, _ = model(batch_s, batch_a, batch_s1)
                val_loss_sum += ((z_pred - z_t1) ** 2).mean().item() * len(batch_s)

        val_loss = val_loss_sum / n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["pred_loss"].append(pred_loss)
        history["recon_loss"].append(recon_loss)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"total={train_loss:.4f} pred={pred_loss:.4f} "
                f"recon={recon_loss:.4f} val={val_loss:.4f}",
                flush=True,
            )

        # Checkpoint
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"lewm_epoch{epoch:03d}.pt",
            )

    print(f"Training complete. Checkpoints in {checkpoint_dir}/")
    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LeWM on tentacle data")
    parser.add_argument("--data", type=str, default="phase4/data/tentacle_data.h5")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-pred", type=float, default=1.0)
    args = parser.parse_args()

    model, history = train(
        data_path=args.data,
        latent_dim=args.latent_dim,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_recon=args.lambda_recon,
        lambda_pred=args.lambda_pred,
    )
