"""Training for execution-aware encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exec_aware.model import ExecutionAwareEncoder
from experiments.exec_aware.dataset import ExecAwareDataset


def train_exec_aware(
    dataset: ExecAwareDataset,
    lambda_exec: float = 0.5,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[ExecutionAwareEncoder, dict]:
    """Train execution-aware encoder."""
    model = ExecutionAwareEncoder(lambda_exec=lambda_exec).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"nce": [], "exec": [], "total": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = {"nce": 0.0, "exec": 0.0, "total": 0.0}
        n_batches = 0

        for batch in loader:
            s_t, s_t1, s_neg, s_before, lever_idx, s_after = [
                x.to(device) for x in batch
            ]

            loss, losses = model.compute_loss(
                s_t, s_t1, s_neg, s_before, lever_idx, s_after,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_loss[k] += v
            n_batches += 1

        scheduler.step()

        for k in epoch_loss:
            epoch_loss[k] /= max(n_batches, 1)
            history[k].append(epoch_loss[k])

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d} | "
                  + " | ".join(f"{k}={v:.4f}" for k, v in epoch_loss.items()),
                  flush=True)

    model.eval()
    return model, history


def train_contrastive_baseline(
    dataset: ExecAwareDataset,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[ExecutionAwareEncoder, dict]:
    """Train with lambda_exec=0 (pure contrastive, no execution consistency)."""
    return train_exec_aware(
        dataset, lambda_exec=0.0,
        n_epochs=n_epochs, batch_size=batch_size, lr=lr, device=device,
    )
