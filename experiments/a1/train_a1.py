"""Unified training script for A1 ablation experiment.

Trains all three variants with identical conditions except the loss function.
Same data, same architecture backbone, same optimizer, same schedule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from phase4.data.generate_tentacle_data import load_tentacle_dataset


def make_dataset(data_path: str, n_negatives: int = 4, max_trajectories: int | None = None):
    """Load data and prepare tensors including negative samples for contrastive."""
    s_t_np, a_t_np, s_t1_np = load_tentacle_dataset(data_path, max_trajectories=max_trajectories)
    s_t = torch.tensor(s_t_np, dtype=torch.float32)
    a_t = torch.tensor(a_t_np, dtype=torch.float32)
    s_t1 = torch.tensor(s_t1_np, dtype=torch.float32)
    del s_t_np, a_t_np, s_t1_np

    # Pre-generate negative sample indices for contrastive variant
    N = len(s_t)
    neg_idx = torch.stack([
        torch.randint(0, N, (N,)) for _ in range(n_negatives)
    ], dim=1)  # (N, K)

    return s_t, a_t, s_t1, neg_idx


def train_variant(
    variant_name: str,
    model: torch.nn.Module,
    s_t: torch.Tensor,
    a_t: torch.Tensor,
    s_t1: torch.Tensor,
    neg_idx: torch.Tensor,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_split: float = 0.1,
    checkpoint_dir: str = "experiments/a1/checkpoints",
    device: str | None = None,
    seed: int = 42,
) -> tuple[torch.nn.Module, dict]:
    """Train one variant with full logging."""
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Train/val split
    N = len(s_t)
    n_val = int(N * val_split)
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    train_idx, val_idx = perm[n_val:], perm[:n_val]

    train_loader = DataLoader(
        TensorDataset(
            s_t[train_idx], a_t[train_idx], s_t1[train_idx], neg_idx[train_idx],
        ),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            s_t[val_idx], a_t[val_idx], s_t1[val_idx], neg_idx[val_idx],
        ),
        batch_size=batch_size,
    )

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "component_losses": []}

    needs_negatives = variant_name == "contrastive"

    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        comp_sums = {}

        for batch_s, batch_a, batch_s1, batch_neg_idx in train_loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_s1 = batch_s1.to(device)

            kwargs = {}
            if needs_negatives:
                # Gather negative samples from full dataset
                s_neg = s_t[batch_neg_idx.long()].to(device)  # (B, K, state_dim)
                kwargs["s_neg"] = s_neg

            loss, components = model.loss(batch_s, batch_a, batch_s1, **kwargs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_s)
            for k, v in components.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v * len(batch_s)

        scheduler.step()

        n_train = len(train_idx)
        avg_train = total_loss / n_train
        avg_comps = {k: v / n_train for k, v in comp_sums.items()}

        # --- Validate (prediction loss only for fair comparison) ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_s, batch_a, batch_s1, _ in val_loader:
                batch_s = batch_s.to(device)
                batch_a = batch_a.to(device)
                batch_s1 = batch_s1.to(device)
                z_t = model.encode(batch_s)
                z_t1 = model.encode(batch_s1)
                z_pred = model.predict(z_t, batch_a)
                val_loss_sum += ((z_pred - z_t1) ** 2).mean().item() * len(batch_s)

        avg_val = val_loss_sum / len(val_idx)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["component_losses"].append(avg_comps)

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            comp_str = " | ".join(f"{k}={v:.4f}" for k, v in avg_comps.items())
            print(
                f"  [{variant_name}] Epoch {epoch:3d}/{n_epochs} | "
                f"total={avg_train:.4f} {comp_str} val_pred={avg_val:.4f}",
                flush=True,
            )

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            torch.save(
                model.state_dict(),
                ckpt_dir / f"a1_{variant_name}_epoch{epoch:03d}.pt",
            )

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  [{variant_name}] Training complete.")
    return model, history
