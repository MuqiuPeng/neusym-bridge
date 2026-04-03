"""Two-stage training for the interface layer.

Stage 1 (Supervised warmup):
    Use simulator ground truth (curvature, tension, tip distance)
    as supervision to bootstrap a reasonable initial mapping.

Stage 2 (Consistency fine-tuning):
    Use consistency between LeWM rollout predictions and true
    next-state in the symbolic layer as an unsupervised signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.interface.probe_interface import InterfaceLayer
from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.data.generate_tentacle_data import load_tentacle_dataset
from phase4.envs.tentacle_env import STATE_DIM


def compute_physics_labels(states: np.ndarray) -> np.ndarray:
    """Compute binary physics labels for each predicate from states.

    Labels (per sample):
    - curvature_high:    max curvature across segments > threshold
    - tension_saturated: (approximated from velocity patterns)
    - tip_deviation:     tip far from center (proxy for deviation)

    Thresholds are calibrated to the actual data distribution after
    extract_state normalization (curvature/10, velocity/0.1, pos/rod_length).

    Args:
        states: (N, 140) state vectors.

    Returns:
        (N, 3) binary label array.
    """
    n = len(states)
    labels = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        # Curvature high: max curvature across segments
        # After normalization (curvature/10), values are in [0, ~0.1]
        curvatures = [abs(states[i, seg * 7 + 6]) for seg in range(20)]
        labels[i, 0] = 1.0 if max(curvatures) > 0.095 else 0.0

        # Tension saturated: high velocity magnitude (proxy)
        # After normalization (v/0.1), values range from ~1.4 to ~133
        velocities = []
        for seg in range(20):
            offset = seg * 7 + 3
            v = np.sqrt(sum(states[i, offset + j] ** 2 for j in range(3)))
            velocities.append(v)
        labels[i, 1] = 1.0 if max(velocities) > 60.0 else 0.0

        # Tip deviation: tip position far from resting
        # After normalization (pos/rod_length), lateral values in [0, ~0.2]
        tip_offset = 19 * 7
        tip_x = states[i, tip_offset]
        tip_z = states[i, tip_offset + 2]
        tip_lateral = np.sqrt(tip_x**2 + tip_z**2)
        labels[i, 2] = 1.0 if tip_lateral > 0.05 else 0.0

    return labels


def train_interface(
    lewm_model: LeWMTentacle,
    data_path: str | Path = "phase4/data/tentacle_data.h5",
    latent_dim: int = 64,
    warmup_epochs: int = 10,
    finetune_epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    lambda_sparse: float = 0.01,
    device: str | None = None,
) -> tuple[InterfaceLayer, dict]:
    """Two-stage interface layer training.

    Args:
        lewm_model: Trained (frozen) LeWM model.
        data_path: Path to tentacle dataset.
        latent_dim: Latent space dimensionality.
        warmup_epochs: Epochs for supervised warmup.
        finetune_epochs: Epochs for consistency fine-tuning.
        batch_size: Training batch size.
        lr: Learning rate.
        lambda_sparse: L1 sparsity regularization weight.
        device: Device string, auto-detected if None.

    Returns:
        Tuple of (trained InterfaceLayer, training history).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Freeze LeWM
    lewm_model.eval()
    lewm_model.to(device)
    for param in lewm_model.parameters():
        param.requires_grad = False

    # Load data
    s_t, a_t, s_t1 = load_tentacle_dataset(data_path, max_trajectories=500)
    physics_labels = compute_physics_labels(s_t)

    s_t_t = torch.tensor(s_t, dtype=torch.float32)
    a_t_t = torch.tensor(a_t, dtype=torch.float32)
    s_t1_t = torch.tensor(s_t1, dtype=torch.float32)
    labels_t = torch.tensor(physics_labels, dtype=torch.float32)
    del s_t, a_t, s_t1, physics_labels  # free numpy copies

    # Data loaders
    supervised_loader = DataLoader(
        TensorDataset(s_t_t, labels_t),
        batch_size=batch_size,
        shuffle=True,
    )
    consistency_loader = DataLoader(
        TensorDataset(s_t_t, a_t_t, s_t1_t),
        batch_size=batch_size,
        shuffle=True,
    )

    # Interface layer
    interface = InterfaceLayer(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(interface.parameters(), lr=lr)

    history = {
        "warmup_loss": [],
        "consistency_loss": [],
        "sparse_loss": [],
    }

    # === Stage 1: Supervised warmup ===
    print("Stage 1: Supervised warmup")
    bce = nn.BCELoss()

    for epoch in range(warmup_epochs):
        epoch_loss = 0.0
        for batch_s, batch_labels in supervised_loader:
            batch_s = batch_s.to(device)
            batch_labels = batch_labels.to(device)

            with torch.no_grad():
                z = lewm_model.encode(batch_s)

            confs = interface(z)

            loss_supervised = bce(confs, batch_labels)
            loss_sparse = lambda_sparse * confs.abs().mean()
            loss = loss_supervised + loss_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_s)

        avg_loss = epoch_loss / len(s_t_t)
        history["warmup_loss"].append(avg_loss)
        if epoch % 5 == 0:
            print(f"  Warmup epoch {epoch}: loss={avg_loss:.4f}", flush=True)

    # === Stage 2: Consistency fine-tuning ===
    # Combine consistency with supervised signal to prevent collapse.
    # When consistency loss is near-zero (good predictor), pure sparsity
    # drives all confidences to 0. Retaining BCE anchors the probes.
    print("\nStage 2: Consistency fine-tuning")

    # Build index mapping from consistency_loader samples to labels
    # (both loaders share the same underlying data)
    supervised_iter = iter(supervised_loader)

    for epoch in range(finetune_epochs):
        epoch_cons_loss = 0.0
        epoch_sparse_loss = 0.0
        supervised_iter = iter(supervised_loader)

        for batch_s, batch_a, batch_s1 in consistency_loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_s1 = batch_s1.to(device)

            with torch.no_grad():
                z_t = lewm_model.encode(batch_s)
                z_t1_pred = lewm_model.predict(z_t, batch_a)
                z_t1_true = lewm_model.encode(batch_s1)

            # Interface outputs for predicted and true next latents
            facts_pred = interface(z_t1_pred)
            facts_true = interface(z_t1_true)
            facts_curr = interface(z_t)

            # Consistency: predicted and true next-state should produce
            # similar symbolic confidence distributions
            loss_consistency = nn.MSELoss()(facts_pred, facts_true)

            # Supervised anchor: keep probes aligned with physics labels
            try:
                sup_s, sup_labels = next(supervised_iter)
            except StopIteration:
                supervised_iter = iter(supervised_loader)
                sup_s, sup_labels = next(supervised_iter)
            sup_s = sup_s.to(device)
            sup_labels = sup_labels.to(device)
            with torch.no_grad():
                z_sup = lewm_model.encode(sup_s)
            loss_supervised = bce(interface(z_sup), sup_labels)

            # Sparsity: not all predicates should be active simultaneously
            loss_sparse = lambda_sparse * facts_curr.abs().mean()

            loss = loss_consistency + 0.5 * loss_supervised + loss_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_cons_loss += loss_consistency.item() * len(batch_s)
            epoch_sparse_loss += loss_sparse.item() * len(batch_s)

        avg_cons = epoch_cons_loss / len(s_t_t)
        avg_sparse = epoch_sparse_loss / len(s_t_t)
        history["consistency_loss"].append(avg_cons)
        history["sparse_loss"].append(avg_sparse)

        if epoch % 5 == 0:
            print(
                f"  Finetune epoch {epoch}: "
                f"consistency={avg_cons:.4f} sparse={avg_sparse:.4f}",
                flush=True,
            )

    print("Interface training complete.")
    return interface, history
