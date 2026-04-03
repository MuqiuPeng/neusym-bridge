"""Validate LeWM latent quality on tentacle data.

Reuses Phase 1 methods:
- Effective rank (should be > 5, otherwise representation collapse)
- Linear probes for physics quantities (tension, curvature)

Acceptance criteria:
- Effective rank > 5
- Tension probe R^2 > 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.data.generate_tentacle_data import load_tentacle_dataset
from src.neusym_bridge.analysis.representation import effective_rank


def collect_latents(
    model: LeWMTentacle,
    states: np.ndarray,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Collect latent representations for a set of states.

    Args:
        model: Trained LeWM model.
        states: (N, state_dim) array of states.
        device: Device for inference.
        batch_size: Batch size for inference.

    Returns:
        (N, latent_dim) array of latent vectors.
    """
    model.eval()
    model.to(device)

    loader = DataLoader(
        TensorDataset(torch.tensor(states, dtype=torch.float32)),
        batch_size=batch_size,
    )

    latents = []
    with torch.no_grad():
        for (batch,) in loader:
            z = model.encode(batch.to(device))
            latents.append(z.cpu().numpy())

    return np.concatenate(latents, axis=0)


def extract_physics_labels(states: np.ndarray) -> dict[str, np.ndarray]:
    """Extract physics-meaningful labels from state vectors.

    For probing whether the latent encodes physical information:
    - curvature: per-segment curvature (index 6 of each 7-dim segment)
    - velocity_magnitude: speed of each segment
    - tip_position: position of the last segment (end effector)

    Args:
        states: (N, 140) state vectors.

    Returns:
        Dict mapping label name to (N, label_dim) arrays.
    """
    n = len(states)
    n_segments = 20

    curvatures = np.zeros((n, n_segments))
    velocities = np.zeros((n, n_segments))
    tip_positions = np.zeros((n, 3))

    for i in range(n_segments):
        offset = i * 7
        # Curvature is at index 6 of each segment
        curvatures[:, i] = states[:, offset + 6]
        # Velocity magnitude from indices 3-5
        vx = states[:, offset + 3]
        vy = states[:, offset + 4]
        vz = states[:, offset + 5]
        velocities[:, i] = np.sqrt(vx**2 + vy**2 + vz**2)

    # Tip position: last segment's position (indices 0-2)
    tip_offset = 19 * 7
    tip_positions[:, 0] = states[:, tip_offset]
    tip_positions[:, 1] = states[:, tip_offset + 1]
    tip_positions[:, 2] = states[:, tip_offset + 2]

    return {
        "curvature": curvatures,
        "velocity": velocities,
        "tip_position": tip_positions,
    }


def train_probe(
    Z: np.ndarray,
    labels: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Train a linear probe and return R^2 score.

    Uses Ridge regression with 5-fold cross-validation.

    Args:
        Z: (N, latent_dim) latent representations.
        labels: (N, label_dim) target labels.

    Returns:
        Mean R^2 across folds.
    """
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    # Average R^2 across output dimensions
    r2_scores = []
    for dim in range(labels.shape[1]):
        model = Ridge(alpha=alpha)
        scores = cross_val_score(
            model, Z, labels[:, dim], cv=5, scoring="r2"
        )
        r2_scores.append(scores.mean())

    return float(np.mean(r2_scores))


def validate_lewm_latent(
    model: LeWMTentacle,
    data_path: str | Path = "phase4/data/tentacle_data.h5",
    max_samples: int = 10000,
    device: str = "cpu",
) -> dict:
    """Full validation of LeWM latent quality.

    Checks:
    1. Effective rank > 5 (no collapse)
    2. Curvature probe R^2 > 0.5 (physics information encoded)
    3. Tip position probe R^2 > 0.5 (spatial information encoded)

    Args:
        model: Trained LeWM model.
        data_path: Path to dataset.
        max_samples: Max samples for validation.
        device: Device for inference.

    Returns:
        Dict with validation results.
    """
    print("Loading validation data...")
    s_t, _, _ = load_tentacle_dataset(data_path, max_trajectories=100)

    if len(s_t) > max_samples:
        idx = np.random.choice(len(s_t), max_samples, replace=False)
        s_t = s_t[idx]

    print(f"Validation samples: {len(s_t)}")

    # Collect latents
    print("Collecting latent representations...")
    Z = collect_latents(model, s_t, device=device)

    # 1. Effective rank
    er = effective_rank(Z)
    print(f"Effective rank: {er:.2f}")

    # 2. Extract physics labels
    labels = extract_physics_labels(s_t)

    # 3. Curvature probe
    print("Training curvature probe...")
    r2_curvature = train_probe(Z, labels["curvature"])
    print(f"Curvature probe R^2: {r2_curvature:.3f}")

    # 4. Velocity probe
    print("Training velocity probe...")
    r2_velocity = train_probe(Z, labels["velocity"])
    print(f"Velocity probe R^2: {r2_velocity:.3f}")

    # 5. Tip position probe
    print("Training tip position probe...")
    r2_tip = train_probe(Z, labels["tip_position"])
    print(f"Tip position probe R^2: {r2_tip:.3f}")

    results = {
        "effective_rank": er,
        "r2_curvature": r2_curvature,
        "r2_velocity": r2_velocity,
        "r2_tip_position": r2_tip,
        "er_pass": er > 5,
        "curvature_pass": r2_curvature > 0.5,
        "tip_pass": r2_tip > 0.5,
    }

    # Summary
    passed = sum([results["er_pass"], results["curvature_pass"], results["tip_pass"]])
    print(f"\nLatent validation: {passed}/3 checks passed")
    for check in ["er_pass", "curvature_pass", "tip_pass"]:
        status = "PASS" if results[check] else "FAIL"
        print(f"  [{status}] {check}")

    return results
