"""Tests for HeatWorldModel and training."""

import torch
import numpy as np

from neusym_bridge.models.baseline_mlp import HeatWorldModel, create_model, SEED_CONFIGS
from neusym_bridge.models.trainer import make_pairs, train_model


def test_model_forward_shapes():
    """Model should produce correct output shapes."""
    model = HeatWorldModel(latent_dim=32)
    T_t = torch.randn(4, 32, 32)
    T_t1 = torch.randn(4, 32, 32)
    z_pred, z_t1, T_t_recon, T_t1_recon = model(T_t, T_t1)
    assert z_pred.shape == (4, 32)
    assert z_t1.shape == (4, 32)
    assert T_t_recon.shape == (4, 32, 32)
    assert T_t1_recon.shape == (4, 32, 32)


def test_encoder_output():
    """Encoder should map 2D fields to latent vectors."""
    model = HeatWorldModel(latent_dim=16)
    x = torch.randn(8, 32, 32)
    z = model.encode(x)
    assert z.shape == (8, 16)


def test_make_pairs():
    """make_pairs should create consecutive frame pairs."""
    traj = torch.randn(3, 5, 32, 32)  # 3 trajectories, 5 steps
    inputs, targets = make_pairs(traj)
    assert inputs.shape == (12, 32, 32)  # 3 * 4 pairs
    assert targets.shape == (12, 32, 32)


def test_deterministic_init():
    """Same seed should produce identical models."""
    m1 = create_model("model_a")
    m2 = create_model("model_a")
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        torch.testing.assert_close(p1, p2)


def test_training_reduces_loss():
    """Training should reduce loss from first to last epoch."""
    torch.manual_seed(42)
    model = HeatWorldModel(latent_dim=8)
    # Small synthetic data
    traj = torch.randn(10, 11, 32, 32)
    history = train_model(model, traj, n_epochs=10, batch_size=32)
    assert history["train_loss"][-1] < history["train_loss"][0]
