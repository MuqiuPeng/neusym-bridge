"""Tests for 2D heat equation data generation."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from neusym_bridge.data.heat_equation import HeatConfig, solve_heat_2d, generate_dataset


def test_cfl_violation():
    """Should raise when CFL condition is violated."""
    T0 = np.ones((8, 8))
    bc = np.zeros(4)
    with pytest.raises(ValueError, match="CFL"):
        solve_heat_2d(T0, alpha=1.0, dt=1.0, n_steps=1, bc=bc)


def test_boundary_conditions_hold():
    """Dirichlet BCs should be maintained at every timestep."""
    gs = 16
    T0 = np.random.randn(gs, gs).astype(np.float32)
    bc = np.array([1.0, -1.0, 0.5, -0.5])
    T0[0, :] = bc[0]
    T0[-1, :] = bc[1]
    T0[:, 0] = bc[2]
    T0[:, -1] = bc[3]

    traj = solve_heat_2d(T0, alpha=0.05, dt=0.01, n_steps=50, bc=bc)

    # Check interior boundary points (excluding corners where BCs overlap)
    for t in range(traj.shape[0]):
        np.testing.assert_allclose(traj[t, 0, 1:-1], bc[0], atol=1e-6)
        np.testing.assert_allclose(traj[t, -1, 1:-1], bc[1], atol=1e-6)
        np.testing.assert_allclose(traj[t, 1:-1, 0], bc[2], atol=1e-6)
        np.testing.assert_allclose(traj[t, 1:-1, -1], bc[3], atol=1e-6)


def test_heat_smoothing():
    """Interior variance should decrease over time (diffusion)."""
    gs = 32
    T0 = np.random.randn(gs, gs).astype(np.float32)
    bc = np.zeros(4)
    T0[0, :] = T0[-1, :] = T0[:, 0] = T0[:, -1] = 0

    traj = solve_heat_2d(T0, alpha=0.1, dt=0.01, n_steps=100, bc=bc)
    # Interior variance should decrease
    var_start = traj[0, 1:-1, 1:-1].var()
    var_end = traj[-1, 1:-1, 1:-1].var()
    assert var_end < var_start


def test_generate_dataset():
    """Should produce valid HDF5 with correct shapes."""
    config = HeatConfig(grid_size=8, n_steps=10, n_trajectories=5)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_dataset(config, Path(tmpdir) / "test.h5")
        import h5py

        with h5py.File(path, "r") as f:
            assert f["trajectories"].shape == (5, 11, 8, 8)
            assert f["alphas"].shape == (5,)
            assert f["boundary_conditions"].shape == (5, 4)
            # Check alpha values are from the choices
            alphas = f["alphas"][:]
            for a in alphas:
                assert any(abs(a - c) < 1e-5 for c in config.alpha_choices)
