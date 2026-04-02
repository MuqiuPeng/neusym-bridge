"""Tests for Phase 2: structure extraction, SINDy, transfer."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from neusym_bridge.analysis.structure_extraction import (
    svcca,
    filter_common_directions,
    build_common_basis,
    build_sindy_timeseries,
    run_sindy,
    analyze_sindy_coefficients,
)
from neusym_bridge.data.heat_nonlinear import (
    NonlinearHeatConfig,
    solve_nonlinear_heat_2d,
    generate_nonlinear_dataset,
)
from neusym_bridge.analysis.phase2_verdict import (
    sindy_to_relatum,
    format_relatum_prolog,
    phase2_verdict,
)


def test_svcca_identical():
    """SVCCA of identical matrices should give correlations ~1."""
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((100, 16))
    result = svcca(Z, Z, n_components=5)
    assert result["correlations"][0] > 0.99


def test_svcca_random():
    """SVCCA of random matrices should give low correlations."""
    rng = np.random.default_rng(42)
    Z_A = rng.standard_normal((200, 16))
    Z_B = rng.standard_normal((200, 16))
    result = svcca(Z_A, Z_B, n_components=5)
    assert result["correlations"][0] < 0.7  # random CCA can be moderate in small dims


def test_filter_common_directions():
    corrs = np.array([0.95, 0.82, 0.71, 0.45, 0.12])
    mask = filter_common_directions(corrs, threshold=0.7)
    assert mask.sum() == 3


def test_build_common_basis_orthogonal():
    """Common basis should be orthogonal."""
    rng = np.random.default_rng(42)
    V_A = rng.standard_normal((16, 8))
    V_B = rng.standard_normal((16, 8))
    Q = build_common_basis(V_A, V_B, n_dims=4)
    assert Q.shape == (16, 4)
    # Check orthogonality
    np.testing.assert_allclose(Q.T @ Q, np.eye(4), atol=1e-10)


def test_build_sindy_timeseries_shape():
    """SINDy time series should have correct shapes."""
    rng = np.random.default_rng(42)
    n_traj, n_steps, d = 5, 10, 8
    Z_all = rng.standard_normal((n_traj * (n_steps + 1), d))
    V = np.eye(d)[:, :3]  # project to 3 dims
    X, Xd = build_sindy_timeseries(Z_all, V, dt=0.01, n_traj=n_traj, n_steps=n_steps)
    expected_rows = n_traj * (n_steps + 1 - 2)  # skip endpoints per traj
    assert X.shape == (expected_rows, 3)
    assert Xd.shape == (expected_rows, 3)


def test_sindy_linear_system():
    """SINDy should recover a simple linear ODE."""
    # Generate data from dx/dt = -0.5*x
    dt = 0.01
    t = np.arange(0, 5, dt)
    x = np.exp(-0.5 * t).reshape(-1, 1)
    xdot = (-0.5 * x)
    model, score = run_sindy(x, xdot, threshold=0.01)
    assert score > 0.9


def test_nonlinear_heat_runs():
    """Nonlinear heat solver should not crash or produce NaN."""
    gs = 8
    T0 = np.random.randn(gs, gs).astype(np.float32)
    bc = np.zeros(4, dtype=np.float32)
    T0[0, :] = T0[-1, :] = T0[:, 0] = T0[:, -1] = 0
    traj = solve_nonlinear_heat_2d(T0, kappa_0=0.1, beta=0.3, dt=0.01, n_steps=50, bc=bc)
    assert not np.any(np.isnan(traj))
    assert traj.shape == (51, gs, gs)


def test_nonlinear_dataset():
    """Should produce valid HDF5."""
    config = NonlinearHeatConfig(grid_size=8, n_steps=10, n_trajectories=3)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_nonlinear_dataset(config, Path(tmpdir) / "test_nl.h5")
        import h5py
        with h5py.File(path, "r") as f:
            assert f["trajectories"].shape == (3, 11, 8, 8)


def test_sindy_to_relatum():
    """Should convert SINDy analysis to Relatum relations."""
    analysis = {
        "top_terms": [
            {"equation": 0, "feature": "x0", "coefficient": -0.5},
            {"equation": 0, "feature": "x0^2", "coefficient": 0.1},
            {"equation": 0, "feature": "1", "coefficient": 0.001},
        ],
    }
    rels = sindy_to_relatum(analysis, strength_threshold=0.01)
    assert len(rels) == 2  # "1" has coef 0.001, below threshold
    assert rels[0]["name"] == "x0"  # highest strength


def test_phase2_verdict_pass():
    results = {
        "max_correlation": 0.95,
        "score_plan_a": 0.6,
        "score_plan_b": 0.85,
        "has_linear_decay": True,
        "transfer_retention": 0.75,
        "intervention_systematic": True,
        "n_relations": 5,
    }
    v = phase2_verdict(results)
    assert v["overall_pass"] is True
    assert v["passed_count"] >= 4
