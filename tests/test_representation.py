"""Tests for representation analysis tools."""

import numpy as np

from neusym_bridge.analysis.representation import (
    linear_cka,
    cka_matrix,
    effective_rank,
    spectrum_analysis,
    procrustes_residual,
)


def test_cka_identical():
    """CKA of identical representations should be 1.0."""
    X = np.random.randn(50, 10)
    assert abs(linear_cka(X, X) - 1.0) < 1e-6


def test_cka_random():
    """CKA of unrelated random matrices should be low."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    Y = rng.standard_normal((200, 10))
    assert linear_cka(X, Y) < 0.3


def test_cka_matrix_symmetric():
    """CKA matrix should be symmetric with 1s on diagonal."""
    rng = np.random.default_rng(42)
    Z = {"a": rng.standard_normal((100, 10)), "b": rng.standard_normal((100, 10))}
    M, names = cka_matrix(Z)
    np.testing.assert_allclose(M, M.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(M), 1.0, atol=1e-6)


def test_effective_rank_identity():
    """Effective rank of identity-like data should equal dimensionality."""
    Z = np.eye(10)
    er = effective_rank(Z)
    assert abs(er - 10.0) < 0.5


def test_effective_rank_low_rank():
    """Effective rank of rank-1 data should be ~1."""
    Z = np.outer(np.random.randn(50), np.random.randn(10))
    er = effective_rank(Z)
    assert er < 2.0


def test_spectrum_analysis():
    """Spectrum analysis should return reasonable values."""
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((100, 10))
    result = spectrum_analysis(Z)
    assert result["effective_rank"] > 0
    assert result["n_signal_dims"] >= 0
    assert len(result["top_eigenvalues"]) == 10


def test_procrustes_aligned():
    """Procrustes residual of rotated copy should be ~0."""
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((100, 10))
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    Z_rot = Z @ Q
    residual, _ = procrustes_residual(Z, Z_rot)
    assert residual < 0.01
