"""Representation analysis: CKA, effective rank, Procrustes alignment.

Phase 1 tools for measuring common structure across models.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment (linear) between two representation matrices.

    Uses centered Gram matrices for unbiased comparison.

    Args:
        X: Representation matrix, shape (n_samples, d1).
        Y: Representation matrix, shape (n_samples, d2).

    Returns:
        CKA similarity in [0, 1].
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtX = X @ X.T
    YtY = Y @ Y.T

    numerator = (XtX * YtY).sum()
    denominator = np.sqrt((XtX * XtX).sum() * (YtY * YtY).sum())

    return float(numerator / (denominator + 1e-8))


def cka_matrix(Z_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise CKA matrix for all model representations.

    Args:
        Z_dict: Mapping from model name to representation matrix.

    Returns:
        Tuple of (CKA matrix, list of model names).
    """
    names = list(Z_dict.keys())
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = linear_cka(Z_dict[names[i]], Z_dict[names[j]])
    return M, names


def effective_rank(Z: np.ndarray) -> float:
    """Effective rank via entropy of normalized singular values.

    Args:
        Z: Representation matrix, shape (n_samples, d).

    Returns:
        Effective rank (continuous measure of dimensionality).
    """
    _, s, _ = np.linalg.svd(Z, full_matrices=False)
    s = s / s.sum()
    return float(np.exp(-np.sum(s * np.log(s + 1e-8))))


def mp_threshold(n_samples: int, n_features: int, sigma: float = 1.0) -> float:
    """Marchenko-Pastur upper edge for noise singular values.

    Eigenvalues above this threshold correspond to signal.

    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        sigma: Noise standard deviation.

    Returns:
        Threshold eigenvalue.
    """
    ratio = n_features / n_samples
    return sigma**2 * (1 + np.sqrt(ratio)) ** 2


def spectrum_analysis(Z: np.ndarray) -> dict:
    """Full spectral analysis of a representation matrix.

    Returns effective rank, signal dimensions (above MP threshold),
    and top eigenvalues.
    """
    n, d = Z.shape
    _, s, _ = np.linalg.svd(Z, full_matrices=False)
    eigenvalues = s**2 / n

    threshold = mp_threshold(n, d)
    n_signal = int((eigenvalues > threshold).sum())
    er = effective_rank(Z)

    return {
        "effective_rank": er,
        "mp_threshold": threshold,
        "n_signal_dims": n_signal,
        "top_eigenvalues": eigenvalues[:10].tolist(),
    }


def procrustes_residual(Z_A: np.ndarray, Z_B: np.ndarray) -> tuple[float, np.ndarray]:
    """Procrustes alignment residual between two representations.

    Normalizes both matrices by Frobenius norm before alignment.

    Args:
        Z_A: Source representation, shape (n_samples, d).
        Z_B: Target representation, shape (n_samples, d).

    Returns:
        Tuple of (normalized residual, rotation matrix R).
    """
    # Normalize for scale-invariant comparison
    Z_A_n = Z_A / (np.linalg.norm(Z_A, "fro") + 1e-8)
    Z_B_n = Z_B / (np.linalg.norm(Z_B, "fro") + 1e-8)

    R, _ = orthogonal_procrustes(Z_A_n, Z_B_n)
    residual = float(np.linalg.norm(Z_A_n @ R - Z_B_n, "fro"))
    return residual, R
