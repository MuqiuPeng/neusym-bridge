"""Structure extraction: SVCCA, SINDy equation recovery, transfer validation.

Phase 2 tools for extracting physics from common representations.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA


# ---------------------------------------------------------------------------
# Task 2.1: SVCCA — extract common subspace
# ---------------------------------------------------------------------------

def svcca(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    n_components: int = 10,
    variance_threshold: float = 0.99,
) -> dict:
    """Singular Vector CCA.

    Step 1: SVD-reduce each matrix to retain variance_threshold of variance.
    Step 2: CCA in the reduced space to find common directions.

    Args:
        Z_A: Latent matrix from model A, shape (N, d).
        Z_B: Latent matrix from model B, shape (N, d).
        n_components: Max number of canonical components.
        variance_threshold: Cumulative variance to retain in SVD step.

    Returns:
        Dict with keys: V_A, V_B (common directions in original space),
        correlations, n_svd_A, n_svd_B.
    """
    def svd_reduce(Z, threshold):
        Z_centered = Z - Z.mean(axis=0)
        U, s, Vt = np.linalg.svd(Z_centered, full_matrices=False)
        var_ratio = (s ** 2) / (s ** 2).sum()
        cum_var = np.cumsum(var_ratio)
        k = int(np.searchsorted(cum_var, threshold)) + 1
        k = min(k, len(s))
        return U[:, :k], s[:k], Vt[:k, :], k

    U_A, s_A, Vt_A, k_A = svd_reduce(Z_A, variance_threshold)
    U_B, s_B, Vt_B, k_B = svd_reduce(Z_B, variance_threshold)

    n_comp = min(n_components, k_A, k_B)
    cca = CCA(n_components=n_comp)
    cca.fit(U_A, U_B)

    # Canonical correlations
    X_c, Y_c = cca.transform(U_A, U_B)
    correlations = np.array([
        np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        for i in range(n_comp)
    ])

    # Map back to original latent space
    V_A = Vt_A.T @ cca.x_weights_
    V_B = Vt_B.T @ cca.y_weights_

    return {
        "V_A": V_A,
        "V_B": V_B,
        "correlations": correlations,
        "n_svd_A": k_A,
        "n_svd_B": k_B,
        "n_components": n_comp,
    }


def filter_common_directions(
    correlations: np.ndarray,
    threshold: float = 0.7,
) -> np.ndarray:
    """Return boolean mask for significant common directions.

    Args:
        correlations: Canonical correlation coefficients.
        threshold: Minimum correlation to keep.

    Returns:
        Boolean mask, shape (n_components,).
    """
    return correlations > threshold


def build_common_basis(
    V_A: np.ndarray,
    V_B: np.ndarray,
    n_dims: int,
) -> np.ndarray:
    """Average the common directions from two models.

    Args:
        V_A: Common directions from model A, shape (d, n_components).
        V_B: Common directions from model B, shape (d, n_components).
        n_dims: Number of dimensions to keep.

    Returns:
        Averaged common basis, shape (d, n_dims).
    """
    V = (V_A[:, :n_dims] + V_B[:, :n_dims]) / 2
    # Orthogonalize via QR for numerical stability
    Q, _ = np.linalg.qr(V)
    return Q


# ---------------------------------------------------------------------------
# Task 2.2: SINDy equation recovery
# ---------------------------------------------------------------------------

def build_sindy_timeseries(
    Z_all: np.ndarray,
    V_common: np.ndarray,
    dt: float,
    n_traj: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project latent trajectories onto common subspace for SINDy.

    Returns X only (no pre-computed derivatives). SINDy should compute
    derivatives internally with SmoothedFiniteDifference for noise robustness.

    Args:
        Z_all: All latent vectors in trajectory order,
               shape (n_traj * (n_steps+1), latent_dim).
        V_common: Common basis, shape (latent_dim, n_common).
        dt: Time step.
        n_traj: Number of trajectories.
        n_steps: Steps per trajectory (frames = n_steps + 1).

    Returns:
        X: States, shape (N_valid, n_common).
        Xdot: Time derivatives (np.gradient fallback), shape (N_valid, n_common).
    """
    frames = n_steps + 1
    Z_reshaped = Z_all.reshape(n_traj, frames, -1)

    X_list, Xd_list = [], []
    for i in range(n_traj):
        Z_proj = Z_reshaped[i] @ V_common  # (frames, n_common)
        Zd = np.gradient(Z_proj, dt, axis=0)
        X_list.append(Z_proj[1:-1])
        Xd_list.append(Zd[1:-1])

    return np.vstack(X_list), np.vstack(Xd_list)


def build_sindy_trajectories(
    Z_all: np.ndarray,
    V_common: np.ndarray,
    n_traj: int,
    n_steps: int,
) -> list[np.ndarray]:
    """Project latent trajectories for SINDy's multiple_trajectories mode.

    Returns list of per-trajectory arrays for SINDy to differentiate internally.

    Returns:
        List of arrays, each shape (n_steps+1, n_common).
    """
    frames = n_steps + 1
    Z_reshaped = Z_all.reshape(n_traj, frames, -1)
    return [Z_reshaped[i] @ V_common for i in range(n_traj)]


def adaptive_threshold(Xdot: np.ndarray, factor: float = 0.1) -> float:
    """Compute SINDy threshold adapted to the scale of derivatives.

    Args:
        Xdot: Time derivatives array.
        factor: Fraction of mean absolute derivative to use as threshold.

    Returns:
        Threshold value.
    """
    scale = float(np.abs(Xdot).mean())
    return scale * factor


def run_sindy(
    X: np.ndarray,
    Xdot: np.ndarray | None = None,
    threshold: float = 0.05,
    poly_degree: int = 2,
    alpha: float = 0.05,
    dt: float | None = None,
    smooth: bool = False,
):
    """Run SINDy with STLSQ optimizer.

    Args:
        X: State matrix, shape (N, d).
        Xdot: Pre-computed derivatives. If None, SINDy computes them (requires dt).
        threshold: Sparsity threshold for STLSQ.
        poly_degree: Maximum polynomial degree.
        alpha: L2 regularization.
        dt: Time step (used when Xdot is None).
        smooth: Use SmoothedFiniteDifference instead of raw finite diff.

    Returns:
        Tuple of (fitted PySINDy model, R² score).
    """
    import pysindy as ps

    diff_method = (
        ps.SmoothedFiniteDifference(smoother_kws={"window_length": 7})
        if smooth else ps.FiniteDifference()
    )

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=100),
        feature_library=ps.PolynomialLibrary(degree=poly_degree),
        differentiation_method=diff_method,
    )

    if Xdot is not None:
        model.fit(X, x_dot=Xdot)
        score = model.score(X, x_dot=Xdot)
    else:
        model.fit(X, t=dt)
        score = model.score(X, t=dt)
    return model, score


def run_sindy_multi_trajectory(
    trajectories: list[np.ndarray],
    dt: float,
    threshold: float = 0.05,
    poly_degree: int = 2,
    alpha: float = 0.05,
):
    """Run SINDy on multiple trajectories with SmoothedFiniteDifference.

    SINDy handles differentiation internally per trajectory, avoiding
    cross-trajectory derivative artifacts.

    Args:
        trajectories: List of arrays, each (n_steps+1, n_dims).
        dt: Time step.
        threshold: Sparsity threshold.
        poly_degree: Polynomial degree.
        alpha: L2 regularization.

    Returns:
        Tuple of (fitted model, R² score).
    """
    import pysindy as ps

    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=alpha, max_iter=100),
        feature_library=ps.PolynomialLibrary(degree=poly_degree),
        differentiation_method=ps.SmoothedFiniteDifference(
            smoother_kws={"window_length": 7}
        ),
    )
    model.fit(trajectories, t=dt, multiple_trajectories=True)
    score = model.score(trajectories, t=dt, multiple_trajectories=True)
    return model, score


def run_wsindy(
    X: np.ndarray,
    Xdot: np.ndarray | None = None,
    threshold: float = 0.05,
    poly_degree: int = 2,
    dt: float | None = None,
):
    """Run WSINDy with SR3 optimizer + SmoothedFiniteDifference (noise-robust)."""
    import pysindy as ps

    model = ps.SINDy(
        optimizer=ps.SR3(threshold=threshold, thresholder="l1", max_iter=1000),
        feature_library=ps.PolynomialLibrary(degree=poly_degree),
        differentiation_method=ps.SmoothedFiniteDifference(
            smoother_kws={"window_length": 7}
        ),
    )
    if Xdot is not None:
        model.fit(X, x_dot=Xdot)
        score = model.score(X, x_dot=Xdot)
    else:
        model.fit(X, t=dt)
        score = model.score(X, t=dt)
    return model, score


def analyze_sindy_coefficients(sindy_model) -> dict:
    """Analyze SINDy coefficient matrix for physics interpretation.

    Returns dict with sparsity, top terms, and whether linear decay is present.
    """
    coefs = sindy_model.coefficients()
    names = sindy_model.get_feature_names()

    sparsity = float((np.abs(coefs) < 1e-10).mean())

    # Collect all nonzero terms
    nonzero_terms = []
    for i in range(coefs.shape[0]):
        for j in range(coefs.shape[1]):
            if abs(coefs[i, j]) > 1e-10:
                nonzero_terms.append({
                    "equation": i,
                    "feature": names[j],
                    "coefficient": float(coefs[i, j]),
                })
    nonzero_terms.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

    # Check for linear decay: negative diagonal-like terms (z_i in dz_i/dt)
    has_linear_decay = False
    for i in range(coefs.shape[0]):
        var_name = f"x{i}" if coefs.shape[0] > 1 else "x0"
        for j, name in enumerate(names):
            if name == var_name and coefs[i, j] < -1e-10:
                has_linear_decay = True
                break

    return {
        "sparsity": sparsity,
        "n_nonzero": len(nonzero_terms),
        "top_terms": nonzero_terms[:15],
        "has_linear_decay": has_linear_decay,
        "coefficients": coefs.tolist(),
        "feature_names": names,
    }


# ---------------------------------------------------------------------------
# Task 2.3: Transfer test
# ---------------------------------------------------------------------------

def transfer_score(
    sindy_model,
    X_target: np.ndarray,
    Xdot_target: np.ndarray,
) -> float:
    """Score a SINDy model on out-of-distribution data.

    Args:
        sindy_model: SINDy model trained on source domain.
        X_target: States from target domain.
        Xdot_target: Derivatives from target domain.

    Returns:
        R² score on target domain.
    """
    return sindy_model.score(X_target, x_dot=Xdot_target)
