"""PCC state extraction and distance metrics.

Extracts PCC (θ, φ) parameters from a rod's current node positions
by fitting constant-curvature arcs to each module's segment positions.
"""

from __future__ import annotations

import numpy as np

from .pcc_env import (
    N_MODULES, SEGS_PER_MOD_ARR, PCC_DIM, THETA_MAX,
    pcc_to_node_positions,
)
from .tentacle_env import N_SEGMENTS, ROD_LENGTH


# ── PCC state extraction ──────────────────────────────────────────────────────

def fit_pcc_module(points: np.ndarray) -> tuple[float, float]:
    """Fit (θ, φ) to a sequence of 3-D points for one PCC module.

    Strategy:
      1. Compute the chord from first to last point.
      2. θ ≈ arc-length / radius; approximate radius from maximum
         lateral deviation.
      3. φ = azimuth of the bending plane from the deviation direction.

    Args:
        points: (SEGS_PER_MOD+1, 3) node positions for this module.

    Returns:
        (theta, phi): bending magnitude [rad] and plane angle [rad].
    """
    # Chord vector
    start = points[0]
    end   = points[-1]
    chord = end - start
    chord_len = np.linalg.norm(chord)

    if chord_len < 1e-9:
        return 0.0, 0.0

    # Tangent at start = direction from first to second point
    t0_vec = points[1] - points[0]
    t0_len = np.linalg.norm(t0_vec)
    if t0_len < 1e-9:
        return 0.0, 0.0
    t0 = t0_vec / t0_len

    # Deviation: largest lateral offset from the straight chord
    lateral_max   = 0.0
    lateral_dir_3d = np.zeros(3)
    for p in points[1:-1]:
        proj = start + np.dot(p - start, chord / (chord_len + 1e-15)) * chord / (chord_len + 1e-15)
        dev  = p - proj
        dev_norm = np.linalg.norm(dev)
        if dev_norm > lateral_max:
            lateral_max   = dev_norm
            lateral_dir_3d = dev / dev_norm

    # Estimate θ from chord geometry: chord = 2R sin(θ/2) → θ = 2 arcsin(chord/(2R))
    # R ≈ chord²/(8·max_dev) for a circular arc
    if lateral_max > 1e-6:
        R     = chord_len ** 2 / (8.0 * lateral_max + 1e-15)
        # arc length ≈ chord for small angles; for larger: arc = R*θ
        # θ from: chord = 2R sin(θ/2)
        ratio = chord_len / (2.0 * R + 1e-15)
        ratio = np.clip(ratio, -1.0, 1.0)
        theta = float(2.0 * np.arcsin(ratio))
    else:
        theta = 0.0

    theta = float(np.clip(theta, 0.0, THETA_MAX))

    # Bending plane angle φ: azimuth of lateral deviation in the x-z plane
    # (We project lateral_dir_3d onto the x-z plane relative to initial tangent)
    # Use global x and z as reference axes (rod starts along y)
    phi = float(np.arctan2(lateral_dir_3d[2], lateral_dir_3d[0]) % (2 * np.pi))

    return theta, phi


def extract_pcc_state(rod, n_modules: int = N_MODULES) -> np.ndarray:
    """Extract PCC state vector from rod's current node positions.

    Args:
        rod:       SimplifiedRod (or CosseratRod).
        n_modules: Number of PCC modules.

    Returns:
        (PCC_DIM,) = [θ₀, φ₀, θ₁, φ₁, …, θ₇, φ₇]
    """
    pos = rod.position_collection.T   # (n_nodes, 3)
    curvatures = np.zeros(PCC_DIM)

    # Use same per-module segment allocation as pcc_to_node_positions
    base_spm = N_SEGMENTS // n_modules
    extra    = N_SEGMENTS % n_modules
    spm_arr  = [base_spm + (1 if m < extra else 0) for m in range(n_modules)]

    node_cursor = 0
    for mod_idx in range(n_modules):
        n_sub      = spm_arr[mod_idx]
        node_start = node_cursor
        node_end   = min(node_start + n_sub + 1, pos.shape[0])
        node_cursor = node_start + n_sub   # next module starts at this node

        points = pos[node_start:node_end]
        if len(points) < 2:
            continue

        theta, phi = fit_pcc_module(points)
        curvatures[mod_idx * 2]     = theta
        curvatures[mod_idx * 2 + 1] = phi

    return curvatures


# ── distance metrics ──────────────────────────────────────────────────────────

def pcc_distance(state_a: np.ndarray, state_b: np.ndarray,
                 w_theta: float = 1.0, w_phi: float = 0.1) -> float:
    """Geometric distance between two PCC states.

    θ-difference is weighted more than φ-difference because θ controls
    the bending magnitude (larger effect on shape) while φ only rotates
    the plane (smaller effect when θ is small).

    Args:
        state_a, state_b: (PCC_DIM,) PCC parameter vectors.
        w_theta: Weight for bending magnitude difference.
        w_phi:   Weight for bending plane difference.

    Returns:
        Scalar distance ≥ 0.
    """
    thetas_a = state_a[0::2]
    phis_a   = state_a[1::2]
    thetas_b = state_b[0::2]
    phis_b   = state_b[1::2]

    # θ: simple absolute difference
    theta_diff = np.abs(thetas_a - thetas_b).mean()

    # φ: angular difference (handles 0/2π wrapping)
    phi_diff = np.abs(
        np.arctan2(np.sin(phis_a - phis_b),
                   np.cos(phis_a - phis_b))
    ).mean()

    return float(w_theta * theta_diff + w_phi * phi_diff)


def pcc_node_distance(state_a: np.ndarray, state_b: np.ndarray) -> float:
    """Tip-position distance between two PCC states (meters).

    Computes node positions for both states and returns the distance
    between their tip nodes (last node).

    Useful as an absolute success criterion.
    """
    pos_a = pcc_to_node_positions(state_a)
    pos_b = pcc_to_node_positions(state_b)
    return float(np.linalg.norm(pos_a[-1] - pos_b[-1]))


# ── task generation ───────────────────────────────────────────────────────────

def random_pcc_state(rng: np.random.Generator | None = None,
                     theta_scale: float = 0.5) -> np.ndarray:
    """Sample a random PCC state.

    Args:
        rng:         NumPy random generator.
        theta_scale: Max θ as fraction of THETA_MAX.

    Returns:
        (PCC_DIM,) PCC parameter vector.
    """
    if rng is None:
        rng = np.random.default_rng()

    state = np.zeros(PCC_DIM)
    for m in range(N_MODULES):
        state[m * 2]     = rng.uniform(0.0, THETA_MAX * theta_scale)
        state[m * 2 + 1] = rng.uniform(0.0, 2.0 * np.pi)

    return state


def generate_pcc_tasks(n_tasks: int = 50,
                       seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (start, target) PCC state pairs for evaluation.

    Starts are always the straight configuration (all zeros).
    Targets are random moderately-bent configurations.

    Returns:
        List of (start_pcc, target_pcc) tuples.
    """
    rng   = np.random.default_rng(seed)
    start = np.zeros(PCC_DIM)         # straight rod
    tasks = []
    for _ in range(n_tasks):
        target = random_pcc_state(rng, theta_scale=0.6)
        tasks.append((start.copy(), target))
    return tasks


def set_pcc_state(rod, curvatures: np.ndarray) -> None:
    """Set rod node positions to match a PCC state (zero velocity).

    Args:
        rod:        SimplifiedRod to modify in-place.
        curvatures: (PCC_DIM,) target PCC parameters.
    """
    target_pos = pcc_to_node_positions(curvatures)   # (n_seg+1, 3)
    n_nodes    = rod.position_collection.shape[1]

    for i in range(min(n_nodes, len(target_pos))):
        rod.position_collection[:, i] = target_pos[i]

    rod.velocity_collection[:] = 0.0
