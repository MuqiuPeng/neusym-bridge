"""PCC (Piecewise Constant Curvature) environment for continuum robot control.

Action space: 16-dim PCC parameters (8 modules × (θ_bend, φ_plane))
  θ ∈ [0, θ_max]  — bending magnitude per module
  φ ∈ [0, 2π]     — bending plane angle

Advantages over 80-dim tension space:
  - Direct geometric meaning (module curvatures)
  - Planning in 16-dim instead of 80-dim
  - Distance metric has physical interpretation
  - No encoder needed for state representation
"""

from __future__ import annotations

import numpy as np

from .tentacle_env import (
    N_SEGMENTS, N_CABLES, ROD_LENGTH, MAX_TENSION,
    extract_state,
)
from .cable_geometry import cable_direction

# ── PCC parameters ─────────────────────────────────────────────────────────────
N_MODULES    = 8
PCC_DIM      = N_MODULES * 2             # 16: (θ, φ) per module
THETA_MAX    = np.pi * 0.8              # max bend per module [rad]

# Distribute N_SEGMENTS (20) over N_MODULES (8) evenly:
# first (20 % 8 = 4) modules get 3 segs, rest get 2 → 4×3 + 4×2 = 20 ✓
_EXTRA       = N_SEGMENTS % N_MODULES   # 4
_BASE_SPM    = N_SEGMENTS // N_MODULES  # 2
SEGS_PER_MOD_ARR = np.array(
    [_BASE_SPM + (1 if m < _EXTRA else 0) for m in range(N_MODULES)],
    dtype=int,
)  # [3,3,3,3,2,2,2,2]
SEGS_PER_MOD = int(SEGS_PER_MOD_ARR.mean())   # kept for compat (≈2)


# ── forward kinematics ────────────────────────────────────────────────────────

def pcc_to_node_positions(curvatures: np.ndarray,
                           rod_length: float = ROD_LENGTH,
                           n_segments: int = N_SEGMENTS,
                           n_modules: int = N_MODULES) -> np.ndarray:
    """PCC parameters → node positions (3-D).

    Args:
        curvatures: (PCC_DIM,) = [theta0, phi0, theta1, phi1, ..., theta7, phi7]

    Returns:
        (n_segments+1, 3) node positions; node 0 is the fixed base.
    """
    # Per-module segment counts (handles non-divisible N_SEGMENTS / N_MODULES)
    base_spm = n_segments // n_modules
    extra    = n_segments % n_modules
    spm_arr  = [base_spm + (1 if m < extra else 0) for m in range(n_modules)]

    seg_len  = rod_length / n_segments

    positions = [np.zeros(3)]
    # Frame: columns are [d1, d2, d3]; d3 = tangent direction (starts as +y)
    frame = np.eye(3)          # column 0=x, 1=y(tangent), 2=z

    for mod_idx in range(n_modules):
        theta = float(curvatures[mod_idx * 2])
        phi   = float(curvatures[mod_idx * 2 + 1])

        # Bending direction in the cross-section plane
        # phi=0 → bend toward +x; phi=pi/2 → bend toward +z
        bend_local = np.array([np.cos(phi), 0.0, np.sin(phi)])  # local x-z plane
        bend_world = frame @ bend_local                           # world frame

        n_sub = spm_arr[mod_idx]   # segments in this module (2 or 3)
        for _ in range(n_sub):
            if theta < 1e-6:
                # Straight extension along current tangent
                tip = positions[-1] + seg_len * frame[:, 1]
            else:
                # Apply theta/n_sub arc per sub-segment
                arc_angle = theta / n_sub
                radius    = seg_len / arc_angle if arc_angle > 1e-9 else 1e9

                # Arc in bending plane: move along tangent + bend toward bend_world
                tangent  = frame[:, 1]
                tip = (positions[-1]
                       + radius * np.sin(arc_angle) * tangent
                       + radius * (1.0 - np.cos(arc_angle)) * bend_world)

                # Rotate frame: rotate d3(tangent) toward bend_world
                # Rodrigues rotation around axis = tangent × bend_world
                rot_axis = np.cross(tangent, bend_world)
                rot_norm = np.linalg.norm(rot_axis)
                if rot_norm > 1e-9:
                    rot_axis /= rot_norm
                    c, s = np.cos(arc_angle), np.sin(arc_angle)
                    K = np.array([[      0, -rot_axis[2],  rot_axis[1]],
                                  [ rot_axis[2],        0, -rot_axis[0]],
                                  [-rot_axis[1],  rot_axis[0],       0]])
                    R = c * np.eye(3) + s * K + (1 - c) * np.outer(rot_axis, rot_axis)
                    frame = R @ frame
                    # Re-ortho-normalise to prevent drift
                    frame[:, 1] /= (np.linalg.norm(frame[:, 1]) + 1e-15)
                    frame[:, 0] -= np.dot(frame[:, 0], frame[:, 1]) * frame[:, 1]
                    frame[:, 0] /= (np.linalg.norm(frame[:, 0]) + 1e-15)
                    frame[:, 2] = np.cross(frame[:, 0], frame[:, 1])

            positions.append(tip)

    return np.array(positions)   # (n_segments+1, 3)


# ── inverse kinematics: curvatures → tensions ─────────────────────────────────

def pcc_to_tensions(curvatures: np.ndarray, rod,
                    stiffness: float = 120.0) -> np.ndarray:
    """Convert PCC parameters to cable tensions via proportional IK.

    Computes target node positions from PCC, then distributes tension
    proportionally to move each segment toward its target.

    Args:
        curvatures: (PCC_DIM,) PCC parameters.
        rod:        Current SimplifiedRod (for reading current positions).
        stiffness:  Proportional gain [N/m].

    Returns:
        (N_SEGMENTS * N_CABLES,) tension vector, clipped to [0, MAX_TENSION].
    """
    target_pos = pcc_to_node_positions(curvatures)   # (n_seg+1, 3)
    current_pos = rod.position_collection.T           # (n_nodes, 3)

    tensions = np.zeros(N_SEGMENTS * N_CABLES)

    for seg in range(N_SEGMENTS):
        # Error vector: move node seg+1 toward target
        err = target_pos[seg + 1] - current_pos[min(seg + 1, len(current_pos) - 1)]
        for cable in range(N_CABLES):
            d = cable_direction(seg, cable)           # (3,) unit pull direction
            # Positive tension when cable pull direction aligns with error
            t = stiffness * np.dot(err, d)
            tensions[seg * N_CABLES + cable] = max(0.0, t)

    return np.clip(tensions, 0.0, MAX_TENSION)


# ── direct-kinematics PCC step ────────────────────────────────────────────────
#
# Design rationale
# ----------------
# The original approach (IK proportional control + physics integration) fails
# because the SimplifiedRod elastic equilibrium does not match the PCC target:
# the elastic restoring force opposes cable forces, so the rod only reaches a
# fraction of the desired curvature.
#
# With high damping (zeta >> 1) the system is quasi-static: the rod snaps
# instantly to the equilibrium determined by the applied cable tensions.  For a
# real cable-driven robot the low-level controller maps desired curvatures to
# cable lengths, making the mapping exact.  We model this ideal controller as
# direct forward kinematics: just set the node positions to the PCC target.
#
# This is consistent with the stated goal:
#     "规划问题退化为纯运动学"  (planning reduces to pure kinematics)
#
# Energy proxy
# ------------
# Real cable work ∝ Σ |θ_i| (bending angle) × EI / L_module.
# We use E = Σ |θ_i| (sum of bending magnitudes) as a dimensionless proxy,
# normalised later by _MAX_PCC_ENERGY in the planner.

def step_pcc(rod,
             curvatures: np.ndarray,
             **_ignored) -> tuple[np.ndarray, float, int]:
    """Execute a PCC action via direct kinematics (quasi-static ideal control).

    Sets rod node positions to match ``curvatures`` exactly (forward kinematics)
    and returns the energy proxy = sum of absolute bending angles.

    Args:
        rod:        SimplifiedRod (positions updated in-place).
        curvatures: (PCC_DIM,) target PCC parameters [theta0, phi0, ...].

    Returns:
        (new_state (140,), energy_proxy, 0)
        energy_proxy = sum(|theta_i|) over all modules.
    """
    from .pcc_state import set_pcc_state

    set_pcc_state(rod, curvatures)
    rod.velocity_collection[:] = 0.0   # enforce quasi-static (v=0)

    # Energy proxy: total bending work proportional to Σ|θ_i|
    energy = float(np.sum(np.abs(curvatures[0::2])))

    new_state = extract_state(rod)
    return new_state, energy, 0
