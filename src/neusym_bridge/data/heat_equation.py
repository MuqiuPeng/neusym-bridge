"""2D heat equation data generation.

Solves the 2D heat/diffusion equation with Dirichlet boundary conditions
using an explicit finite-difference scheme.  Produces trajectory datasets
stored as HDF5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class HeatConfig:
    grid_size: int = 32
    n_steps: int = 100
    n_trajectories: int = 200
    dt: float = 0.0005
    alpha_choices: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])


def solve_heat_2d(
    T0: np.ndarray,
    alpha: float,
    dt: float,
    n_steps: int,
    bc: np.ndarray,
) -> np.ndarray:
    """Solve 2D heat equation with explicit finite differences.

    Args:
        T0: Initial temperature field, shape (gs, gs).
        alpha: Thermal diffusivity.
        dt: Time step.
        n_steps: Number of time steps.
        bc: Boundary conditions [top, bottom, left, right].

    Returns:
        Trajectory array, shape (n_steps+1, gs, gs).
    """
    gs = T0.shape[0]
    dx = 1.0 / (gs - 1)
    r = alpha * dt / dx ** 2

    if r > 0.25:
        raise ValueError(
            f"CFL condition violated: r={r:.4f} > 0.25. "
            f"Reduce dt or alpha."
        )

    traj = np.zeros((n_steps + 1, gs, gs), dtype=np.float32)
    traj[0] = T0.copy()

    for t in range(n_steps):
        T = traj[t]
        T_new = T.copy()

        # Interior update (explicit 5-point stencil)
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + r * (
            T[2:, 1:-1] + T[:-2, 1:-1]
            + T[1:-1, 2:] + T[1:-1, :-2]
            - 4 * T[1:-1, 1:-1]
        )

        # Enforce Dirichlet BCs
        T_new[0, :] = bc[0]
        T_new[-1, :] = bc[1]
        T_new[:, 0] = bc[2]
        T_new[:, -1] = bc[3]

        traj[t + 1] = T_new

    return traj


def generate_dataset(config: HeatConfig, path: str | Path) -> Path:
    """Generate a dataset of 2D heat equation trajectories.

    Returns:
        Path to the saved HDF5 file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    gs = config.grid_size
    all_traj = []
    all_alpha = []
    all_bc = []

    for i in range(config.n_trajectories):
        alpha = rng.choice(config.alpha_choices)
        bc = rng.uniform(-1, 1, size=4).astype(np.float32)

        T0 = rng.standard_normal((gs, gs)).astype(np.float32)
        # Apply BCs to initial condition
        T0[0, :] = bc[0]
        T0[-1, :] = bc[1]
        T0[:, 0] = bc[2]
        T0[:, -1] = bc[3]

        traj = solve_heat_2d(T0, alpha, config.dt, config.n_steps, bc)
        all_traj.append(traj)
        all_alpha.append(alpha)
        all_bc.append(bc)

    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=np.stack(all_traj))
        f.create_dataset("alphas", data=np.array(all_alpha))
        f.create_dataset("boundary_conditions", data=np.stack(all_bc))
        f.attrs["dt"] = config.dt
        f.attrs["n_steps"] = config.n_steps
        f.attrs["grid_size"] = gs

    return path
