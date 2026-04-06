"""Nonlinear heat equation data generation.

Solves a nonlinear variant where thermal diffusivity depends on temperature:
  kappa(T) = kappa_0 * (1 + beta * T^2)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class NonlinearHeatConfig:
    grid_size: int = 32
    n_steps: int = 100
    n_trajectories: int = 100
    dt: float = 0.005
    kappa_0: float = 0.1
    beta: float = 0.5


def solve_nonlinear_heat_2d(
    T0: np.ndarray,
    kappa_0: float,
    beta: float,
    dt: float,
    n_steps: int,
    bc: np.ndarray,
) -> np.ndarray:
    """Solve nonlinear heat equation with temperature-dependent diffusivity."""
    gs = T0.shape[0]
    dx = 1.0 / (gs - 1)

    traj = np.zeros((n_steps + 1, gs, gs), dtype=np.float32)
    traj[0] = T0.copy()

    for t in range(n_steps):
        T = traj[t]
        kappa = kappa_0 * (1 + beta * T ** 2)

        laplacian = (
            np.roll(T, 1, 0) + np.roll(T, -1, 0)
            + np.roll(T, 1, 1) + np.roll(T, -1, 1)
            - 4 * T
        ) / dx ** 2

        T_new = T + dt * kappa * laplacian

        # Enforce BCs
        T_new[0, :] = bc[0]
        T_new[-1, :] = bc[1]
        T_new[:, 0] = bc[2]
        T_new[:, -1] = bc[3]

        traj[t + 1] = T_new

    return traj


def generate_nonlinear_dataset(
    config: NonlinearHeatConfig,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    gs = config.grid_size
    all_traj = []
    all_bc = []

    for i in range(config.n_trajectories):
        bc = rng.uniform(-0.5, 0.5, size=4).astype(np.float32)
        T0 = rng.standard_normal((gs, gs)).astype(np.float32) * 0.5
        T0[0, :] = bc[0]
        T0[-1, :] = bc[1]
        T0[:, 0] = bc[2]
        T0[:, -1] = bc[3]

        traj = solve_nonlinear_heat_2d(
            T0, config.kappa_0, config.beta, config.dt, config.n_steps, bc,
        )
        all_traj.append(traj)
        all_bc.append(bc)

    with h5py.File(path, "w") as f:
        f.create_dataset("trajectories", data=np.stack(all_traj))
        f.create_dataset("boundary_conditions", data=np.stack(all_bc))
        f.attrs["dt"] = config.dt
        f.attrs["n_steps"] = config.n_steps
        f.attrs["grid_size"] = gs
        f.attrs["kappa_0"] = config.kappa_0
        f.attrs["beta"] = config.beta

    return path
