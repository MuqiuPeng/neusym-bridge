"""Lever definitions: 8 discrete macro-actions for tentacle control.

Each lever maps to a fixed 80-dim cable tension pattern. The agent
selects levers (not raw tensions), reducing the action space from
continuous R^80 to 8 discrete choices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import (
    N_SEGMENTS, N_CABLES, ACTION_DIM, MAX_TENSION,
    make_tentacle, step as sim_step, extract_state, set_state,
)

LEVERS = [
    "bend_up", "bend_down", "bend_left", "bend_right",
    "twist_cw", "twist_ccw", "extend", "retract",
]


def make_lever_tensions(lever_name: str) -> np.ndarray:
    """Convert lever name to 80-dim cable tension vector.

    Cable layout per segment: [+x, +z, -x, -z].
    """
    t = np.zeros(ACTION_DIM)

    if lever_name == "bend_up":
        for seg in range(N_SEGMENTS):
            t[seg * N_CABLES + 0] = 3.0  # +x cable
    elif lever_name == "bend_down":
        for seg in range(N_SEGMENTS):
            t[seg * N_CABLES + 2] = 3.0  # -x cable
    elif lever_name == "bend_left":
        for seg in range(N_SEGMENTS):
            t[seg * N_CABLES + 3] = 3.0  # -z cable
    elif lever_name == "bend_right":
        for seg in range(N_SEGMENTS):
            t[seg * N_CABLES + 1] = 3.0  # +z cable
    elif lever_name == "twist_cw":
        for seg in range(N_SEGMENTS):
            t[seg * N_CABLES + 0] = 2.0
            t[seg * N_CABLES + 3] = 2.0
    elif lever_name == "twist_ccw":
        for seg in range(N_SEGMENTS):
            t[seg * N_CABLES + 1] = 2.0
            t[seg * N_CABLES + 2] = 2.0
    elif lever_name == "extend":
        for seg in range(N_SEGMENTS):
            for c in range(N_CABLES):
                t[seg * N_CABLES + c] = 1.5
    elif lever_name == "retract":
        pass  # zero tensions — rely on elastic restoring force

    return np.clip(t, 0.0, MAX_TENSION)


def tension_energy(tensions: np.ndarray, n_steps: int = 500, dt: float = 1e-4) -> float:
    """Compute energy as integral of tension magnitudes over time.

    Energy = sum(|T|) * dt * n_steps.  This is a proxy for actuator
    effort: higher tensions cost more energy regardless of whether
    they produce motion (isometric effort counts).
    """
    return float(np.sum(np.abs(tensions)) * dt * n_steps)


def execute_lever(
    env, rod, lever_name: str, n_sim_steps: int = 50,
) -> tuple[np.ndarray, float]:
    """Execute a lever action via the physics simulator.

    Returns:
        (new_state (140,), energy consumed).
    """
    tensions = make_lever_tensions(lever_name)
    new_state, _ = sim_step(env, rod, tensions, dt=1e-4, n_steps=n_sim_steps)
    # Use tension-based energy (the simulator's power calc can be zero
    # for SimplifiedRod because forces are cleared before power measurement)
    energy = tension_energy(tensions, n_sim_steps)
    return new_state, energy


def calibrate_levers(n_states: int = 20) -> dict[str, list[float]]:
    """Measure energy cost of each lever across random initial states."""
    from phase4.envs.tentacle_env import random_valid_state

    costs: dict[str, list[float]] = {lev: [] for lev in LEVERS}

    for i in range(n_states):
        init_state = random_valid_state(seed=i)

        for lever in LEVERS:
            env, rod = make_tentacle()
            set_state(rod, init_state)
            _, energy = execute_lever(env, rod, lever)
            costs[lever].append(energy)

    return costs
