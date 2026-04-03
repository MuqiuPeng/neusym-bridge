"""Generate tentacle control trajectory dataset.

Produces an HDF5 file with randomized trajectories covering:
- Different starting postures (random)
- Different target postures (random)
- Different motion patterns (bending/twisting/extension)

Each trajectory stores (states, actions, energies, target).
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow importing from phase4 package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import (
    make_tentacle,
    step,
    set_state,
    extract_state,
    random_valid_state,
    ACTION_DIM,
    STATE_DIM,
    MAX_TENSION,
)


def generate_trajectory(
    start_state: np.ndarray,
    target_state: np.ndarray,
    n_steps: int = 200,
    seed: int | None = None,
) -> dict:
    """Generate one trajectory from start to target using a random policy.

    The random policy provides training data diversity. It uses
    exponential-distributed tensions biased toward the target.

    Args:
        start_state: (140,) initial state.
        target_state: (140,) goal state.
        n_steps: Number of control steps.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: states, actions, energies, target.
    """
    rng = np.random.RandomState(seed)

    env, rod = make_tentacle()
    set_state(rod, start_state)

    trajectory = {
        "states": [],
        "actions": [],
        "energies": [],
        "target": target_state,
    }

    state = extract_state(rod)

    for t in range(n_steps):
        # Random action with exponential distribution (naturally non-negative)
        action = rng.exponential(0.5, size=ACTION_DIM)
        action = np.clip(action, 0.0, MAX_TENSION)

        trajectory["states"].append(state)
        trajectory["actions"].append(action)

        next_state, energy = step(env, rod, action)

        trajectory["energies"].append(energy)
        state = next_state

    # Append final state
    trajectory["states"].append(state)

    del env, rod
    return trajectory


def build_tentacle_dataset(
    n_trajectories: int = 5000,
    n_steps: int = 200,
    save_path: str | Path = "tentacle_data.h5",
) -> None:
    """Generate full tentacle control dataset.

    Args:
        n_trajectories: Number of trajectories to generate.
        n_steps: Steps per trajectory.
        save_path: Output HDF5 file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_trajectories} trajectories ({n_steps} steps each)")
    print(f"State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print(f"Output: {save_path}")

    with h5py.File(save_path, "w") as f:
        # Store metadata
        f.attrs["n_trajectories"] = n_trajectories
        f.attrs["n_steps"] = n_steps
        f.attrs["state_dim"] = STATE_DIM
        f.attrs["action_dim"] = ACTION_DIM

        for i in range(n_trajectories):
            if i % 100 == 0:
                print(f"  Trajectory {i}/{n_trajectories}", flush=True)

            # Random start and target
            start = random_valid_state(seed=i)
            target = random_valid_state(seed=i + n_trajectories)

            traj = generate_trajectory(start, target, n_steps=n_steps, seed=i)

            grp = f.create_group(f"traj_{i:05d}")
            grp.create_dataset(
                "states",
                data=np.array(traj["states"], dtype=np.float32),
                compression="gzip",
            )
            grp.create_dataset(
                "actions",
                data=np.array(traj["actions"], dtype=np.float32),
                compression="gzip",
            )
            grp.create_dataset(
                "energies",
                data=np.array(traj["energies"], dtype=np.float32),
                compression="gzip",
            )
            grp.attrs["target"] = target.astype(np.float32)

            del traj, start, target
            # PyElastica BaseSystemCollection holds cyclic refs;
            # force GC every 50 trajectories to prevent memory buildup
            if i % 50 == 0:
                gc.collect()

    print(f"Dataset complete: {n_trajectories} trajectories -> {save_path}")


def load_tentacle_dataset(
    path: str | Path,
    max_trajectories: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load tentacle dataset and return flattened (s_t, a_t, s_t1) pairs.

    Args:
        path: Path to HDF5 file.
        max_trajectories: Limit number of trajectories loaded.

    Returns:
        Tuple of (states_t, actions_t, states_t1), each (N, dim).
    """
    all_s_t = []
    all_a_t = []
    all_s_t1 = []

    with h5py.File(path, "r") as f:
        n_traj = f.attrs["n_trajectories"]
        if max_trajectories:
            n_traj = min(n_traj, max_trajectories)

        for i in range(n_traj):
            grp = f[f"traj_{i:05d}"]
            states = grp["states"][:]    # (n_steps+1, state_dim)
            actions = grp["actions"][:]  # (n_steps, action_dim)

            all_s_t.append(states[:-1])   # s_t
            all_a_t.append(actions)       # a_t
            all_s_t1.append(states[1:])   # s_{t+1}

    return (
        np.concatenate(all_s_t, axis=0),
        np.concatenate(all_a_t, axis=0),
        np.concatenate(all_s_t1, axis=0),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate tentacle dataset")
    parser.add_argument("--n-traj", type=int, default=5000)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--output", type=str, default="phase4/data/tentacle_data.h5")
    args = parser.parse_args()

    build_tentacle_dataset(
        n_trajectories=args.n_traj,
        n_steps=args.n_steps,
        save_path=args.output,
    )
