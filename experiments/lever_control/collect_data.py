"""Data collection: random lever exploration + state discretization."""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import (
    make_tentacle, set_state, random_valid_state,
)
from experiments.lever_control.levers import LEVERS, execute_lever


def collect_transitions(
    n_episodes: int = 200,
    levers_per_episode: int = 15,
    seed: int = 0,
) -> list[dict]:
    """Collect transition data by random lever execution.

    Returns list of dicts with keys:
        state_before, lever, state_after, energy.
    """
    rng = np.random.RandomState(seed)
    records = []

    for ep in range(n_episodes):
        env, rod = make_tentacle()
        state = random_valid_state(seed=ep)
        set_state(rod, state)

        for _ in range(levers_per_episode):
            lever = rng.choice(LEVERS)
            new_state, energy = execute_lever(env, rod, lever)

            records.append({
                "state_before": state.copy(),
                "lever": lever,
                "state_after": new_state.copy(),
                "energy": energy,
            })
            state = new_state

        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes "
                  f"({len(records)} transitions)", flush=True)

    return records


def discretize_states(
    encoder: torch.nn.Module,
    records: list[dict],
    k: int = 50,
    device: str = "cpu",
) -> tuple[KMeans, np.ndarray, np.ndarray]:
    """Encode all states and cluster into k discrete nodes.

    Returns:
        (kmeans, node_before_array, node_after_array)
    """
    states_before = np.array([r["state_before"] for r in records])
    states_after = np.array([r["state_after"] for r in records])
    all_states = np.vstack([states_before, states_after])

    encoder.eval().to(device)
    with torch.no_grad():
        Z = encoder.encode(
            torch.tensor(all_states, dtype=torch.float32).to(device)
        ).cpu().numpy()

    print(f"  k-means clustering {len(all_states)} states into {k} nodes...",
          flush=True)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(Z)

    n = len(records)
    labels = kmeans.labels_
    return kmeans, labels[:n], labels[n:]


def build_transition_graph(
    records: list[dict],
    node_before: np.ndarray,
    node_after: np.ndarray,
    min_observations: int = 2,
) -> dict[tuple[int, str, int], list[float]]:
    """Build weighted transition graph from collected data.

    Returns dict mapping (from_node, lever, to_node) -> [energy_list].
    Only includes edges observed at least min_observations times.
    """
    edge_data: dict[tuple[int, str, int], list[float]] = defaultdict(list)

    for i, record in enumerate(records):
        n1 = int(node_before[i])
        n2 = int(node_after[i])
        edge_data[(n1, record["lever"], n2)].append(record["energy"])

    # Filter for reliability
    reliable = {
        k: v for k, v in edge_data.items()
        if len(v) >= min_observations
    }

    print(f"  Total edges: {len(edge_data)}, "
          f"reliable (>={min_observations}x): {len(reliable)}", flush=True)

    return reliable
