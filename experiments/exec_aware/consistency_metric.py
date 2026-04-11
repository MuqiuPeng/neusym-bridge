"""Execution consistency measurement: the core diagnostic metric.

For each (state, lever) pair, compare:
  - predicted node (from transition graph)
  - actual node (after physical execution and re-encoding)

Consistency rate = fraction where predicted == actual.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import make_tentacle, set_state, random_valid_state
from experiments.lever_control.levers import LEVERS, execute_lever
from experiments.lever_control.planner import state_to_node


def measure_execution_consistency(
    encoder: torch.nn.Module,
    kmeans,
    graph: dict,
    n_states: int = 100,
    device: str = "cpu",
) -> dict:
    """Measure how often physical execution matches graph prediction.

    Args:
        encoder: Trained encoder with .encode() method.
        kmeans: Fitted KMeans for state discretization.
        graph: Transition graph {(n1, lever, n2): [energies]}.
        n_states: Number of random states to test.

    Returns:
        Dict with consistency_rate, drift stats.
    """
    # Build lookup: (node, lever) -> most common target node
    predicted_target: dict[tuple[int, str], int] = {}
    edge_counts: dict[tuple[int, str, int], int] = {}
    for (n1, lever, n2), energies in graph.items():
        edge_counts[(n1, lever, n2)] = len(energies)

    for (n1, lever, n2), count in edge_counts.items():
        key = (n1, lever)
        if key not in predicted_target or count > edge_counts.get(
            (n1, lever, predicted_target[key]), 0
        ):
            predicted_target[key] = n2

    consistent = 0
    total = 0
    drift_distances = []

    for i in range(n_states):
        state = random_valid_state(seed=30000 + i)
        current_node = state_to_node(encoder, kmeans, state, device)

        for lever in LEVERS:
            key = (current_node, lever)
            if key not in predicted_target:
                continue  # no graph edge for this (node, lever)

            env, rod = make_tentacle()
            set_state(rod, state)
            new_state, _ = execute_lever(env, rod, lever)
            actual_node = state_to_node(encoder, kmeans, new_state, device)

            pred_node = predicted_target[key]
            is_match = actual_node == pred_node
            consistent += int(is_match)
            total += 1

            # Latent drift distance
            encoder.eval()
            with torch.no_grad():
                z_actual = encoder.encode(
                    torch.tensor(new_state, dtype=torch.float32).unsqueeze(0).to(device)
                ).cpu().numpy().squeeze()
            z_predicted_center = kmeans.cluster_centers_[pred_node]
            drift = float(np.linalg.norm(z_actual - z_predicted_center))
            drift_distances.append(drift)

    rate = consistent / total if total > 0 else 0.0
    return {
        "consistency_rate": rate,
        "total_tested": total,
        "drift_mean": float(np.mean(drift_distances)) if drift_distances else 0.0,
        "drift_std": float(np.std(drift_distances)) if drift_distances else 0.0,
    }
