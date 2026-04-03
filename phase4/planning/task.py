"""Tentacle planning task definition.

Task: move tentacle from state A to state B while minimizing energy.
Evaluation metrics:
- Success rate: final state within threshold of target
- Energy efficiency: optimal_energy / actual_energy
- Explanation quality: whether Relatum can diagnose failures
"""

from __future__ import annotations

import numpy as np

from ..envs.tentacle_env import (
    make_tentacle,
    step,
    set_state,
    extract_state,
    random_valid_state,
    STATE_DIM,
    ACTION_DIM,
)


class TentaclePlanningTask:
    """A reach-target planning task for the tentacle.

    Evaluate a planner by how well it moves the tentacle from
    start to target, subject to energy constraints.
    """

    def __init__(
        self,
        start: np.ndarray,
        target: np.ndarray,
        energy_budget: float = 10.0,
        success_threshold: float | None = None,
    ):
        self.start = start
        self.target = target
        self.energy_budget = energy_budget

        # Success threshold relative to initial distance:
        # reaching within 50% of starting distance counts as success
        initial_dist = np.linalg.norm(target - start)
        self.success_threshold = (
            success_threshold if success_threshold is not None
            else max(initial_dist * 0.5, 5.0)
        )

        # Optimal energy is estimated as straight-line distance
        # (lower bound; actual optimal is higher due to dynamics)
        self.optimal_energy = initial_dist * 0.1

    def evaluate(self, trajectory: dict) -> dict:
        """Evaluate a trajectory against this task.

        Args:
            trajectory: Dict with keys "states" and "energies".

        Returns:
            Dict with success, distance, total_energy, efficiency.
        """
        final_state = trajectory["states"][-1]
        total_energy = sum(trajectory["energies"])

        dist = np.linalg.norm(final_state - self.target)
        success = dist < self.success_threshold

        efficiency = (
            self.optimal_energy / total_energy
            if total_energy > 0
            else 0.0
        )
        efficiency = min(efficiency, 1.0)

        return {
            "success": bool(success),
            "distance": float(dist),
            "total_energy": float(total_energy),
            "efficiency": float(efficiency),
        }


def execute_plan(
    actions: list[np.ndarray],
    start_state: np.ndarray,
) -> dict:
    """Execute a sequence of actions in the simulator.

    Args:
        actions: List of (80,) tension vectors.
        start_state: (140,) initial state.

    Returns:
        Trajectory dict with states and energies.
    """
    env, rod = make_tentacle()
    set_state(rod, start_state)

    trajectory = {
        "states": [extract_state(rod)],
        "energies": [],
        "actions": actions,
    }

    for action in actions:
        state, energy = step(env, rod, action)
        trajectory["states"].append(state)
        trajectory["energies"].append(energy)

    del env, rod
    return trajectory


def generate_task_suite(
    n_tasks: int = 100,
    seed: int = 42,
) -> list[TentaclePlanningTask]:
    """Generate a suite of random planning tasks.

    Args:
        n_tasks: Number of tasks.
        seed: Base random seed.

    Returns:
        List of TentaclePlanningTask instances.
    """
    tasks = []
    for i in range(n_tasks):
        start = random_valid_state(seed=seed + i)
        target = random_valid_state(seed=seed + i + 10000)
        tasks.append(TentaclePlanningTask(start=start, target=target))
    return tasks
