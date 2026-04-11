"""Lever planners: two-phase evaluation (explore offline, execute online).

Phase 1 (Explore): Random lever execution builds a transition graph.
    Energy spent here is NOT counted in task evaluation.

Phase 2 (Execute): Given the learned graph, plan and execute.
    Relatum uses min-energy Dijkstra path.
    Greedy uses latent-distance heuristic on the same graph.
    Both have identical information — only edge selection differs.
"""

from __future__ import annotations

import heapq
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import make_tentacle, set_state
from experiments.lever_control.levers import LEVERS, execute_lever


def state_to_node(encoder, kmeans, state: np.ndarray, device: str = "cpu") -> int:
    encoder.eval()
    with torch.no_grad():
        z = encoder.encode(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        ).cpu().numpy()
    return int(kmeans.predict(z)[0])


# ── Graph utilities ──────────────────────────────────────────────────


def dijkstra_min_energy(
    graph: dict[tuple[int, str, int], list[float]],
    start: int,
    target: int,
) -> tuple[list[int], list[str], float] | None:
    """Dijkstra on the learned graph using mean energy as edge weight."""
    adj: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    for (n1, lever, n2), energies in graph.items():
        adj[n1].append((n2, lever, float(np.mean(energies))))

    dist = {start: 0.0}
    prev: dict[int, tuple[int, str]] = {}
    heap = [(0.0, start)]

    while heap:
        d, node = heapq.heappop(heap)
        if node == target:
            path_nodes = [target]
            path_levers = []
            cur = target
            while cur in prev:
                p, lev = prev[cur]
                path_nodes.append(p)
                path_levers.append(lev)
                cur = p
            return list(reversed(path_nodes)), list(reversed(path_levers)), d
        if d > dist.get(node, float("inf")):
            continue
        for nb, lever, energy in adj[node]:
            nd = d + energy
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                prev[nb] = (node, lever)
                heapq.heappush(heap, (nd, nb))

    return None


def greedy_path(
    graph: dict[tuple[int, str, int], list[float]],
    kmeans,
    start: int,
    target: int,
    max_steps: int = 30,
) -> tuple[list[int], list[str]] | None:
    """Greedy path: each step picks the neighbor closest to target in latent space."""
    adj: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
    for (n1, lever, n2), energies in graph.items():
        adj[n1].append((n2, lever, float(np.mean(energies))))

    target_center = kmeans.cluster_centers_[target]
    node = start
    visited = {node}
    path_nodes = [node]
    path_levers = []

    for _ in range(max_steps):
        if node == target:
            break
        neighbors = adj[node]
        if not neighbors:
            break

        best_lever = None
        best_dist = float("inf")
        best_next = None

        for nb, lever, _ in neighbors:
            if nb in visited and nb != target:
                continue
            center = kmeans.cluster_centers_[nb]
            dist = float(np.linalg.norm(center - target_center))
            if dist < best_dist:
                best_dist = dist
                best_lever = lever
                best_next = nb

        if best_lever is None:
            break

        path_nodes.append(best_next)
        path_levers.append(best_lever)
        visited.add(best_next)
        node = best_next

    if node == target:
        return path_nodes, path_levers
    return None


# ── Execution (Phase 2) ─────────────────────────────────────────────


def execute_path(
    encoder, kmeans, path_levers: list[str],
    initial_state: np.ndarray,
    target_node: int,
    device: str = "cpu",
) -> dict:
    """Execute a planned lever sequence on the physical simulator.

    Only records energy consumed during execution (not exploration).
    """
    env, rod = make_tentacle()
    set_state(rod, initial_state)

    state = initial_state.copy()
    total_energy = 0.0
    steps_taken = 0

    for lever in path_levers:
        new_state, energy = execute_lever(env, rod, lever)
        total_energy += energy
        state = new_state
        steps_taken += 1

        current_node = state_to_node(encoder, kmeans, state, device)
        if current_node == target_node:
            break

    final_node = state_to_node(encoder, kmeans, state, device)
    return {
        "success": final_node == target_node,
        "total_energy": total_energy,
        "steps": steps_taken,
    }


class RelatumMinEnergyPlanner:
    """Plan using Dijkstra min-energy path on the learned graph."""

    def __init__(self, encoder, kmeans, graph, device="cpu"):
        self.encoder = encoder
        self.kmeans = kmeans
        self.graph = graph
        self.device = device

    def plan_and_execute(
        self, start_state: np.ndarray, target_state: np.ndarray,
    ) -> dict:
        start_node = state_to_node(self.encoder, self.kmeans, start_state, self.device)
        target_node = state_to_node(self.encoder, self.kmeans, target_state, self.device)

        if start_node == target_node:
            return {"success": True, "total_energy": 0.0, "steps": 0,
                    "reason": "already_at_target", "planned_energy": 0.0}

        result = dijkstra_min_energy(self.graph, start_node, target_node)
        if result is None:
            return {"success": False, "total_energy": 0.0, "steps": 0,
                    "reason": "no_path_in_graph", "planned_energy": None}

        path_nodes, path_levers, planned_energy = result
        exec_result = execute_path(
            self.encoder, self.kmeans, path_levers,
            start_state, target_node, self.device,
        )
        exec_result["reason"] = "reached" if exec_result["success"] else "drift"
        exec_result["planned_energy"] = planned_energy
        exec_result["planned_steps"] = len(path_levers)
        return exec_result


class GreedyGraphPlanner:
    """Plan using latent-distance greedy on the same graph."""

    def __init__(self, encoder, kmeans, graph, device="cpu"):
        self.encoder = encoder
        self.kmeans = kmeans
        self.graph = graph
        self.device = device

    def plan_and_execute(
        self, start_state: np.ndarray, target_state: np.ndarray,
    ) -> dict:
        start_node = state_to_node(self.encoder, self.kmeans, start_state, self.device)
        target_node = state_to_node(self.encoder, self.kmeans, target_state, self.device)

        if start_node == target_node:
            return {"success": True, "total_energy": 0.0, "steps": 0,
                    "reason": "already_at_target"}

        result = greedy_path(self.graph, self.kmeans, start_node, target_node)
        if result is None:
            return {"success": False, "total_energy": 0.0, "steps": 0,
                    "reason": "no_path_in_graph"}

        path_nodes, path_levers = result
        exec_result = execute_path(
            self.encoder, self.kmeans, path_levers,
            start_state, target_node, self.device,
        )
        exec_result["reason"] = "reached" if exec_result["success"] else "drift"
        return exec_result


class RandomGraphPlanner:
    """Random walk on the graph (baseline)."""

    def __init__(self, encoder, kmeans, graph, device="cpu"):
        self.encoder = encoder
        self.kmeans = kmeans
        self.graph = graph
        self.device = device

    def plan_and_execute(
        self, start_state: np.ndarray, target_state: np.ndarray,
        max_steps: int = 30,
    ) -> dict:
        start_node = state_to_node(self.encoder, self.kmeans, start_state, self.device)
        target_node = state_to_node(self.encoder, self.kmeans, target_state, self.device)

        if start_node == target_node:
            return {"success": True, "total_energy": 0.0, "steps": 0,
                    "reason": "already_at_target"}

        adj: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for (n1, lever, n2) in self.graph:
            adj[n1].append((n2, lever))

        rng = np.random.RandomState(7)
        env, rod = make_tentacle()
        set_state(rod, start_state)

        state = start_state.copy()
        node = start_node
        total_energy = 0.0

        for step in range(max_steps):
            if node == target_node:
                break
            neighbors = adj[node]
            if not neighbors:
                break
            nb, lever = neighbors[rng.randint(len(neighbors))]
            new_state, energy = execute_lever(env, rod, lever)
            total_energy += energy
            state = new_state
            node = state_to_node(self.encoder, self.kmeans, state, self.device)

        return {
            "success": node == target_node,
            "total_energy": total_energy,
            "steps": step + 1 if node != start_node else 0,
            "reason": "reached" if node == target_node else "max_steps",
        }
