"""Lever planners: Relatum exploration + min-energy path vs baselines.

The Relatum planner explores by executing random levers, learns a
transition graph with energy costs, and uses Dijkstra on the learned
graph once Relatum confirms a path to the target exists (collapse).
"""

from __future__ import annotations

import heapq
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import make_tentacle, set_state, extract_state
from experiments.lever_control.levers import LEVERS, execute_lever
from src.neusym_bridge.relatum.interface import RelatumInterface


LEVER_RULE = "solved(C) :- reachable_target(C), at_current(C).\n"


class LeverKB:
    """Knowledge base learned through lever exploration.

    Stores a weighted directed graph: node -> [(neighbor, lever, energy)].
    """

    def __init__(self):
        self.edges: dict[int, list[tuple[int, str, float]]] = defaultdict(list)
        self.visited_nodes: set[int] = set()
        self.target_node: int | None = None

    def observe_transition(
        self, from_node: int, lever: str, to_node: int, energy: float,
    ) -> None:
        self.visited_nodes.add(from_node)
        self.visited_nodes.add(to_node)
        # Avoid duplicate edges (same from, lever, to)
        for nb, lev, _ in self.edges[from_node]:
            if nb == to_node and lev == lever:
                return
        self.edges[from_node].append((to_node, lever, energy))

    def reachable_from(self, start: int) -> set[int]:
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb, _, _ in self.edges[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited

    def min_energy_path(
        self, start: int, target: int,
    ) -> tuple[list[int], list[str], float] | None:
        """Dijkstra shortest path by energy on the learned graph."""
        dist = {start: 0.0}
        prev: dict[int, tuple[int, str]] = {}
        heap = [(0.0, start)]

        while heap:
            d, node = heapq.heappop(heap)
            if node == target:
                # Reconstruct path
                path_nodes = [target]
                path_levers = []
                cur = target
                while cur in prev:
                    p, lev = prev[cur]
                    path_nodes.append(p)
                    path_levers.append(lev)
                    cur = p
                return (
                    list(reversed(path_nodes)),
                    list(reversed(path_levers)),
                    d,
                )
            if d > dist.get(node, float("inf")):
                continue
            for nb, lever, energy in self.edges[node]:
                nd = d + energy
                if nd < dist.get(nb, float("inf")):
                    dist[nb] = nd
                    prev[nb] = (node, lever)
                    heapq.heappush(heap, (nd, nb))

        return None

    def inject_into_relatum(self, ri: RelatumInterface, current_node: int) -> None:
        cid = str(current_node)
        ri.assert_probabilistic("at_current", (cid,), 1.0)
        if self.target_node is not None:
            reachable = self.reachable_from(current_node)
            if self.target_node in reachable:
                ri.assert_probabilistic("reachable_target", (cid,), 1.0)


def _make_lever_relatum() -> RelatumInterface:
    ri = RelatumInterface()
    ri.load_rules_from_text(LEVER_RULE)
    ri.set_collapse_threshold("solved", 0.99)
    return ri


class RelatumLeverPlanner:
    """Explore with levers, learn graph, Relatum collapse, min-energy execute."""

    def __init__(self, encoder, kmeans, device="cpu"):
        self.encoder = encoder
        self.kmeans = kmeans
        self.device = device

    def state_to_node(self, state: np.ndarray) -> int:
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder.encode(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).cpu().numpy()
        return int(self.kmeans.predict(z)[0])

    def plan(
        self,
        start_state: np.ndarray,
        target_state: np.ndarray,
        max_steps: int = 60,
    ) -> dict:
        env, rod = make_tentacle()
        set_state(rod, start_state)

        kb = LeverKB()
        current_node = self.state_to_node(start_state)
        target_node = self.state_to_node(target_state)
        kb.target_node = target_node
        kb.visited_nodes.add(current_node)

        state = start_state.copy()
        trajectory = [state.copy()]
        lever_sequence = []
        total_energy = 0.0
        explore_steps = 0
        execute_steps = 0
        relatum_checks = 0
        rng = np.random.RandomState(42)

        phase = "explore"

        for step_idx in range(max_steps):
            if current_node == target_node:
                break

            if phase == "explore":
                # Check Relatum
                ri = _make_lever_relatum()
                kb.inject_into_relatum(ri, current_node)
                ri.update_closure([])
                relatum_checks += 1

                cid = str(current_node)
                solved_id = f"solved({cid})"
                if ri.is_collapsed(solved_id) or ri.is_known(solved_id):
                    phase = "execute"
                    continue

                # Pick exploration lever: prefer untried from current node
                tried_levers = {lev for _, lev, _ in kb.edges[current_node]}
                untried = [l for l in LEVERS if l not in tried_levers]
                lever = rng.choice(untried) if untried else rng.choice(LEVERS)

                new_state, energy = execute_lever(env, rod, lever)
                new_node = self.state_to_node(new_state)
                kb.observe_transition(current_node, lever, new_node, energy)

                total_energy += energy
                state = new_state
                current_node = new_node
                trajectory.append(state.copy())
                lever_sequence.append(lever)
                explore_steps += 1

            elif phase == "execute":
                result = kb.min_energy_path(current_node, target_node)
                if result is None:
                    phase = "explore"
                    continue

                path_nodes, path_levers, path_energy = result
                for lev in path_levers:
                    new_state, energy = execute_lever(env, rod, lev)
                    new_node = self.state_to_node(new_state)
                    total_energy += energy
                    state = new_state
                    current_node = new_node
                    trajectory.append(state.copy())
                    lever_sequence.append(lev)
                    execute_steps += 1
                    if current_node == target_node:
                        break
                break

        return {
            "success": current_node == target_node,
            "total_energy": total_energy,
            "total_steps": len(lever_sequence),
            "explore_steps": explore_steps,
            "execute_steps": execute_steps,
            "relatum_checks": relatum_checks,
            "nodes_visited": len(kb.visited_nodes),
            "edges_learned": sum(len(v) for v in kb.edges.values()),
        }


class GreedyLeverPlanner:
    """Pick the lever that moves closest to target in latent space."""

    def __init__(self, encoder, kmeans, device="cpu"):
        self.encoder = encoder
        self.kmeans = kmeans
        self.device = device

    def _encode(self, state: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder.encode(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).cpu().numpy().squeeze()

    def plan(
        self,
        start_state: np.ndarray,
        target_state: np.ndarray,
        max_steps: int = 60,
    ) -> dict:
        env, rod = make_tentacle()
        set_state(rod, start_state)

        z_target = self._encode(target_state)
        state = start_state.copy()
        total_energy = 0.0
        lever_sequence = []
        target_node = int(self.kmeans.predict(z_target.reshape(1, -1))[0])

        for _ in range(max_steps):
            current_node = int(self.kmeans.predict(
                self._encode(state).reshape(1, -1))[0])
            if current_node == target_node:
                break

            # Try all levers, pick best by latent distance
            best_lever = None
            best_dist = float("inf")
            best_state = None
            best_energy = 0.0

            for lever in LEVERS:
                env_t, rod_t = make_tentacle()
                set_state(rod_t, state)
                ns, en = execute_lever(env_t, rod_t, lever)
                z_ns = self._encode(ns)
                dist = float(np.linalg.norm(z_ns - z_target))
                if dist < best_dist:
                    best_dist = dist
                    best_lever = lever
                    best_state = ns
                    best_energy = en

            # Execute best lever on the real rod
            state_new, energy = execute_lever(env, rod, best_lever)
            total_energy += energy
            state = state_new
            lever_sequence.append(best_lever)

        current_node = int(self.kmeans.predict(
            self._encode(state).reshape(1, -1))[0])

        return {
            "success": current_node == target_node,
            "total_energy": total_energy,
            "total_steps": len(lever_sequence),
            "explore_steps": 0,
            "execute_steps": len(lever_sequence),
        }


class RandomLeverPlanner:
    """Random lever baseline."""

    def __init__(self, encoder, kmeans, device="cpu"):
        self.encoder = encoder
        self.kmeans = kmeans
        self.device = device

    def plan(
        self,
        start_state: np.ndarray,
        target_state: np.ndarray,
        max_steps: int = 60,
    ) -> dict:
        rng = np.random.RandomState(7)
        env, rod = make_tentacle()
        set_state(rod, start_state)

        self.encoder.eval()
        with torch.no_grad():
            z_t = self.encoder.encode(
                torch.tensor(target_state, dtype=torch.float32).unsqueeze(0)
            ).cpu().numpy()
        target_node = int(self.kmeans.predict(z_t)[0])

        state = start_state.copy()
        total_energy = 0.0
        lever_sequence = []

        for _ in range(max_steps):
            with torch.no_grad():
                z = self.encoder.encode(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                ).cpu().numpy()
            current_node = int(self.kmeans.predict(z)[0])
            if current_node == target_node:
                break

            lever = rng.choice(LEVERS)
            state, energy = execute_lever(env, rod, lever)
            total_energy += energy
            lever_sequence.append(lever)

        with torch.no_grad():
            z = self.encoder.encode(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            ).cpu().numpy()
        current_node = int(self.kmeans.predict(z)[0])

        return {
            "success": current_node == target_node,
            "total_energy": total_energy,
            "total_steps": len(lever_sequence),
            "explore_steps": 0,
            "execute_steps": 0,
        }
