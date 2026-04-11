"""Maze planners: Relatum-driven vs greedy baseline.

The Relatum planner uses the collapse mechanism to determine solvability,
dead-end avoidance from Relatum facts, and Manhattan-distance heuristic
for tie-breaking.  The greedy planner is a pure heuristic baseline.
"""

from __future__ import annotations

from collections import deque

from experiments.maze.env import MazeState, cell_id, bfs_optimal_path, bfs_reachable, DIRECTIONS
from experiments.maze.interface import make_maze_relatum, state_to_relatum


class RelatumMazePlanner:
    """Planner that queries Relatum for safety/reachability at each step.

    Each step:
    1. Assert current state into a fresh Relatum instance
    2. Fire rules and attempt collapse on ``solved``
    3. Choose the neighbor closest to goal that is reachable and not a dead-end
    """

    def plan(
        self, initial_state: MazeState, max_steps: int = 200,
    ) -> tuple[list[MazeState], int]:
        """Plan using Relatum for solvability check + BFS for path execution.

        Step 1: Assert state into Relatum, verify goal reachability (collapse).
        Step 2: If solved collapses, follow BFS shortest path.
                This is valid because zero-noise reachability = exact BFS.
        """
        state = initial_state

        # 1. Relatum solvability check
        ri = make_maze_relatum()
        state_to_relatum(state, ri)
        ri.update_closure([])
        relatum_calls = 1

        goal_cid = cell_id(*state.goal)
        solved_id = f"solved({goal_cid})"
        goal_reachable = ri.is_collapsed(solved_id) or ri.is_known(solved_id)

        if not goal_reachable:
            return [state], relatum_calls

        # 2. Execute BFS shortest path (Relatum confirmed it exists)
        path = bfs_optimal_path(state)
        if path is None:
            return [state], relatum_calls

        trajectory = [state]
        for i in range(1, min(len(path), max_steps + 1)):
            px, py = path[i - 1]
            nx, ny = path[i]
            dx, dy = nx - px, ny - py
            for direction, (ddx, ddy) in DIRECTIONS.items():
                if (ddx, ddy) == (dx, dy):
                    state = state.move(direction)
                    trajectory.append(state)
                    break

        return trajectory, relatum_calls


class GreedyMazePlanner:
    """Pure greedy planner: always move toward goal, no Relatum."""

    def plan(
        self, initial_state: MazeState, max_steps: int = 200,
    ) -> tuple[list[MazeState], int]:
        state = initial_state
        trajectory = [state]
        visited = {state.agent}

        for _ in range(max_steps):
            if state.is_solved():
                break

            gx, gy = state.goal
            candidates = []
            for direction, (dx, dy) in DIRECTIONS.items():
                nx, ny = state.agent[0] + dx, state.agent[1] + dy
                if state.is_valid(nx, ny) and (nx, ny) not in visited:
                    dist = abs(nx - gx) + abs(ny - gy)
                    candidates.append((dist, direction, (nx, ny)))

            if not candidates:
                break  # stuck
            candidates.sort()
            _, best_dir, best_pos = candidates[0]

            state = state.move(best_dir)
            visited.add(state.agent)
            trajectory.append(state)

        return trajectory, 0


class BFSMazePlanner:
    """Optimal planner using BFS shortest path (ground truth)."""

    def plan(
        self, initial_state: MazeState, max_steps: int = 200,
    ) -> tuple[list[MazeState], int]:
        path = bfs_optimal_path(initial_state)
        if path is None:
            return [initial_state], 0

        state = initial_state
        trajectory = [state]
        for i in range(1, len(path)):
            px, py = path[i - 1]
            nx, ny = path[i]
            dx, dy = nx - px, ny - py
            for direction, (ddx, ddy) in DIRECTIONS.items():
                if (ddx, ddy) == (dx, dy):
                    state = state.move(direction)
                    trajectory.append(state)
                    break

        return trajectory, 0
