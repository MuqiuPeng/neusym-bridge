"""Maze planners: exploration-based Relatum vs baselines.

The Relatum planner has NO prior knowledge of the maze. It must:
  1. Explore by taking actions and observing state transitions
  2. Build a knowledge base of learned adjacency rules
  3. Use Relatum to check if the goal is reachable (collapse)
  4. Plan on the learned graph once a path is known

Baselines:
  - Random walk: take random valid actions
  - Greedy: always move toward goal (Manhattan), no exploration memory
  - BFS oracle: has full map knowledge (upper bound)
"""

from __future__ import annotations

import numpy as np

from experiments.maze.env import MazeState, cell_id, bfs_optimal_path, DIRECTIONS
from experiments.maze.interface import ExplorationKB, make_maze_relatum


class RelatumExplorationPlanner:
    """Two-phase planner: explore until goal path found, then execute.

    Phase 1 (Explore):
      - Pick frontier cells (cells with untried actions)
      - Navigate to closest frontier on learned graph
      - Try unexplored actions, learn edges/walls
      - After each new observation, check Relatum: did `solved` collapse?

    Phase 2 (Execute):
      - Once Relatum collapses `solved`, plan on learned graph
      - Follow shortest path to goal
    """

    def plan(
        self, initial_state: MazeState, max_steps: int = 500,
    ) -> tuple[list[MazeState], dict]:
        state = initial_state
        kb = ExplorationKB()
        kb.visited.add(state.agent)
        kb.observe_goal(state.goal)  # agent knows where the goal is

        trajectory = [state]
        relatum_checks = 0
        explore_steps = 0
        execute_steps = 0
        phase = "explore"

        for step in range(max_steps):
            if state.is_solved():
                break

            if phase == "explore":
                # Check Relatum: has the learned graph connected agent to goal?
                ri = make_maze_relatum()
                kb.inject_into_relatum(ri, state.agent)
                ri.update_closure([])
                relatum_checks += 1

                goal_cid = cell_id(*state.agent)
                solved_id = f"solved({goal_cid})"
                if ri.is_collapsed(solved_id) or ri.is_known(solved_id):
                    # Goal reachable on learned graph — switch to execute
                    phase = "execute"
                    continue

                # Explore: find next action to try
                action = self._pick_explore_action(state, kb)
                if action is None:
                    break  # fully explored, goal unreachable

                new_state = state.move(action)
                kb.observe_transition(state.agent, action, new_state.agent)
                state = new_state
                trajectory.append(state)
                explore_steps += 1

            elif phase == "execute":
                # Plan on learned graph
                path = kb.path_to(state.agent, state.goal)
                if path is None:
                    # Lost connectivity (shouldn't happen) — fall back to explore
                    phase = "explore"
                    continue

                for i in range(1, len(path)):
                    action = kb.action_for_edge(path[i - 1], path[i])
                    if action is None:
                        break
                    state = state.move(action)
                    trajectory.append(state)
                    execute_steps += 1
                    if state.is_solved():
                        break
                break  # done executing

        stats = {
            "explore_steps": explore_steps,
            "execute_steps": execute_steps,
            "total_steps": len(trajectory) - 1,
            "relatum_checks": relatum_checks,
            "cells_visited": kb.n_visited,
            "edges_learned": kb.n_edges,
            "coverage": kb.coverage,
        }
        return trajectory, stats

    def _pick_explore_action(
        self, state: MazeState, kb: ExplorationKB,
    ) -> str | None:
        """Choose the next exploration action.

        Priority:
        1. Try an unexplored action from current cell
        2. Navigate to closest frontier cell (has unexplored actions)
        3. None if fully explored
        """
        # 1. Unexplored action from current position
        unexplored = kb.unexplored_actions(state.agent)
        if unexplored:
            # Prefer directions toward the goal
            gx, gy = state.goal
            ax, ay = state.agent

            def goal_priority(d):
                dx, dy = DIRECTIONS[d]
                # Negative = closer to goal = better
                return abs(ax + dx - gx) + abs(ay + dy - gy)

            unexplored.sort(key=goal_priority)
            return unexplored[0]

        # 2. Navigate to closest frontier cell
        target = kb.closest_frontier_cell(state.agent)
        if target is None:
            return None  # fully explored

        path = kb.path_to(state.agent, target)
        if path is None or len(path) < 2:
            return None

        # Return action for first step toward frontier
        return kb.action_for_edge(path[0], path[1])


class RandomWalkPlanner:
    """Random walk baseline — no memory, no learning."""

    def plan(
        self, initial_state: MazeState, max_steps: int = 500,
    ) -> tuple[list[MazeState], dict]:
        rng = np.random.RandomState(42)
        state = initial_state
        trajectory = [state]

        for _ in range(max_steps):
            if state.is_solved():
                break
            actions = list(DIRECTIONS.keys())
            rng.shuffle(actions)
            moved = False
            for action in actions:
                new_state = state.move(action)
                if new_state.agent != state.agent:
                    state = new_state
                    trajectory.append(state)
                    moved = True
                    break
            if not moved:
                break

        return trajectory, {"total_steps": len(trajectory) - 1}


class GreedyPlanner:
    """Greedy planner — always move toward goal, no memory."""

    def plan(
        self, initial_state: MazeState, max_steps: int = 500,
    ) -> tuple[list[MazeState], dict]:
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
                    candidates.append((dist, direction))
            if not candidates:
                break
            candidates.sort()
            state = state.move(candidates[0][1])
            visited.add(state.agent)
            trajectory.append(state)

        return trajectory, {"total_steps": len(trajectory) - 1}


class BFSOraclePlanner:
    """BFS with full map knowledge — upper bound."""

    def plan(
        self, initial_state: MazeState, max_steps: int = 500,
    ) -> tuple[list[MazeState], dict]:
        path = bfs_optimal_path(initial_state)
        if path is None:
            return [initial_state], {"total_steps": 0}

        state = initial_state
        trajectory = [state]
        for i in range(1, len(path)):
            dx = path[i][0] - path[i - 1][0]
            dy = path[i][1] - path[i - 1][1]
            for direction, (ddx, ddy) in DIRECTIONS.items():
                if (ddx, ddy) == (dx, dy):
                    state = state.move(direction)
                    trajectory.append(state)
                    break

        return trajectory, {"total_steps": len(trajectory) - 1}
