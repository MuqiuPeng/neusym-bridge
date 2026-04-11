"""Maze environment: fully discrete, no neural network required.

Provides maze generation with solvability guarantee, BFS ground-truth
shortest path, and ASCII rendering.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

DIRECTIONS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass
class MazeState:
    width: int
    height: int
    agent: tuple[int, int]
    goal: tuple[int, int]
    walls: set[tuple[int, int]]
    steps: int = 0
    path: list[tuple[int, int]] = field(default_factory=list)

    def is_valid(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and 0 <= y < self.height
                and (x, y) not in self.walls)

    def neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Return valid non-wall neighbors of (x, y)."""
        result = []
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                result.append((nx, ny))
        return result

    def move(self, direction: str) -> MazeState:
        dx, dy = DIRECTIONS[direction]
        nx, ny = self.agent[0] + dx, self.agent[1] + dy
        if not self.is_valid(nx, ny):
            return self
        return MazeState(
            width=self.width, height=self.height,
            agent=(nx, ny), goal=self.goal,
            walls=self.walls, steps=self.steps + 1,
            path=self.path + [self.agent],
        )

    def is_solved(self) -> bool:
        return self.agent == self.goal

    def render(self) -> str:
        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.agent:
                    row += "A"
                elif (x, y) == self.goal:
                    row += "G"
                elif (x, y) in self.walls:
                    row += "#"
                else:
                    row += "."
            lines.append(row)
        return "\n".join(lines)


def cell_id(x: int, y: int) -> str:
    """Canonical string identifier for a cell, used as Relatum arg."""
    return f"{x}_{y}"


# ── BFS utilities ────────────────────────────────────────────────────


def bfs_solvable(state: MazeState) -> bool:
    visited = {state.agent}
    queue = deque([state.agent])
    while queue:
        x, y = queue.popleft()
        if (x, y) == state.goal:
            return True
        for nx, ny in state.neighbors(x, y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def bfs_optimal_path(state: MazeState) -> list[tuple[int, int]] | None:
    """Return shortest path from agent to goal (inclusive), or None."""
    parent = {state.agent: None}
    queue = deque([state.agent])
    while queue:
        pos = queue.popleft()
        if pos == state.goal:
            path = []
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            return list(reversed(path))
        x, y = pos
        for nx, ny in state.neighbors(x, y):
            if (nx, ny) not in parent:
                parent[(nx, ny)] = pos
                queue.append((nx, ny))
    return None


def bfs_reachable(state: MazeState) -> set[tuple[int, int]]:
    """Return all cells reachable from agent position."""
    visited = {state.agent}
    queue = deque([state.agent])
    while queue:
        x, y = queue.popleft()
        for nx, ny in state.neighbors(x, y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited


# ── Maze generation ──────────────────────────────────────────────────


def make_maze(
    width: int = 8,
    height: int = 8,
    seed: int = 42,
    wall_density: float = 0.2,
) -> MazeState:
    """Generate a random maze guaranteed to be solvable."""
    rng = np.random.RandomState(seed)

    walls = set()
    for x in range(width):
        for y in range(height):
            if rng.random() < wall_density:
                walls.add((x, y))

    start = (0, 0)
    goal = (width - 1, height - 1)
    walls.discard(start)
    walls.discard(goal)

    state = MazeState(width, height, start, goal, walls)

    if not bfs_solvable(state):
        wall_list = list(walls)
        rng.shuffle(wall_list)
        for w in wall_list:
            walls.discard(w)
            state = MazeState(width, height, start, goal, walls)
            if bfs_solvable(state):
                break

    return state
