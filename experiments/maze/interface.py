"""Exploration-based interface: agent acts, observes, and learns rules.

The agent does NOT have a map. It can only:
  1. Take an action (up/down/left/right)
  2. Observe the resulting state (current position, whether it moved)

From these observations, it incrementally builds a knowledge base:
  - visited(C):      cell C has been visited
  - adjacent(C1,C2): moving from C1 reaches C2 (learned by doing)
  - blocked(C1,D):   action D from C1 hits a wall (learned by failing)
  - goal_cell(C):    cell C is the goal (observed when agent is there)

Rules are LEARNED, not pre-programmed. The agent discovers the graph
structure through trial and error.
"""

from __future__ import annotations

from collections import deque

from experiments.maze.env import MazeState, cell_id, DIRECTIONS

from src.neusym_bridge.relatum.interface import RelatumInterface


# The only pre-loaded rule: if we know a path from current to goal, we're solved.
# The adjacency graph that feeds this rule is learned from exploration.
MAZE_RULE = "solved(C) :- reachable_goal(C), at_current(C).\n"


class ExplorationKB:
    """Knowledge base built incrementally through exploration.

    This is the "learned rule" equivalent: instead of injecting
    pre-computed BFS reachability, the agent discovers adjacency
    by taking actions and observing outcomes.
    """

    def __init__(self):
        # Learned graph: cell -> set of (neighbor_cell, action)
        self.edges: dict[tuple[int, int], set[tuple[tuple[int, int], str]]] = {}
        # Cells where action D is known to hit a wall
        self.blocked: dict[tuple[int, int], set[str]] = {}
        # All visited cells
        self.visited: set[tuple[int, int]] = set()
        # Goal location (learned when agent steps on it or sees it)
        self.goal: tuple[int, int] | None = None
        # Frontier: visited cells with unexplored actions
        self.frontier: set[tuple[int, int]] = set()

    def observe_transition(
        self,
        from_pos: tuple[int, int],
        action: str,
        to_pos: tuple[int, int],
    ) -> None:
        """Learn from one action-observation pair."""
        self.visited.add(from_pos)
        self.visited.add(to_pos)

        if from_pos == to_pos:
            # Action failed (wall or boundary)
            self.blocked.setdefault(from_pos, set()).add(action)
        else:
            # Learned a new edge
            self.edges.setdefault(from_pos, set()).add((to_pos, action))
            self.edges.setdefault(to_pos, set()).add((from_pos, _reverse(action)))

        # Update frontier
        for cell in (from_pos, to_pos):
            if self._has_unexplored(cell):
                self.frontier.add(cell)
            else:
                self.frontier.discard(cell)

    def observe_goal(self, pos: tuple[int, int]) -> None:
        self.goal = pos

    def _has_unexplored(self, pos: tuple[int, int]) -> bool:
        """Does this cell have any action we haven't tried?"""
        tried = set()
        for _, act in self.edges.get(pos, set()):
            tried.add(act)
        tried |= self.blocked.get(pos, set())
        return len(tried) < 4  # 4 directions

    def known_neighbors(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Return cells reachable from pos via learned edges."""
        return [nb for nb, _ in self.edges.get(pos, set())]

    def path_to(
        self, start: tuple[int, int], target: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        """BFS on the LEARNED graph (not the true maze)."""
        if start == target:
            return [start]
        parent = {start: None}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            for nb in self.known_neighbors(pos):
                if nb not in parent:
                    parent[nb] = pos
                    if nb == target:
                        path = []
                        cur = nb
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        return list(reversed(path))
                    queue.append(nb)
        return None

    def reachable_from(self, start: tuple[int, int]) -> set[tuple[int, int]]:
        """All cells reachable from start on the learned graph."""
        visited = {start}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            for nb in self.known_neighbors(pos):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return visited

    def action_for_edge(
        self, from_pos: tuple[int, int], to_pos: tuple[int, int],
    ) -> str | None:
        """Return the action that moves from from_pos to to_pos."""
        for nb, act in self.edges.get(from_pos, set()):
            if nb == to_pos:
                return act
        return None

    def unexplored_actions(self, pos: tuple[int, int]) -> list[str]:
        """Actions not yet tried from this cell."""
        tried = set()
        for _, act in self.edges.get(pos, set()):
            tried.add(act)
        tried |= self.blocked.get(pos, set())
        return [d for d in DIRECTIONS if d not in tried]

    def closest_frontier_cell(
        self, start: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Find the closest frontier cell (has unexplored actions) via learned graph."""
        if self._has_unexplored(start):
            return start
        visited = {start}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            if self._has_unexplored(pos):
                return pos
            for nb in self.known_neighbors(pos):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        return None

    def inject_into_relatum(self, ri: RelatumInterface, current: tuple[int, int]) -> None:
        """Inject learned knowledge into Relatum for collapse check."""
        cid = cell_id(*current)
        ri.assert_probabilistic("at_current", (cid,), 1.0)

        if self.goal is not None:
            # Check if goal is reachable on learned graph
            reachable = self.reachable_from(current)
            if self.goal in reachable:
                ri.assert_probabilistic("reachable_goal", (cid,), 1.0)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.edges.values()) // 2

    @property
    def n_visited(self) -> int:
        return len(self.visited)

    @property
    def coverage(self) -> float:
        """Fraction of frontier exhausted (no more unexplored actions)."""
        if not self.visited:
            return 0.0
        exhausted = sum(1 for c in self.visited if not self._has_unexplored(c))
        return exhausted / len(self.visited)


def _reverse(action: str) -> str:
    return {"up": "down", "down": "up", "left": "right", "right": "left"}[action]


def make_maze_relatum() -> RelatumInterface:
    ri = RelatumInterface()
    ri.load_rules_from_text(MAZE_RULE)
    ri.set_collapse_threshold("solved", 0.99)
    return ri
