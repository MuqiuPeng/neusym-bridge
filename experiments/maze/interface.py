"""Zero-noise interface layer: direct mapping from maze state to Relatum facts.

Unlike the tentacle domain where confidences are neural-network outputs
(ECE=0.021), every fact here has confidence=1.0. This isolates Relatum's
symbolic reasoning from any perception noise.
"""

from __future__ import annotations

from experiments.maze.env import MazeState, cell_id, bfs_reachable

from src.neusym_bridge.relatum.interface import RelatumInterface


# Relatum rule for the maze domain.
# Uses single-arg predicates keyed by cell_id so they work with
# RelatumInterface._find_groundings (shared-arg assumption).
MAZE_RULE = (
    "solved(C) :- reachable(C), goal_cell(C).\n"
)


def state_to_relatum(state: MazeState, ri: RelatumInterface) -> list[str]:
    """Assert all maze facts into a RelatumInterface instance.

    Because RelatumInterface doesn't support multi-arity recursive rules,
    reachability is pre-computed in Python (BFS) and injected as ground
    facts.  This is semantically equivalent to Datalog forward-chaining
    on ``reachable(C2) :- reachable(C1), adjacent(C1, C2)``.

    Returns:
        List of asserted fact_ids.
    """
    fact_ids = []

    # Agent position
    ax, ay = state.agent
    fid = ri.assert_probabilistic("at", (cell_id(ax, ay),), 1.0)
    fact_ids.append(fid)

    # Goal
    gx, gy = state.goal
    fid = ri.assert_probabilistic("goal_cell", (cell_id(gx, gy),), 1.0)
    fact_ids.append(fid)

    # Walls (informational — not consumed by rules, but useful for queries)
    for wx, wy in state.walls:
        fid = ri.assert_probabilistic("wall", (cell_id(wx, wy),), 1.0)
        fact_ids.append(fid)

    # Pre-computed reachability (BFS from agent)
    reachable = bfs_reachable(state)
    for rx, ry in reachable:
        fid = ri.assert_probabilistic("reachable", (cell_id(rx, ry),), 1.0)
        fact_ids.append(fid)

    # Dead-end detection: cells with only 1 neighbor
    for rx, ry in reachable:
        n = len(state.neighbors(rx, ry))
        if n <= 1 and (rx, ry) != state.goal:
            ri.assert_probabilistic("dead_end", (cell_id(rx, ry),), 1.0)

    return fact_ids


def make_maze_relatum(collapse_threshold: float = 0.99) -> RelatumInterface:
    """Create a fresh RelatumInterface configured for maze reasoning."""
    ri = RelatumInterface()
    ri.load_rules_from_text(MAZE_RULE)
    ri.set_collapse_threshold("solved", collapse_threshold)
    return ri
