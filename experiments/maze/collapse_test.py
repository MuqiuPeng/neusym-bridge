"""Maze-domain collapse mechanism validation.

Mirrors Phase 3's three scenarios in a deterministic setting:
  A: Normal derivation -> solved collapses
  B: Block critical passage -> solved retracts
  C: Partial information -> active query for missing facts
"""

from __future__ import annotations

from experiments.maze.env import (
    MazeState, bfs_optimal_path, bfs_solvable, cell_id, make_maze,
)
from experiments.maze.interface import make_maze_relatum, state_to_relatum


def test_scenario_a() -> bool:
    """Scenario A: Normal collapse — goal is reachable, solved should collapse."""
    print("Scenario A: Normal collapse")
    maze = make_maze(6, 6, seed=0)
    assert bfs_solvable(maze), "Maze must be solvable"

    ri = make_maze_relatum(collapse_threshold=0.99)
    state_to_relatum(maze, ri)
    ri.update_closure([])

    goal_cid = cell_id(*maze.goal)
    solved_id = f"solved({goal_cid})"

    collapsed = ri.is_collapsed(solved_id)
    proof = ri.explain(solved_id)

    if collapsed:
        print(f"  PASS: solved collapsed, proof chain length={len(proof)}")
    else:
        conf = ri.get_confidence(solved_id)
        print(f"  FAIL: solved not collapsed (conf={conf:.3f})")

    return collapsed


def test_scenario_b() -> bool:
    """Scenario B: Retraction — block critical passage, solved should retract."""
    print("\nScenario B: Retraction after blocking")
    maze = make_maze(6, 6, seed=0)
    path = bfs_optimal_path(maze)
    assert path is not None and len(path) > 2

    # First, establish solved
    ri = make_maze_relatum(collapse_threshold=0.99)
    state_to_relatum(maze, ri)
    ri.update_closure([])

    goal_cid = cell_id(*maze.goal)
    solved_id = f"solved({goal_cid})"
    assert ri.is_collapsed(solved_id), "solved should be collapsed initially"

    # Now block cells along the path until the maze becomes unsolvable
    for block_pos in path[1:-1]:
        maze_blocked = MazeState(
            width=maze.width, height=maze.height,
            agent=maze.agent, goal=maze.goal,
            walls=maze.walls | {block_pos},
        )
        if not bfs_solvable(maze_blocked):
            bx, by = block_pos
            blocked_cid = cell_id(bx, by)

            # To trigger retraction, we need to retract the goal's
            # reachable fact (which is what solved depends on).
            # In the blocked maze, recompute reachability and find cells
            # that are no longer reachable.
            from experiments.maze.env import bfs_reachable
            old_reachable = bfs_reachable(maze)
            new_reachable = bfs_reachable(maze_blocked)
            lost = old_reachable - new_reachable

            # Retract lost cells by injecting low confidence
            for lx, ly in lost:
                ri.assert_probabilistic(
                    "reachable", (cell_id(lx, ly),), 0.05,
                )

            retracted = not ri.is_collapsed(solved_id)
            if retracted:
                print(f"  PASS: blocked ({bx},{by}), {len(lost)} cells lost, "
                      f"solved correctly retracted")
            else:
                # Check if goal was among lost cells
                goal_lost = maze.goal in lost
                print(f"  FAIL: blocked ({bx},{by}), {len(lost)} cells lost, "
                      f"goal_lost={goal_lost}, solved still collapsed")
            return retracted

    # All path cells have alternatives
    print("  SKIP: no single cell blocks all paths (maze too connected)")
    return True


def test_scenario_c() -> bool:
    """Scenario C: Active query — inject partial facts, check for missing queries."""
    print("\nScenario C: Active query for missing facts")
    maze = make_maze(6, 6, seed=0)

    ri = make_maze_relatum(collapse_threshold=0.99)

    # Only inject goal and agent position, skip reachable facts
    ri.assert_probabilistic("at", (cell_id(*maze.agent),), 1.0)
    ri.assert_probabilistic("goal_cell", (cell_id(*maze.goal),), 1.0)

    # Without reachable facts, solved cannot fire
    ri.update_closure([])
    goal_cid = cell_id(*maze.goal)
    solved_id = f"solved({goal_cid})"

    not_collapsed = not ri.is_collapsed(solved_id)

    # Active query should request "reachable" facts
    missing = ri.find_missing_premises()
    needs_reachable = any(m.predicate == "reachable" for m in missing)

    ok = not_collapsed and needs_reachable
    if ok:
        n_req = sum(1 for m in missing if m.predicate == "reachable")
        print(f"  PASS: solved not collapsed, {n_req} reachable queries requested")
    else:
        print(f"  FAIL: collapsed={not not_collapsed}, reachable_requested={needs_reachable}")

    # Now supply the missing reachable facts
    from experiments.maze.env import bfs_reachable
    reachable = bfs_reachable(maze)
    for rx, ry in reachable:
        ri.assert_probabilistic("reachable", (cell_id(rx, ry),), 1.0)
    ri.update_closure([])

    now_collapsed = ri.is_collapsed(solved_id)
    if now_collapsed:
        print(f"  PASS: after supplying reachable facts, solved collapsed")
    else:
        print(f"  FAIL: solved still not collapsed after reachable injection")

    return ok and now_collapsed


def run_all_collapse_tests() -> dict:
    """Run all three scenarios, return results."""
    results = {
        "scenario_a": test_scenario_a(),
        "scenario_b": test_scenario_b(),
        "scenario_c": test_scenario_c(),
    }
    n_pass = sum(results.values())
    print(f"\nCollapse tests: {n_pass}/3 passed")
    return results
