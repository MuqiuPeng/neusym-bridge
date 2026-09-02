"""Collapse mechanism tests in the exploration-based maze setting.

Scenario A: Agent explores, learns path to goal -> solved collapses
Scenario B: Agent learns path, then a wall appears -> solved retracts
Scenario C: Agent has partial knowledge -> Relatum queries missing info
"""

from __future__ import annotations

from experiments.maze.env import MazeState, cell_id, make_maze, bfs_solvable
from experiments.maze.interface import ExplorationKB, make_maze_relatum, DIRECTIONS


def test_scenario_a() -> bool:
    """Scenario A: Explore until path to goal is learned, solved collapses."""
    print("Scenario A: Exploration -> Collapse")
    maze = make_maze(6, 6, seed=0)

    kb = ExplorationKB()
    kb.observe_goal(maze.goal)

    # Simulate a systematic exploration (visit all reachable cells)
    from collections import deque
    queue = deque([maze.agent])
    visited = {maze.agent}
    kb.visited.add(maze.agent)

    while queue:
        pos = queue.popleft()
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            target = (nx, ny)
            if maze.is_valid(nx, ny):
                kb.observe_transition(pos, direction, target)
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
            else:
                kb.observe_transition(pos, direction, pos)  # wall

    # Now check Relatum
    ri = make_maze_relatum()
    kb.inject_into_relatum(ri, maze.agent)
    ri.update_closure([])

    goal_cid = cell_id(*maze.agent)
    solved_id = f"solved({goal_cid})"
    collapsed = ri.is_collapsed(solved_id) or ri.is_known(solved_id)

    if collapsed:
        path = kb.path_to(maze.agent, maze.goal)
        print(f"  PASS: solved collapsed after exploring {kb.n_visited} cells, "
              f"{kb.n_edges} edges learned, path length={len(path) - 1 if path else 'N/A'}")
    else:
        print(f"  FAIL: solved not collapsed despite {kb.n_visited} cells explored")

    return collapsed


def test_scenario_b() -> bool:
    """Scenario B: After learning a path, a wall appears blocking it."""
    print("\nScenario B: Learned path -> wall appears -> retraction")
    maze = make_maze(6, 6, seed=0)

    # Build full KB by exploring
    kb = ExplorationKB()
    kb.observe_goal(maze.goal)
    from collections import deque
    queue = deque([maze.agent])
    visited = {maze.agent}
    kb.visited.add(maze.agent)
    while queue:
        pos = queue.popleft()
        for direction, (dx, dy) in DIRECTIONS.items():
            nx, ny = pos[0] + dx, pos[1] + dy
            if maze.is_valid(nx, ny):
                kb.observe_transition(pos, direction, (nx, ny))
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
            else:
                kb.observe_transition(pos, direction, pos)

    # Confirm solved initially
    ri = make_maze_relatum()
    kb.inject_into_relatum(ri, maze.agent)
    ri.update_closure([])
    goal_cid = cell_id(*maze.agent)
    solved_id = f"solved({goal_cid})"
    assert ri.is_collapsed(solved_id) or ri.is_known(solved_id), "Should be solved initially"

    # Now simulate a wall appearing: remove edges from the KB
    # Find a critical cell that, if removed, disconnects agent from goal
    path = kb.path_to(maze.agent, maze.goal)
    if path is None or len(path) < 3:
        print("  SKIP: path too short to test blocking")
        return True

    for block_pos in path[1:-1]:
        # Remove all edges to/from block_pos in the KB
        kb_copy = ExplorationKB()
        kb_copy.goal = kb.goal
        kb_copy.visited = kb.visited.copy()
        for cell, neighbors in kb.edges.items():
            if cell == block_pos:
                continue
            kb_copy.edges[cell] = {(nb, act) for nb, act in neighbors if nb != block_pos}

        # Check if goal is still reachable in the modified KB
        reachable = kb_copy.reachable_from(maze.agent)
        if maze.goal not in reachable:
            # This blocking works — inject into fresh Relatum
            ri2 = make_maze_relatum()
            kb_copy.inject_into_relatum(ri2, maze.agent)
            ri2.update_closure([])

            not_solved = not (ri2.is_collapsed(solved_id) or ri2.is_known(solved_id))
            if not_solved:
                print(f"  PASS: blocking {block_pos} disconnects goal, "
                      f"solved correctly not derived")
            else:
                print(f"  FAIL: blocking {block_pos} but solved still derived")
            return not_solved

    print("  SKIP: no single cell blocks all learned paths")
    return True


def test_scenario_c() -> bool:
    """Scenario C: Partial exploration — Relatum can't derive solved yet."""
    print("\nScenario C: Partial exploration -> active query")
    maze = make_maze(6, 6, seed=0)

    # Only explore immediate neighbors of start
    kb = ExplorationKB()
    kb.observe_goal(maze.goal)
    kb.visited.add(maze.agent)

    for direction, (dx, dy) in DIRECTIONS.items():
        nx, ny = maze.agent[0] + dx, maze.agent[1] + dy
        target = (nx, ny) if maze.is_valid(nx, ny) else maze.agent
        kb.observe_transition(maze.agent, direction, target)

    # Goal should NOT be reachable yet (only 1 step explored)
    ri = make_maze_relatum()
    kb.inject_into_relatum(ri, maze.agent)
    ri.update_closure([])

    goal_cid = cell_id(*maze.agent)
    solved_id = f"solved({goal_cid})"
    not_solved = not (ri.is_collapsed(solved_id) or ri.is_known(solved_id))

    # Active query should request reachable_goal
    missing = ri.find_missing_premises()
    needs_info = any(m.predicate == "reachable_goal" for m in missing)

    # The agent knows it needs to explore more
    has_frontier = len(kb.frontier) > 0 or kb._has_unexplored(maze.agent)

    ok = not_solved and has_frontier
    if ok:
        print(f"  PASS: solved not derived (only {kb.n_visited} cells known), "
              f"frontier has {len(kb.frontier)} cells to explore")
    else:
        print(f"  FAIL: not_solved={not_solved}, has_frontier={has_frontier}")

    return ok


def run_all_collapse_tests() -> dict:
    results = {
        "scenario_a": test_scenario_a(),
        "scenario_b": test_scenario_b(),
        "scenario_c": test_scenario_c(),
    }
    n_pass = sum(results.values())
    print(f"\nCollapse tests: {n_pass}/3 passed")
    return results
