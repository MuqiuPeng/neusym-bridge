"""Maze validation: rule learning through exploration.

The agent has NO prior map. It learns adjacency rules by taking actions
(up/down/left/right) and observing state transitions. Relatum checks
whether the learned knowledge is sufficient to derive "solved".

Usage:
    python -m experiments.maze.run_maze [--n-mazes 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.maze.env import make_maze, bfs_optimal_path
from experiments.maze.planner import (
    RelatumExplorationPlanner,
    RandomWalkPlanner,
    GreedyPlanner,
    BFSOraclePlanner,
)
from experiments.maze.collapse_test import run_all_collapse_tests

OUTPUT_DIR = Path("experiments/maze/outputs")


def evaluate_planners(
    n_mazes: int = 100,
    sizes: list[int] | None = None,
    wall_densities: list[float] | None = None,
) -> dict:
    if sizes is None:
        sizes = [6, 8, 10]
    if wall_densities is None:
        wall_densities = [0.1, 0.2, 0.3]

    planners = {
        "relatum_explore": RelatumExplorationPlanner(),
        "greedy": GreedyPlanner(),
        "random_walk": RandomWalkPlanner(),
        "bfs_oracle": BFSOraclePlanner(),
    }

    results = {name: [] for name in planners}
    rng = np.random.RandomState(0)

    for maze_id in range(n_mazes):
        size = int(rng.choice(sizes))
        density = float(rng.choice(wall_densities))
        maze = make_maze(size, size, seed=maze_id, wall_density=density)

        optimal = bfs_optimal_path(maze)
        optimal_steps = len(optimal) - 1 if optimal else None

        for name, planner in planners.items():
            traj, stats = planner.plan(maze, max_steps=size * size * 4)
            solved = traj[-1].is_solved()
            total_steps = len(traj) - 1

            entry = {
                "solved": solved,
                "total_steps": total_steps if solved else None,
                "optimal_steps": optimal_steps,
                "optimality": (
                    optimal_steps / total_steps
                    if solved and total_steps > 0 and optimal_steps
                    else None
                ),
                "maze_size": size,
                "wall_density": density,
            }
            entry.update(stats)
            results[name].append(entry)

        if (maze_id + 1) % 20 == 0:
            print(f"  {maze_id + 1}/{n_mazes} mazes evaluated", flush=True)

    return results


def print_results(results: dict, n_mazes: int) -> dict:
    print("\n" + "=" * 80)
    print("Maze Planning Results (Exploration-Based)")
    print("=" * 80)
    print(f"{'Planner':20} {'Success':>8} {'Total Steps':>12} "
          f"{'Optimality':>11} {'Explore':>8} {'Execute':>8}")
    print("-" * 80)

    summary = {}
    for name in ["relatum_explore", "greedy", "random_walk", "bfs_oracle"]:
        r = results[name]
        solved = [x for x in r if x["solved"]]
        success_rate = len(solved) / n_mazes

        if solved:
            avg_total = float(np.mean([x["total_steps"] for x in solved]))
            opt_vals = [x["optimality"] for x in solved if x["optimality"]]
            avg_opt = float(np.mean(opt_vals)) if opt_vals else 0.0
        else:
            avg_total = float("inf")
            avg_opt = 0.0

        # Exploration-specific stats
        explore_vals = [x.get("explore_steps", 0) for x in solved]
        execute_vals = [x.get("execute_steps", 0) for x in solved]
        avg_explore = float(np.mean(explore_vals)) if explore_vals else 0
        avg_execute = float(np.mean(execute_vals)) if execute_vals else 0

        total_str = f"{avg_total:.1f}" if avg_total != float("inf") else "N/A"
        print(f"{name:20} {success_rate:>8.3f} {total_str:>12} "
              f"{avg_opt:>11.3f} {avg_explore:>8.1f} {avg_execute:>8.1f}")

        summary[name] = {
            "success_rate": success_rate,
            "avg_total_steps": avg_total if avg_total != float("inf") else None,
            "avg_optimality": avg_opt,
            "avg_explore_steps": avg_explore,
            "avg_execute_steps": avg_execute,
        }

    # Per-size breakdown for relatum
    print(f"\nRelatum Explorer by maze size:")
    for size in sorted(set(x["maze_size"] for x in results["relatum_explore"])):
        subset = [x for x in results["relatum_explore"] if x["maze_size"] == size]
        solved = [x for x in subset if x["solved"]]
        sr = len(solved) / len(subset)
        avg_explore = np.mean([x.get("explore_steps", 0) for x in solved]) if solved else 0
        avg_cells = np.mean([x.get("cells_visited", 0) for x in solved]) if solved else 0
        avg_edges = np.mean([x.get("edges_learned", 0) for x in solved]) if solved else 0
        print(f"  {size}x{size}: success={sr:.3f}  explore={avg_explore:.0f} steps  "
              f"cells={avg_cells:.0f}  edges={avg_edges:.0f}  (n={len(subset)})")

    print("=" * 80)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Maze Validation: Rule Learning")
    parser.add_argument("--n-mazes", type=int, default=100)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Maze Validation: Rule Learning Through Exploration", flush=True)
    print("=" * 60, flush=True)

    # Collapse tests
    print("\n" + "=" * 60, flush=True)
    print("Collapse Mechanism Tests", flush=True)
    print("=" * 60, flush=True)
    collapse_results = run_all_collapse_tests()

    # Planning evaluation
    print("\n" + "=" * 60, flush=True)
    print(f"Planning Evaluation ({args.n_mazes} mazes)", flush=True)
    print("=" * 60, flush=True)
    plan_results = evaluate_planners(n_mazes=args.n_mazes)
    summary = print_results(plan_results, args.n_mazes)

    # Comparison
    print("\n" + "=" * 60, flush=True)
    print("Key Insight", flush=True)
    print("=" * 60, flush=True)

    rel = summary["relatum_explore"]
    bfs = summary["bfs_oracle"]
    print(f"  Relatum (explore+plan): success={rel['success_rate']:.3f}, "
          f"total={rel['avg_total_steps']:.1f} steps")
    print(f"    of which: explore={rel['avg_explore_steps']:.1f} + "
          f"execute={rel['avg_execute_steps']:.1f}")
    if bfs["avg_total_steps"]:
        overhead = (rel["avg_total_steps"] / bfs["avg_total_steps"]) - 1
        print(f"  BFS oracle: {bfs['avg_total_steps']:.1f} steps (100% optimal)")
        print(f"  Exploration overhead: {overhead:.1%}")
    print(f"\n  The agent LEARNS the maze structure through interaction,")
    print(f"  then uses Relatum collapse to detect when it has enough")
    print(f"  knowledge to solve the problem.")

    # Save
    output = {
        "collapse_tests": {k: bool(v) for k, v in collapse_results.items()},
        "planning": summary,
    }
    out_path = OUTPUT_DIR / "maze_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Maze experiment complete.", flush=True)


if __name__ == "__main__":
    main()
