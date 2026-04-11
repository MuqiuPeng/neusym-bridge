"""Maze validation experiment — orchestration script.

Validates Relatum symbolic reasoning in a zero-noise setting:
  1. Collapse mechanism (Scenario A/B/C)
  2. Planning comparison (Relatum vs Greedy vs BFS optimal)
  3. Scaling analysis across maze sizes and wall densities

Usage:
    python -m experiments.maze.run_maze [--n-mazes 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.maze.env import make_maze, bfs_optimal_path
from experiments.maze.planner import RelatumMazePlanner, GreedyMazePlanner, BFSMazePlanner
from experiments.maze.collapse_test import run_all_collapse_tests

OUTPUT_DIR = Path("experiments/maze/outputs")


def evaluate_planners(
    n_mazes: int = 100,
    sizes: list[int] | None = None,
    wall_densities: list[float] | None = None,
) -> dict:
    """Run all planners on a suite of random mazes."""
    if sizes is None:
        sizes = [6, 8, 10]
    if wall_densities is None:
        wall_densities = [0.1, 0.2, 0.3]

    planners = {
        "relatum": RelatumMazePlanner(),
        "greedy": GreedyMazePlanner(),
        "bfs": BFSMazePlanner(),
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
            traj, n_calls = planner.plan(maze, max_steps=size * size * 2)
            solved = traj[-1].is_solved()
            steps = len(traj) - 1

            results[name].append({
                "solved": solved,
                "steps": steps if solved else None,
                "optimal_steps": optimal_steps,
                "optimality": (
                    optimal_steps / steps
                    if solved and steps > 0 and optimal_steps is not None
                    else None
                ),
                "relatum_calls": n_calls,
                "maze_size": size,
                "wall_density": density,
            })

        if (maze_id + 1) % 20 == 0:
            print(f"  {maze_id + 1}/{n_mazes} mazes evaluated", flush=True)

    return results


def print_results(results: dict, n_mazes: int) -> dict:
    """Print summary table and return summary dict."""
    print("\n" + "=" * 70)
    print("Maze Planning Results")
    print("=" * 70)
    print(f"{'Planner':15} {'Success':>8} {'Avg Steps':>10} "
          f"{'Optimality':>11} {'Relatum Calls':>14}")
    print("-" * 70)

    summary = {}
    for name in ["relatum", "greedy", "bfs"]:
        r = results[name]
        solved = [x for x in r if x["solved"]]
        success_rate = len(solved) / n_mazes
        avg_steps = float(np.mean([x["steps"] for x in solved])) if solved else float("inf")
        opt_vals = [x["optimality"] for x in solved if x["optimality"] is not None]
        avg_opt = float(np.mean(opt_vals)) if opt_vals else 0.0
        avg_calls = float(np.mean([x["relatum_calls"] for x in r]))

        print(f"{name:15} {success_rate:>8.3f} {avg_steps:>10.1f} "
              f"{avg_opt:>11.3f} {avg_calls:>14.1f}")

        summary[name] = {
            "success_rate": success_rate,
            "avg_steps": avg_steps if avg_steps != float("inf") else None,
            "avg_optimality": avg_opt,
            "avg_relatum_calls": avg_calls,
        }

    # Per-size breakdown for relatum
    print(f"\nRelatum planner by maze size:")
    for size in sorted(set(x["maze_size"] for x in results["relatum"])):
        subset = [x for x in results["relatum"] if x["maze_size"] == size]
        solved = [x for x in subset if x["solved"]]
        sr = len(solved) / len(subset) if subset else 0
        opt_vals = [x["optimality"] for x in solved if x["optimality"]]
        ao = float(np.mean(opt_vals)) if opt_vals else 0
        print(f"  {size}x{size}: success={sr:.3f}  optimality={ao:.3f}  (n={len(subset)})")

    print("=" * 70)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Maze Validation Experiment")
    parser.add_argument("--n-mazes", type=int, default=100)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Maze Validation Experiment", flush=True)
    print("=" * 60, flush=True)

    # ── Task 1: Collapse mechanism tests ──
    print("\n" + "=" * 60, flush=True)
    print("Collapse Mechanism Tests (Scenario A/B/C)", flush=True)
    print("=" * 60, flush=True)

    collapse_results = run_all_collapse_tests()

    # ── Task 2: Planning evaluation ──
    print("\n" + "=" * 60, flush=True)
    print(f"Planning Evaluation ({args.n_mazes} mazes)", flush=True)
    print("=" * 60, flush=True)

    plan_results = evaluate_planners(n_mazes=args.n_mazes)
    summary = print_results(plan_results, args.n_mazes)

    # ── Task 3: Comparison table ──
    print("\n" + "=" * 60, flush=True)
    print("Maze vs Tentacle Domain Comparison", flush=True)
    print("=" * 60, flush=True)
    print(f"{'':20} {'Maze':>15} {'Tentacle':>15}")
    print("-" * 55)
    print(f"{'State type':20} {'discrete':>15} {'continuous':>15}")
    print(f"{'Interface noise':20} {'zero':>15} {'ECE=0.021':>15}")
    print(f"{'Relatum role':20} {'core planning':>15} {'interpretability':>15}")
    print(f"{'Collapse':20} {'deterministic':>15} {'Noisy-OR':>15}")

    relatum_sr = summary["relatum"]["success_rate"]
    relatum_opt = summary["relatum"]["avg_optimality"]
    greedy_sr = summary["greedy"]["success_rate"]

    if relatum_sr > 0.95 and relatum_opt > 0.90:
        verdict = "Relatum achieves near-optimal planning under zero noise"
    elif relatum_sr > 0.80:
        verdict = "Relatum effective but sub-optimal under zero noise"
    else:
        verdict = "Relatum planning limited even under zero noise"

    print(f"\nVerdict: {verdict}")
    print(f"  Relatum: success={relatum_sr:.3f}, optimality={relatum_opt:.3f}")
    print(f"  Greedy:  success={greedy_sr:.3f}")

    # ── Save ──
    output = {
        "collapse_tests": {k: bool(v) for k, v in collapse_results.items()},
        "planning": summary,
        "verdict": verdict,
    }
    out_path = OUTPUT_DIR / "maze_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Maze experiment complete.", flush=True)


if __name__ == "__main__":
    main()
