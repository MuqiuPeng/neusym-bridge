"""Lever Control: Minimum-Energy Tentacle Planning — orchestration.

Usage:
    python -m experiments.lever_control.run_lever [--n-tasks 50] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import random_valid_state
from phase4.models.lewm_tentacle import LeWMTentacle

from experiments.lever_control.levers import LEVERS, calibrate_levers
from experiments.lever_control.collect_data import (
    collect_transitions, discretize_states, build_transition_graph,
)
from experiments.lever_control.planner import (
    RelatumLeverPlanner, GreedyLeverPlanner, RandomLeverPlanner,
)

OUTPUT_DIR = Path("experiments/lever_control/outputs")
LEWM_CKPT = Path("phase4/checkpoints/lewm_epoch049.pt")
DATA_PATH = Path("phase4/data/tentacle_data.h5")


def load_or_train_lewm(device: str = "cpu") -> LeWMTentacle:
    model = LeWMTentacle()
    if LEWM_CKPT.exists():
        model.load_state_dict(torch.load(LEWM_CKPT, weights_only=True))
        print(f"Loaded LeWM from {LEWM_CKPT}")
    else:
        # Train from scratch (reuse B4 pattern)
        print("Training LeWM from scratch...")
        from experiments.b4.run_b4 import load_data, train_lewm_if_needed
        s_t, a_t, s_t1 = load_data(max_traj=500)
        model = train_lewm_if_needed(s_t, a_t, s_t1, device=device)
    model.eval().to(device)
    return model


def generate_tasks(
    records: list[dict],
    node_before: np.ndarray,
    node_after: np.ndarray,
    n_tasks: int = 50,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Generate tasks from collected data ensuring different start/target nodes.

    Returns list of (start_state, target_state, start_node, target_node).
    """
    rng = np.random.RandomState(seed)
    all_states = np.array([r["state_before"] for r in records])
    all_nodes = node_before

    # Group states by node
    node_to_indices: dict[int, list[int]] = {}
    for i, n in enumerate(all_nodes):
        node_to_indices.setdefault(int(n), []).append(i)

    nodes = list(node_to_indices.keys())
    tasks = []
    attempts = 0
    while len(tasks) < n_tasks and attempts < n_tasks * 10:
        attempts += 1
        n1, n2 = rng.choice(nodes, size=2, replace=False)
        i1 = rng.choice(node_to_indices[n1])
        i2 = rng.choice(node_to_indices[n2])
        tasks.append((all_states[i1].copy(), all_states[i2].copy(), int(n1), int(n2)))

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Lever Control Experiment")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--n-episodes", type=int, default=200,
                        help="Episodes for data collection")
    parser.add_argument("--k-nodes", type=int, default=30,
                        help="Number of discrete state nodes")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Lever Control: Minimum-Energy Tentacle Planning", flush=True)
    print("=" * 60, flush=True)

    # ── Load encoder ──
    lewm = load_or_train_lewm(args.device)

    # ── Calibrate levers ──
    print("\n--- Lever Energy Calibration ---", flush=True)
    costs = calibrate_levers(n_states=10)
    print(f"{'Lever':15} {'Mean Energy':>12} {'Std':>10}")
    print("-" * 40)
    for lever in LEVERS:
        vals = costs[lever]
        print(f"{lever:15} {np.mean(vals):>12.6f} {np.std(vals):>10.6f}")

    # ── Collect transitions ──
    print("\n--- Data Collection ---", flush=True)
    records = collect_transitions(
        n_episodes=args.n_episodes, levers_per_episode=15,
    )
    print(f"Total transitions: {len(records)}")

    # ── Discretize states ──
    print("\n--- State Discretization ---", flush=True)
    kmeans, node_before, node_after = discretize_states(
        lewm, records, k=args.k_nodes, device=args.device,
    )

    # ── Build graph ──
    print("\n--- Transition Graph ---", flush=True)
    graph = build_transition_graph(records, node_before, node_after)

    # Node connectivity stats
    nodes_with_outgoing = set()
    for (n1, _, _) in graph:
        nodes_with_outgoing.add(n1)
    print(f"  Nodes with outgoing edges: {len(nodes_with_outgoing)}/{args.k_nodes}")

    # ── Evaluate planners ──
    print("\n--- Planning Evaluation ---", flush=True)
    tasks = generate_tasks(records, node_before, node_after, n_tasks=args.n_tasks)
    print(f"  Generated {len(tasks)} tasks with distinct start/target nodes", flush=True)

    planners = {
        "relatum_minE": RelatumLeverPlanner(lewm, kmeans, args.device),
        "greedy": GreedyLeverPlanner(lewm, kmeans, args.device),
        "random": RandomLeverPlanner(lewm, kmeans, args.device),
    }

    results = {name: [] for name in planners}

    for ti, (start, target, sn, tn) in enumerate(tasks):
        if (ti + 1) % 10 == 0:
            print(f"  Task {ti + 1}/{args.n_tasks}", flush=True)

        for name, planner in planners.items():
            result = planner.plan(start, target, max_steps=60)
            results[name].append(result)

    # ── Summary ──
    print("\n" + "=" * 75, flush=True)
    print("Lever Control Results", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Planner':20} {'Success':>8} {'Avg Energy':>12} "
          f"{'Avg Steps':>10} {'Explore':>8} {'Execute':>8}")
    print("-" * 75)

    summary = {}
    for name in planners:
        r = results[name]
        solved = [x for x in r if x["success"]]
        sr = len(solved) / len(r)
        avg_energy = float(np.mean([x["total_energy"] for x in solved])) if solved else 0
        avg_steps = float(np.mean([x["total_steps"] for x in solved])) if solved else 0
        avg_explore = float(np.mean([x.get("explore_steps", 0) for x in solved])) if solved else 0
        avg_execute = float(np.mean([x.get("execute_steps", 0) for x in solved])) if solved else 0

        print(f"{name:20} {sr:>8.3f} {avg_energy:>12.4f} "
              f"{avg_steps:>10.1f} {avg_explore:>8.1f} {avg_execute:>8.1f}")

        summary[name] = {
            "success_rate": sr,
            "avg_energy": avg_energy,
            "avg_steps": avg_steps,
            "avg_explore_steps": avg_explore,
            "avg_execute_steps": avg_execute,
        }

    # Energy savings
    if summary["relatum_minE"]["avg_energy"] > 0 and summary["greedy"]["avg_energy"] > 0:
        savings = 1 - summary["relatum_minE"]["avg_energy"] / summary["greedy"]["avg_energy"]
        print(f"\nRelatum vs Greedy energy savings: {savings:.1%}")

    # Save
    output = {
        "calibration": {lev: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for lev, v in costs.items()},
        "graph_stats": {
            "n_transitions": len(records),
            "n_reliable_edges": len(graph),
            "n_nodes_with_edges": len(nodes_with_outgoing),
        },
        "planning": summary,
    }
    out_path = OUTPUT_DIR / "lever_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Lever control experiment complete.", flush=True)


if __name__ == "__main__":
    main()
