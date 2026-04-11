"""Lever Control: Two-Phase Evaluation.

Phase 1 (Explore): Random lever execution builds transition graph (offline).
Phase 2 (Execute): Plan on the learned graph (online, this is what we evaluate).

Usage:
    python -m experiments.lever_control.run_lever [--n-tasks 50] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle

from experiments.lever_control.levers import LEVERS, calibrate_levers
from experiments.lever_control.collect_data import (
    collect_transitions, discretize_states, build_transition_graph,
)
from experiments.lever_control.planner import (
    RelatumMinEnergyPlanner, GreedyGraphPlanner, RandomGraphPlanner,
    state_to_node,
)

OUTPUT_DIR = Path("experiments/lever_control/outputs")
LEWM_CKPT = Path("phase4/checkpoints/lewm_epoch049.pt")


def load_or_train_lewm(device: str = "cpu") -> LeWMTentacle:
    model = LeWMTentacle()
    if LEWM_CKPT.exists():
        model.load_state_dict(torch.load(LEWM_CKPT, weights_only=True))
        print(f"Loaded LeWM from {LEWM_CKPT}")
    else:
        from experiments.b4.run_b4 import load_data, train_lewm_if_needed
        s_t, a_t, s_t1 = load_data(max_traj=500)
        model = train_lewm_if_needed(s_t, a_t, s_t1, device=device)
    model.eval().to(device)
    return model


def generate_tasks(
    records: list[dict],
    node_before: np.ndarray,
    n_tasks: int = 50,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Generate tasks from collected data ensuring different start/target nodes."""
    rng = np.random.RandomState(seed)
    all_states = np.array([r["state_before"] for r in records])

    node_to_indices: dict[int, list[int]] = {}
    for i, n in enumerate(node_before):
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
    parser = argparse.ArgumentParser(description="Lever Control: Two-Phase Evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--k-nodes", type=int, default=30)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Lever Control: Two-Phase Evaluation", flush=True)
    print("=" * 60, flush=True)

    lewm = load_or_train_lewm(args.device)

    # ── Calibration ──
    print("\n--- Lever Energy Calibration ---", flush=True)
    costs = calibrate_levers(n_states=10)
    for lever in LEVERS:
        vals = costs[lever]
        print(f"  {lever:15} energy={np.mean(vals):.4f}", flush=True)

    # ══════════════════════════════════════════════════════
    # Phase 1: Offline Exploration (energy NOT counted)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("Phase 1: Offline Exploration (energy NOT counted)", flush=True)
    print("=" * 60, flush=True)

    records = collect_transitions(n_episodes=args.n_episodes, levers_per_episode=15)
    exploration_energy = sum(r["energy"] for r in records)
    print(f"  Total transitions: {len(records)}")
    print(f"  Exploration energy: {exploration_energy:.2f} (NOT counted in eval)")

    kmeans, node_before, node_after = discretize_states(
        lewm, records, k=args.k_nodes, device=args.device,
    )
    graph = build_transition_graph(records, node_before, node_after)

    # Graph stats
    nodes_with_out = set(k[0] for k in graph)
    all_edge_energies = [np.mean(v) for v in graph.values()]
    print(f"\n  Graph: {len(nodes_with_out)}/{args.k_nodes} nodes with edges, "
          f"{len(graph)} reliable edges")
    print(f"  Edge energy: mean={np.mean(all_edge_energies):.4f}, "
          f"range=[{np.min(all_edge_energies):.4f}, {np.max(all_edge_energies):.4f}]")

    # ══════════════════════════════════════════════════════
    # Phase 2: Execution (THIS is what we evaluate)
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60, flush=True)
    print("Phase 2: Execution (only this energy counts)", flush=True)
    print("=" * 60, flush=True)

    tasks = generate_tasks(records, node_before, n_tasks=args.n_tasks)
    print(f"  Generated {len(tasks)} tasks with distinct start/target nodes")

    planners = {
        "relatum_minE": RelatumMinEnergyPlanner(lewm, kmeans, graph, args.device),
        "greedy_graph": GreedyGraphPlanner(lewm, kmeans, graph, args.device),
        "random_graph": RandomGraphPlanner(lewm, kmeans, graph, args.device),
    }

    results = {name: [] for name in planners}

    for ti, (start, target, sn, tn) in enumerate(tasks):
        if (ti + 1) % 10 == 0:
            print(f"  Task {ti + 1}/{len(tasks)}", flush=True)
        for name, planner in planners.items():
            result = planner.plan_and_execute(start, target)
            results[name].append(result)

    # ── Summary ──
    print("\n" + "=" * 75, flush=True)
    print("Two-Phase Evaluation Results", flush=True)
    print("=" * 75, flush=True)

    print(f"\nExploration (Phase 1, offline):")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Transitions: {len(records)}")
    print(f"  Energy spent: {exploration_energy:.2f} (NOT counted)")
    print(f"  Graph: {len(graph)} edges, {len(nodes_with_out)} nodes")

    print(f"\nExecution (Phase 2, EVALUATED):")
    print(f"{'Planner':20} {'Success':>8} {'Exec Energy':>12} "
          f"{'Exec Steps':>11} {'No-Path':>8}")
    print("-" * 70)

    summary = {}
    for name in planners:
        r = results[name]
        solved = [x for x in r if x["success"]]
        no_path = sum(1 for x in r if x.get("reason") == "no_path_in_graph")
        sr = len(solved) / len(r)
        avg_energy = float(np.mean([x["total_energy"] for x in solved])) if solved else 0
        avg_steps = float(np.mean([x["steps"] for x in solved])) if solved else 0

        print(f"{name:20} {sr:>8.3f} {avg_energy:>12.4f} "
              f"{avg_steps:>11.1f} {no_path:>8}")

        summary[name] = {
            "success_rate": sr,
            "avg_energy": avg_energy,
            "avg_steps": avg_steps,
            "no_path_failures": no_path,
        }

    # Energy savings
    rel_e = summary["relatum_minE"]["avg_energy"]
    grd_e = summary["greedy_graph"]["avg_energy"]
    if rel_e > 0 and grd_e > 0:
        savings = 1 - rel_e / grd_e
        print(f"\nRelatum vs Greedy execution energy savings: {savings:.1%}")

    # Save
    output = {
        "exploration": {
            "episodes": args.n_episodes,
            "transitions": len(records),
            "energy_not_counted": exploration_energy,
            "graph_edges": len(graph),
            "graph_coverage": len(nodes_with_out) / args.k_nodes,
        },
        "calibration": {lev: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for lev, v in costs.items()},
        "execution": summary,
    }
    out_path = OUTPUT_DIR / "lever_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("Lever control experiment complete.", flush=True)


if __name__ == "__main__":
    main()
