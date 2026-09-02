"""Sweep k (number of clusters) to verify graph connectivity is the bottleneck.

If success rate increases as k decreases, the bottleneck is graph sparsity.
If k=5 still fails, the bottleneck is exploration volume.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exec_aware.model import ExecutionAwareEncoder
from experiments.exec_aware.dataset import collect_exec_pairs, collect_temporal_pairs, ExecAwareDataset
from experiments.exec_aware.train import train_exec_aware
from experiments.lever_control.collect_data import collect_transitions, build_transition_graph
from experiments.lever_control.planner import (
    RelatumMinEnergyPlanner, GreedyGraphPlanner, state_to_node,
)

OUTPUT_DIR = Path("experiments/exec_aware/outputs")


def sweep_k(
    encoder,
    lever_records: list[dict],
    k_values: list[int],
    n_tasks: int = 50,
    device: str = "cpu",
) -> dict:
    results = {}

    for k in k_values:
        print(f"\n{'=' * 50}", flush=True)
        print(f"k = {k}", flush=True)
        print(f"{'=' * 50}", flush=True)

        # Discretize
        all_states_before = np.array([r["state_before"] for r in lever_records])
        all_states_after = np.array([r["state_after"] for r in lever_records])
        all_states = np.vstack([all_states_before, all_states_after])

        encoder.eval()
        with torch.no_grad():
            Z = encoder.encode(
                torch.tensor(all_states, dtype=torch.float32).to(device)
            ).cpu().numpy()

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(Z)

        n = len(lever_records)
        node_before = kmeans.labels_[:n]
        node_after = kmeans.labels_[n:]

        # Build graph
        graph = build_transition_graph(lever_records, node_before, node_after, min_observations=2)
        nodes_with_out = set(key[0] for key in graph)

        # Generate tasks from data (ensuring different nodes)
        node_to_indices: dict[int, list[int]] = {}
        for i, nd in enumerate(node_before):
            node_to_indices.setdefault(int(nd), []).append(i)

        nodes = list(node_to_indices.keys())
        rng = np.random.RandomState(9999)
        tasks = []
        attempts = 0
        while len(tasks) < n_tasks and attempts < n_tasks * 20:
            attempts += 1
            if len(nodes) < 2:
                break
            n1, n2 = rng.choice(nodes, size=2, replace=False)
            i1 = rng.choice(node_to_indices[n1])
            i2 = rng.choice(node_to_indices[n2])
            tasks.append((all_states_before[i1], all_states_before[i2]))

        # Count no-path
        from experiments.lever_control.planner import dijkstra_min_energy
        no_path = 0
        for start, target in tasks:
            sn = state_to_node(encoder, kmeans, start, device)
            tn = state_to_node(encoder, kmeans, target, device)
            if sn == tn:
                continue
            path = dijkstra_min_energy(graph, sn, tn)
            if path is None:
                no_path += 1

        # Evaluate with Dijkstra planner
        planner = RelatumMinEnergyPlanner(encoder, kmeans, graph, device)
        plan_results = []
        for start, target in tasks:
            r = planner.plan_and_execute(start, target)
            plan_results.append(r)

        solved = [r for r in plan_results if r["success"]]
        sr = len(solved) / len(tasks) if tasks else 0
        avg_e = float(np.mean([r["total_energy"] for r in solved])) if solved else 0

        print(f"  Edges: {len(graph)}, nodes w/ out: {len(nodes_with_out)}/{k}")
        print(f"  No-path: {no_path}/{len(tasks)}")
        print(f"  Success: {sr:.3f}, Energy: {avg_e:.4f}")

        results[k] = {
            "reliable_edges": len(graph),
            "nodes_with_outedge": len(nodes_with_out),
            "no_path_failures": no_path,
            "success_rate": sr,
            "avg_energy": avg_e,
            "n_tasks": len(tasks),
        }

    # Summary
    print(f"\n{'=' * 65}", flush=True)
    print("k-sweep Results", flush=True)
    print(f"{'=' * 65}", flush=True)
    print(f"{'k':>5} {'Edges':>8} {'No-Path':>10} {'Success':>10} {'Energy':>10}")
    print("-" * 50)
    for k in sorted(results):
        r = results[k]
        e_str = f"{r['avg_energy']:.4f}" if r['avg_energy'] else "N/A"
        print(f"{k:>5} {r['reliable_edges']:>8} "
              f"{r['no_path_failures']:>10} {r['success_rate']:>10.3f} {e_str:>10}")

    best_k = max(results, key=lambda k: results[k]["success_rate"])
    print(f"\nBest k: {best_k}, success: {results[best_k]['success_rate']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--n-data-episodes", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("k-sweep: Graph Connectivity Verification", flush=True)
    print("=" * 60, flush=True)

    # Train a fresh exec-aware encoder
    print("\n--- Data Collection ---", flush=True)
    temporal = collect_temporal_pairs(n_episodes=args.n_data_episodes)
    exec_pairs = collect_exec_pairs(n_episodes=args.n_data_episodes)
    dataset = ExecAwareDataset(temporal, exec_pairs)

    print("\n--- Training Exec-Aware Encoder ---", flush=True)
    model, _ = train_exec_aware(dataset, lambda_exec=0.5, n_epochs=50, device=args.device)

    # Wrap for .encode() at top level
    class Wrapper:
        def __init__(self, m, d):
            self.m = m; self.d = d
        def encode(self, x):
            return self.m.encode(x.to(self.d))
        def eval(self):
            self.m.eval(); return self
        def to(self, d):
            self.m.to(d); return self

    encoder = Wrapper(model, args.device)

    # Collect lever transitions
    print("\n--- Lever Transitions ---", flush=True)
    lever_records = collect_transitions(n_episodes=args.n_data_episodes, levers_per_episode=15)

    # Sweep
    results = sweep_k(
        encoder, lever_records,
        k_values=[5, 8, 10, 15, 20, 30],
        n_tasks=args.n_tasks, device=args.device,
    )

    out_path = OUTPUT_DIR / "k_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
