"""Execution-Aware Encoder: full pipeline.

1. Collect temporal + execution pair data
2. Train Contrastive baseline (lambda=0) and Exec-Aware (lambda=0.5)
3. Build transition graphs for both
4. Measure execution consistency
5. Evaluate lever control success rate (two-phase: explore offline, execute online)
6. Optionally: re-planning variant

Usage:
    python -m experiments.exec_aware.run_exec_aware [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exec_aware.model import ExecutionAwareEncoder
from experiments.exec_aware.dataset import (
    collect_exec_pairs, collect_temporal_pairs, ExecAwareDataset,
)
from experiments.exec_aware.train import train_exec_aware, train_contrastive_baseline
from experiments.exec_aware.consistency_metric import measure_execution_consistency

from experiments.lever_control.collect_data import (
    collect_transitions, discretize_states, build_transition_graph,
)
from experiments.lever_control.planner import (
    RelatumMinEnergyPlanner, GreedyGraphPlanner, state_to_node,
)
from experiments.lever_control.levers import LEVERS

OUTPUT_DIR = Path("experiments/exec_aware/outputs")


def build_graph_for_encoder(
    encoder, records, k: int = 30, device: str = "cpu",
):
    """Discretize states with this encoder's latent space and build graph."""
    kmeans, node_before, node_after = discretize_states(
        encoder, records, k=k, device=device,
    )
    graph = build_transition_graph(records, node_before, node_after)
    return kmeans, graph, node_before


def generate_tasks(records, node_before, n_tasks=50, seed=42):
    """Generate tasks ensuring different start/target nodes."""
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
        tasks.append((all_states[i1].copy(), all_states[i2].copy()))
    return tasks


def evaluate_planner(planner, tasks):
    """Run planner on all tasks, return summary."""
    results = []
    for start, target in tasks:
        r = planner.plan_and_execute(start, target)
        results.append(r)
    solved = [r for r in results if r["success"]]
    return {
        "success_rate": len(solved) / len(results),
        "avg_energy": float(np.mean([r["total_energy"] for r in solved])) if solved else 0,
        "avg_steps": float(np.mean([r["steps"] for r in solved])) if solved else 0,
        "no_path": sum(1 for r in results if r.get("reason") == "no_path_in_graph"),
    }


def main():
    parser = argparse.ArgumentParser(description="Execution-Aware Encoder Experiment")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--k-nodes", type=int, default=30)
    parser.add_argument("--n-data-episodes", type=int, default=200)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("Execution-Aware Encoder Experiment", flush=True)
    print("=" * 60, flush=True)

    # ── Step 1: Collect data ──
    print("\n--- Data Collection ---", flush=True)
    print("Temporal pairs (for InfoNCE):", flush=True)
    temporal = collect_temporal_pairs(n_episodes=args.n_data_episodes)
    print(f"  {len(temporal)} temporal pairs")

    print("Execution pairs (for exec consistency):", flush=True)
    exec_pairs = collect_exec_pairs(n_episodes=args.n_data_episodes)
    print(f"  {len(exec_pairs)} execution pairs")

    dataset = ExecAwareDataset(temporal, exec_pairs)

    # Also collect lever transitions for graph building
    print("Lever transitions (for graph):", flush=True)
    lever_records = collect_transitions(n_episodes=args.n_data_episodes, levers_per_episode=15)
    print(f"  {len(lever_records)} lever transitions")

    # ── Step 2: Train both variants ──
    print("\n--- Training Contrastive Baseline (lambda=0) ---", flush=True)
    model_contrastive, hist_c = train_contrastive_baseline(
        dataset, n_epochs=args.n_epochs, device=args.device,
    )

    print("\n--- Training Exec-Aware (lambda=0.5) ---", flush=True)
    model_exec_aware, hist_e = train_exec_aware(
        dataset, lambda_exec=0.5, n_epochs=args.n_epochs, device=args.device,
    )

    # ── Step 3: Build graphs for both ──
    print("\n--- Building Graphs ---", flush=True)
    # Wrap models to have .encode() at top level
    class EncoderWrapper:
        def __init__(self, model, device):
            self.model = model
            self.device = device
        def encode(self, x):
            return self.model.encode(x.to(self.device))
        def eval(self):
            self.model.eval()
            return self
        def to(self, device):
            self.model.to(device)
            return self

    enc_c = EncoderWrapper(model_contrastive, args.device)
    enc_e = EncoderWrapper(model_exec_aware, args.device)

    print("  Contrastive:", flush=True)
    kmeans_c, graph_c, nb_c = build_graph_for_encoder(
        enc_c, lever_records, k=args.k_nodes, device=args.device,
    )
    print("  Exec-Aware:", flush=True)
    kmeans_e, graph_e, nb_e = build_graph_for_encoder(
        enc_e, lever_records, k=args.k_nodes, device=args.device,
    )

    # ── Step 4: Execution consistency ──
    print("\n--- Execution Consistency ---", flush=True)
    print("  Contrastive:", flush=True)
    cons_c = measure_execution_consistency(
        enc_c, kmeans_c, graph_c, n_states=50, device=args.device,
    )
    print(f"    Consistency: {cons_c['consistency_rate']:.3f}, "
          f"drift: {cons_c['drift_mean']:.4f}", flush=True)

    print("  Exec-Aware:", flush=True)
    cons_e = measure_execution_consistency(
        enc_e, kmeans_e, graph_e, n_states=50, device=args.device,
    )
    print(f"    Consistency: {cons_e['consistency_rate']:.3f}, "
          f"drift: {cons_e['drift_mean']:.4f}", flush=True)

    improvement = cons_e["consistency_rate"] - cons_c["consistency_rate"]
    print(f"\n  Consistency improvement: {improvement:+.3f} "
          f"({improvement / max(cons_c['consistency_rate'], 1e-8):.0%})", flush=True)

    # ── Step 5: Lever control evaluation ──
    print("\n--- Lever Control Evaluation (Phase 2 only) ---", flush=True)

    # Generate tasks from contrastive graph (shared task set)
    tasks_c = generate_tasks(lever_records, nb_c, n_tasks=args.n_tasks)
    tasks_e = generate_tasks(lever_records, nb_e, n_tasks=args.n_tasks)

    configs = {
        "Contrastive + Dijkstra": (
            RelatumMinEnergyPlanner(enc_c, kmeans_c, graph_c, args.device), tasks_c,
        ),
        "Contrastive + Greedy": (
            GreedyGraphPlanner(enc_c, kmeans_c, graph_c, args.device), tasks_c,
        ),
        "ExecAware + Dijkstra": (
            RelatumMinEnergyPlanner(enc_e, kmeans_e, graph_e, args.device), tasks_e,
        ),
        "ExecAware + Greedy": (
            GreedyGraphPlanner(enc_e, kmeans_e, graph_e, args.device), tasks_e,
        ),
    }

    eval_results = {}
    for name, (planner, tasks) in configs.items():
        print(f"  {name}...", flush=True)
        eval_results[name] = evaluate_planner(planner, tasks)

    # ── Summary ──
    print("\n" + "=" * 75, flush=True)
    print("Execution-Aware Encoder Results", flush=True)
    print("=" * 75, flush=True)

    print(f"\nExecution Consistency:")
    print(f"  {'Encoder':20} {'Rate':>10} {'Drift':>10}")
    print(f"  {'Contrastive':20} {cons_c['consistency_rate']:>10.3f} "
          f"{cons_c['drift_mean']:>10.4f}")
    print(f"  {'Exec-Aware':20} {cons_e['consistency_rate']:>10.3f} "
          f"{cons_e['drift_mean']:>10.4f}")

    print(f"\nLever Control (Phase 2):")
    print(f"  {'Config':30} {'Success':>8} {'Energy':>10} "
          f"{'Steps':>8} {'No-Path':>8}")
    print("-" * 70)
    for name, r in eval_results.items():
        print(f"  {name:30} {r['success_rate']:>8.3f} "
              f"{r['avg_energy']:>10.4f} {r['avg_steps']:>8.1f} "
              f"{r['no_path']:>8}")

    # ── Save ──
    output = {
        "consistency": {
            "contrastive": cons_c,
            "exec_aware": cons_e,
            "improvement": improvement,
        },
        "lever_control": eval_results,
        "training": {
            "contrastive_final_loss": hist_c["total"][-1] if hist_c["total"] else None,
            "exec_aware_final_loss": hist_e["total"][-1] if hist_e["total"] else None,
            "exec_aware_final_exec_loss": hist_e["exec"][-1] if hist_e["exec"] else None,
        },
    }
    out_path = OUTPUT_DIR / "exec_aware_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("Exec-aware experiment complete.", flush=True)


if __name__ == "__main__":
    main()
