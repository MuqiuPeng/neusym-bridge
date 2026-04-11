"""PyElastica 500ep with pre-trained LeWM encoder (not from-scratch).

The previous run used a freshly trained encoder that barely learned
(NCE 0.693->0.661). This run uses the pre-trained LeWM which already
has 30 effective clusters.

Usage:
    python -m experiments.exec_aware.run_pyela_lewm
"""

from __future__ import annotations

import io
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Suppress PyElastica tqdm
import elastica
_orig_integrate = elastica.integrate
def _quiet_integrate(*args, **kwargs):
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return _orig_integrate(*args, **kwargs)
elastica.integrate = _quiet_integrate

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle
from experiments.lever_control.collect_data import (
    collect_transitions, discretize_states, build_transition_graph,
)
from experiments.lever_control.planner import (
    RelatumMinEnergyPlanner, GreedyGraphPlanner,
    state_to_node, dijkstra_min_energy,
)

OUTPUT_DIR = Path("experiments/exec_aware/outputs")
LEWM_CKPT = Path("phase4/checkpoints/lewm_epoch049.pt")


def _elapsed(t0):
    m, s = divmod(int(time.time() - t0), 60)
    return f"{m:02d}:{s:02d}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60, flush=True)
    print("PyElastica + Pre-trained LeWM Encoder", flush=True)
    print("=" * 60, flush=True)

    # Load pre-trained LeWM
    print(f"[{_elapsed(t0)}] Loading LeWM from {LEWM_CKPT}...", flush=True)
    lewm = LeWMTentacle()
    lewm.load_state_dict(torch.load(LEWM_CKPT, weights_only=True))
    lewm.eval()

    # Collect transitions with PyElastica
    print(f"[{_elapsed(t0)}] Collecting 500 episodes (PyElastica)...", flush=True)
    records = collect_transitions(n_episodes=500, levers_per_episode=15)
    print(f"[{_elapsed(t0)}] {len(records)} transitions collected", flush=True)

    # Verify latent diversity
    all_states = np.array([r["state_before"] for r in records])
    with torch.no_grad():
        Z = lewm.encode(torch.tensor(all_states[:1000], dtype=torch.float32)).numpy()
    print(f"[{_elapsed(t0)}] Latent stats (first 1000): "
          f"std={Z.std(axis=0).mean():.4f}, range=[{Z.min():.3f}, {Z.max():.3f}]",
          flush=True)

    # k-sweep
    results = {}
    for k in [5, 10, 15, 20]:
        print(f"\n[{_elapsed(t0)}] === k={k} ===", flush=True)

        kmeans, nb, na = discretize_states(lewm, records, k=k)
        graph = build_transition_graph(records, nb, na, min_observations=2)
        nodes_out = set(key[0] for key in graph)

        # Cluster distribution
        unique, counts = np.unique(nb, return_counts=True)
        print(f"  Cluster distribution: {dict(zip(unique.tolist(), counts.tolist()))}", flush=True)
        print(f"  Edges: {len(graph)}, nodes_out: {len(nodes_out)}/{k}", flush=True)

        # Generate tasks
        node_to_idx: dict[int, list[int]] = {}
        for i, n in enumerate(nb):
            node_to_idx.setdefault(int(n), []).append(i)
        nodes = [n for n in node_to_idx if len(node_to_idx[n]) > 0]

        rng = np.random.RandomState(9999)
        tasks = []
        att = 0
        while len(tasks) < 50 and att < 500:
            att += 1
            if len(nodes) < 2:
                break
            n1, n2 = rng.choice(nodes, size=2, replace=False)
            tasks.append((
                all_states[rng.choice(node_to_idx[n1])],
                all_states[rng.choice(node_to_idx[n2])],
            ))

        # Count no-path
        no_path = 0
        for s, t in tasks:
            sn = state_to_node(lewm, kmeans, s)
            tn = state_to_node(lewm, kmeans, t)
            if sn != tn and dijkstra_min_energy(graph, sn, tn) is None:
                no_path += 1
        print(f"  No-path: {no_path}/{len(tasks)}", flush=True)

        # Evaluate Dijkstra planner
        planner = RelatumMinEnergyPlanner(lewm, kmeans, graph)
        solved = 0
        energies = []
        steps_list = []
        for ti, (s, t) in enumerate(tasks):
            r = planner.plan_and_execute(s, t)
            if r["success"]:
                solved += 1
                energies.append(r["total_energy"])
                steps_list.append(r["steps"])
            if (ti + 1) % 10 == 0:
                print(f"    Task {ti+1}/50: solved so far {solved}", flush=True)

        sr = solved / len(tasks)
        avg_e = float(np.mean(energies)) if energies else 0
        avg_s = float(np.mean(steps_list)) if steps_list else 0
        print(f"  SUCCESS: {sr:.3f} ({solved}/{len(tasks)}), "
              f"energy={avg_e:.4f}, steps={avg_s:.1f}", flush=True)

        results[k] = {
            "edges": len(graph),
            "nodes_out": len(nodes_out),
            "no_path": no_path,
            "success_rate": sr,
            "avg_energy": avg_e,
            "avg_steps": avg_s,
        }

    # Summary
    print(f"\n[{_elapsed(t0)}] SUMMARY:", flush=True)
    print(f"{'k':>5} {'Edges':>8} {'Out':>5} {'No-Path':>8} {'Success':>8} {'Energy':>8} {'Steps':>6}", flush=True)
    print("-" * 55, flush=True)
    for k in sorted(results):
        r = results[k]
        print(f"{k:>5} {r['edges']:>8} {r['nodes_out']:>5} {r['no_path']:>8} "
              f"{r['success_rate']:>8.3f} {r['avg_energy']:>8.4f} {r['avg_steps']:>6.1f}", flush=True)

    out_path = OUTPUT_DIR / "pyela_500ep_lewm.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{_elapsed(t0)}] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
