"""Retrain LeWM on PyElastica data, then re-evaluate lever control.

The previous LeWM was trained on SimplifiedRod synthetic data.
This script trains a fresh LeWM on actual CosseratRod transitions,
then runs the k-sweep evaluation.

Usage:
    python -m experiments.exec_aware.run_pyela_retrain
"""

from __future__ import annotations

import io
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import elastica
_orig_integrate = elastica.integrate
def _quiet_integrate(*args, **kwargs):
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return _orig_integrate(*args, **kwargs)
elastica.integrate = _quiet_integrate

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.envs.tentacle_env import make_tentacle, set_state, random_valid_state, ACTION_DIM
from experiments.lever_control.levers import LEVERS, execute_lever, make_lever_tensions
from experiments.lever_control.collect_data import build_transition_graph
from experiments.lever_control.planner import (
    RelatumMinEnergyPlanner, GreedyGraphPlanner,
    state_to_node, dijkstra_min_energy,
)
from sklearn.cluster import KMeans

OUTPUT_DIR = Path("experiments/exec_aware/outputs")


def _elapsed(t0):
    m, s = divmod(int(time.time() - t0), 60)
    return f"{m:02d}:{s:02d}"


def collect_pyela_transitions(n_episodes=500, levers_per_ep=15):
    """Collect (s_t, a_t, s_t1) triples for LeWM training + graph building."""
    rng = np.random.RandomState(0)
    records = []

    for ep in range(n_episodes):
        env, rod = make_tentacle()
        state = random_valid_state(seed=ep)
        set_state(rod, state)

        for _ in range(levers_per_ep):
            lever = rng.choice(LEVERS)
            tensions = make_lever_tensions(lever)
            new_state, energy = execute_lever(env, rod, lever)

            records.append({
                "state_before": state.copy(),
                "action": tensions.copy(),
                "lever": lever,
                "state_after": new_state.copy(),
                "energy": energy,
            })
            state = new_state

        if (ep + 1) % 50 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes "
                  f"({len(records)} transitions)", flush=True)

    return records


def train_lewm_on_pyela(records, n_epochs=50, device="cpu"):
    """Train LeWM from scratch on PyElastica transition data."""
    s_t = np.array([r["state_before"] for r in records])
    a_t = np.array([r["action"] for r in records])
    s_t1 = np.array([r["state_after"] for r in records])

    s_t_t = torch.tensor(s_t, dtype=torch.float32)
    a_t_t = torch.tensor(a_t, dtype=torch.float32)
    s_t1_t = torch.tensor(s_t1, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(s_t_t, a_t_t, s_t1_t),
        batch_size=256, shuffle=True,
    )

    model = LeWMTentacle().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for bs, ba, bs1 in loader:
            bs, ba, bs1 = bs.to(device), ba.to(device), bs1.to(device)
            z_pred, z_t1, s_recon, s_orig = model(bs, ba, bs1)
            loss_pred = nn.MSELoss()(z_pred, z_t1)
            loss_recon = nn.MSELoss()(s_recon, s_orig)
            loss = loss_pred + loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bs)

        avg = total_loss / len(s_t_t)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={avg:.6f}", flush=True)

    model.eval()
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60, flush=True)
    print("Retrain LeWM on PyElastica Data + Lever Evaluation", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Collect PyElastica transitions
    print(f"\n[{_elapsed(t0)}] Collecting 500 episodes...", flush=True)
    records = collect_pyela_transitions(n_episodes=500, levers_per_ep=15)
    print(f"[{_elapsed(t0)}] {len(records)} transitions", flush=True)

    # Step 2: Train LeWM
    print(f"\n[{_elapsed(t0)}] Training LeWM on PyElastica data...", flush=True)
    lewm = train_lewm_on_pyela(records, n_epochs=50)

    # Check latent diversity
    states = np.array([r["state_before"] for r in records[:1000]])
    with torch.no_grad():
        Z = lewm.encode(torch.tensor(states, dtype=torch.float32)).numpy()
    print(f"[{_elapsed(t0)}] Latent: std={Z.std(axis=0).mean():.4f}, "
          f"range=[{Z.min():.3f}, {Z.max():.3f}]", flush=True)

    # Save checkpoint
    ckpt_path = OUTPUT_DIR / "lewm_pyela.pt"
    torch.save(lewm.state_dict(), ckpt_path)
    print(f"[{_elapsed(t0)}] Saved to {ckpt_path}", flush=True)

    # Step 3: Discretize + build graph + evaluate for each k
    # Assign node labels
    all_states_before = np.array([r["state_before"] for r in records])
    all_states_after = np.array([r["state_after"] for r in records])
    all_states = np.vstack([all_states_before, all_states_after])
    with torch.no_grad():
        Z_all = lewm.encode(torch.tensor(all_states, dtype=torch.float32)).numpy()

    results = {}
    for k in [5, 10, 15, 20, 30]:
        print(f"\n[{_elapsed(t0)}] === k={k} ===", flush=True)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(Z_all)

        n = len(records)
        nb = kmeans.labels_[:n]
        na = kmeans.labels_[n:]

        # Cluster distribution
        unique, counts = np.unique(nb, return_counts=True)
        top3 = sorted(zip(counts, unique), reverse=True)[:3]
        print(f"  Top-3 clusters: {[(int(u), int(c)) for c, u in top3]}", flush=True)

        graph = build_transition_graph(records, nb, na, min_observations=2)
        nodes_out = set(key[0] for key in graph)
        print(f"  Edges: {len(graph)}, nodes_out: {len(nodes_out)}/{k}", flush=True)

        # Generate tasks
        node_to_idx: dict[int, list[int]] = {}
        for i, nd in enumerate(nb):
            node_to_idx.setdefault(int(nd), []).append(i)
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
                all_states_before[rng.choice(node_to_idx[n1])],
                all_states_before[rng.choice(node_to_idx[n2])],
            ))

        # No-path count
        no_path = 0
        for s, t in tasks:
            sn = state_to_node(lewm, kmeans, s)
            tn = state_to_node(lewm, kmeans, t)
            if sn != tn and dijkstra_min_energy(graph, sn, tn) is None:
                no_path += 1
        print(f"  No-path: {no_path}/{len(tasks)}", flush=True)

        # Evaluate
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
                print(f"    Task {ti+1}/50: solved={solved}", flush=True)

        sr = solved / len(tasks) if tasks else 0
        avg_e = float(np.mean(energies)) if energies else 0
        avg_s = float(np.mean(steps_list)) if steps_list else 0
        print(f"  SUCCESS: {sr:.3f} ({solved}/{len(tasks)}), "
              f"energy={avg_e:.4f}, steps={avg_s:.1f}", flush=True)

        results[k] = {
            "edges": len(graph), "nodes_out": len(nodes_out),
            "no_path": no_path, "success_rate": sr,
            "avg_energy": avg_e, "avg_steps": avg_s,
        }

    # Summary
    print(f"\n[{_elapsed(t0)}] SUMMARY:", flush=True)
    print(f"{'k':>5} {'Edges':>8} {'Out':>5} {'No-Path':>8} "
          f"{'Success':>8} {'Energy':>8} {'Steps':>6}", flush=True)
    print("-" * 60, flush=True)
    for k in sorted(results):
        r = results[k]
        print(f"{k:>5} {r['edges']:>8} {r['nodes_out']:>5} {r['no_path']:>8} "
              f"{r['success_rate']:>8.3f} {r['avg_energy']:>8.4f} {r['avg_steps']:>6.1f}",
              flush=True)

    out_path = OUTPUT_DIR / "pyela_retrain_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{_elapsed(t0)}] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
