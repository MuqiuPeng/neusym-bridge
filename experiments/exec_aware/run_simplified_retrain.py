"""Retrain LeWM on SimplifiedRod (with improved warmup) + k-sweep evaluation.

Now that random_valid_state uses 500 warmup steps with tension=1.5,
states are diverse. This tests whether the lever control pipeline
works with proper state diversity.

Usage:
    python -m experiments.exec_aware.run_simplified_retrain
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Force SimplifiedRod backend
sys.modules["elastica"] = None  # type: ignore

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.envs.tentacle_env import make_tentacle, set_state, random_valid_state, ACTION_DIM
from experiments.lever_control.levers import LEVERS, execute_lever, make_lever_tensions
from experiments.lever_control.collect_data import build_transition_graph
from experiments.lever_control.planner import (
    RelatumMinEnergyPlanner, GreedyGraphPlanner,
    state_to_node, dijkstra_min_energy,
)

OUTPUT_DIR = Path("experiments/exec_aware/outputs")


def _elapsed(t0):
    m, s = divmod(int(time.time() - t0), 60)
    return f"{m:02d}:{s:02d}"


def collect_transitions(n_episodes=500, levers_per_ep=15):
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
        if (ep + 1) % 100 == 0:
            print(f"  Collected {ep+1}/{n_episodes} ({len(records)} transitions)",
                  flush=True)
    return records


def train_lewm(records, n_epochs=50, device="cpu"):
    s_t = torch.tensor(np.array([r["state_before"] for r in records]), dtype=torch.float32)
    a_t = torch.tensor(np.array([r["action"] for r in records]), dtype=torch.float32)
    s_t1 = torch.tensor(np.array([r["state_after"] for r in records]), dtype=torch.float32)

    loader = DataLoader(TensorDataset(s_t, a_t, s_t1), batch_size=256, shuffle=True)
    model = LeWMTentacle().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total = 0.0
        for bs, ba, bs1 in loader:
            bs, ba, bs1 = bs.to(device), ba.to(device), bs1.to(device)
            z_pred, z_t1, s_recon, s_orig = model(bs, ba, bs1)
            loss = nn.MSELoss()(z_pred, z_t1) + nn.MSELoss()(s_recon, s_orig)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(bs)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}: loss={total/len(s_t):.6f}", flush=True)

    model.eval()
    return model


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60, flush=True)
    print("SimplifiedRod + Improved Warmup: Full Pipeline", flush=True)
    print("=" * 60, flush=True)

    # Collect
    print(f"\n[{_elapsed(t0)}] Collecting 500 episodes...", flush=True)
    records = collect_transitions(500, 15)
    print(f"[{_elapsed(t0)}] {len(records)} transitions", flush=True)

    # Train LeWM
    print(f"\n[{_elapsed(t0)}] Training LeWM...", flush=True)
    lewm = train_lewm(records, n_epochs=50)

    # Check latent diversity
    states = np.array([r["state_before"] for r in records[:2000]])
    with torch.no_grad():
        Z = lewm.encode(torch.tensor(states, dtype=torch.float32)).numpy()
    print(f"[{_elapsed(t0)}] Latent: std={Z.std(axis=0).mean():.4f}, "
          f"range=[{Z.min():.3f}, {Z.max():.3f}]", flush=True)

    # Save
    torch.save(lewm.state_dict(), OUTPUT_DIR / "lewm_simplified_v2.pt")

    # Discretize all states
    all_sb = np.array([r["state_before"] for r in records])
    all_sa = np.array([r["state_after"] for r in records])
    all_s = np.vstack([all_sb, all_sa])
    with torch.no_grad():
        Z_all = lewm.encode(torch.tensor(all_s, dtype=torch.float32)).numpy()

    # k-sweep
    results = {}
    for k in [5, 10, 15, 20, 30]:
        print(f"\n[{_elapsed(t0)}] === k={k} ===", flush=True)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(Z_all)

        n = len(records)
        nb, na = kmeans.labels_[:n], kmeans.labels_[n:]

        unique, counts = np.unique(nb, return_counts=True)
        top3 = sorted(zip(counts, unique), reverse=True)[:3]
        print(f"  Top-3 clusters: {[(int(u), int(c)) for c, u in top3]}", flush=True)

        graph = build_transition_graph(records, nb, na, min_observations=2)
        nodes_out = set(key[0] for key in graph)
        print(f"  Edges: {len(graph)}, nodes_out: {len(nodes_out)}/{k}", flush=True)

        # Tasks
        node_to_idx: dict[int, list[int]] = {}
        for i, nd in enumerate(nb):
            node_to_idx.setdefault(int(nd), []).append(i)
        nodes = [n for n in node_to_idx if len(node_to_idx[n]) > 0]
        rng = np.random.RandomState(9999)
        tasks = []
        att = 0
        while len(tasks) < 50 and att < 500:
            att += 1
            if len(nodes) < 2: break
            n1, n2 = rng.choice(nodes, size=2, replace=False)
            tasks.append((all_sb[rng.choice(node_to_idx[n1])],
                          all_sb[rng.choice(node_to_idx[n2])]))

        no_path = sum(
            1 for s, t in tasks
            if state_to_node(lewm, kmeans, s) != state_to_node(lewm, kmeans, t)
            and dijkstra_min_energy(graph, state_to_node(lewm, kmeans, s),
                                    state_to_node(lewm, kmeans, t)) is None
        )
        print(f"  No-path: {no_path}/{len(tasks)}", flush=True)

        # Evaluate both planners
        for pname, PlannerCls in [("Dijkstra", RelatumMinEnergyPlanner),
                                   ("Greedy", GreedyGraphPlanner)]:
            planner = PlannerCls(lewm, kmeans, graph)
            solved, energies, steps_list = 0, [], []
            for ti, (s, t) in enumerate(tasks):
                r = planner.plan_and_execute(s, t)
                if r["success"]:
                    solved += 1
                    energies.append(r["total_energy"])
                    steps_list.append(r["steps"])
                if (ti + 1) % 25 == 0:
                    print(f"    {pname} task {ti+1}/50: solved={solved}", flush=True)

            sr = solved / len(tasks) if tasks else 0
            avg_e = float(np.mean(energies)) if energies else 0
            avg_s = float(np.mean(steps_list)) if steps_list else 0
            print(f"  {pname}: success={sr:.3f}, energy={avg_e:.4f}, steps={avg_s:.1f}",
                  flush=True)

            results[f"k{k}_{pname.lower()}"] = {
                "k": k, "planner": pname,
                "edges": len(graph), "nodes_out": len(nodes_out),
                "no_path": no_path, "success_rate": sr,
                "avg_energy": avg_e, "avg_steps": avg_s,
            }

    # Summary
    print(f"\n[{_elapsed(t0)}] SUMMARY:", flush=True)
    print(f"{'k':>3} {'Planner':>10} {'Edges':>6} {'Out':>4} "
          f"{'NoPath':>7} {'Success':>8} {'Energy':>8} {'Steps':>6}", flush=True)
    print("-" * 65, flush=True)
    for key in sorted(results):
        r = results[key]
        print(f"{r['k']:>3} {r['planner']:>10} {r['edges']:>6} {r['nodes_out']:>4} "
              f"{r['no_path']:>7} {r['success_rate']:>8.3f} "
              f"{r['avg_energy']:>8.4f} {r['avg_steps']:>6.1f}", flush=True)

    out_path = OUTPUT_DIR / "simplified_v2_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[{_elapsed(t0)}] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
