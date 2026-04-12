"""Diagnostic: failure reason breakdown for LeWM at k=15.

Loads lewm_simplified_v2.pt, rebuilds graph (caching records to disk),
then runs 50 tasks with full reason tracking:
  - already_at_target
  - no_path_in_graph
  - reached  (true success)
  - drift     (path found, but physical execution missed target)

Usage:
    python -m experiments.exec_aware.run_diagnostic
"""

from __future__ import annotations

import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.modules["elastica"] = None  # type: ignore

import numpy as np
import torch
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
RECORDS_CACHE = OUTPUT_DIR / "diag_records.pkl"
K = 15
N_TASKS = 50


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
            print(f"  {ep+1}/{n_episodes} ({len(records)} transitions)", flush=True)
    return records


def main():
    t0 = time.time()

    # ── Load or collect records ───────────────────────────────────────
    if RECORDS_CACHE.exists():
        print(f"[{_elapsed(t0)}] Loading cached records from {RECORDS_CACHE}", flush=True)
        with open(RECORDS_CACHE, "rb") as f:
            records = pickle.load(f)
        print(f"[{_elapsed(t0)}] {len(records)} transitions", flush=True)
    else:
        print(f"[{_elapsed(t0)}] Collecting 500 episodes (will cache)...", flush=True)
        records = collect_transitions(500, 15)
        print(f"[{_elapsed(t0)}] {len(records)} transitions — saving cache", flush=True)
        with open(RECORDS_CACHE, "wb") as f:
            pickle.dump(records, f)

    # ── Load model ────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "lewm_simplified_v2.pt"
    print(f"[{_elapsed(t0)}] Loading model from {model_path}", flush=True)
    lewm = LeWMTentacle()
    lewm.load_state_dict(torch.load(model_path, map_location="cpu"))
    lewm.eval()

    # ── Encode all states ─────────────────────────────────────────────
    all_sb = np.array([r["state_before"] for r in records])
    all_sa = np.array([r["state_after"] for r in records])
    all_s = np.vstack([all_sb, all_sa])
    with torch.no_grad():
        Z_all = lewm.encode(torch.tensor(all_s, dtype=torch.float32)).numpy()

    # ── Cluster ───────────────────────────────────────────────────────
    print(f"[{_elapsed(t0)}] KMeans k={K}...", flush=True)
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    kmeans.fit(Z_all)

    n = len(records)
    nb = kmeans.labels_[:n]   # node_before for each record
    na = kmeans.labels_[n:]   # node_after

    # ── Graph ─────────────────────────────────────────────────────────
    graph = build_transition_graph(records, nb, na, min_observations=2)
    reliable = len(graph)
    all_edges_raw = sum(
        1 for (n1, lever, n2), energies in graph.items()
    )
    nodes_out = len(set(k[0] for k in graph))
    print(f"[{_elapsed(t0)}] Graph: {reliable} reliable edges, {nodes_out}/{K} nodes with out-edges",
          flush=True)

    # ── Task generation (same logic as retrain script) ────────────────
    node_to_idx: dict[int, list[int]] = {}
    for i, nd in enumerate(nb):
        node_to_idx.setdefault(int(nd), []).append(i)
    nodes = [nd for nd in node_to_idx if len(node_to_idx[nd]) > 0]

    rng = np.random.RandomState(9999)
    tasks = []
    att = 0
    while len(tasks) < N_TASKS and att < 500:
        att += 1
        if len(nodes) < 2:
            break
        n1, n2 = rng.choice(nodes, size=2, replace=False)
        s = all_sb[rng.choice(node_to_idx[n1])]
        t = all_sb[rng.choice(node_to_idx[n2])]
        tasks.append((s, t, n1, n2))

    print(f"[{_elapsed(t0)}] {len(tasks)} tasks generated", flush=True)

    # ── Check start==target after encoding ───────────────────────────
    same_node = sum(
        1 for s, t, n1, n2 in tasks
        if state_to_node(lewm, kmeans, s) == state_to_node(lewm, kmeans, t)
    )
    print(f"  start==target (post-encode): {same_node}/{len(tasks)} ({same_node/len(tasks)*100:.0f}%)",
          flush=True)

    # ── Evaluate with reason tracking ─────────────────────────────────
    for pname, PlannerCls in [("Dijkstra", RelatumMinEnergyPlanner),
                               ("Greedy", GreedyGraphPlanner)]:
        planner = PlannerCls(lewm, kmeans, graph)
        reasons = []
        steps_success = []
        steps_drift = []
        planned_vs_executed = []  # (planned_steps, executed_steps) for drift cases

        for s, t, _, _ in tasks:
            r = planner.plan_and_execute(s, t)
            reasons.append(r["reason"])
            if r["reason"] == "reached":
                steps_success.append(r["steps"])
            elif r["reason"] == "drift":
                steps_drift.append(r["steps"])
                if "planned_steps" in r:
                    planned_vs_executed.append((r["planned_steps"], r["steps"]))

        cnt = Counter(reasons)
        total = len(tasks)
        print(f"\n  [{pname}] Failure breakdown ({total} tasks):", flush=True)
        for reason in ["already_at_target", "no_path_in_graph", "reached", "drift"]:
            c = cnt.get(reason, 0)
            print(f"    {reason:25s}: {c:3d} ({c/total*100:5.1f}%)", flush=True)

        if steps_success:
            from collections import Counter as C
            sc = C(steps_success)
            print(f"    steps (success): {dict(sorted(sc.items()))}", flush=True)
        if steps_drift:
            sc = Counter(steps_drift)
            print(f"    steps (drift):   {dict(sorted(sc.items()))}", flush=True)
        if planned_vs_executed:
            print(f"    planned_steps (drift): "
                  f"min={min(p for p,_ in planned_vs_executed)} "
                  f"max={max(p for p,_ in planned_vs_executed)} "
                  f"mean={np.mean([p for p,_ in planned_vs_executed]):.1f}",
                  flush=True)

    print(f"\n[{_elapsed(t0)}] Done.", flush=True)


if __name__ == "__main__":
    main()
