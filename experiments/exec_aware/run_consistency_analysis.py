"""Transition consistency analysis.

For each (from_node, lever) pair, computes:
    consistency = count(mode_destination) / total_observations

If most pairs have consistency < 0.70, it proves the discretization is
ill-conditioned: the same lever from the same cluster leads to different
physical outcomes depending on where within the cluster the rod actually is.

This is the fundamental difficulty of discrete lever planning on a
continuous physical system.

Usage:
    python -m experiments.exec_aware.run_consistency_analysis
"""

from __future__ import annotations

import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.modules["elastica"] = None  # type: ignore

import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle

OUTPUT_DIR = Path("experiments/exec_aware/outputs")
RECORDS_CACHE = OUTPUT_DIR / "diag_records.pkl"


def compute_consistency(edge_data_full: dict) -> dict:
    """edge_data_full: (n1, lever, n2) -> [energies], NO min_obs filter.

    Returns: (n1, lever) -> consistency score [0..1]
    """
    # Group observations by (from_node, lever)
    pair_outcomes: dict[tuple[int, str], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for (n1, lever, n2), energies in edge_data_full.items():
        pair_outcomes[(n1, lever)][n2] += len(energies)

    consistency = {}
    for (n1, lever), outcomes in pair_outcomes.items():
        total = sum(outcomes.values())
        mode_count = max(outcomes.values())
        consistency[(n1, lever)] = mode_count / total
    return consistency


def main():
    # ── Load records ────────────────────────────────────────────────
    with open(RECORDS_CACHE, "rb") as f:
        records = pickle.load(f)
    print(f"Loaded {len(records)} transitions", flush=True)

    # ── Load model and encode ────────────────────────────────────────
    lewm = LeWMTentacle()
    lewm.load_state_dict(torch.load(OUTPUT_DIR / "lewm_simplified_v2.pt", map_location="cpu"))
    lewm.eval()

    all_sb = np.array([r["state_before"] for r in records])
    all_sa = np.array([r["state_after"] for r in records])
    all_s = np.vstack([all_sb, all_sa])
    with torch.no_grad():
        Z = lewm.encode(torch.tensor(all_s, dtype=torch.float32)).numpy()

    # ── Cluster at multiple k values ─────────────────────────────────
    for k in [10, 15, 20, 30]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(Z)
        n = len(records)
        nb = kmeans.labels_[:n]
        na = kmeans.labels_[n:]

        # Build FULL edge data (no min_obs filter)
        edge_data_full: dict = defaultdict(list)
        for i, r in enumerate(records):
            edge_data_full[(int(nb[i]), r["lever"], int(na[i]))].append(r["energy"])

        # Compute consistency per (from_node, lever) pair
        consistency = compute_consistency(edge_data_full)
        scores = list(consistency.values())

        print(f"\n{'='*55}")
        print(f"k={k}: {len(consistency)} unique (node, lever) pairs")
        print(f"  Consistency stats:")
        print(f"    mean  = {np.mean(scores):.3f}")
        print(f"    median= {np.median(scores):.3f}")
        print(f"    min   = {np.min(scores):.3f}")
        print(f"    max   = {np.max(scores):.3f}")

        # Fraction surviving various thresholds
        print(f"  Pairs surviving threshold (of {len(scores)} total):")
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
            n_survive = sum(1 for s in scores if s >= thresh)
            pct = n_survive / len(scores) * 100
            bar = "#" * int(pct / 2)
            print(f"    >= {thresh:.0%}: {n_survive:4d} / {len(scores)} ({pct:5.1f}%)  {bar}")

        # What does >= 0.70 look like on the graph?
        # (n1, lever) pairs with consistency >= 0.70, keep only mode edge
        filtered_edges = {}
        pair_outcomes: dict = defaultdict(lambda: defaultdict(list))
        for (n1, lever, n2), energies in edge_data_full.items():
            pair_outcomes[(n1, lever)][n2].extend(energies)

        for (n1, lever), outcomes in pair_outcomes.items():
            if consistency[(n1, lever)] < 0.70:
                continue
            mode_n2 = max(outcomes, key=lambda n: len(outcomes[n]))
            filtered_edges[(n1, lever, mode_n2)] = outcomes[mode_n2]

        nodes_with_out = len(set(k_[0] for k_ in filtered_edges))
        print(f"  After >= 0.70 filter: {len(filtered_edges)} edges, "
              f"{nodes_with_out}/{k} nodes reachable")

        # Distribution of consistency scores (histogram)
        bins = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        labels = ["<30%", "30-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
        hist, _ = np.histogram(scores, bins=bins)
        print(f"  Consistency distribution:")
        for lbl, cnt in zip(labels, hist):
            bar = "#" * cnt
            print(f"    {lbl:12s}: {cnt:4d}  {bar}")


if __name__ == "__main__":
    main()
