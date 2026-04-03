"""B3: Intervention Robustness Scan — orchestration script.

Replicates Phase 2's intervention check at scale (4 dirs × 50 samples × 9
amplitudes) and produces statistical conclusions + cross-model consistency.

Usage:
    python -m experiments.b3.run_b3 [--n-samples 50] [--device cpu]

Prerequisites:
    - data/heat_2d.h5           (Phase 0)
    - checkpoints/model_b.pt    (Phase 0)
    - checkpoints/model_c.pt    (Phase 0)
    - results/phase1_verdict.json (Phase 1)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Project imports
from neusym_bridge.models.trainer import load_model
from neusym_bridge.analysis.latent_collector import collect_latents
from neusym_bridge.analysis.structure_extraction import (
    svcca,
    build_common_basis,
)

# B3 modules
from experiments.b3.intervention_robust import run_robust_intervention
from experiments.b3.analyze import (
    analyze_intervention_results,
    plot_intervention_results,
)
from experiments.b3.cross_model import cross_model_consistency
from experiments.b3.report import b3_summary_table, save_summary


DATA_PATH = Path("data/heat_2d.h5")
MODEL_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("experiments/b3/outputs")


def main():
    parser = argparse.ArgumentParser(description="B3 Intervention Robustness Scan")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Initial samples per direction (default: 50)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-cross-model", action="store_true",
                        help="Skip the cross-model consistency check")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    alphas = np.linspace(-4, 4, 9)

    # ------------------------------------------------------------------
    # Load data and Phase 1 verdict
    # ------------------------------------------------------------------
    print("=" * 60, flush=True)
    print("B3: Intervention Robustness Scan", flush=True)
    print("=" * 60, flush=True)

    with open(RESULTS_DIR / "phase1_verdict.json") as f:
        p1 = json.load(f)
    pair = p1["phase2_inputs"]["best_model_pair"]
    print(f"Model pair from Phase 1: {pair}", flush=True)

    with h5py.File(DATA_PATH, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    print(f"Trajectories shape: {trajectories.shape}", flush=True)

    model_1 = load_model(MODEL_DIR / f"{pair[0]}.pt")
    model_2 = load_model(MODEL_DIR / f"{pair[1]}.pt")
    print("Models loaded.", flush=True)

    # ------------------------------------------------------------------
    # Recompute SVCCA common basis (same as Phase 2)
    # ------------------------------------------------------------------
    print("\nRecomputing SVCCA common basis ...", flush=True)
    t0_frames = trajectories[:, 0]  # all initial frames
    Z_1 = collect_latents(model_1, t0_frames)
    Z_2 = collect_latents(model_2, t0_frames)

    svcca_result = svcca(Z_1, Z_2, n_components=10)
    V_common = build_common_basis(
        svcca_result["V_A"], svcca_result["V_B"], n_dims=4,
    )
    print(f"V_common shape: {V_common.shape}", flush=True)
    print(f"Top-4 canonical correlations: "
          f"{svcca_result['correlations'][:4].round(3)}", flush=True)

    # ------------------------------------------------------------------
    # Task 1: Large-scale intervention on primary model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("Task 1: Robust intervention scan (primary model)", flush=True)
    print("=" * 60, flush=True)

    results = run_robust_intervention(
        model_1, V_common, trajectories,
        n_samples=args.n_samples, alphas=alphas, device=args.device,
    )

    # Save raw results
    raw_path = OUTPUT_DIR / "results_raw.pkl"
    with open(raw_path, "wb") as f:
        pickle.dump({"results": results, "alphas": alphas.tolist()}, f)
    print(f"\nRaw results saved to {raw_path}", flush=True)

    # ------------------------------------------------------------------
    # Task 2: Statistical analysis + visualization
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("Task 2: Statistical analysis", flush=True)
    print("=" * 60, flush=True)

    causal_dirs = analyze_intervention_results(results)
    plot_intervention_results(results, alphas)

    # ------------------------------------------------------------------
    # Task 3: Cross-model consistency (optional)
    # ------------------------------------------------------------------
    cross_results = None
    if not args.skip_cross_model:
        print("\n" + "=" * 60, flush=True)
        print("Task 3: Cross-model consistency", flush=True)
        print("=" * 60, flush=True)

        cross_results = cross_model_consistency(
            models={pair[0]: model_1, pair[1]: model_2},
            V_commons={pair[0]: V_common, pair[1]: V_common},
            trajectories=trajectories,
            n_samples=args.n_samples,
            alphas=alphas,
            device=args.device,
        )

    # ------------------------------------------------------------------
    # Task 4: Summary report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("Task 4: Summary", flush=True)
    print("=" * 60, flush=True)

    summary = b3_summary_table(results, cross_results)
    save_summary(summary)

    print("\nB3 complete.", flush=True)


if __name__ == "__main__":
    main()
