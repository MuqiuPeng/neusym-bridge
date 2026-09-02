# -*- coding: utf-8 -*-
"""Evaluate PCC planners on 50 tasks and compare variants.

Planners:
  PCC_EnergyOpt (lam=1.0)  -- energy + progress joint objective
  PCC_EnergyOpt (lam=0.5)  -- energy-biased variant
  PCC_Greedy               -- direct approach, no energy opt
  PCC_Random               -- random baseline

Usage:
    python -m phase4.planning.evaluate_pcc
    python -m phase4.planning.evaluate_pcc --n-tasks 20 --max-steps 15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import SimplifiedRod
from phase4.envs.pcc_state    import (
    generate_pcc_tasks, set_pcc_state,
    pcc_distance, pcc_to_node_positions,
)
from phase4.envs.pcc_env import PCC_DIM
from phase4.planning.pcc_planner import (
    PCCGreedyPlanner, PCCEnergyPlanner, PCCRandomPlanner,
)

OUTPUT_DIR = Path("phase4/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -- single-task runner --------------------------------------------------------

def run_task(planner, start_pcc: np.ndarray, target_pcc: np.ndarray,
             max_steps: int, tip_tol: float = 0.05) -> dict:
    """Run one task and return metrics.

    Success criterion: Euclidean tip distance < tip_tol [m].
    Uses actual rod node positions (not extract_pcc_state) to avoid
    fit_pcc_module roundtrip errors for large bend angles.
    """
    rod = SimplifiedRod()   # quasi-static (damping=160 already set in __init__)
    set_pcc_state(rod, start_pcc)

    t0           = time.perf_counter()
    traj, energy = planner.plan(rod, target_pcc, max_steps=max_steps)
    elapsed      = time.perf_counter() - t0

    # Use trajectory's last applied_pcc for distance metrics
    # (avoids roundtrip error of extract_pcc_state for large angles)
    final_applied = traj[-1]["applied_pcc"]
    final_dist    = pcc_distance(final_applied, target_pcc)

    # Actual tip distance: rod.position_collection vs target tip
    target_nodes = pcc_to_node_positions(target_pcc)        # (21, 3)
    actual_tip   = rod.position_collection[:, -1]            # (3,)
    tip_dist_m   = float(np.linalg.norm(actual_tip - target_nodes[-1]))
    success      = tip_dist_m < tip_tol

    # Progress ratio in PCC angular distance
    init_dist      = pcc_distance(start_pcc, target_pcc)
    progress_ratio = 1.0 - final_dist / (init_dist + 1e-9)

    return {
        "success":        bool(success),
        "final_dist":     float(final_dist),     # PCC angular dist (applied)
        "tip_dist_m":     float(tip_dist_m),     # actual rod tip distance
        "energy":         float(energy),
        "steps":          len(traj),
        "progress_ratio": float(progress_ratio),
        "elapsed_s":      float(elapsed),
    }


# -- main evaluation -----------------------------------------------------------

def evaluate_pcc_vs_baseline(n_tasks: int = 50, max_steps: int = 20,
                              verbose: bool = True) -> dict:
    """Run all planners on the same task suite and print results."""
    tasks = generate_pcc_tasks(n_tasks=n_tasks, seed=42)

    planners = {
        "PCC_EnergyOpt (lam=1.0)": PCCEnergyPlanner(n_candidates=10, lambda_energy=1.0),
        "PCC_EnergyOpt (lam=0.5)": PCCEnergyPlanner(n_candidates=10, lambda_energy=0.5),
        "PCC_Greedy":              PCCGreedyPlanner(step_fraction=0.30),
        "PCC_Random":              PCCRandomPlanner(),
    }

    all_results: dict[str, list[dict]] = {}

    for name, planner in planners.items():
        if verbose:
            print(f"\n-- {name} --")
        results = []
        for i, (start, target) in enumerate(tasks):
            r = run_task(planner, start, target, max_steps=max_steps)
            results.append(r)
            if verbose and (i % 10 == 0 or i == n_tasks - 1):
                print(f"  task {i+1:>3}/{n_tasks}  "
                      f"success={r['success']}  "
                      f"dist={r['final_dist']:.3f}  "
                      f"energy={r['energy']:.4f}")
        all_results[name] = results

    return all_results


def print_summary(results: dict[str, list[dict]]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 78)
    print(f"{'Planner':<28} {'Success':>8} {'Energy':>10} {'TipDist':>10} {'Progress':>10} {'Steps':>7}")
    print("-" * 78)

    for name, rlist in results.items():
        suc   = np.mean([r["success"]        for r in rlist])
        eng   = np.mean([r["energy"]         for r in rlist])
        tdist = np.mean([r["tip_dist_m"]     for r in rlist])
        prog  = np.mean([r["progress_ratio"] for r in rlist])
        steps = np.mean([r["steps"]          for r in rlist])

        print(f"{name:<28} {suc:>8.2%} {eng:>10.4f} {tdist:>9.4f}m {prog:>10.2%} {steps:>7.1f}")

    print("=" * 78)

    # Energy savings: EnergyOpt (lam=1.0) vs Greedy
    k_opt   = "PCC_EnergyOpt (lam=1.0)"
    k_greed = "PCC_Greedy"
    if k_opt in results and k_greed in results:
        e_opt   = np.mean([r["energy"] for r in results[k_opt]])
        e_greed = np.mean([r["energy"] for r in results[k_greed]])
        if e_greed > 1e-9:
            saving = (1.0 - e_opt / e_greed) * 100
            print(f"\nEnergyOpt (lam=1.0) saves {saving:.1f}% energy vs Greedy")


def save_results(results: dict, output_path: Path) -> None:
    def _conv(obj):
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating,)):  return float(obj)
        if isinstance(obj, np.ndarray):      return obj.tolist()
        if isinstance(obj, (np.bool_,)):     return bool(obj)
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: [{kk: _conv(vv) for kk, vv in r.items()} for r in v]
             for k, v in results.items()},
            f, indent=2,
        )
    print(f"\nSaved: {output_path}")


# -- quick smoke-test (1 task) -------------------------------------------------

def smoke_test() -> None:
    """Verify the full pipeline runs without error on a single task."""
    print("-- Smoke test (1 task each) --")
    from phase4.envs.pcc_state import random_pcc_state

    rng    = np.random.default_rng(7)
    start  = np.zeros(16)
    target = random_pcc_state(rng, theta_scale=0.4)

    for name, planner in [
        ("Greedy",    PCCGreedyPlanner()),
        ("EnergyOpt", PCCEnergyPlanner(n_candidates=5)),
        ("Random",    PCCRandomPlanner()),
    ]:
        r = run_task(planner, start, target, max_steps=5)
        print(f"  {name:<12}  success={r['success']}  "
              f"dist={r['final_dist']:.4f}  energy={r['energy']:.5f}  "
              f"steps={r['steps']}  time={r['elapsed_s']:.2f}s")

    print("Smoke test passed.\n")


# -- CLI -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PCC planners")
    parser.add_argument("--n-tasks",   type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--smoke",     action="store_true",
                        help="Run quick 1-task smoke test only")
    parser.add_argument("--output",    type=str,
                        default="phase4/results/pcc_eval_results.json")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    print(f"Evaluating PCC planners: {args.n_tasks} tasks, {args.max_steps} steps each")
    results = evaluate_pcc_vs_baseline(
        n_tasks   = args.n_tasks,
        max_steps = args.max_steps,
    )
    print_summary(results)
    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
