"""Energy optimization comparison.

Compares PureLEWM (single candidate, no energy awareness) vs
EnergyOptimalPlanner (N candidates, hard progress constraint + min energy).

Reaching the target is a HARD CONSTRAINT, not a soft penalty.
EnergyOptimal selects the minimum-energy candidate that still makes
progress toward the target in latent space; no lambda tuning needed.

Metrics:
  - avg_distance    : mean latent distance to target at end of rollout
  - avg_energy      : mean total physical energy (sum of tensions * steps)
  - success_rate    : fraction of tasks within success_threshold
  - efficiency      : optimal_energy / actual_energy (capped at 1.0)

Usage:
    python -m experiments.energy_opt.run_lambda_sweep
"""

from __future__ import annotations

import sys
sys.modules["elastica"] = None  # Force SimplifiedRod backend (PyElastica forces have no effect)

import json
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.planning.task import generate_task_suite, execute_plan
from phase4.planning.planners import PureLEWMPlanner, EnergyOptimalPlanner, GradientPlanner

OUTPUT_DIR = Path("experiments/energy_opt/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LEWM_CHECKPOINT = Path("phase4/checkpoints/lewm_epoch049.pt")
N_TASKS = 100
N_STEPS = 50
N_CANDIDATES = 20


def _elapsed(t0):
    m, s = divmod(int(time.time() - t0), 60)
    return f"{m:02d}:{s:02d}"


def evaluate_planner(planner, tasks, n_steps, label, t0):
    results = []
    for i, task in enumerate(tasks):
        if (i + 1) % 20 == 0:
            print(f"  [{_elapsed(t0)}] {label}: task {i+1}/{len(tasks)}", flush=True)
        initial_dist = float(np.linalg.norm(task.start - task.target))
        actions = planner.plan(task.start, task.target, n_steps=n_steps)
        trajectory = execute_plan(actions, task.start)
        metrics = task.evaluate(trajectory)
        # progress_ratio: 1.0 = fully solved, 0 = no change, <0 = moved away
        metrics["progress_ratio"] = (initial_dist - metrics["distance"]) / (initial_dist + 1e-8)
        metrics["initial_dist"] = initial_dist
        results.append(metrics)
    return results


def summarize(results: list[dict]) -> dict:
    return {
        "avg_distance":    float(np.mean([r["distance"] for r in results])),
        "avg_energy":      float(np.mean([r["total_energy"] for r in results])),
        "avg_progress":    float(np.mean([r.get("progress_ratio", 0.0) for r in results])),
        "success_rate":    float(np.mean([r["success"] for r in results])),
        "avg_efficiency":  float(np.mean([r["efficiency"] for r in results])),
        "n_tasks": len(results),
    }


def main():
    t0 = time.time()

    # ── Load model ────────────────────────────────────────────────────
    if not LEWM_CHECKPOINT.exists():
        print(f"ERROR: {LEWM_CHECKPOINT} not found.", flush=True)
        print("Run phase4 training first (scripts/run_phase4.py).", flush=True)
        sys.exit(1)

    print(f"[{_elapsed(t0)}] Loading LeWM from {LEWM_CHECKPOINT}", flush=True)
    lewm = LeWMTentacle()
    lewm.load_state_dict(torch.load(LEWM_CHECKPOINT, map_location="cpu"))
    lewm.eval()

    # ── Tasks ─────────────────────────────────────────────────────────
    print(f"[{_elapsed(t0)}] Generating {N_TASKS} tasks...", flush=True)
    tasks = generate_task_suite(n_tasks=N_TASKS, seed=0)

    # ── Planners ──────────────────────────────────────────────────────
    # GradientPlanner: backprop through LeWM, adaptive lr ∝ dist
    # EnergyOptimal λ=0.1: best from previous sweep (included as reference)
    planners = {
        "PureLEWM":              PureLEWMPlanner(lewm),
        "EnergyOpt_ld=0.1":      EnergyOptimalPlanner(lewm, N_CANDIDATES, lambda_dist=0.1),
        "Gradient_lr=0.01":      GradientPlanner(lewm, n_grad_steps=10, lr_scale=0.01),
        "Gradient_lr=0.05":      GradientPlanner(lewm, n_grad_steps=10, lr_scale=0.05),
        "Gradient_lr=0.1":       GradientPlanner(lewm, n_grad_steps=10, lr_scale=0.1),
    }

    # ── Evaluate ──────────────────────────────────────────────────────
    all_results = {}
    for label, planner in planners.items():
        print(f"\n[{_elapsed(t0)}] Evaluating: {label}", flush=True)
        results = evaluate_planner(planner, tasks, N_STEPS, label, t0)
        all_results[label] = results
        s = summarize(results)
        print(f"  dist={s['avg_distance']:.1f}  energy={s['avg_energy']:.3f}  "
              f"success={s['success_rate']:.3f}  efficiency={s['avg_efficiency']:.3f}",
              flush=True)

    # ── Summary table ─────────────────────────────────────────────────
    print(f"\n{'='*72}", flush=True)
    print(f"{'Planner':<25} {'Progress':>10} {'Avg Dist':>10} {'Avg Energy':>12} "
          f"{'Success':>9}", flush=True)
    print(f"{'-'*72}", flush=True)

    baseline = summarize(all_results["PureLEWM"])

    summary = {}
    for label, results in all_results.items():
        s = summarize(results)
        tag = "(baseline)" if label == "PureLEWM" else \
              f"({(s['avg_progress']-baseline['avg_progress'])*100:+.1f}pp)"
        print(f"{label:<25} {s['avg_progress']:>+9.3f}{tag:>12} "
              f"{s['avg_distance']:>10.1f} {s['avg_energy']:>12.3f} "
              f"{s['success_rate']:>9.3f}", flush=True)
        summary[label] = s

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "summary": summary,
        "config": {
            "n_tasks": N_TASKS,
            "n_steps": N_STEPS,
            "n_candidates": N_CANDIDATES,
            "lambda_dist_values": [0.1, 0.5, 1.0, 2.0],
        },
    }
    out_path = OUTPUT_DIR / "lambda_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[{_elapsed(t0)}] Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
