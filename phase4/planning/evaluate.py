"""Ablation experiment runner and evaluation.

Runs all four planner variants on a suite of planning tasks
and compares success rate, energy efficiency, and explanation quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.planning.task import (
    TentaclePlanningTask,
    execute_plan,
    generate_task_suite,
)
from phase4.planning.planners import (
    FullSystemPlanner,
    PureLEWMPlanner,
    PureRelatumPlanner,
    HardThresholdPlanner,
)
from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.interface.probe_interface import InterfaceLayer
from src.neusym_bridge.relatum.interface import RelatumInterface


def run_ablation(
    lewm: LeWMTentacle,
    interface: InterfaceLayer,
    n_tasks: int = 100,
    n_plan_steps: int = 50,
    device: str = "cpu",
) -> dict:
    """Run the full ablation experiment.

    Evaluates four planner variants on n_tasks random tasks.

    Args:
        lewm: Trained LeWM model.
        interface: Trained interface layer.
        n_tasks: Number of evaluation tasks.
        n_plan_steps: Steps per planning episode.
        device: Device for inference.

    Returns:
        Dict mapping planner name to list of per-task results.
    """
    print(f"Generating {n_tasks} planning tasks...")
    tasks = generate_task_suite(n_tasks=n_tasks)

    planners = {
        "full_system": FullSystemPlanner(lewm, interface, device=device),
        "pure_lewm": PureLEWMPlanner(lewm, device=device),
        "pure_relatum": PureRelatumPlanner(),
        "hard_threshold": HardThresholdPlanner(lewm, interface, device=device),
    }

    results = {name: [] for name in planners}

    for i, task in enumerate(tasks):
        if i % 10 == 0:
            print(f"  Task {i}/{n_tasks}", flush=True)

        for name, planner in planners.items():
            actions = planner.plan(task.start, task.target, n_steps=n_plan_steps)
            trajectory = execute_plan(actions, task.start)
            metrics = task.evaluate(trajectory)

            # For full system, also capture Relatum diagnostics
            if name == "full_system":
                metrics["relatum_diagnosis"] = _get_diagnosis(
                    lewm, interface, trajectory, device
                )

            results[name].append(metrics)

    return results


def _get_diagnosis(
    lewm: LeWMTentacle,
    interface: InterfaceLayer,
    trajectory: dict,
    device: str,
) -> dict | None:
    """Get Relatum diagnosis for the final state of a trajectory."""
    import torch

    final_state = trajectory["states"][-1]
    s = torch.tensor(final_state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        z = lewm.encode(s).squeeze(0)

    ri = RelatumInterface()
    ri.load_rules_from_text(
        "structural_risk(N) :- curvature_high(N), tension_saturated(N), tip_deviation(N).\n"
    )
    ri.set_collapse_threshold("structural_risk", 0.6)

    interface.to_relatum_assertions(z, "final", ri)
    ri.update_closure([])

    if ri.is_collapsed("structural_risk(final)"):
        proof = ri.explain("structural_risk(final)")
        return {
            "collapsed": True,
            "confidence": ri.get_confidence("structural_risk(final)"),
            "proof_steps": len(proof),
        }
    return None


def summarize_results(results: dict) -> dict:
    """Compute summary statistics for ablation results.

    Returns:
        Dict with per-planner summary stats.
    """
    summary = {}

    print("\nAblation Results:")
    print(f"{'Method':<20} {'Success Rate':>12} {'Efficiency':>12} {'Avg Distance':>14}")
    print("-" * 60)

    for name, rlist in results.items():
        success_rate = np.mean([r["success"] for r in rlist])
        efficiency = np.mean([r["efficiency"] for r in rlist])
        avg_dist = np.mean([r["distance"] for r in rlist])

        print(f"{name:<20} {success_rate:>12.3f} {efficiency:>12.3f} {avg_dist:>14.4f}")

        summary[name] = {
            "success_rate": float(success_rate),
            "avg_efficiency": float(efficiency),
            "avg_distance": float(avg_dist),
            "n_tasks": len(rlist),
        }

    return summary


def evaluate_explanation_quality(results: dict) -> float:
    """Evaluate explanation quality on failed cases.

    For tasks where the full system failed, check if Relatum
    provided a useful diagnosis.

    Returns:
        Fraction of failed cases with a Relatum explanation.
    """
    full_results = results.get("full_system", [])
    failed = [r for r in full_results if not r["success"]]

    if not failed:
        print("No failed cases to evaluate.")
        return 1.0

    n_explained = sum(
        1 for r in failed
        if r.get("relatum_diagnosis") and r["relatum_diagnosis"].get("proof_steps", 0) > 0
    )

    rate = n_explained / len(failed)
    print(f"\nExplanation quality:")
    print(f"  Failed cases: {len(failed)}")
    print(f"  With Relatum diagnosis: {n_explained}")
    print(f"  Explanation rate: {rate:.3f}")

    return rate


def save_ablation_results(
    results: dict,
    summary: dict,
    explanation_rate: float,
    output_path: str | Path = "phase4/results/ablation_results.json",
) -> None:
    """Save ablation results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    output = {
        "summary": summary,
        "explanation_rate": explanation_rate,
        "per_task": {
            name: [{k: convert(v) for k, v in r.items()} for r in rlist]
            for name, rlist in results.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
