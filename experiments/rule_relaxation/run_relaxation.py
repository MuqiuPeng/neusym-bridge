"""Conjunction Rule Relaxation experiment.

Tests whether Relatum's +4.8% negative impact comes from overly strict
conjunction rules or from inherent inference latency.

4 Relatum rule variants + NoRelatum baseline, same 100 tasks.

Usage:
    python experiments/rule_relaxation/run_relaxation.py
    python experiments/rule_relaxation/run_relaxation.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.a1.variants import TemporalContrastiveModel, ReconstructionAE
from experiments.contrastive_full.run_contrastive_full import (
    load_models, train_interface_for_encoder,
)
from experiments.ablation_norelatum.run_norelatum import NoRelatumPlanner
from phase4.interface.probe_interface import InterfaceLayer
from phase4.data.generate_tentacle_data import load_tentacle_dataset
from phase4.planning.task import generate_task_suite, execute_plan
from phase4.envs.tentacle_env import ACTION_DIM, MAX_TENSION
from src.neusym_bridge.relatum.interface import RelatumInterface

DATA_PATH = "phase4/data/tentacle_data.h5"
RESULTS_DIR = Path("experiments/rule_relaxation/results")

# =====================================================================
# Rule variants (using tentacle domain predicates)
# =====================================================================

# Strict: all 3 predicates required (current Phase 4 / contrastive_full)
RULE_STRICT = """
structural_risk(N) :- curvature_high(N), tension_saturated(N), tip_deviation(N).
"""

# Medium: any 2 of 3 predicates
RULE_MEDIUM = """
structural_risk(N) :- curvature_high(N), tension_saturated(N).
structural_risk(N) :- curvature_high(N), tip_deviation(N).
structural_risk(N) :- tension_saturated(N), tip_deviation(N).
"""

# Loose: any 1 predicate (closest to NoRelatum behavior)
RULE_LOOSE = """
structural_risk(N) :- curvature_high(N).
structural_risk(N) :- tension_saturated(N).
structural_risk(N) :- tip_deviation(N).
"""


# =====================================================================
# Relatum-based planner with configurable rules
# =====================================================================

class ConfigurableRelatumPlanner:
    """FullSystemPlanner with configurable rules and threshold."""

    def __init__(self, model, interface, rules, threshold=0.6, device="cpu"):
        self.model = model.eval().to(device)
        self.interface = interface.eval().to(device)
        self.device = device
        self.rules = rules
        self.threshold = threshold

    def plan(self, start, target, n_steps=50):
        with torch.no_grad():
            z_current = self.model.encode(
                torch.tensor(start, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)
            z_target = self.model.encode(
                torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)

        actions = []
        n_safe = 0

        ri = RelatumInterface()
        ri.load_rules_from_text(self.rules)
        ri.set_collapse_threshold("structural_risk", self.threshold)

        for t in range(n_steps):
            node_id = f"step_{t}"
            self.interface.to_relatum_assertions(z_current, node_id, ri)
            ri.update_closure([])
            risk = ri.is_collapsed(f"structural_risk({node_id})")

            if risk:
                action = self._safe_action(z_current, z_target)
                n_safe += 1
            else:
                action = self._energy_optimal_action(z_current, z_target)

            actions.append(action)

            with torch.no_grad():
                a_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(self.device)
                z_current = self.model.predict(z_current.unsqueeze(0), a_tensor).squeeze(0)

        return actions, n_safe

    def _energy_optimal_action(self, z_current, z_target):
        delta = (z_target - z_current).cpu().numpy()
        error_norm = np.linalg.norm(delta) + 1e-8
        scale = min(2.0, error_norm) * MAX_TENSION * 0.3
        delta_norm = delta / error_norm
        bias = np.tile(delta_norm, ACTION_DIM // len(delta_norm) + 1)[:ACTION_DIM]
        bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)
        rng = np.random.RandomState(hash(tuple(delta[:5].tolist())) % 2**31)
        noise = rng.exponential(0.3, size=ACTION_DIM)
        action = scale * (0.6 * bias + 0.4 * noise)
        return np.clip(action, 0.0, MAX_TENSION)

    def _safe_action(self, z_current, z_target):
        return self._energy_optimal_action(z_current, z_target) * 0.3


# =====================================================================
# Main experiment
# =====================================================================

def run_configs(model_c, interface_c, n_tasks=100, device="cpu"):
    """Run all 5 configurations on the same task set."""
    print(f"Generating {n_tasks} planning tasks...")
    tasks = generate_task_suite(n_tasks=n_tasks, seed=42)

    configs = {
        "norelatum":   ("norelatum", None, None),
        "strict_060":  ("relatum", RULE_STRICT, 0.60),   # current baseline
        "strict_040":  ("relatum", RULE_STRICT, 0.40),   # lower threshold
        "medium_060":  ("relatum", RULE_MEDIUM, 0.60),   # 2-of-3
        "loose_060":   ("relatum", RULE_LOOSE,  0.60),   # 1-of-3
    }

    results = {}

    for name, (mode, rules, threshold) in configs.items():
        print(f"\n  Running: {name}", flush=True)
        distances = []
        safe_counts = []

        for i, task in enumerate(tasks):
            if i % 20 == 0:
                print(f"    Task {i}/{n_tasks}", flush=True)

            if mode == "norelatum":
                planner = NoRelatumPlanner(model_c, interface_c, device=device)
                actions, n_safe = planner.plan(task.start, task.target)
            else:
                planner = ConfigurableRelatumPlanner(
                    model_c, interface_c, rules, threshold, device=device,
                )
                actions, n_safe = planner.plan(task.start, task.target)

            traj = execute_plan(actions, task.start)
            metrics = task.evaluate(traj)
            distances.append(metrics["distance"])
            safe_counts.append(n_safe)

        avg_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))
        avg_safe = float(np.mean(safe_counts))

        results[name] = {
            "avg_distance": avg_dist,
            "std_distance": std_dist,
            "avg_safe_steps": avg_safe,
            "safe_rate": avg_safe / 50.0,
        }

        print(f"    dist={avg_dist:.1f} +/- {std_dist:.1f}, "
              f"safe={avg_safe:.1f}/50 ({avg_safe/50*100:.0f}%)")

    return results


def print_results(results):
    print(f"\n{'='*70}")
    print("Conjunction Rule Relaxation Results")
    print(f"{'='*70}")
    print(f"{'Config':20} {'Avg Dist':>10} {'Std':>8} {'Safe Rate':>10} {'vs NoRelatum':>14}")
    print("-" * 70)

    baseline = results["norelatum"]["avg_distance"]

    for name, r in results.items():
        delta = r["avg_distance"] - baseline
        pct = delta / baseline * 100
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:20} {r['avg_distance']:>10.1f} {r['std_distance']:>8.1f} "
            f"{r['safe_rate']:>10.1%} {sign}{delta:>8.1f} ({sign}{pct:.1f}%)"
        )

    print("=" * 70)

    # Determine outcome
    norel = baseline
    strict = results["strict_060"]["avg_distance"]
    loose = results["loose_060"]["avg_distance"]

    if abs(loose - norel) < norel * 0.03:
        outcome = "A"
        desc = "Loose rule matches NoRelatum -> conjunction is the cause"
    elif abs(results["strict_040"]["avg_distance"] - norel) < norel * 0.03:
        outcome = "B"
        desc = "Lower threshold matches NoRelatum -> threshold is the cause"
    else:
        outcome = "C"
        desc = "All Relatum variants worse -> inherent inference latency cost"

    print(f"\nOutcome: Scenario {outcome}")
    print(f"  {desc}")

    return outcome


def main():
    parser = argparse.ArgumentParser(description="Rule Relaxation Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_tasks = 20 if args.quick else 100
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    model_c, _ = load_models(device)

    # Train interface
    print("\nTraining Interface Layer...")
    s_t, a_t, s_t1 = load_tentacle_dataset(DATA_PATH, max_trajectories=500)
    interface_c = train_interface_for_encoder(
        model_c, s_t, a_t, s_t1, device=device,
    )
    del s_t, a_t, s_t1
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Run experiment
    print(f"\n{'='*60}")
    print("Rule Relaxation Experiment")
    print(f"{'='*60}")
    results = run_configs(model_c, interface_c, n_tasks=n_tasks, device=device)
    outcome = print_results(results)

    # Save
    output = {"results": results, "outcome": outcome, "n_tasks": n_tasks}
    out_path = RESULTS_DIR / "rule_relaxation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
