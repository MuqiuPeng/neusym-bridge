"""NoRelatum ablation: isolate Relatum's contribution in Contrastive Full System.

Adds the missing ablation point:
  Contrastive Pure       (no Interface, no Relatum)  = 337.5
  Contrastive NoRelatum  (Interface, no Relatum)     = ???
  Contrastive Full       (Interface + Relatum)       = 97.8

This decomposes the symbolic layer contribution into:
  Interface contribution  = Pure - NoRelatum
  Relatum contribution    = NoRelatum - Full

Usage:
    python experiments/ablation_norelatum/run_norelatum.py
    python experiments/ablation_norelatum/run_norelatum.py --quick
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

from experiments.a1.variants import ReconstructionAE, TemporalContrastiveModel
from experiments.contrastive_full.run_contrastive_full import (
    load_models, train_interface_for_encoder,
)
from phase4.interface.probe_interface import InterfaceLayer
from phase4.data.generate_tentacle_data import load_tentacle_dataset
from phase4.planning.task import generate_task_suite, execute_plan
from phase4.planning.planners import FullSystemPlanner, PureLEWMPlanner
from phase4.planning.evaluate import _get_diagnosis
from phase4.envs.tentacle_env import ACTION_DIM, MAX_TENSION

DATA_PATH = "phase4/data/tentacle_data.h5"
RESULTS_DIR = Path("experiments/ablation_norelatum/results")


# =====================================================================
# NoRelatum Planner
# =====================================================================

class NoRelatumPlanner:
    """Interface-only planner: uses confidence thresholds directly, no Relatum.

    Has Interface Layer for perception but no Noisy-OR, no collapse,
    no provenance tracking. Risk is determined by hard threshold on
    individual predicate confidences.
    """

    def __init__(self, model, interface, device="cpu", threshold=0.5):
        self.model = model.eval().to(device)
        self.interface = interface.eval().to(device)
        self.device = device
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
        n_safe_steps = 0

        for t in range(n_steps):
            with torch.no_grad():
                confs = self.interface(z_current.unsqueeze(0)).squeeze(0)

            # Direct hard threshold on ANY predicate — no Relatum reasoning
            any_risk = (confs > self.threshold).any().item()

            if any_risk:
                action = self._safe_action(z_current, z_target)
                n_safe_steps += 1
            else:
                action = self._energy_optimal_action(z_current, z_target)

            actions.append(action)

            with torch.no_grad():
                a_tensor = torch.tensor(
                    action, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                z_current = self.model.predict(
                    z_current.unsqueeze(0), a_tensor
                ).squeeze(0)

        return actions, n_safe_steps

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

def run_all_configs(model_c, model_r, interface_c, interface_r,
                    n_tasks=100, n_plan_steps=50, device="cpu"):
    """Run all 5 planner configurations on the same task set."""
    print(f"Generating {n_tasks} planning tasks...")
    tasks = generate_task_suite(n_tasks=n_tasks, seed=42)

    # Build planners
    planners = {
        "recon_pure": ("plan_only", PureLEWMPlanner(model_r, device=device)),
        "recon_full": ("plan_only", FullSystemPlanner(model_r, interface_r, device=device)),
        "contrastive_pure": ("plan_only", PureLEWMPlanner(model_c, device=device)),
        "contrastive_norelatum": ("plan_safe", NoRelatumPlanner(model_c, interface_c, device=device)),
        "contrastive_full": ("plan_only", FullSystemPlanner(model_c, interface_c, device=device)),
    }

    results = {name: [] for name in planners}
    safe_counts = {name: [] for name in planners}

    for i, task in enumerate(tasks):
        if i % 10 == 0:
            print(f"  Task {i}/{n_tasks}", flush=True)

        for name, (mode, planner) in planners.items():
            if mode == "plan_safe":
                actions, n_safe = planner.plan(task.start, task.target, n_steps=n_plan_steps)
                safe_counts[name].append(n_safe)
            else:
                actions = planner.plan(task.start, task.target, n_steps=n_plan_steps)

            traj = execute_plan(actions, task.start)
            metrics = task.evaluate(traj)
            results[name].append(metrics)

    return results, tasks, safe_counts


def print_ablation_table(results, safe_counts):
    """Print full 5-way ablation table with contribution decomposition."""
    print(f"\n{'='*75}")
    print("Full Ablation Table")
    print(f"{'='*75}")
    print(f"{'Config':30} {'Avg Dist':>10} {'Interface':>10} {'Relatum':>8}")
    print("-" * 75)

    summaries = {}
    for name, rlist in results.items():
        avg_dist = float(np.mean([r["distance"] for r in rlist]))
        has_if = "norelatum" in name or "full" in name
        has_rel = "full" in name
        summaries[name] = avg_dist
        print(f"{name:30} {avg_dist:>10.1f} {'Y' if has_if else 'N':>10} {'Y' if has_rel else 'N':>8}")

    print("=" * 75)

    # Contribution decomposition
    d_rp = summaries["recon_pure"]
    d_rf = summaries["recon_full"]
    d_cp = summaries["contrastive_pure"]
    d_cn = summaries["contrastive_norelatum"]
    d_cf = summaries["contrastive_full"]

    print(f"\nContribution Decomposition (Contrastive path):")
    print(f"  Encoder:    recon_pure -> contrastive_pure    {d_rp:.1f} -> {d_cp:.1f}  ({d_cp - d_rp:+.1f}, {(d_cp - d_rp) / d_rp * 100:+.1f}%)")
    print(f"  Interface:  contrastive_pure -> norelatum     {d_cp:.1f} -> {d_cn:.1f}  ({d_cn - d_cp:+.1f}, {(d_cn - d_cp) / d_cp * 100:+.1f}%)")
    print(f"  Relatum:    norelatum -> full                 {d_cn:.1f} -> {d_cf:.1f}  ({d_cf - d_cn:+.1f}, {(d_cf - d_cn) / d_cn * 100:+.1f}%)")
    print(f"  Total:      recon_pure -> contrastive_full    {d_rp:.1f} -> {d_cf:.1f}  ({d_cf - d_rp:+.1f}, {(d_cf - d_rp) / d_rp * 100:+.1f}%)")

    # NoRelatum safe step stats
    if safe_counts.get("contrastive_norelatum"):
        avg_safe = np.mean(safe_counts["contrastive_norelatum"])
        print(f"\n  NoRelatum safe steps: {avg_safe:.1f}/50 ({avg_safe/50*100:.0f}% of steps)")

    # Paired comparison
    print(f"\nPaired Task Analysis (NoRelatum vs Full, same 100 tasks):")
    helped = neutral = hurt = 0
    for r_nr, r_full in zip(results["contrastive_norelatum"], results["contrastive_full"]):
        diff = r_nr["distance"] - r_full["distance"]
        if diff > 10:
            helped += 1
        elif diff < -10:
            hurt += 1
        else:
            neutral += 1
    print(f"  Relatum helped (NoRelatum worse by >10): {helped}")
    print(f"  Neutral (within 10):                     {neutral}")
    print(f"  Relatum hurt (NoRelatum better by >10):  {hurt}")

    return summaries


def main():
    parser = argparse.ArgumentParser(description="NoRelatum Ablation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_tasks = 20 if args.quick else 100
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load encoders
    model_c, model_r = load_models(device)

    # Train interface layers (same as contrastive_full experiment)
    print("\nTraining Interface Layers...")
    s_t, a_t, s_t1 = load_tentacle_dataset(DATA_PATH, max_trajectories=500)

    print("  Contrastive interface:")
    interface_c = train_interface_for_encoder(
        model_c, s_t.copy(), a_t.copy(), s_t1.copy(), device=device,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    print("  Reconstruction interface:")
    interface_r = train_interface_for_encoder(
        model_r, s_t.copy(), a_t.copy(), s_t1.copy(), device=device,
    )
    del s_t, a_t, s_t1
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Run all configs
    print(f"\n{'='*60}")
    print("Running 5-way ablation")
    print(f"{'='*60}")
    results, tasks, safe_counts = run_all_configs(
        model_c, model_r, interface_c, interface_r,
        n_tasks=n_tasks, device=device,
    )

    # Print results
    summaries = print_ablation_table(results, safe_counts)

    # Save
    output = {
        "summaries": {k: float(v) for k, v in summaries.items()},
        "safe_step_stats": {
            "mean": float(np.mean(safe_counts.get("contrastive_norelatum", [0]))),
            "std": float(np.std(safe_counts.get("contrastive_norelatum", [0]))),
        },
        "n_tasks": n_tasks,
    }
    out_path = RESULTS_DIR / "norelatum_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
