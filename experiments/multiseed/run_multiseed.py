"""Multi-seed replication of A1 training objective ablation.

3 variants x 3 seeds = 9 runs. Seed=42 results reused from A1.
Each run: train 50 epochs + full evaluation (ER, SINDy, probes, planning).

Usage:
    python experiments/multiseed/run_multiseed.py          # full run
    python experiments/multiseed/run_multiseed.py --quick   # reduced epochs
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

from experiments.a1.variants import ReconstructionAE, PredictiveModel, TemporalContrastiveModel
from experiments.a1.train_a1 import make_dataset, train_variant
from experiments.a1.evaluate_a1 import (
    collect_latents_generic,
    collect_trajectory_latents,
    eval_effective_rank,
    eval_probes,
    eval_sindy,
    eval_planning,
)
from phase4.data.generate_tentacle_data import load_tentacle_dataset

DATA_PATH = "phase4/data/tentacle_data.h5"
CKPT_DIR = Path("experiments/multiseed/checkpoints")
RESULTS_DIR = Path("experiments/multiseed/results")

VARIANTS = ["reconstruction", "predictive", "contrastive"]
SEEDS = [42, 137, 999]


def build_model(variant_name, latent_dim=64):
    if variant_name == "reconstruction":
        return ReconstructionAE(latent_dim=latent_dim)
    elif variant_name == "predictive":
        return PredictiveModel(latent_dim=latent_dim)
    elif variant_name == "contrastive":
        return TemporalContrastiveModel(latent_dim=latent_dim)
    raise ValueError(f"Unknown variant: {variant_name}")


def evaluate_model(name, model, data_path, device, n_sindy_traj=50, n_planning_tasks=50):
    """Run full evaluation, return results dict (no Z array)."""
    s_t, _, _ = load_tentacle_dataset(data_path, max_trajectories=100)
    idx = np.random.RandomState(0).choice(len(s_t), min(10000, len(s_t)), replace=False)
    states = s_t[idx]

    Z = collect_latents_generic(model, states, device=device)

    results = {}
    results["effective_rank"] = eval_effective_rank(Z)
    results.update(eval_probes(Z, states))

    traj_latents = collect_trajectory_latents(model, data_path, n_traj=n_sindy_traj, device=device)
    sindy_res = eval_sindy(traj_latents)
    results.update(sindy_res)

    results["planning_distance"] = eval_planning(model, n_tasks=n_planning_tasks, device=device)

    return results


def run_single(variant, seed, s_t, a_t, s_t1, neg_idx, n_epochs, device, n_sindy_traj, n_planning_tasks):
    """Train + evaluate one variant/seed combination."""
    result_path = RESULTS_DIR / f"{variant}_seed{seed}.json"
    if result_path.exists():
        print(f"  Skipping {variant} seed={seed} (already exists)")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'='*50}")
    print(f"Training: {variant}  seed={seed}")
    print(f"{'='*50}")

    model = build_model(variant)
    trained_model, history = train_variant(
        variant_name=variant,
        model=model,
        s_t=s_t, a_t=a_t, s_t1=s_t1, neg_idx=neg_idx,
        n_epochs=n_epochs,
        batch_size=256,
        checkpoint_dir=str(CKPT_DIR),
        device=device,
        seed=seed,
    )

    # Save checkpoint
    ckpt_path = CKPT_DIR / f"{variant}_seed{seed}.pt"
    torch.save(trained_model.state_dict(), ckpt_path)

    # Evaluate
    print(f"\n  Evaluating {variant} seed={seed}...")
    results = evaluate_model(
        variant, trained_model, DATA_PATH, device,
        n_sindy_traj=n_sindy_traj,
        n_planning_tasks=n_planning_tasks,
    )
    results["variant"] = variant
    results["seed"] = seed
    results["final_train_loss"] = history["train_loss"][-1]
    results["final_val_loss"] = history["val_loss"][-1]

    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {result_path}")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def copy_seed42_results():
    """Reuse A1's seed=42 results."""
    a1_path = Path("experiments/a1/results/a1_results.json")
    if not a1_path.exists():
        return

    with open(a1_path) as f:
        a1_data = json.load(f)

    for variant in VARIANTS:
        out_path = RESULTS_DIR / f"{variant}_seed42.json"
        if out_path.exists():
            continue
        if variant not in a1_data:
            continue

        a1_res = a1_data[variant]
        result = {
            "variant": variant,
            "seed": 42,
            "effective_rank": a1_res["effective_rank"],
            "sindy_r2": a1_res["sindy_r2"],
            "r2_curvature": a1_res["r2_curvature"],
            "r2_velocity": a1_res["r2_velocity"],
            "r2_tip_position": a1_res["r2_tip_position"],
            "planning_distance": a1_res["planning_distance"],
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Copied A1 seed=42 result for {variant}")


def analyze(all_results):
    """Statistical analysis across seeds."""
    from scipy import stats

    print(f"\n{'='*80}")
    print("Multi-Seed Statistical Summary (mean +/- std, n=3)")
    print(f"{'='*80}")

    metrics = ["effective_rank", "sindy_r2", "planning_distance", "r2_curvature", "r2_velocity"]
    header = f"{'Variant':18}"
    for m in metrics:
        header += f" {m[:14]:>16}"
    print(header)
    print("-" * 80)

    summary = {}
    for variant in VARIANTS:
        vals = {m: [] for m in metrics}
        for seed in SEEDS:
            key = f"{variant}_seed{seed}"
            r = all_results.get(key)
            if r:
                for m in metrics:
                    if m in r:
                        vals[m].append(r[m])

        row = f"{variant:18}"
        summary[variant] = {}
        for m in metrics:
            if vals[m]:
                mean = np.mean(vals[m])
                std = np.std(vals[m])
                summary[variant][m] = {"mean": mean, "std": std, "values": vals[m]}
                row += f" {mean:>8.3f}+/-{std:.3f}"
            else:
                row += f" {'N/A':>16}"
        print(row)
    print("=" * 80)

    # Significance tests
    print(f"\nSignificance Tests (Welch t-test, n=3 per group):")
    tests = [
        ("SINDy R2: Recon vs Predictive", "reconstruction", "predictive", "sindy_r2"),
        ("SINDy R2: Recon vs Contrastive", "reconstruction", "contrastive", "sindy_r2"),
        ("Planning: Recon vs Contrastive", "reconstruction", "contrastive", "planning_distance"),
        ("Eff.Rank: Recon vs Predictive", "reconstruction", "predictive", "effective_rank"),
    ]

    test_results = {}
    for label, v1, v2, metric in tests:
        vals1 = summary.get(v1, {}).get(metric, {}).get("values", [])
        vals2 = summary.get(v2, {}).get(metric, {}).get("values", [])
        if len(vals1) >= 2 and len(vals2) >= 2:
            t_stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
            sig = "**" if p_val < 0.05 else ("*" if p_val < 0.1 else "ns")
            print(f"\n  {label}:")
            print(f"    {v1}: {np.mean(vals1):.3f} +/- {np.std(vals1):.3f}")
            print(f"    {v2}: {np.mean(vals2):.3f} +/- {np.std(vals2):.3f}")
            print(f"    t={t_stat:.3f}, p={p_val:.4f}  {sig}")
            test_results[label] = {"t": float(t_stat), "p": float(p_val), "sig": sig}

    # Consistency check
    print(f"\nSeed Consistency Check:")
    claims = [
        ("Predictive SINDy > Recon SINDy", "predictive", "reconstruction", "sindy_r2", "greater"),
        ("Contrastive Planning < Recon Planning", "contrastive", "reconstruction", "planning_distance", "less"),
        ("Predictive ER < Recon ER", "predictive", "reconstruction", "effective_rank", "less"),
    ]

    consistency = {}
    for desc, v1, v2, metric, direction in claims:
        count = 0
        for seed in SEEDS:
            r1 = all_results.get(f"{v1}_seed{seed}", {})
            r2 = all_results.get(f"{v2}_seed{seed}", {})
            if metric in r1 and metric in r2:
                ok = r1[metric] > r2[metric] if direction == "greater" else r1[metric] < r2[metric]
                count += ok
                status = "Y" if ok else "N"
                print(f"  seed={seed}: {v1}={r1[metric]:.3f} vs {v2}={r2[metric]:.3f} -> {status}")
        print(f"  {desc}: {count}/3 consistent\n")
        consistency[desc] = f"{count}/3"

    return {"summary": summary, "tests": test_results, "consistency": consistency}


def main():
    parser = argparse.ArgumentParser(description="Multi-seed A1 replication")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_epochs = 10 if args.quick else 50
    n_sindy_traj = 20 if args.quick else 50
    n_planning_tasks = 20 if args.quick else 50

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy seed=42 from A1
    print("Checking for existing seed=42 results...")
    copy_seed42_results()

    # Load data once
    print("\nLoading data...")
    s_t, a_t, s_t1, neg_idx = make_dataset(DATA_PATH)
    print(f"Data: {len(s_t)} transitions")

    # Run all combinations
    all_results = {}
    total = len(VARIANTS) * len(SEEDS)
    done = 0

    for seed in SEEDS:
        for variant in VARIANTS:
            result = run_single(
                variant, seed,
                s_t, a_t, s_t1, neg_idx,
                n_epochs=n_epochs,
                device=device,
                n_sindy_traj=n_sindy_traj,
                n_planning_tasks=n_planning_tasks,
            )
            all_results[f"{variant}_seed{seed}"] = result
            done += 1
            print(f"\nProgress: {done}/{total}")

    # Free training data
    del s_t, a_t, s_t1, neg_idx
    gc.collect()

    # Statistical analysis
    analysis = analyze(all_results)

    # Save full analysis
    out_path = RESULTS_DIR / "multiseed_analysis.json"

    # Make serializable
    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(json.dumps(analysis, default=serialize))
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()
