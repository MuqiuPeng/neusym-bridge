"""Phase 2: Does the common structure correspond to physical laws?

Fix order from first run diagnosis:
  1. Sanity check: SINDy directly on temperature field (known answer)
  2. SmoothedFiniteDifference (most critical fix)
  3. Adaptive threshold (calibrate to Xdot scale)
  4. Reduce to 4 dims (avoid feature explosion)
  5. degree=1 first, then degree=2
  6. Re-run transfer test with working SINDy
"""

from pathlib import Path
import json

import torch
import h5py
import numpy as np

from neusym_bridge.models.baseline_mlp import SEED_CONFIGS
from neusym_bridge.models.trainer import load_model
from neusym_bridge.analysis.latent_collector import collect_latents
from neusym_bridge.analysis.structure_extraction import (
    svcca,
    filter_common_directions,
    build_common_basis,
    build_sindy_timeseries,
    build_sindy_trajectories,
    run_sindy,
    run_sindy_multi_trajectory,
    run_wsindy,
    analyze_sindy_coefficients,
    adaptive_threshold,
    transfer_score,
)
from neusym_bridge.data.heat_nonlinear import (
    NonlinearHeatConfig,
    generate_nonlinear_dataset,
)
from neusym_bridge.analysis.phase2_verdict import (
    sindy_to_relatum,
    format_relatum_prolog,
    phase2_verdict,
    print_phase2_verdict,
    save_phase2_verdict,
)

DATA_PATH = Path("data/heat_2d.h5")
MODEL_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 verdict
    with open(RESULTS_DIR / "phase1_verdict.json") as f:
        p1 = json.load(f)
    pair = p1["phase2_inputs"]["best_model_pair"]
    print(f"Using model pair: {pair}", flush=True)

    # Load data
    with h5py.File(DATA_PATH, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
        dt = float(f.attrs["dt"])
        n_steps = int(f.attrs["n_steps"])
    n_traj = trajectories.shape[0]

    # Load models
    model_1 = load_model(MODEL_DIR / f"{pair[0]}.pt")
    model_2 = load_model(MODEL_DIR / f"{pair[1]}.pt")

    # =========================================================================
    # Step 0: Sanity check — SINDy on raw temperature field
    # =========================================================================
    print(f"\n{'='*50}", flush=True)
    print("Step 0: Sanity Check — SINDy on Temperature Field", flush=True)
    print(f"{'='*50}", flush=True)

    # Extract center-point time series from 20 trajectories
    n_sanity = 20
    temp_trajs = []
    for i in range(n_sanity):
        center_ts = trajectories[i, :, 16, 16].numpy().reshape(-1, 1)
        temp_trajs.append(center_ts)

    sanity_model, sanity_score = run_sindy_multi_trajectory(
        temp_trajs, dt=dt, threshold=1e-3, poly_degree=1, alpha=0.01
    )
    print(f"Temperature-field SINDy (center point, degree=1):", flush=True)
    sanity_model.print()
    print(f"R² = {sanity_score:.4f}", flush=True)

    if sanity_score < 0.5:
        print("WARNING: SINDy toolchain sanity check failed!", flush=True)
        print("Trying degree=2...", flush=True)
        sanity_model2, sanity_score2 = run_sindy_multi_trajectory(
            temp_trajs, dt=dt, threshold=1e-3, poly_degree=2, alpha=0.01
        )
        sanity_model2.print()
        print(f"R² (degree=2) = {sanity_score2:.4f}", flush=True)

    # =========================================================================
    # Task 2.1: SVCCA
    # =========================================================================
    print(f"\n{'='*50}", flush=True)
    print("Task 2.1: SVCCA Common Subspace Extraction", flush=True)
    print(f"{'='*50}", flush=True)

    # Encode all frames
    all_frames = trajectories.reshape(-1, 32, 32)
    print(f"Encoding {len(all_frames)} frames...", flush=True)
    Z_1 = collect_latents(model_1, all_frames)
    Z_2 = collect_latents(model_2, all_frames)

    # SVCCA on t=0 samples
    test_inputs = trajectories[:, 0, :, :]
    Z_1_t0 = collect_latents(model_1, test_inputs)
    Z_2_t0 = collect_latents(model_2, test_inputs)

    svcca_result = svcca(Z_1_t0, Z_2_t0, n_components=10)
    corrs = svcca_result["correlations"]
    print(f"SVD retained: A={svcca_result['n_svd_A']}, B={svcca_result['n_svd_B']}", flush=True)
    for i, r in enumerate(corrs):
        print(f"  Direction {i+1}: r = {r:.4f}", flush=True)

    n_sig = int(filter_common_directions(corrs, threshold=0.7).sum())
    print(f"Significant directions: {n_sig}", flush=True)

    # =========================================================================
    # Task 2.2: SINDy with fixes
    # =========================================================================
    print(f"\n{'='*50}", flush=True)
    print("Task 2.2: SINDy Equation Recovery (Fixed)", flush=True)
    print(f"{'='*50}", flush=True)

    # --- Plan A: 1 dim ---
    V_a = build_common_basis(svcca_result["V_A"], svcca_result["V_B"], n_dims=1)
    trajs_a = build_sindy_trajectories(Z_1, V_a, n_traj, n_steps)

    # Adaptive threshold from first trajectory derivatives
    X_a_sample, Xd_a_sample = build_sindy_timeseries(Z_1, V_a, dt, n_traj, n_steps)
    thresh_a = adaptive_threshold(Xd_a_sample, factor=0.1)
    print(f"\nPlan A (1 dim): adaptive threshold = {thresh_a:.6f}", flush=True)

    print("\n--- Plan A, degree=1, SmoothedFiniteDiff ---", flush=True)
    sindy_a1, score_a1 = run_sindy_multi_trajectory(
        trajs_a, dt=dt, threshold=thresh_a, poly_degree=1, alpha=0.1
    )
    sindy_a1.print()
    print(f"R² = {score_a1:.4f}", flush=True)

    print("\n--- Plan A, degree=2, SmoothedFiniteDiff ---", flush=True)
    sindy_a2, score_a2 = run_sindy_multi_trajectory(
        trajs_a, dt=dt, threshold=thresh_a, poly_degree=2, alpha=0.1
    )
    sindy_a2.print()
    print(f"R² = {score_a2:.4f}", flush=True)

    # --- Plan C: 4 dims (new recommended plan) ---
    V_c = build_common_basis(svcca_result["V_A"], svcca_result["V_B"], n_dims=4)
    trajs_c = build_sindy_trajectories(Z_1, V_c, n_traj, n_steps)

    X_c_sample, Xd_c_sample = build_sindy_timeseries(Z_1, V_c, dt, n_traj, n_steps)
    thresh_c = adaptive_threshold(Xd_c_sample, factor=0.1)
    print(f"\nPlan C (4 dims): adaptive threshold = {thresh_c:.6f}", flush=True)
    print(f"  Feature count: degree=1 → {4+1}, degree=2 → {(4+1)*(4+2)//2}", flush=True)

    print("\n--- Plan C, degree=1, SmoothedFiniteDiff ---", flush=True)
    sindy_c1, score_c1 = run_sindy_multi_trajectory(
        trajs_c, dt=dt, threshold=thresh_c, poly_degree=1, alpha=0.1
    )
    sindy_c1.print()
    print(f"R² = {score_c1:.4f}", flush=True)

    print("\n--- Plan C, degree=2, SmoothedFiniteDiff ---", flush=True)
    sindy_c2, score_c2 = run_sindy_multi_trajectory(
        trajs_c, dt=dt, threshold=thresh_c, poly_degree=2, alpha=0.1
    )
    sindy_c2.print()
    print(f"R² = {score_c2:.4f}", flush=True)

    # WSINDy fallback on Plan C if needed
    best_score = max(score_a1, score_a2, score_c1, score_c2)
    used_wsindy = False
    if best_score < 0.7:
        print(f"\n--- WSINDy fallback (Plan C, degree=2) ---", flush=True)
        wsindy_c, wscore_c = run_wsindy(
            X_c_sample, dt=dt, threshold=thresh_c * 0.5, poly_degree=2
        )
        wsindy_c.print()
        print(f"WSINDy R² = {wscore_c:.4f}", flush=True)
        used_wsindy = True
        if wscore_c > best_score:
            best_score = wscore_c

    # Summary
    print(f"\n--- SINDy Score Summary ---", flush=True)
    scores = {
        "plan_a_deg1": score_a1, "plan_a_deg2": score_a2,
        "plan_c_deg1": score_c1, "plan_c_deg2": score_c2,
    }
    for name, s in scores.items():
        marker = " ← BEST" if s == max(scores.values()) else ""
        print(f"  {name:15s}: R² = {s:.4f}{marker}", flush=True)

    # Select best model for downstream
    best_key = max(scores, key=scores.get)
    best_sindy = {
        "plan_a_deg1": sindy_a1, "plan_a_deg2": sindy_a2,
        "plan_c_deg1": sindy_c1, "plan_c_deg2": sindy_c2,
    }[best_key]
    best_V = V_a if "plan_a" in best_key else V_c
    best_score_val = scores[best_key]

    analysis = analyze_sindy_coefficients(best_sindy)
    print(f"\n--- Coefficient Analysis ({best_key}) ---", flush=True)
    print(f"  Sparsity: {analysis['sparsity']:.3f}", flush=True)
    print(f"  Nonzero terms: {analysis['n_nonzero']}", flush=True)
    print(f"  Has linear decay: {analysis['has_linear_decay']}", flush=True)
    for t in analysis["top_terms"][:8]:
        print(f"    eq{t['equation']}: {t['feature']:>12s} = {t['coefficient']:+.6f}", flush=True)

    # =========================================================================
    # Task 2.3: Transfer test
    # =========================================================================
    print(f"\n{'='*50}", flush=True)
    print("Task 2.3: Nonlinear Heat Transfer Test", flush=True)
    print(f"{'='*50}", flush=True)

    nl_config = NonlinearHeatConfig(n_trajectories=100)
    nl_path = Path("data/heat_nonlinear.h5")
    print("Generating nonlinear heat data...", flush=True)
    generate_nonlinear_dataset(nl_config, nl_path)

    with h5py.File(nl_path, "r") as f:
        nl_traj = torch.tensor(f["trajectories"][:], dtype=torch.float32)

    nl_frames = nl_traj.reshape(-1, 32, 32)
    Z_nl = collect_latents(model_1, nl_frames)

    # Build per-trajectory arrays for multi_trajectory scoring
    nl_trajs = build_sindy_trajectories(
        Z_nl, best_V, nl_config.n_trajectories, nl_config.n_steps
    )

    # Transfer: score linear-trained SINDy on nonlinear data
    score_transfer = best_sindy.score(
        nl_trajs, t=nl_config.dt, multiple_trajectories=True
    )
    print(f"Transfer R² (linear → nonlinear): {score_transfer:.4f}", flush=True)

    # In-domain: retrain on nonlinear
    sindy_nl, score_nl = run_sindy_multi_trajectory(
        nl_trajs, dt=nl_config.dt,
        threshold=adaptive_threshold(
            build_sindy_timeseries(Z_nl, best_V, nl_config.dt,
                                   nl_config.n_trajectories, nl_config.n_steps)[1],
            factor=0.1
        ),
        poly_degree=2 if "deg2" in best_key else 1,
        alpha=0.1,
    )
    print(f"Nonlinear in-domain R²: {score_nl:.4f}", flush=True)

    retention = score_transfer / score_nl if score_nl > 1e-6 else 0.0
    print(f"Transfer retention: {retention:.4f}", flush=True)

    # Intervention check
    print(f"\n--- Intervention Check ---", flush=True)
    test_sample = trajectories[0, 0]
    z_ref = model_1.encode(test_sample.unsqueeze(0))
    intervention_systematic = False
    n_common = best_V.shape[1]

    for d in range(min(n_common, 4)):
        direction = torch.tensor(best_V[:, d], dtype=torch.float32)
        recons = []
        for alpha_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            z_pert = z_ref + alpha_val * direction.unsqueeze(0)
            with torch.no_grad():
                T_dec = model_1.decode(z_pert)
            recons.append(T_dec.squeeze().numpy())

        diffs = [float(np.mean(recons[i+1] - recons[i])) for i in range(4)]
        monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
        var_change = float(np.std([r.mean() for r in recons]))
        print(f"  Direction {d+1}: var_change={var_change:.4f}, monotonic={monotonic}", flush=True)
        if var_change > 0.01 and monotonic:
            intervention_systematic = True

    print(f"  Systematic causal effect: {intervention_systematic}", flush=True)

    # =========================================================================
    # Task 2.4: Verdict
    # =========================================================================
    print(f"\n{'='*50}", flush=True)
    print("Task 2.4: Phase 2 Verdict", flush=True)
    print(f"{'='*50}", flush=True)

    relations = sindy_to_relatum(analysis)
    prolog_text = format_relatum_prolog(relations)
    with open(RESULTS_DIR / "relatum_auto_generated.pl", "w") as f:
        f.write(prolog_text)
    print(f"\nRelatum predicates ({len(relations)}) saved", flush=True)
    print(prolog_text, flush=True)

    results_dict = {
        "max_correlation": float(corrs[0]),
        "score_plan_a": float(max(score_a1, score_a2)),
        "score_plan_b": float(max(score_c1, score_c2)),
        "has_linear_decay": analysis["has_linear_decay"],
        "transfer_retention": float(retention),
        "intervention_systematic": intervention_systematic,
        "n_relations": len(relations),
    }

    verdict = phase2_verdict(results_dict)
    print_phase2_verdict(verdict)
    save_phase2_verdict(verdict, RESULTS_DIR / "phase2_verdict.json")

    # Save raw results
    raw = {
        "svcca": {"correlations": corrs.tolist()},
        "sindy_scores": scores,
        "best_plan": best_key,
        "sanity_score": float(sanity_score),
        "analysis": {k: v for k, v in analysis.items() if k != "top_terms"},
        "top_terms": analysis["top_terms"],
        "transfer": {
            "score": float(score_transfer),
            "in_domain": float(score_nl),
            "retention": float(retention),
        },
        "intervention_systematic": intervention_systematic,
        "used_wsindy": used_wsindy,
        "adaptive_thresholds": {
            "plan_a": float(thresh_a),
            "plan_c": float(thresh_c),
        },
    }
    with open(RESULTS_DIR / "phase2_raw_results.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)

    print(f"\nAll results saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
