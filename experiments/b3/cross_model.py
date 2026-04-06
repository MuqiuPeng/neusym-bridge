"""B3 Task 3: Cross-model consistency verification.

Verifies that the same common direction produces consistent causal effects
across both models in the pair (model_b and model_c by default).
"""

from __future__ import annotations

import numpy as np

from experiments.b3.intervention_robust import run_robust_intervention


def cross_model_consistency(
    models: dict,
    V_commons: dict,
    trajectories,
    n_samples: int = 50,
    alphas: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Run the robustness scan on each model independently and compare.

    Args:
        models: {"model_b": HeatWorldModel, "model_c": HeatWorldModel}
        V_commons: {"model_b": V_common_array, "model_c": V_common_array}
            Each V_common is the SVCCA common basis projected into that
            model's latent space.  For a symmetric check, both can use
            the averaged basis from build_common_basis.
        trajectories: Full trajectory tensor.
        n_samples: Samples per direction.

    Returns:
        Dict with per-model results and a consistency summary.
    """
    if alphas is None:
        alphas = np.linspace(-4, 4, 9)

    results_per_model = {}
    for model_name, model in models.items():
        print(f"\n--- {model_name} ---", flush=True)
        V = V_commons[model_name]
        results_per_model[model_name] = run_robust_intervention(
            model, V, trajectories,
            n_samples=n_samples, alphas=alphas, device=device,
        )

    # Compare monotone rates across models
    model_names = list(models.keys())
    n_directions = len(results_per_model[model_names[0]])

    print("\n" + "=" * 50)
    print("Cross-model monotone rate comparison")
    print("=" * 50)
    header = f"{'Dir':>4}"
    for mn in model_names:
        header += f"  {mn:>10}"
    header += f"  {'consistent':>10}"
    print(header)
    print("-" * len(header))

    n_consistent = 0
    consistency_details = []
    for dir_idx in range(n_directions):
        mono_vals = {}
        for mn in model_names:
            mono_vals[mn] = results_per_model[mn][dir_idx]["monotone_rate"]

        # Consistent if both above threshold or both below
        verdicts = [v > 0.7 for v in mono_vals.values()]
        consistent = len(set(verdicts)) == 1
        if consistent:
            n_consistent += 1

        row = f"  {dir_idx + 1:>2}"
        for mn in model_names:
            row += f"  {mono_vals[mn]:>10.3f}"
        row += f"  {'Y' if consistent else 'N':>10}"
        print(row)

        consistency_details.append({
            "direction": dir_idx,
            "mono_rates": mono_vals,
            "consistent": consistent,
        })

    print(f"\nConsistent directions: {n_consistent}/{n_directions}")

    return {
        "per_model": results_per_model,
        "consistency": consistency_details,
        "n_consistent": n_consistent,
    }
