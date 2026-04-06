"""B3 Task 4: Summary report generation.

Produces the final summary table and JSON for inclusion in the paper.
"""

from __future__ import annotations

import json
import numpy as np
from scipy import stats


PHASE2_LABELS = ["aux temp", "non-causal", "spatial struct", "main temp"]


def b3_summary_table(
    results: dict,
    cross_model_results: dict | None = None,
) -> dict:
    """Print and return the final B3 summary.

    Returns:
        Dict suitable for JSON serialisation.
    """
    n_directions = len(results)
    n_samples = len(results[0]["effect_sizes"])

    print()
    print("=" * 75)
    print("B3 Final Summary")
    print("=" * 75)
    print(
        f"{'Dir':>4} {'mono':>8} {'effect_mu':>10} {'effect_sd':>10} "
        f"{'rho_mu':>8} {'p(effect)':>10} {'verdict':>10}"
    )
    print("-" * 75)

    summary_rows = []
    causal_count = 0

    for dir_idx in range(n_directions):
        r = results[dir_idx]
        sizes = np.array(r["effect_sizes"])
        rhos = np.array(r["spearman_rhos"])
        mono = r["monotone_rate"]

        _, p_effect = stats.ttest_1samp(sizes, 0)
        is_causal = (mono > 0.7) and (p_effect < 0.05)
        if is_causal:
            causal_count += 1

        label = PHASE2_LABELS[dir_idx] if dir_idx < len(PHASE2_LABELS) else "?"
        verdict = "CAUSAL" if is_causal else "-"

        print(
            f"  {dir_idx + 1:>2} {mono:>8.3f} {sizes.mean():>10.4f} "
            f"{sizes.std():>10.4f} {rhos.mean():>8.3f} "
            f"{p_effect:>10.4f} {verdict:>10}  ({label})"
        )

        summary_rows.append({
            "direction": dir_idx + 1,
            "label": label,
            "monotone_rate": round(float(mono), 4),
            "effect_mean": round(float(sizes.mean()), 6),
            "effect_std": round(float(sizes.std()), 6),
            "rho_mean": round(float(rhos.mean()), 4),
            "rho_std": round(float(rhos.std()), 4),
            "p_effect": round(float(p_effect), 6),
            "is_causal": is_causal,
        })

    print("=" * 75)
    print(f"\nPhase 2 conclusion: 3/4 directions monotonic (visual inspection)")
    print(f"B3 conclusion    : {causal_count}/4 directions statistically causal "
          f"(n={n_samples})")

    summary = {
        "n_samples": n_samples,
        "causal_count": causal_count,
        "n_directions": n_directions,
        "directions": summary_rows,
    }

    if cross_model_results is not None:
        summary["cross_model_consistent"] = cross_model_results["n_consistent"]
        summary["cross_model_details"] = cross_model_results["consistency"]

    return summary


def save_summary(summary: dict, path: str = "experiments/b3/outputs/b3_summary.json"):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {path}")
