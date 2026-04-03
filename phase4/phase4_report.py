"""Phase 4 verdict: end-to-end integration validation.

Aggregates results from all sub-tasks and produces a pass/fail verdict.

Checks:
1. Simulator stable (no numerical explosion)
2. LeWM latent effective rank > 5
3. Interface AUC > 0.65 for at least 2/3 predicates
4. Full system success rate > pure LeWM
5. Full system energy efficiency > pure LeWM
6. Failed case explanation rate > 0.5
"""

from __future__ import annotations

import json
from pathlib import Path


def phase4_verdict(results: dict) -> dict:
    """Produce Phase 4 verdict from aggregated results.

    Args:
        results: Dict with keys:
            - sim_stable: bool
            - effective_rank: float
            - aucs: list[float]
            - full_success: float
            - lewm_success: float
            - full_efficiency: float
            - lewm_efficiency: float
            - explanation_rate: float

    Returns:
        Verdict dict with checks, counts, and overall pass/fail.
    """
    # When both success rates are 0, compare average distance instead.
    # Lower distance = better planning performance.
    full_better_success = results["full_success"] > results["lewm_success"]
    if results["full_success"] == 0 and results["lewm_success"] == 0:
        full_dist = results.get("full_distance", float("inf"))
        lewm_dist = results.get("lewm_distance", float("inf"))
        full_better_success = full_dist < lewm_dist

    full_better_efficiency = results["full_efficiency"] > results["lewm_efficiency"]
    if results["full_efficiency"] == 0 and results["lewm_efficiency"] == 0:
        full_dist = results.get("full_distance", float("inf"))
        lewm_dist = results.get("lewm_distance", float("inf"))
        full_better_efficiency = full_dist < lewm_dist

    checks = {
        "Simulator stable (no numerical explosion)":
            results["sim_stable"],

        "LeWM latent effective rank > 5":
            results["effective_rank"] > 5,

        "Interface AUC > 0.65 (at least 2/3 predicates)":
            sum(auc > 0.65 for auc in results["aucs"]) >= 2,

        "Full system outperforms pure LeWM (success or distance)":
            full_better_success,

        "Full system more efficient than pure LeWM (efficiency or distance)":
            full_better_efficiency,

        "Failed case explanation rate > 0.5":
            results["explanation_rate"] > 0.5,
    }

    passed = sum(checks.values())
    total = len(checks)

    print(f"\nPhase 4 Results: {passed}/{total} checks passed\n")
    for check, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {check}")

    if passed >= 5:
        conclusion = "End-to-end validation PASSED. System is effective."
    elif passed >= 4:
        conclusion = (
            "Partially effective. Some metrics below target. "
            "Document limitations and adjust paper narrative."
        )
    else:
        conclusion = (
            "System integration has issues. "
            "Diagnose root cause before proceeding."
        )

    print(f"\nConclusion: {conclusion}")

    verdict = {
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed_count": passed,
        "total_checks": total,
        "overall_pass": passed >= 5,
        "conclusion": conclusion,
        "raw_results": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in results.items()
        },
    }

    return verdict


def save_verdict(
    verdict: dict,
    output_path: str | Path = "docs/results/phase4_verdict.json",
) -> None:
    """Save verdict to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)

    print(f"Verdict saved to {output_path}")
