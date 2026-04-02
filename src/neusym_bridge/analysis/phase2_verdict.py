"""Phase 2 verdict: aggregate SINDy results, transfer test, generate Relatum predicates."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def sindy_to_relatum(sindy_analysis: dict, strength_threshold: float = 0.01) -> list[dict]:
    """Convert SINDy coefficients to Relatum relation declarations.

    Args:
        sindy_analysis: Output from analyze_sindy_coefficients().
        strength_threshold: Minimum absolute coefficient to include.

    Returns:
        List of relation dicts with type, name, strength, direction, dim.
    """
    relations = []
    for term in sindy_analysis["top_terms"]:
        if abs(term["coefficient"]) < strength_threshold:
            continue

        feat = term["feature"]
        is_nonlinear = "^" in feat or " " in feat  # "x0 x1" or "x0^2"

        relations.append({
            "type": "binary" if is_nonlinear else "unary",
            "name": feat.replace(" ", "_"),
            "strength": abs(term["coefficient"]),
            "direction": "positive" if term["coefficient"] > 0 else "negative",
            "equation": term["equation"],
        })

    return sorted(relations, key=lambda x: -x["strength"])


def format_relatum_prolog(relations: list[dict]) -> str:
    """Format relations as Prolog-style Relatum declarations."""
    lines = ["% Auto-generated from SINDy (Phase 2)", "% Domain: heat conduction", ""]
    seen = set()
    for rel in relations:
        name = f"sindy_{rel['name']}"
        if name in seen:
            continue
        seen.add(name)
        if rel["type"] == "unary":
            lines.append(f"relation {name}(State).  % coef={rel['direction']} {rel['strength']:.4f}")
        else:
            lines.append(f"relation {name}(State, State).  % coef={rel['direction']} {rel['strength']:.4f}")
    return "\n".join(lines)


def phase2_verdict(results: dict) -> dict:
    """Evaluate Phase 2 results.

    Args:
        results: Dictionary with keys:
            - max_correlation: highest canonical correlation from SVCCA
            - score_plan_a: SINDy R² for 1-dim plan
            - score_plan_b: SINDy R² for 8-dim plan
            - has_linear_decay: whether SINDy found decay terms
            - transfer_retention: transfer R² / in-domain R²
            - intervention_systematic: whether interventions show causal effect
            - n_relations: number of auto-generated predicates

    Returns:
        Verdict dict with checks, pass/fail, and Phase 3 inputs.
    """
    checks = {
        "Common directions extracted (corr > 0.9)":
            results["max_correlation"] > 0.9,
        "SINDy R² > 0.7 (at least one plan)":
            max(results["score_plan_a"], results["score_plan_b"]) > 0.7,
        "Recovered equation has linear decay":
            results["has_linear_decay"],
        "Transfer retention > 0.6":
            results["transfer_retention"] > 0.6,
        "Intervention shows causal effect":
            results["intervention_systematic"],
    }

    passed = sum(checks.values())
    best_plan = "plan_b" if results["score_plan_b"] > results["score_plan_a"] else "plan_a"

    verdict = {
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed_count": passed,
        "total_checks": len(checks),
        "overall_pass": passed >= 4,
        "metrics": {
            "max_correlation": results["max_correlation"],
            "score_plan_a": results["score_plan_a"],
            "score_plan_b": results["score_plan_b"],
            "transfer_retention": results["transfer_retention"],
            "n_relations": results["n_relations"],
        },
        "phase3_inputs": {
            "best_plan": best_plan,
            "best_score": max(results["score_plan_a"], results["score_plan_b"]),
            "n_relations": results["n_relations"],
            "transfer_retention": results["transfer_retention"],
        },
    }
    return verdict


def print_phase2_verdict(verdict: dict) -> None:
    print(f"\n{'='*50}")
    print(f"Phase 2 结果: {verdict['passed_count']}/{verdict['total_checks']} 项通过")
    print(f"{'='*50}\n")

    for check, passed in verdict["checks"].items():
        print(f"  {'✓' if passed else '✗'}  {check}")

    print(f"\n关键指标:")
    for k, v in verdict["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if verdict["overall_pass"]:
        p3 = verdict["phase3_inputs"]
        print(f"\n结论: 公共结构对应物理规律，进入 Phase 3")
        print(f"  推荐方案: {p3['best_plan']} (R²={p3['best_score']:.4f})")
        print(f"  自动谓词数: {p3['n_relations']}")
    elif verdict["passed_count"] >= 3:
        print(f"\n结论: 弱阳性，部分对应物理规律")
    else:
        print(f"\n结论: 公共结构未显著对应物理规律")


def save_phase2_verdict(verdict: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
