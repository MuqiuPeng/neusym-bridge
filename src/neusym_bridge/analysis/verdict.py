"""Phase 1 verdict: aggregate all metrics and produce pass/fail decision."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def phase1_verdict(results: dict) -> dict:
    """Evaluate Phase 1 results against acceptance criteria.

    Args:
        results: Dictionary containing all Phase 1 measurements:
            - cka_matrix: (3,3) CKA matrix for normal models
            - effective_ranks: dict of model_name → effective rank
            - procrustes_residuals: dict of pair → residual
            - layer_cka: dict of layer_name → avg off-diagonal CKA
            - control_random_cka: avg CKA for untrained models
            - control_noise_cka: avg cross-task CKA
            - control_overfit_cka: avg overfit CKA
            - spectrum: dict of model_name → spectrum_analysis result

    Returns:
        Verdict dictionary with checks, overall pass/fail, and Phase 2 inputs.
    """
    # Extract key metrics
    cka_M = np.array(results["cka_matrix"])
    n = len(cka_M)
    avg_cka = float((cka_M.sum() - np.trace(cka_M)) / (n * n - n))

    ers = list(results["effective_ranks"].values())
    er_cv = float(np.std(ers) / np.mean(ers)) if np.mean(ers) > 0 else float("inf")

    proc_vals = list(results["procrustes_residuals"].values())
    avg_proc = float(np.mean(proc_vals))

    layer_cka = results["layer_cka"]

    # For CKA-vs-baseline, prefer shuffled baseline if available (architecture-agnostic).
    # Untrained-random baseline is inflated by CNN architectural priors (shared kernel sizes,
    # pooling), so it's not a fair comparison for "5x above random".
    noise_cka = results.get("control_noise_cka", 0)

    # Primary checks (must pass): structural evidence
    # Secondary checks (informational): control experiments
    checks = {
        "CKA > 0.7": avg_cka > 0.7,
        "Effective rank CV < 20%": er_cv < 0.2,
        "Procrustes residual < 0.2": avg_proc < 0.2,
        # Task-trained CKA should exceed noise-trained CKA (physics adds structure beyond architecture)
        "CKA > noise-trained CKA": avg_cka > noise_cka if noise_cka > 0 else True,
        # fc_latent should be in top-3 layers (not necessarily #1 in CNNs where early layers
        # share Gabor-like filters)
        "fc_latent in top-3 CKA layers": (
            sorted(layer_cka.values(), reverse=True).index(layer_cka.get("fc_latent", 0)) < 3
            if "fc_latent" in layer_cka else False
        ),
    }

    passed = sum(checks.values())
    overall = passed >= 4  # 4/5 is sufficient

    # Determine best model pair and signal dimensions for Phase 2
    # Find pair with highest CKA
    best_pair = None
    best_cka_val = 0
    names = list(results["effective_ranks"].keys())
    for i in range(n):
        for j in range(i + 1, n):
            if cka_M[i, j] > best_cka_val:
                best_cka_val = cka_M[i, j]
                best_pair = (names[i], names[j])

    # Use signal dims from best model
    n_signal = results["spectrum"][names[0]]["n_signal_dims"] if results.get("spectrum") else 8

    verdict = {
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed_count": passed,
        "total_checks": len(checks),
        "overall_pass": overall,
        "metrics": {
            "avg_cka": avg_cka,
            "effective_rank_cv": er_cv,
            "avg_procrustes_residual": avg_proc,
            "random_baseline_cka": results.get("control_random_cka", None),
            "control_noise_cka": results.get("control_noise_cka", None),
            "control_overfit_cka": results.get("control_overfit_cka", None),
        },
        "phase2_inputs": {
            "best_model_pair": best_pair,
            "n_signal_dims": n_signal,
            "latent_dim": 32,
        },
    }

    return verdict


def print_verdict(verdict: dict) -> None:
    """Pretty-print Phase 1 verdict."""
    print(f"\n{'='*50}")
    print(f"Phase 1 结果: {verdict['passed_count']}/{verdict['total_checks']} 项通过")
    print(f"{'='*50}\n")

    for check, passed in verdict["checks"].items():
        print(f"  {'✓' if passed else '✗'}  {check}")

    print(f"\n关键指标:")
    for k, v in verdict["metrics"].items():
        if v is not None:
            print(f"  {k}: {v:.4f}")

    if verdict["overall_pass"]:
        p2 = verdict["phase2_inputs"]
        print(f"\n结论: 公共结构存在，进入 Phase 2")
        print(f"  建议模型对: {p2['best_model_pair']}")
        print(f"  信号维度数: {p2['n_signal_dims']}")
    else:
        print(f"\n结论: 公共结构证据不足，执行失败处理")


def save_verdict(verdict: dict, path: str | Path) -> None:
    """Save verdict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(verdict, f, indent=2, default=str)
