"""B4 Task 3: Compare confidence aggregation methods.

Compares Noisy-OR (used by Relatum) vs hard threshold vs mean vs max
in terms of calibration quality and decision accuracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from experiments.b4.calibration import compute_ece


AGGREGATION_METHODS = {
    "noisy_or": lambda confs: 1.0 - np.prod(1.0 - confs),
    "hard_threshold": lambda confs: float(np.any(confs > 0.5)),
    "mean": lambda confs: float(np.mean(confs)),
    "max": lambda confs: float(np.max(confs)),
}


def compare_aggregation(
    confs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compare aggregation methods for combining per-predicate confidences.

    For each sample, the "ground truth risk" is defined as: any predicate
    is active (logical OR of binary labels). This matches the Relatum rule
    `structural_risk :- curvature_high, tension_saturated, tip_deviation`
    in a relaxed form (any vs all).

    We test two ground-truth definitions:
    - any_risk: OR of labels (any predicate active)
    - all_risk: AND of labels (all predicates active, matching the actual rule)

    Returns:
        Dict with per-method ECE, accuracy, F1.
    """
    n = len(confs)

    # Ground truth: structural_risk requires ALL predicates
    gt_all = (labels.sum(axis=1) == labels.shape[1]).astype(float)
    # Relaxed: ANY predicate active
    gt_any = (labels.sum(axis=1) > 0).astype(float)

    results = {}

    print("\nAggregation Method Comparison:")
    print(f"{'Method':18} {'ECE(all)':>10} {'F1(all)':>8} "
          f"{'ECE(any)':>10} {'F1(any)':>8} {'Acc(any)':>8}")
    print("-" * 70)

    for method_name, aggregator in AGGREGATION_METHODS.items():
        agg_confs = np.array([aggregator(confs[i]) for i in range(n)])
        decisions = (agg_confs > threshold).astype(float)

        ece_all = compute_ece(gt_all, agg_confs)
        ece_any = compute_ece(gt_any, agg_confs)
        f1_all = float(f1_score(gt_all, decisions, zero_division=0))
        f1_any = float(f1_score(gt_any, decisions, zero_division=0))
        acc_any = float((decisions == gt_any).mean())

        results[method_name] = {
            "ece_all": ece_all,
            "ece_any": ece_any,
            "f1_all": f1_all,
            "f1_any": f1_any,
            "acc_any": acc_any,
        }

        print(f"{method_name:18} {ece_all:>10.4f} {f1_all:>8.3f} "
              f"{ece_any:>10.4f} {f1_any:>8.3f} {acc_any:>8.3f}")

    return results
