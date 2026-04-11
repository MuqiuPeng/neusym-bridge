"""B4 Task 2: Threshold sensitivity analysis.

Scans precision/recall across thresholds to find the optimal operating point
for each predicate, and compares it to the default 0.5 threshold.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score


def threshold_sensitivity(
    confs: np.ndarray,
    labels: np.ndarray,
    predicate_names: list[str],
) -> dict:
    """Analyze threshold sensitivity for each predicate.

    Returns:
        Dict mapping predicate name to {best_threshold, best_f1,
        precision_05, recall_05, f1_05}.
    """
    results = {}

    print("\nThreshold Sensitivity Analysis:")
    print(f"{'Predicate':25} {'Best Thresh':>12} {'Best F1':>8} "
          f"{'F1@0.5':>8} {'P@0.5':>8} {'R@0.5':>8}")
    print("-" * 75)

    for i, name in enumerate(predicate_names):
        c = confs[:, i]
        y = labels[:, i]

        # Skip if only one class
        if len(np.unique(y)) < 2:
            results[name] = {
                "best_threshold": 0.5,
                "best_f1": 0.0,
                "f1_05": 0.0,
                "precision_05": 0.0,
                "recall_05": 0.0,
            }
            print(f"{name:25} {'N/A':>12} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")
            continue

        precision, recall, thresholds = precision_recall_curve(y, c)
        # F1 at each threshold
        f1s = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1s[:-1])  # last entry has no threshold
        best_thresh = float(thresholds[best_idx])
        best_f1 = float(f1s[best_idx])

        # Performance at threshold=0.5
        preds_05 = (c > 0.5).astype(int)
        f1_05 = float(f1_score(y, preds_05, zero_division=0))
        tp = (preds_05 * y).sum()
        p_05 = float(tp / (preds_05.sum() + 1e-8))
        r_05 = float(tp / (y.sum() + 1e-8))

        results[name] = {
            "best_threshold": best_thresh,
            "best_f1": best_f1,
            "f1_05": f1_05,
            "precision_05": p_05,
            "recall_05": r_05,
        }

        print(f"{name:25} {best_thresh:>12.3f} {best_f1:>8.3f} "
              f"{f1_05:>8.3f} {p_05:>8.3f} {r_05:>8.3f}")

    return results
