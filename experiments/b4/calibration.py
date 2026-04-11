"""B4 Task 1: Reliability diagrams and Expected Calibration Error.

Evaluates whether the Interface Layer's confidence outputs are calibrated:
a model that says "confidence 0.7" should be correct ~70% of the time.
"""

from __future__ import annotations

import numpy as np


def compute_ece(labels: np.ndarray, confs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Lower is better; 0 = perfectly calibrated.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return float(ece)


def reliability_curve(
    labels: np.ndarray,
    confs: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability diagram data.

    Returns:
        (bin_confs, bin_accs, bin_counts) for non-empty bins.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confs, bin_accs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_confs.append(confs[mask].mean())
        bin_accs.append(labels[mask].mean())
        bin_counts.append(int(mask.sum()))

    return np.array(bin_confs), np.array(bin_accs), np.array(bin_counts)


def plot_reliability_diagrams(
    confs: np.ndarray,
    labels: np.ndarray,
    predicate_names: list[str],
    output_path: str = "experiments/b4/outputs/b4_reliability_diagram.pdf",
) -> list[float]:
    """Plot reliability diagrams for each predicate.

    Args:
        confs: (N, n_predicates) confidence values.
        labels: (N, n_predicates) binary ground truth.
        predicate_names: Names for each predicate.

    Returns:
        List of ECE scores per predicate.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_pred = len(predicate_names)
    fig, axes = plt.subplots(1, n_pred, figsize=(4 * n_pred, 4))
    if n_pred == 1:
        axes = [axes]

    ece_scores = []

    for i, (ax, name) in enumerate(zip(axes, predicate_names)):
        prob_pred, prob_true, counts = reliability_curve(labels[:, i], confs[:, i])
        ece = compute_ece(labels[:, i], confs[:, i])
        ece_scores.append(ece)

        ax.plot(prob_pred, prob_true, "o-", linewidth=2, label="Interface Layer")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.set_title(f"{name}\nECE={ece:.3f}", fontsize=11)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Interface Layer Calibration\n"
        "(closer to diagonal = better calibrated)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagram saved to {output_path}")

    return ece_scores
