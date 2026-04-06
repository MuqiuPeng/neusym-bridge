"""B3 Task 1: Large-scale intervention robustness scan.

Upgrades Phase 2's 20-point visual inspection to a statistically grounded
analysis: 4 directions × 50 samples × 9 amplitudes = 1800 intervention
experiments, each tested for monotonicity via Spearman correlation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


def run_robust_intervention(
    model,
    V_common: np.ndarray,
    trajectories: torch.Tensor,
    n_samples: int = 50,
    alphas: np.ndarray | None = None,
    device: str = "cpu",
) -> dict:
    """Run intervention experiments across many samples and amplitudes.

    For each common direction, perturb n_samples initial states along that
    direction at each amplitude in alphas, decode, and measure the mean
    temperature effect.  Monotonicity is assessed per-sample via Spearman rho.

    Args:
        model: Trained HeatWorldModel (must have .encode / .decode).
        V_common: Common direction matrix, shape (latent_dim, n_directions).
        trajectories: Full trajectory tensor, shape (n_traj, n_steps, 32, 32).
        n_samples: Number of initial samples to draw from trajectories.
        alphas: Perturbation amplitudes.  Defaults to linspace(-4, 4, 9).
        device: torch device string.

    Returns:
        Dict keyed by direction index, each containing:
            monotone_rate, effect_sizes, spearman_rhos, spearman_pvals,
            direction_consistency, effect_curves.
    """
    if alphas is None:
        alphas = np.linspace(-4, 4, 9)

    n_directions = V_common.shape[1]
    model.eval()
    model.to(device)

    # Draw n_samples distinct initial frames (trajectory_idx, step=0)
    rng = np.random.RandomState(42)
    n_traj = trajectories.shape[0]
    sample_indices = rng.choice(n_traj, size=min(n_samples, n_traj), replace=False)
    samples = trajectories[sample_indices, 0]  # (n_samples, 32, 32)

    results = {}
    for dir_idx in range(n_directions):
        results[dir_idx] = {
            "monotone_rate": None,
            "effect_sizes": [],
            "spearman_rhos": [],
            "spearman_pvals": [],
            "direction_consistency": None,
            "effect_curves": [],
        }

        direction = torch.tensor(
            V_common[:, dir_idx], dtype=torch.float32
        ).to(device)

        print(f"\n  Direction {dir_idx + 1}/{n_directions} ...", flush=True)

        for si in range(len(samples)):
            sample = samples[si].to(device)

            with torch.no_grad():
                z_ref = model.encode(sample.unsqueeze(0)).squeeze(0)

            effects = []
            for alpha in alphas:
                z_pert = z_ref + alpha * direction
                with torch.no_grad():
                    T_dec = model.decode(z_pert.unsqueeze(0)).squeeze()
                effects.append(float(T_dec.mean()))

            effects = np.array(effects)
            results[dir_idx]["effect_curves"].append(effects.tolist())

            # Spearman rho for monotonicity
            rho, pval = spearmanr(alphas, effects)
            results[dir_idx]["spearman_rhos"].append(float(rho))
            results[dir_idx]["spearman_pvals"].append(float(pval))

            # Effect size: range of mean temperature across amplitudes
            effect_size = float(np.max(effects) - np.min(effects))
            results[dir_idx]["effect_sizes"].append(effect_size)

        # Aggregate
        rhos = np.array(results[dir_idx]["spearman_rhos"])
        pvals = np.array(results[dir_idx]["spearman_pvals"])
        sizes = np.array(results[dir_idx]["effect_sizes"])

        # Monotone if |rho| > 0.8 and p < 0.05
        monotone_mask = (np.abs(rhos) > 0.8) & (pvals < 0.05)
        results[dir_idx]["monotone_rate"] = float(monotone_mask.mean())

        # Direction consistency: fraction of monotone samples that agree on sign
        pos_mono = float(((rhos > 0.8) & (pvals < 0.05)).mean())
        neg_mono = float(((rhos < -0.8) & (pvals < 0.05)).mean())
        denom = pos_mono + neg_mono + 1e-8
        results[dir_idx]["direction_consistency"] = float(
            max(pos_mono, neg_mono) / denom
        )

        print(
            f"    monotone_rate={results[dir_idx]['monotone_rate']:.3f}  "
            f"effect={sizes.mean():.4f}±{sizes.std():.4f}  "
            f"rho={rhos.mean():.3f}±{rhos.std():.3f}",
            flush=True,
        )

    return results
