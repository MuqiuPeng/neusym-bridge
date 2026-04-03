"""B3 Task 2: Statistical analysis of intervention robustness results.

Produces summary statistics, hypothesis tests, and publication-ready
visualization (2×2 panel showing per-direction effect curves with
confidence bands).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def analyze_intervention_results(
    results: dict,
    alpha_threshold: float = 0.05,
    monotone_rho_threshold: float = 0.8,
    monotone_rate_threshold: float = 0.7,
) -> list[int]:
    """Full statistical analysis of the robustness scan.

    For each direction, reports monotone rate, effect size with CI, and
    Spearman rho distribution.  A direction is deemed "statistically causal"
    when its monotone rate exceeds the threshold AND the effect size is
    significantly > 0 (one-sample t-test).

    Returns:
        List of direction indices judged as causal.
    """
    n_directions = len(results)
    n_samples = len(results[0]["effect_sizes"])

    print()
    print("=" * 65)
    print("B3 Intervention Robustness Analysis")
    print(f"(n={n_samples} samples/direction, "
          f"|rho|>{monotone_rho_threshold}, "
          f"rate>{monotone_rate_threshold})")
    print("=" * 65)

    causal_directions: list[int] = []

    for dir_idx in range(n_directions):
        r = results[dir_idx]
        rhos = np.array(r["spearman_rhos"])
        pvals = np.array(r["spearman_pvals"])
        sizes = np.array(r["effect_sizes"])
        mono_rate = r["monotone_rate"]

        # One-sample t-test: effect size > 0
        t_effect, p_effect = stats.ttest_1samp(sizes, 0)
        # One-sample t-test: mean rho != 0
        t_rho, p_rho = stats.ttest_1samp(rhos, 0)

        # 95% CI for effect size
        ci_effect = stats.t.interval(
            0.95, df=len(sizes) - 1,
            loc=sizes.mean(), scale=stats.sem(sizes),
        )

        # 95% CI for monotone rate (Wilson score interval)
        n_mono = int(round(mono_rate * n_samples))
        ci_mono = _wilson_ci(n_mono, n_samples)

        is_causal = (mono_rate > monotone_rate_threshold) and (p_effect < alpha_threshold)
        if is_causal:
            causal_directions.append(dir_idx)

        tag = "CAUSAL" if is_causal else "non-causal"
        print(f"\nDirection {dir_idx + 1}: {tag}")
        print(f"  Monotone rate : {mono_rate:.3f}  "
              f"95% CI [{ci_mono[0]:.3f}, {ci_mono[1]:.3f}]")
        print(f"  Effect size   : {sizes.mean():.4f} +/- {sizes.std():.4f}  "
              f"95% CI [{ci_effect[0]:.4f}, {ci_effect[1]:.4f}]  "
              f"p={p_effect:.4f}")
        print(f"  Spearman rho  : {rhos.mean():.3f} +/- {rhos.std():.3f}  "
              f"p={p_rho:.4f}")
        print(f"  Dir consistency: {r['direction_consistency']:.3f}")

    print(f"\nCausal directions: {len(causal_directions)}/{n_directions}")
    print(f"  Indices: {[d + 1 for d in causal_directions]}")
    return causal_directions


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z ** 2 / n
    center = (p_hat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def plot_intervention_results(
    results: dict,
    alphas: np.ndarray | None = None,
    output_path: str = "experiments/b3/outputs/b3_intervention_robustness.pdf",
) -> None:
    """Generate 2x2 panel figure for the paper.

    Each panel shows individual sample curves (semi-transparent), mean curve,
    and +/- 1 std confidence band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if alphas is None:
        alphas = np.linspace(-4, 4, 9)

    n_directions = len(results)
    ncols = 2
    nrows = (n_directions + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for dir_idx in range(n_directions):
        ax = axes[dir_idx]
        r = results[dir_idx]
        curves = np.array(r["effect_curves"])  # (n_samples, n_alphas)
        rhos = np.array(r["spearman_rhos"])

        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)

        # Individual traces
        for curve in curves:
            ax.plot(alphas, curve, alpha=0.08, color="steelblue", linewidth=0.5)

        # Mean +/- std
        ax.plot(alphas, mean_curve, color="steelblue", linewidth=2, label="mean")
        ax.fill_between(
            alphas, mean_curve - std_curve, mean_curve + std_curve,
            alpha=0.25, color="steelblue",
        )

        mono_rate = r["monotone_rate"]
        rho_mean = rhos.mean()
        ax.set_title(
            f"Direction {dir_idx + 1}   "
            f"mono={mono_rate:.2f}   "
            f"rho={rho_mean:+.2f}",
            fontsize=11,
        )
        ax.set_xlabel("Perturbation amplitude alpha")
        ax.set_ylabel("Effect (mean T)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

    # Hide unused axes
    for idx in range(n_directions, len(axes)):
        axes[idx].set_visible(False)

    n_samples = len(results[0]["effect_sizes"])
    fig.suptitle(
        "Intervention Robustness Scan\n"
        f"(n={n_samples} samples x {len(alphas)} amplitudes per direction)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")
