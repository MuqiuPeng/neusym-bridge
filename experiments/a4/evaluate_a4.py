"""A4 evaluation: CKA analysis, effective rank, cross-arch intervention."""

from __future__ import annotations

import json

import numpy as np
import torch

from neusym_bridge.analysis.representation import (
    cka_matrix,
    effective_rank,
    linear_cka,
    spectrum_analysis,
)
from neusym_bridge.analysis.structure_extraction import (
    build_common_basis,
    svcca,
)
from experiments.b3.intervention_robust import run_robust_intervention


# ── CKA ─────────────────────────────────────────────────────────────


def collect_all_latents(
    models: dict[str, torch.nn.Module],
    test_inputs: torch.Tensor,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Collect latent vectors from all models."""
    Z = {}
    for key, model in models.items():
        model.eval().to(device)
        with torch.no_grad():
            z = model.encode(test_inputs.to(device))
        Z[key] = z.cpu().numpy()
    return Z


def analyze_cka(
    Z: dict[str, np.ndarray],
    arch_groups: dict[str, list[str]],
) -> dict:
    """Compute CKA matrix and break down within/cross architecture."""
    M, keys = cka_matrix(Z)

    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Within-architecture CKA
    within = {}
    for arch, arch_keys in arch_groups.items():
        pairs = [
            M[key_to_idx[ki], key_to_idx[kj]]
            for i, ki in enumerate(arch_keys)
            for jj, kj in enumerate(arch_keys)
            if jj > i
        ]
        if pairs:
            within[arch] = {
                "mean": float(np.mean(pairs)),
                "std": float(np.std(pairs)),
                "values": [float(v) for v in pairs],
            }

    # Cross-architecture CKA
    arch_names = list(arch_groups.keys())
    cross = {}
    all_cross_vals = []
    for i, ai in enumerate(arch_names):
        for aj in arch_names[i + 1 :]:
            pairs = [
                M[key_to_idx[ki], key_to_idx[kj]]
                for ki in arch_groups[ai]
                for kj in arch_groups[aj]
            ]
            if pairs:
                pair_key = f"{ai}_vs_{aj}"
                cross[pair_key] = {
                    "mean": float(np.mean(pairs)),
                    "std": float(np.std(pairs)),
                }
                all_cross_vals.extend(pairs)

    return {
        "matrix": M,
        "keys": keys,
        "within_arch": within,
        "cross_arch": cross,
        "cross_arch_mean": float(np.mean(all_cross_vals)) if all_cross_vals else 0.0,
        "cross_arch_std": float(np.std(all_cross_vals)) if all_cross_vals else 0.0,
    }


def print_cka_summary(cka_result: dict, random_baseline: float = 0.389) -> None:
    print("\n" + "=" * 60)
    print("Within-architecture CKA (replicates Phase 1)")
    print("=" * 60)
    for arch, stats in cka_result["within_arch"].items():
        print(f"  {arch:10}: {stats['mean']:.3f} +/- {stats['std']:.3f}")

    print("\n" + "=" * 60)
    print("Cross-architecture CKA (core new contribution)")
    print("=" * 60)
    for pair, stats in cka_result["cross_arch"].items():
        print(f"  {pair:20}: {stats['mean']:.3f} +/- {stats['std']:.3f}")
    print(f"\n  Overall cross-arch: {cka_result['cross_arch_mean']:.3f} "
          f"+/- {cka_result['cross_arch_std']:.3f}")
    print(f"  Random baseline:    {random_baseline:.3f}")

    if cka_result["cross_arch_mean"] > 0.6:
        print("\n  -> Cross-arch CKA well above random baseline")
        print("     Common structure is DATA-DRIVEN, not architecture-specific")
    elif cka_result["cross_arch_mean"] > 0.4:
        print("\n  -> Cross-arch CKA moderately above random baseline")
        print("     Partial common structure across architectures")
    else:
        print("\n  -> Cross-arch CKA near random baseline")
        print("     Common structure appears architecture-specific")


# ── Effective Rank ──────────────────────────────────────────────────


def analyze_effective_ranks(Z: dict[str, np.ndarray]) -> dict[str, dict]:
    results = {}
    for key, z in Z.items():
        sa = spectrum_analysis(z)
        results[key] = {
            "effective_rank": sa["effective_rank"],
            "n_signal_dims": sa["n_signal_dims"],
        }
    return results


# ── Cross-arch Intervention ─────────────────────────────────────────


def cross_arch_intervention(
    models_by_arch: dict[str, list],
    Z_by_arch: dict[str, list[np.ndarray]],
    trajectories: torch.Tensor,
    n_samples: int = 20,
    device: str = "cpu",
) -> dict:
    """Run B3-style intervention for each architecture using its own SVCCA basis.

    For each architecture, compute SVCCA between its first two seed models,
    build a 4-dim common basis, then run intervention on the first model.
    """
    results = {}

    for arch, model_list in models_by_arch.items():
        if len(model_list) < 2:
            continue

        z_list = Z_by_arch[arch]

        # SVCCA between first two seeds
        svcca_result = svcca(z_list[0], z_list[1], n_components=10)
        V_common = build_common_basis(
            svcca_result["V_A"], svcca_result["V_B"], n_dims=4,
        )

        # Intervention on first seed model
        model = model_list[0]
        interv = run_robust_intervention(
            model, V_common, trajectories,
            n_samples=n_samples, device=device,
        )

        n_dirs = len(interv)
        mono_rates = [interv[d]["monotone_rate"] for d in range(n_dirs)]
        causal_count = sum(r > 0.7 for r in mono_rates)

        results[arch] = {
            "causal_count": causal_count,
            "n_directions": n_dirs,
            "mono_rates": mono_rates,
        }
        print(f"  {arch:10}: {causal_count}/{n_dirs} causal  "
              f"mono_rates={[f'{r:.2f}' for r in mono_rates]}")

    return results


# ── Heatmap ─────────────────────────────────────────────────────────


def plot_cka_heatmap(
    M: np.ndarray,
    keys: list[str],
    arch_order: list[str],
    output_path: str = "experiments/a4/outputs/a4_cka_heatmap.pdf",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Reorder keys by architecture group
    ordered_keys = []
    for arch in arch_order:
        ordered_keys += sorted(k for k in keys if k.startswith(arch + "_"))

    idx = [keys.index(k) for k in ordered_keys]
    M_ord = M[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(M_ord, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    # Annotations
    for i in range(len(ordered_keys)):
        for j in range(len(ordered_keys)):
            ax.text(j, i, f"{M_ord[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if M_ord[i, j] > 0.7 else "black")

    # Red boxes for same-architecture blocks
    offset = 0
    for arch in arch_order:
        n = sum(1 for k in ordered_keys if k.startswith(arch + "_"))
        if n > 0:
            rect = patches.Rectangle(
                (offset - 0.5, offset - 0.5), n, n,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
            offset += n

    labels = [k.replace("_seed", "\ns") for k in ordered_keys]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    plt.colorbar(im, ax=ax, shrink=0.8, label="CKA")
    ax.set_title(
        "CKA Matrix: Within- and Cross-Architecture\n"
        "(Red boxes = same architecture)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to {output_path}")
