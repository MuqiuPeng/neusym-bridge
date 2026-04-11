"""A4: Cross-Architecture Replication — orchestration script.

Trains 4 architectures x 3 seeds = 12 models, computes 12x12 CKA matrix,
and runs cross-architecture intervention analysis.

Usage:
    python -m experiments.a4.run_a4 [--device cpu] [--n-epochs 30]

Prerequisites:
    - data/heat_2d.h5 (Phase 0)
    - checkpoints/model_b.pt, model_c.pt (Phase 0, for CNN baseline)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from neusym_bridge.models.trainer import train_model
from neusym_bridge.models.baseline_mlp import SEED_CONFIGS

from experiments.a4.models import (
    ALL_ENCODERS,
    SEEDS,
    HeatWorldModelA4,
    create_a4_model,
    ckpt_path,
    load_a4_model,
    model_key,
    save_a4_model,
)
from experiments.a4.evaluate_a4 import (
    analyze_cka,
    analyze_effective_ranks,
    collect_all_latents,
    cross_arch_intervention,
    plot_cka_heatmap,
    print_cka_summary,
)


DATA_PATH = Path("data/heat_2d.h5")
OUTPUT_DIR = Path("experiments/a4/outputs")
ARCH_ORDER = ["cnn", "mlp", "vit", "cnn_wide"]


def main():
    parser = argparse.ArgumentParser(description="A4 Cross-Architecture Replication")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing checkpoints")
    parser.add_argument("--interv-samples", type=int, default=20,
                        help="Samples per direction for intervention (default 20)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("A4: Cross-Architecture Replication", flush=True)
    print("=" * 60, flush=True)

    with h5py.File(DATA_PATH, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)
    print(f"Trajectories: {trajectories.shape}", flush=True)

    # ── Train 12 models ─────────────────────────────────────────────
    if not args.skip_train:
        print("\n" + "=" * 60, flush=True)
        print("Training: 4 architectures x 3 seeds", flush=True)
        print("=" * 60, flush=True)

        for arch in ARCH_ORDER:
            for seed in SEEDS:
                cp = ckpt_path(arch, seed)

                # Reuse existing Phase 0 CNN checkpoints
                if arch == "cnn":
                    # Map seeds to Phase 0 model names
                    seed_to_name = {v: k for k, v in SEED_CONFIGS.items()}
                    phase0_name = seed_to_name.get(seed)
                    phase0_path = Path(f"checkpoints/{phase0_name}.pt")
                    if phase0_path.exists():
                        print(f"\n  {model_key(arch, seed)}: "
                              f"reusing Phase 0 checkpoint {phase0_path}",
                              flush=True)
                        # Load Phase 0 model and re-save as A4 checkpoint
                        from neusym_bridge.models.trainer import load_model
                        m = load_model(phase0_path)
                        save_a4_model(m, cp)
                        continue

                if cp.exists():
                    print(f"\n  {model_key(arch, seed)}: checkpoint exists, skipping",
                          flush=True)
                    continue

                print(f"\n  Training {model_key(arch, seed)} ...", flush=True)
                model = create_a4_model(arch, seed)
                train_model(
                    model, trajectories,
                    n_epochs=args.n_epochs,
                    batch_size=512,
                    device=args.device,
                )
                save_a4_model(model, cp)

    # ── Load all models ──────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Loading all 12 models ...", flush=True)
    print("=" * 60, flush=True)

    models = {}
    for arch in ARCH_ORDER:
        for seed in SEEDS:
            key = model_key(arch, seed)
            cp = ckpt_path(arch, seed)
            if arch == "cnn":
                # CNN uses HeatWorldModel, need to load as A4 wrapper
                # But we saved it as A4 format above
                try:
                    models[key] = load_a4_model(arch, cp)
                except RuntimeError:
                    # Fallback: load as original HeatWorldModel
                    from neusym_bridge.models.trainer import load_model as load_orig
                    models[key] = load_orig(cp)
            else:
                models[key] = load_a4_model(arch, cp)
            print(f"  Loaded {key}", flush=True)

    # ── Collect latents ──────────────────────────────────────────────
    print("\nCollecting latents ...", flush=True)
    test_inputs = trajectories[:, 0]  # all t=0 frames
    Z = collect_all_latents(models, test_inputs, device=args.device)
    for key, z in Z.items():
        print(f"  {key}: shape={z.shape}, "
              f"eff_rank={float(np.exp(-np.sum((sv := (np.linalg.svd(z, compute_uv=False) / np.linalg.svd(z, compute_uv=False).sum())) * np.log(sv + 1e-8)))):.1f}",
              flush=True)

    # ── CKA analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("CKA Analysis", flush=True)
    print("=" * 60, flush=True)

    arch_groups = {
        arch: [model_key(arch, s) for s in SEEDS]
        for arch in ARCH_ORDER
    }
    cka_result = analyze_cka(Z, arch_groups)
    print_cka_summary(cka_result)

    # Save CKA matrix
    np.save(OUTPUT_DIR / "a4_cka_matrix.npy", cka_result["matrix"])

    # Heatmap
    plot_cka_heatmap(
        cka_result["matrix"], cka_result["keys"], ARCH_ORDER,
    )

    # ── Effective rank ───────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Effective Rank", flush=True)
    print("=" * 60, flush=True)

    er_results = analyze_effective_ranks(Z)
    for arch in ARCH_ORDER:
        ers = [er_results[model_key(arch, s)]["effective_rank"] for s in SEEDS]
        print(f"  {arch:10}: eff_rank = {np.mean(ers):.1f} +/- {np.std(ers):.1f}",
              flush=True)

    # ── Cross-arch intervention ──────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("Cross-Architecture Intervention", flush=True)
    print("=" * 60, flush=True)

    models_by_arch = {
        arch: [models[model_key(arch, s)] for s in SEEDS]
        for arch in ARCH_ORDER
    }
    Z_by_arch = {
        arch: [Z[model_key(arch, s)] for s in SEEDS]
        for arch in ARCH_ORDER
    }
    interv_results = cross_arch_intervention(
        models_by_arch, Z_by_arch, trajectories,
        n_samples=args.interv_samples, device=args.device,
    )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("A4 Final Summary", flush=True)
    print("=" * 60, flush=True)

    summary = {
        "within_arch_cka": {
            arch: cka_result["within_arch"].get(arch, {})
            for arch in ARCH_ORDER
        },
        "cross_arch_cka": cka_result["cross_arch"],
        "cross_arch_mean": cka_result["cross_arch_mean"],
        "cross_arch_std": cka_result["cross_arch_std"],
        "random_baseline_cka": 0.389,
        "effective_ranks": {
            k: v["effective_rank"] for k, v in er_results.items()
        },
        "intervention": interv_results,
    }

    with open(OUTPUT_DIR / "a4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {OUTPUT_DIR / 'a4_summary.json'}", flush=True)

    # Conclusion
    cross_mean = cka_result["cross_arch_mean"]
    if cross_mean > 0.6:
        conclusion = "STRONG: Common structure is data-driven, not architecture-specific"
    elif cross_mean > 0.4:
        conclusion = "MODERATE: Partial cross-architecture common structure"
    else:
        conclusion = "WEAK: Common structure appears architecture-dependent"

    print(f"\nConclusion: {conclusion}", flush=True)
    print("A4 complete.", flush=True)


if __name__ == "__main__":
    main()
