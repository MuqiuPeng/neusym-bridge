"""A1 Experiment Runner: Training Objective Ablation Study.

Trains three variants with identical conditions except the loss function,
then evaluates all on the same pipeline.

Usage:
    python experiments/a1/run_a1.py
    python experiments/a1/run_a1.py --skip-train          # evaluate only
    python experiments/a1/run_a1.py --quick                # reduced for testing
    python experiments/a1/run_a1.py --variants predictive  # single variant
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.a1.variants import ReconstructionAE, PredictiveModel, TemporalContrastiveModel
from experiments.a1.train_a1 import make_dataset, train_variant
from experiments.a1.evaluate_a1 import (
    evaluate_variant,
    compute_cross_variant_cka,
    summarize_a1,
)

DATA_PATH = "phase4/data/tentacle_data.h5"
CHECKPOINT_DIR = "experiments/a1/checkpoints"
RESULTS_DIR = "experiments/a1/results"


def build_variants(latent_dim=64):
    return {
        "reconstruction": ReconstructionAE(latent_dim=latent_dim),
        "predictive": PredictiveModel(latent_dim=latent_dim),
        "contrastive": TemporalContrastiveModel(latent_dim=latent_dim),
    }


def main():
    parser = argparse.ArgumentParser(description="A1: Training Objective Ablation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, load checkpoints")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Subset of variants to run (reconstruction, predictive, contrastive)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    # Quick mode
    if args.quick:
        n_epochs = 10
        n_sindy_traj = 20
        n_planning_tasks = 20
        max_traj = 200
    else:
        n_epochs = args.n_epochs or 50
        n_sindy_traj = 50
        n_planning_tasks = 50
        max_traj = None  # use all

    if args.n_epochs is not None:
        n_epochs = args.n_epochs

    # Build variants
    all_variants = build_variants(latent_dim=args.latent_dim)
    if args.variants:
        all_variants = {k: v for k, v in all_variants.items() if k in args.variants}

    print(f"\nVariants: {list(all_variants.keys())}")
    print(f"Epochs: {n_epochs}")

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # Training
    # =====================================================================
    trained_models = {}

    if not args.skip_train:
        print(f"\n{'='*60}")
        print("Phase 1: Loading data")
        print(f"{'='*60}")
        s_t, a_t, s_t1, neg_idx = make_dataset(DATA_PATH, max_trajectories=max_traj)
        print(f"Data: {len(s_t)} transitions")

        for name, model in all_variants.items():
            print(f"\n{'='*60}")
            print(f"Phase 1: Training {name}")
            print(f"{'='*60}")

            trained_model, history = train_variant(
                variant_name=name,
                model=model,
                s_t=s_t, a_t=a_t, s_t1=s_t1, neg_idx=neg_idx,
                n_epochs=n_epochs,
                batch_size=args.batch_size,
                checkpoint_dir=CHECKPOINT_DIR,
                device=device,
                seed=42,
            )
            trained_models[name] = trained_model

            # Save history
            hist_path = Path(RESULTS_DIR) / f"history_{name}.json"
            serializable = {
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
                "component_losses": history["component_losses"],
            }
            with open(hist_path, "w") as f:
                json.dump(serializable, f, indent=2)

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Free training data
        del s_t, a_t, s_t1, neg_idx
        gc.collect()
    else:
        # Load from checkpoints
        print(f"\n{'='*60}")
        print("Loading trained models from checkpoints")
        print(f"{'='*60}")

        ckpt_dir = Path(CHECKPOINT_DIR)
        for name, model in all_variants.items():
            # Find latest checkpoint
            ckpts = sorted(ckpt_dir.glob(f"a1_{name}_epoch*.pt"))
            if not ckpts:
                print(f"  WARNING: No checkpoint found for {name}, skipping")
                continue
            ckpt = ckpts[-1]
            print(f"  Loading {name} from {ckpt}")
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            trained_models[name] = model

    # =====================================================================
    # Evaluation
    # =====================================================================
    print(f"\n{'='*60}")
    print("Phase 2: Evaluation")
    print(f"{'='*60}")

    all_results = {}
    for name, model in trained_models.items():
        results = evaluate_variant(
            name=name,
            model=model,
            data_path=DATA_PATH,
            device=device,
            n_sindy_traj=n_sindy_traj,
            n_planning_tasks=n_planning_tasks,
        )
        all_results[name] = results

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # =====================================================================
    # Cross-variant analysis
    # =====================================================================
    if len(all_results) > 1:
        cka_m = compute_cross_variant_cka(all_results)

    # Summary table
    summarize_a1(all_results)

    # =====================================================================
    # Save results
    # =====================================================================
    output = {}
    for name, res in all_results.items():
        output[name] = {k: v for k, v in res.items() if k != "Z"}
        # Convert numpy to list for JSON
        for k, v in output[name].items():
            if isinstance(v, np.ndarray):
                output[name][k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                output[name][k] = float(v)

    if len(all_results) > 1:
        output["cka_matrix"] = cka_m.tolist()
        output["cka_names"] = list(all_results.keys())

    results_path = Path(RESULTS_DIR) / "a1_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
