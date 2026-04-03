"""Phase 4: End-to-end integration validation runner.

Orchestrates the full Phase 4 pipeline:
1. Generate tentacle dataset (or load existing)
2. Train LeWM
3. Validate LeWM latent
4. Train interface layer
5. Validate interface layer
6. Run ablation experiments
7. Produce verdict

Usage:
    python scripts/run_phase4.py
    python scripts/run_phase4.py --skip-datagen  # if data already exists
    python scripts/run_phase4.py --quick          # reduced dataset for testing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from phase4.envs.tentacle_env import make_tentacle, step, extract_state, random_valid_state
from phase4.data.generate_tentacle_data import build_tentacle_dataset, load_tentacle_dataset
from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.models.train_lewm import train as train_lewm
from phase4.models.validate_latent import validate_lewm_latent
from phase4.interface.probe_interface import InterfaceLayer
from phase4.interface.train_interface import train_interface
from phase4.interface.validate_interface import validate_interface
from phase4.planning.evaluate import (
    run_ablation,
    summarize_results,
    evaluate_explanation_quality,
    save_ablation_results,
)
from phase4.phase4_report import phase4_verdict, save_verdict


RESULTS_DIR = Path("docs/results")
DATA_PATH = Path("phase4/data/tentacle_data.h5")
CHECKPOINT_DIR = Path("phase4/checkpoints")


def check_simulator_stability(n_tests: int = 10) -> bool:
    """Verify simulator doesn't produce NaN or Inf."""
    print("=" * 50)
    print("Step 0: Simulator stability check")
    print("=" * 50)

    for i in range(n_tests):
        env, rod = make_tentacle()
        tensions = np.random.exponential(0.5, size=80)
        tensions = np.clip(tensions, 0.0, 5.0)

        state, energy = step(env, rod, tensions, dt=1e-4, n_steps=100)

        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"  FAIL: NaN/Inf detected in test {i}")
            return False

        if np.any(np.abs(state) > 1e6):
            print(f"  FAIL: Numerical explosion in test {i}")
            return False

    print(f"  PASS: {n_tests} stability tests passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 validation")
    parser.add_argument("--skip-datagen", action="store_true",
                        help="Skip data generation (use existing)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: smaller dataset and fewer epochs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu), auto-detect if omitted")
    parser.add_argument("--n-traj", type=int, default=None,
                        help="Override number of trajectories")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Show GPU info if available
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Quick mode parameters
    if args.quick:
        n_traj = 100
        n_epochs_lewm = 10
        n_epochs_warmup = 5
        n_epochs_finetune = 5
        n_ablation_tasks = 20
    else:
        n_traj = 1000
        n_epochs_lewm = 50
        n_epochs_warmup = 10
        n_epochs_finetune = 20
        n_ablation_tasks = 100

    # Allow overrides
    if args.n_traj is not None:
        n_traj = args.n_traj
    batch_size = args.batch_size or 256

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # === Step 0: Simulator stability ===
    sim_stable = check_simulator_stability()
    if not sim_stable:
        print("\nSimulator unstable. Aborting.")
        return

    # === Step 1: Dataset generation ===
    print("\n" + "=" * 50)
    print("Step 1: Dataset generation")
    print("=" * 50)

    if not args.skip_datagen or not DATA_PATH.exists():
        build_tentacle_dataset(
            n_trajectories=n_traj,
            n_steps=200,
            save_path=DATA_PATH,
        )
    else:
        print(f"  Using existing dataset: {DATA_PATH}")

    # === Step 2: Train LeWM ===
    print("\n" + "=" * 50)
    print("Step 2: Train LeWM")
    print("=" * 50)

    lewm_model, lewm_history = train_lewm(
        data_path=str(DATA_PATH),
        latent_dim=64,
        n_epochs=n_epochs_lewm,
        batch_size=batch_size,
        lr=1e-3,
        lambda_recon=1.0,
        lambda_pred=1.0,
        checkpoint_dir=str(CHECKPOINT_DIR),
        device=device,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    # === Step 3: Validate LeWM latent ===
    print("\n" + "=" * 50)
    print("Step 3: Validate LeWM latent quality")
    print("=" * 50)

    latent_results = validate_lewm_latent(
        lewm_model,
        data_path=str(DATA_PATH),
        device=device,
    )

    # === Step 4: Train interface layer ===
    print("\n" + "=" * 50)
    print("Step 4: Train interface layer")
    print("=" * 50)

    interface_layer, interface_history = train_interface(
        lewm_model,
        data_path=str(DATA_PATH),
        latent_dim=64,
        warmup_epochs=n_epochs_warmup,
        finetune_epochs=n_epochs_finetune,
        batch_size=batch_size,
        device=device,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    # === Step 5: Validate interface layer ===
    print("\n" + "=" * 50)
    print("Step 5: Validate interface layer")
    print("=" * 50)

    interface_results = validate_interface(
        interface_layer,
        lewm_model,
        data_path=str(DATA_PATH),
        device=device,
    )

    # === Step 6: Ablation experiments ===
    print("\n" + "=" * 50)
    print("Step 6: Ablation experiments")
    print("=" * 50)

    ablation_results = run_ablation(
        lewm_model,
        interface_layer,
        n_tasks=n_ablation_tasks,
        device=device,
    )

    summary = summarize_results(ablation_results)
    explanation_rate = evaluate_explanation_quality(ablation_results)

    save_ablation_results(
        ablation_results, summary, explanation_rate,
        output_path="phase4/results/ablation_results.json",
    )

    # === Step 7: Verdict ===
    print("\n" + "=" * 50)
    print("Step 7: Phase 4 Verdict")
    print("=" * 50)

    verdict_input = {
        "sim_stable": sim_stable,
        "effective_rank": latent_results["effective_rank"],
        "aucs": interface_results["aucs"],
        "full_success": summary["full_system"]["success_rate"],
        "lewm_success": summary["pure_lewm"]["success_rate"],
        "full_efficiency": summary["full_system"]["avg_efficiency"],
        "lewm_efficiency": summary["pure_lewm"]["avg_efficiency"],
        "full_distance": summary["full_system"]["avg_distance"],
        "lewm_distance": summary["pure_lewm"]["avg_distance"],
        "explanation_rate": explanation_rate,
    }

    verdict = phase4_verdict(verdict_input)
    save_verdict(verdict)

    # Save full raw results
    raw_output = {
        "latent_validation": latent_results,
        "interface_validation": interface_results,
        "ablation_summary": summary,
        "explanation_rate": explanation_rate,
        "verdict": verdict,
    }

    raw_path = RESULTS_DIR / "phase4_raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(raw_output, f, indent=2, default=str)
    print(f"\nFull results saved to {raw_path}")

    # Print overall project status
    print("\n" + "=" * 50)
    print("NEUSYM-BRIDGE Project Status")
    print("=" * 50)
    print("Phase 1 (Common structure):     PASS  CKA=0.944")
    print("Phase 2 (Causal information):   PASS  3/4 interventions")
    print("Phase 3 (Collapse mechanism):   PASS  6/6")
    status = "PASS" if verdict["overall_pass"] else "FAIL"
    print(f"Phase 4 (End-to-end):           {status}  {verdict['passed_count']}/{verdict['total_checks']}")


if __name__ == "__main__":
    main()
