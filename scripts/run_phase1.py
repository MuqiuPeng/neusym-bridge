"""Phase 1: Common structure detection — full experiment pipeline.

Runs all Phase 1 tasks:
  1.4  CKA similarity measurement
  1.5  Effective rank analysis
  1.6  Procrustes alignment
  1.7  Layer-wise CKA
  1.8  Control experiments (random, cross-task, overfit)
  1.9  Verdict and Phase 2 handoff

Prerequisites: run_phase0.py must have completed successfully.
"""

from pathlib import Path
import json

import torch
import h5py
import numpy as np

from neusym_bridge.models.baseline_mlp import SEED_CONFIGS, create_model
from neusym_bridge.models.trainer import load_model
from neusym_bridge.analysis.latent_collector import collect_latents, collect_layer_activations
from neusym_bridge.analysis.representation import (
    linear_cka,
    cka_matrix,
    spectrum_analysis,
    procrustes_residual,
)
from neusym_bridge.analysis.controls import (
    control_random_models,
    control_noise_task,
    control_overfit,
)
from neusym_bridge.analysis.verdict import phase1_verdict, print_verdict, save_verdict

DATA_PATH = Path("data/heat_2d.h5")
MODEL_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
MODEL_NAMES = list(SEED_CONFIGS.keys())


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    with h5py.File(DATA_PATH, "r") as f:
        trajectories = torch.tensor(f["trajectories"][:], dtype=torch.float32)

    # Use first timestep of each trajectory as test inputs (aligned across models)
    test_inputs = trajectories[:, 0, :, :]
    print(f"Test inputs: {test_inputs.shape}")

    # Load models and collect latents
    models = {}
    Z = {}
    for name in MODEL_NAMES:
        models[name] = load_model(MODEL_DIR / f"{name}.pt")
        Z[name] = collect_latents(models[name], test_inputs)
        print(f"  {name}: Z shape {Z[name].shape}, "
              f"mean={Z[name].mean():.3f}, std={Z[name].std():.3f}")

    # === Task 1.4: CKA ===
    print("\n" + "=" * 50)
    print("Task 1.4: CKA Similarity")
    print("=" * 50)
    M, names = cka_matrix(Z)
    n = len(names)
    print(f"\nCKA Matrix:")
    for i, name in enumerate(names):
        row = "  ".join(f"{M[i, j]:.3f}" for j in range(n))
        print(f"  {name}: {row}")

    avg_cka = (M.sum() - np.trace(M)) / (n * n - n)
    print(f"\nAvg off-diagonal CKA: {avg_cka:.4f}")

    # Random baseline (shuffled samples)
    rng = np.random.default_rng(0)
    Z_shuffled = {name: Z[name][rng.permutation(len(Z[name]))] for name in names}
    M_shuf, _ = cka_matrix(Z_shuffled)
    shuf_avg = (M_shuf.sum() - np.trace(M_shuf)) / (n * n - n)
    print(f"Shuffled baseline CKA: {shuf_avg:.4f}")

    np.save(RESULTS_DIR / "cka_matrix.npy", M)

    # === Task 1.5: Effective Rank ===
    print("\n" + "=" * 50)
    print("Task 1.5: Effective Rank & Spectrum Analysis")
    print("=" * 50)
    spectra = {}
    for name in MODEL_NAMES:
        spectra[name] = spectrum_analysis(Z[name])
        s = spectra[name]
        print(f"\n  {name}:")
        print(f"    Effective rank:  {s['effective_rank']:.2f}")
        print(f"    Signal dims:     {s['n_signal_dims']}")
        print(f"    MP threshold:    {s['mp_threshold']:.4f}")
        print(f"    Top-5 eigenvals: {[f'{v:.4f}' for v in s['top_eigenvalues'][:5]]}")

    ers = {name: spectra[name]["effective_rank"] for name in MODEL_NAMES}
    er_vals = list(ers.values())
    er_cv = np.std(er_vals) / np.mean(er_vals)
    print(f"\n  Effective rank CV: {er_cv:.4f}")

    # === Task 1.6: Procrustes ===
    print("\n" + "=" * 50)
    print("Task 1.6: Procrustes Alignment")
    print("=" * 50)
    proc_results = {}
    for i in range(len(MODEL_NAMES)):
        for j in range(i + 1, len(MODEL_NAMES)):
            a, b = MODEL_NAMES[i], MODEL_NAMES[j]
            res, _ = procrustes_residual(Z[a], Z[b])
            proc_results[f"{a}-{b}"] = res
            print(f"  Procrustes({a}, {b}): {res:.4f}")

    # === Task 1.7: Layer-wise CKA ===
    print("\n" + "=" * 50)
    print("Task 1.7: Layer-wise CKA")
    print("=" * 50)
    layer_acts = {}
    for name in MODEL_NAMES:
        layer_acts[name] = collect_layer_activations(models[name], test_inputs)

    layer_names = list(layer_acts[MODEL_NAMES[0]].keys())
    layer_cka = {}
    for layer in layer_names:
        Z_layer = {name: layer_acts[name][layer] for name in MODEL_NAMES}
        M_layer, _ = cka_matrix(Z_layer)
        avg = (M_layer.sum() - np.trace(M_layer)) / (n * n - n)
        layer_cka[layer] = float(avg)
        print(f"  {layer:20s}: avg CKA = {avg:.3f}")

    with open(RESULTS_DIR / "layer_cka.json", "w") as f:
        json.dump(layer_cka, f, indent=2)

    # === Task 1.8: Control Experiments ===
    print("\n" + "=" * 50)
    print("Task 1.8: Control Experiments")
    print("=" * 50)

    # Control A: Random models
    print("\n  Control A: Untrained random models")
    ctrl_random = control_random_models(test_inputs)
    print(f"    Avg CKA: {ctrl_random['avg_cka']:.4f}")

    # Control B: Noise-trained model
    print("\n  Control B: Noise-trained model")
    ctrl_noise = control_noise_task(test_inputs, Z)
    print(f"    Avg cross-task CKA: {ctrl_noise['avg_cross_cka']:.4f}")

    # Control C: Overfitted model
    print("\n  Control C: Overfitted model (500 epochs)")
    ctrl_overfit = control_overfit(trajectories, test_inputs, Z, n_epochs_overfit=200)
    print(f"    Avg overfit CKA: {ctrl_overfit['avg_overfit_cka']:.4f}")
    print(f"    Overfit train/val: {ctrl_overfit['final_train_loss']:.6f} / "
          f"{ctrl_overfit['final_val_loss']:.6f}")

    # === Task 1.9: Verdict ===
    print("\n" + "=" * 50)
    print("Task 1.9: Phase 1 Verdict")
    print("=" * 50)

    results = {
        "cka_matrix": M.tolist(),
        "effective_ranks": ers,
        "procrustes_residuals": proc_results,
        "layer_cka": layer_cka,
        "control_random_cka": ctrl_random["avg_cka"],
        "control_noise_cka": ctrl_noise["avg_cross_cka"],
        "control_overfit_cka": ctrl_overfit["avg_overfit_cka"],
        "spectrum": spectra,
    }

    verdict = phase1_verdict(results)
    print_verdict(verdict)
    save_verdict(verdict, RESULTS_DIR / "phase1_verdict.json")

    # Save all raw results
    with open(RESULTS_DIR / "phase1_raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
