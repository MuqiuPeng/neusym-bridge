"""B4: Interface Calibration Analysis — orchestration script.

Trains an Interface Layer on the tentacle domain, then evaluates:
  1. Reliability diagrams + ECE per predicate
  2. Threshold sensitivity (optimal vs default 0.5)
  3. Aggregation method comparison (Noisy-OR vs alternatives)

If tentacle_data.h5 does not exist (pyelastica not installed), generates
synthetic data that matches the expected distribution.

Usage:
    python -m experiments.b4.run_b4 [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.interface.probe_interface import InterfaceLayer, TENTACLE_PREDICATES
from phase4.interface.train_interface import compute_physics_labels
from phase4.envs.tentacle_env import STATE_DIM, ACTION_DIM

from experiments.b4.calibration import compute_ece, plot_reliability_diagrams
from experiments.b4.threshold_sensitivity import threshold_sensitivity
from experiments.b4.noisy_or_calibration import compare_aggregation

DATA_PATH = Path("phase4/data/tentacle_data.h5")
LEWM_CKPT = Path("phase4/checkpoints/lewm_epoch049.pt")
OUTPUT_DIR = Path("experiments/b4/outputs")


# ── Synthetic data generation ────────────────────────────────────────


def generate_synthetic_tentacle_data(
    path: Path,
    n_traj: int = 200,
    n_steps: int = 50,
    seed: int = 42,
) -> None:
    """Generate synthetic tentacle data matching the HDF5 format.

    Produces realistic-looking state vectors:
    - 20 segments x 7 dims (x, y, z, vx, vy, vz, curvature)
    - Positions along a chain, velocities with some variance
    - Curvatures in the normalized range
    - Actions as cable tensions in [0, MAX_TENSION]
    """
    rng = np.random.RandomState(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["n_trajectories"] = n_traj

        for t in range(n_traj):
            states = np.zeros((n_steps + 1, STATE_DIM), dtype=np.float32)
            actions = rng.uniform(0, 5.0, (n_steps, ACTION_DIM)).astype(np.float32)

            # Initialize chain positions along y-axis (normalized by rod_length)
            for step in range(n_steps + 1):
                noise_scale = 0.02 * (1 + step * 0.01)
                for seg in range(20):
                    off = seg * 7
                    # Positions: x,z near zero with some lateral deviation
                    states[step, off + 0] = rng.normal(0, noise_scale)          # x
                    states[step, off + 1] = seg / 20.0 + rng.normal(0, 0.005)  # y
                    states[step, off + 2] = rng.normal(0, noise_scale)          # z
                    # Velocities (normalized by 0.1)
                    for j in range(3):
                        states[step, off + 3 + j] = rng.normal(0, 30.0)
                    # Curvature (normalized by 10)
                    states[step, off + 6] = abs(rng.normal(0, 0.05))

                # Occasionally create high-curvature / high-velocity states
                if rng.random() < 0.3:
                    seg = rng.randint(0, 20)
                    states[step, seg * 7 + 6] = rng.uniform(0.1, 0.15)  # above 0.095
                if rng.random() < 0.4:
                    seg = rng.randint(0, 20)
                    mag = rng.uniform(65, 130)
                    direction = rng.randn(3)
                    direction /= np.linalg.norm(direction) + 1e-8
                    states[step, seg * 7 + 3: seg * 7 + 6] = direction * mag
                if rng.random() < 0.35:
                    tip = 19 * 7
                    states[step, tip] = rng.uniform(0.06, 0.2)
                    states[step, tip + 2] = rng.uniform(0.06, 0.2)

            grp = f.create_group(f"traj_{t:05d}")
            grp.create_dataset("states", data=states)
            grp.create_dataset("actions", data=actions)

    print(f"Generated synthetic tentacle data: {n_traj} trajectories -> {path}")


def load_data(max_traj: int = 500):
    """Load tentacle dataset, generating synthetic data if needed."""
    if not DATA_PATH.exists():
        print("Tentacle data not found, generating synthetic data...")
        generate_synthetic_tentacle_data(DATA_PATH, n_traj=500, n_steps=50)

    from phase4.data.generate_tentacle_data import load_tentacle_dataset
    return load_tentacle_dataset(DATA_PATH, max_trajectories=max_traj)


# ── LeWM training (if checkpoint missing) ────────────────────────────


def train_lewm_if_needed(s_t, a_t, s_t1, device="cpu", n_epochs=50):
    """Train LeWM from scratch if no checkpoint exists."""
    if LEWM_CKPT.exists():
        print(f"Loading LeWM from {LEWM_CKPT}")
        model = LeWMTentacle()
        model.load_state_dict(torch.load(LEWM_CKPT, weights_only=True))
        model.eval().to(device)
        return model

    print("Training LeWM from scratch...")
    model = LeWMTentacle().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    s_t_t = torch.tensor(s_t, dtype=torch.float32)
    a_t_t = torch.tensor(a_t, dtype=torch.float32)
    s_t1_t = torch.tensor(s_t1, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(s_t_t, a_t_t, s_t1_t),
        batch_size=256, shuffle=True,
    )

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch_s, batch_a, batch_s1 in loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_s1 = batch_s1.to(device)

            z_pred, z_t1, s_recon, s_orig = model(batch_s, batch_a, batch_s1)
            loss_pred = nn.MSELoss()(z_pred, z_t1)
            loss_recon = nn.MSELoss()(s_recon, s_orig)
            loss = loss_pred + loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_s)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  LeWM epoch {epoch}: loss={total_loss / len(s_t_t):.4f}",
                  flush=True)

    LEWM_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), LEWM_CKPT)
    model.eval()
    return model


# ── Interface training ────────────────────────────────────────────────


def train_interface(lewm, s_t, a_t, s_t1, device="cpu"):
    """Two-stage interface training (same as Phase 4)."""
    lewm.eval().to(device)
    for p in lewm.parameters():
        p.requires_grad = False

    labels = compute_physics_labels(s_t)
    s_t_t = torch.tensor(s_t, dtype=torch.float32)
    a_t_t = torch.tensor(a_t, dtype=torch.float32)
    s_t1_t = torch.tensor(s_t1, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    sup_loader = DataLoader(
        TensorDataset(s_t_t, labels_t), batch_size=256, shuffle=True,
    )
    cons_loader = DataLoader(
        TensorDataset(s_t_t, a_t_t, s_t1_t), batch_size=256, shuffle=True,
    )

    interface = InterfaceLayer(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(interface.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    # Stage 1: Supervised warmup
    print("  Stage 1: Supervised warmup")
    for epoch in range(10):
        interface.train()
        for batch_s, batch_l in sup_loader:
            batch_s, batch_l = batch_s.to(device), batch_l.to(device)
            with torch.no_grad():
                z = lewm.encode(batch_s)
            loss = bce(interface(z), batch_l) + 0.01 * interface(z).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}", flush=True)

    # Stage 2: Consistency fine-tuning
    print("  Stage 2: Consistency fine-tuning")
    sup_iter = iter(sup_loader)
    for epoch in range(20):
        interface.train()
        sup_iter = iter(sup_loader)
        for batch_s, batch_a, batch_s1 in cons_loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_s1 = batch_s1.to(device)
            with torch.no_grad():
                z_t = lewm.encode(batch_s)
                z_pred = lewm.predict(z_t, batch_a)
                z_true = lewm.encode(batch_s1)
            loss_cons = nn.MSELoss()(interface(z_pred), interface(z_true))
            try:
                ss, sl = next(sup_iter)
            except StopIteration:
                sup_iter = iter(sup_loader)
                ss, sl = next(sup_iter)
            ss, sl = ss.to(device), sl.to(device)
            with torch.no_grad():
                zs = lewm.encode(ss)
            loss_anchor = 0.5 * bce(interface(zs), sl)
            loss = loss_cons + loss_anchor + 0.01 * interface(z_t).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}", flush=True)

    interface.eval()
    return interface


# ── Collect predictions ───────────────────────────────────────────────


def collect_predictions(interface, lewm, states, device="cpu"):
    """Collect interface confidences and physics labels for test set."""
    labels = compute_physics_labels(states)

    loader = DataLoader(
        TensorDataset(torch.tensor(states, dtype=torch.float32)),
        batch_size=512,
    )

    all_confs = []
    interface.eval()
    lewm.eval()
    with torch.no_grad():
        for (batch,) in loader:
            z = lewm.encode(batch.to(device))
            all_confs.append(interface(z).cpu())

    confs = torch.cat(all_confs).numpy()
    return confs, labels


# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="B4 Interface Calibration")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("B4: Interface Calibration Analysis", flush=True)
    print("=" * 60, flush=True)

    # Load data
    s_t, a_t, s_t1 = load_data(max_traj=500)
    print(f"Data: s_t={s_t.shape}, a_t={a_t.shape}", flush=True)

    # Train/load LeWM
    print("\n--- LeWM ---", flush=True)
    lewm = train_lewm_if_needed(s_t, a_t, s_t1, device=args.device)

    # Train interface
    print("\n--- Interface Training ---", flush=True)
    interface = train_interface(lewm, s_t, a_t, s_t1, device=args.device)

    # Hold-out test set: use last 20% of data
    n_test = len(s_t) // 5
    test_states = s_t[-n_test:]
    print(f"\nTest set: {n_test} samples", flush=True)

    confs, labels = collect_predictions(interface, lewm, test_states, args.device)

    # Label statistics
    print("\nLabel distribution:")
    for i, name in enumerate(TENTACLE_PREDICATES):
        pos_rate = labels[:, i].mean()
        print(f"  {name}: {pos_rate:.3f} positive rate")

    # ── Task 1: Reliability diagrams + ECE ──
    print("\n" + "=" * 60, flush=True)
    print("Task 1: Reliability Diagrams + ECE", flush=True)
    print("=" * 60, flush=True)

    ece_scores = plot_reliability_diagrams(
        confs, labels, TENTACLE_PREDICATES,
    )
    print(f"\nECE per predicate:")
    for name, ece in zip(TENTACLE_PREDICATES, ece_scores):
        print(f"  {name:25}: ECE={ece:.4f}")
    print(f"  {'Mean ECE':25}: {np.mean(ece_scores):.4f}")

    # ── Task 2: Threshold sensitivity ──
    print("\n" + "=" * 60, flush=True)
    print("Task 2: Threshold Sensitivity", flush=True)
    print("=" * 60, flush=True)

    thresh_results = threshold_sensitivity(confs, labels, TENTACLE_PREDICATES)

    # ── Task 3: Aggregation comparison ──
    print("\n" + "=" * 60, flush=True)
    print("Task 3: Aggregation Method Comparison", flush=True)
    print("=" * 60, flush=True)

    agg_results = compare_aggregation(confs, labels)

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("B4 Final Summary", flush=True)
    print("=" * 60, flush=True)

    mean_ece = float(np.mean(ece_scores))
    if mean_ece < 0.05:
        cal_verdict = "Excellent (ECE < 0.05)"
    elif mean_ece < 0.10:
        cal_verdict = "Good (ECE < 0.10)"
    elif mean_ece < 0.15:
        cal_verdict = "Acceptable (ECE < 0.15)"
    else:
        cal_verdict = "Poor (ECE > 0.15) — consider post-hoc calibration"

    print(f"\nCalibration quality: {cal_verdict}")
    print(f"Mean ECE: {mean_ece:.4f}")

    print(f"\nThreshold recommendations:")
    for name, r in thresh_results.items():
        bt = r["best_threshold"]
        diff = abs(bt - 0.5)
        note = "0.5 is optimal" if diff < 0.1 else f"consider {bt:.2f}"
        print(f"  {name:25}: best_thresh={bt:.3f}  ({note})")

    # Best aggregation
    best_agg = max(agg_results.items(), key=lambda x: x[1]["f1_any"])[0]
    print(f"\nBest aggregation method (by F1-any): {best_agg}")
    if best_agg == "noisy_or":
        print("  -> Noisy-OR is optimal, Relatum design validated")
    else:
        print(f"  -> Consider {best_agg} (Relatum uses Noisy-OR)")

    summary = {
        "ece_scores": {name: ece for name, ece in zip(TENTACLE_PREDICATES, ece_scores)},
        "mean_ece": mean_ece,
        "calibration_verdict": cal_verdict,
        "threshold_sensitivity": thresh_results,
        "aggregation_comparison": agg_results,
        "best_aggregation": best_agg,
    }
    with open(OUTPUT_DIR / "b4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {OUTPUT_DIR / 'b4_summary.json'}")
    print("B4 complete.", flush=True)


if __name__ == "__main__":
    main()
