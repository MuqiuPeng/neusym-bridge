"""Retrain LeWM on SimplifiedRod data.

The original lewm_epoch049.pt was trained on PyElastica data where the
force-application bug caused the rod to never move.  The predictor
therefore learned "any action → same state", making gradient-based
planning in latent space meaningless on the real (SimplifiedRod) physics.

This script:
  1. Generates fresh training data using SimplifiedRod (elastica blocked).
  2. Trains a new LeWMTentacle with identical architecture.
  3. Saves to experiments/energy_opt/outputs/lewm_simplified_retrained.pt

Data size: N_TRAJ trajectories × N_STEPS steps = N_TRAJ*N_STEPS transitions.
At N_TRAJ=300, N_STEPS=100 → 30 000 transitions (≈ 15-20 min collection).

Usage:
    python -m experiments.energy_opt.retrain_lewm_simplified
"""

from __future__ import annotations

import sys
sys.modules["elastica"] = None          # Force SimplifiedRod backend

import time
import gc
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import (
    make_tentacle, step, set_state, extract_state,
    random_valid_state, ACTION_DIM, STATE_DIM, MAX_TENSION,
)
from phase4.models.lewm_tentacle import LeWMTentacle

OUTPUT_DIR  = Path("experiments/energy_opt/outputs")
DATA_PATH   = OUTPUT_DIR / "tentacle_simplified.h5"
MODEL_PATH  = OUTPUT_DIR / "lewm_simplified_retrained.pt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TRAJ   = 300          # trajectories to generate
N_STEPS  = 100          # control steps per trajectory
N_EPOCHS = 50
BATCH    = 256
LR       = 1e-3
LAMBDA_PRED  = 1.0
LAMBDA_RECON = 0.5


def _elapsed(t0):
    m, s = divmod(int(time.time() - t0), 60)
    return f"{m:02d}:{s:02d}"


# ── 1. Data generation ────────────────────────────────────────────────

def generate_data(t0):
    if DATA_PATH.exists():
        print(f"[{_elapsed(t0)}] Reusing cached data: {DATA_PATH}", flush=True)
        return

    print(f"[{_elapsed(t0)}] Generating {N_TRAJ} trajectories × {N_STEPS} steps...",
          flush=True)

    rng = np.random.RandomState(42)

    with h5py.File(DATA_PATH, "w") as f:
        f.attrs["n_traj"]  = N_TRAJ
        f.attrs["n_steps"] = N_STEPS

        for i in range(N_TRAJ):
            start  = random_valid_state(seed=i)
            target = random_valid_state(seed=i + 10000)

            env, rod = make_tentacle()
            set_state(rod, start)

            states  = [extract_state(rod)]
            actions = []
            energies= []

            for t in range(N_STEPS):
                # Biased toward target + random exploration
                delta = target - states[-1]
                bias  = np.clip(delta / (np.linalg.norm(delta) + 1e-8), -1, 1)
                a     = rng.exponential(1.0, size=ACTION_DIM).astype(np.float32)
                a    *= np.clip(0.5 + 0.5 * bias[:ACTION_DIM], 0.1, 2.0)
                a     = np.clip(a, 0.0, MAX_TENSION)

                new_state, energy = step(env, rod, a)
                states.append(new_state)
                actions.append(a)
                energies.append(energy)

            del env, rod
            gc.collect()

            grp = f.create_group(f"traj_{i:04d}")
            grp.create_dataset("states",   data=np.array(states,   dtype=np.float32))
            grp.create_dataset("actions",  data=np.array(actions,  dtype=np.float32))
            grp.create_dataset("energies", data=np.array(energies, dtype=np.float32))
            grp.attrs["start"]  = start
            grp.attrs["target"] = target

            if (i + 1) % 50 == 0:
                print(f"  [{_elapsed(t0)}] {i+1}/{N_TRAJ} trajectories", flush=True)

    print(f"[{_elapsed(t0)}] Data saved to {DATA_PATH}", flush=True)


def load_data():
    states_t, actions_t, states_t1 = [], [], []
    with h5py.File(DATA_PATH, "r") as f:
        for key in f:
            s = f[key]["states"][:]   # (N_STEPS+1, 140)
            a = f[key]["actions"][:]  # (N_STEPS,   80)
            states_t.append(s[:-1])
            states_t1.append(s[1:])
            actions_t.append(a)

    S  = torch.tensor(np.concatenate(states_t,  axis=0), dtype=torch.float32)
    A  = torch.tensor(np.concatenate(actions_t, axis=0), dtype=torch.float32)
    S1 = torch.tensor(np.concatenate(states_t1, axis=0), dtype=torch.float32)
    print(f"Dataset: {S.shape[0]:,} transitions, state_dim={S.shape[1]}", flush=True)
    return S, A, S1


# ── 2. Training ───────────────────────────────────────────────────────

def train(S, A, S1, t0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{_elapsed(t0)}] Training on {device}", flush=True)

    model = LeWMTentacle().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    loader = DataLoader(
        TensorDataset(S, A, S1), batch_size=BATCH, shuffle=True, drop_last=True
    )

    for epoch in range(N_EPOCHS):
        model.train()
        total_pred = total_recon = 0.0
        for s_t, a_t, s_t1 in loader:
            s_t  = s_t.to(device)
            a_t  = a_t.to(device)
            s_t1 = s_t1.to(device)

            z_pred, z_t1, s_recon, s_t_ = model(s_t, a_t, s_t1)

            loss_pred  = nn.functional.mse_loss(z_pred, z_t1)
            loss_recon = nn.functional.mse_loss(s_recon, s_t_)
            loss = LAMBDA_PRED * loss_pred + LAMBDA_RECON * loss_recon

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_pred  += loss_pred.item()
            total_recon += loss_recon.item()

        sched.step()

        if (epoch + 1) % 10 == 0:
            n = len(loader)
            print(f"  [{_elapsed(t0)}] epoch {epoch+1:3d}/{N_EPOCHS}  "
                  f"pred={total_pred/n:.4f}  recon={total_recon/n:.4f}", flush=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[{_elapsed(t0)}] Model saved to {MODEL_PATH}", flush=True)
    return model


# ── 3. Quick sanity check ─────────────────────────────────────────────

def sanity_check(model, S, A, S1, t0):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        s_sample  = S[:500].to(device)
        a_sample  = A[:500].to(device)
        s1_sample = S1[:500].to(device)

        z_t  = model.encode(s_sample)
        z_t1_true = model.encode(s1_sample)
        z_t1_pred = model.predict(z_t, a_sample)

        pred_err  = (z_t1_pred - z_t1_true).norm(dim=1).mean().item()
        recon     = model.decode(z_t)
        recon_err = (recon - s_sample).norm(dim=1).mean().item()

        # Latent diversity
        latent_std = z_t.std(dim=0).mean().item()

    print(f"\n[{_elapsed(t0)}] Sanity check (500 samples):", flush=True)
    print(f"  Latent std       : {latent_std:.4f}  (>0.3 = good diversity)", flush=True)
    print(f"  Predictor error  : {pred_err:.4f}   (latent dist units)", flush=True)
    print(f"  Decoder recon err: {recon_err:.4f}  (state units, initial dist ~13)", flush=True)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    generate_data(t0)
    S, A, S1 = load_data()
    model = train(S, A, S1, t0)
    sanity_check(model, S, A, S1, t0)
    print(f"\n[{_elapsed(t0)}] Done. New checkpoint: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
