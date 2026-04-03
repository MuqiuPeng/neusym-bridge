"""Unified evaluation pipeline for A1 ablation experiment.

Runs identical evaluation on all three variants:
1. Effective Rank
2. SINDy R² (core metric)
3. Linear probe R² (physics labels)
4. Planning distance
5. Cross-variant CKA matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.neusym_bridge.analysis.representation import effective_rank, linear_cka, cka_matrix
from src.neusym_bridge.analysis.structure_extraction import (
    build_sindy_timeseries,
    adaptive_threshold,
    run_sindy,
)
from phase4.models.validate_latent import collect_latents, extract_physics_labels, train_probe
from phase4.data.generate_tentacle_data import load_tentacle_dataset
from phase4.planning.task import TentaclePlanningTask, execute_plan, generate_task_suite
from phase4.planning.planners import PureLEWMPlanner


def collect_latents_generic(model, states, device="cpu", batch_size=512):
    """Collect latents from any variant model (all have .encode())."""
    model.eval()
    model.to(device)
    loader = DataLoader(
        TensorDataset(torch.tensor(states, dtype=torch.float32)),
        batch_size=batch_size,
    )
    latents = []
    with torch.no_grad():
        for (batch,) in loader:
            z = model.encode(batch.to(device))
            latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def collect_trajectory_latents(model, data_path, n_traj=50, device="cpu"):
    """Collect latent trajectories (for SINDy) from HDF5 data.

    Returns list of per-trajectory latent arrays.
    """
    import h5py

    model.eval()
    model.to(device)
    trajectories = []

    with h5py.File(data_path, "r") as f:
        total_traj = f.attrs["n_trajectories"]
        n = min(n_traj, total_traj)
        for i in range(n):
            states = f[f"traj_{i:05d}"]["states"][:]  # (n_steps+1, state_dim)
            s_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            with torch.no_grad():
                z = model.encode(s_tensor).cpu().numpy()
            trajectories.append(z)

    return trajectories


def eval_effective_rank(Z):
    er = effective_rank(Z)
    print(f"  Effective rank: {er:.2f}")
    return er


def eval_probes(Z, states):
    labels = extract_physics_labels(states)
    r2_curv = train_probe(Z, labels["curvature"])
    r2_vel = train_probe(Z, labels["velocity"])
    r2_tip = train_probe(Z, labels["tip_position"])
    print(f"  Probe R²: curvature={r2_curv:.3f} velocity={r2_vel:.3f} tip={r2_tip:.3f}")
    return {"r2_curvature": r2_curv, "r2_velocity": r2_vel, "r2_tip_position": r2_tip}


def eval_sindy(latent_trajectories, dt=0.01, n_common_dims=4):
    """Run SINDy on latent trajectories projected to top PCA directions.

    Uses SVD on concatenated latents to find principal directions,
    then runs SINDy on projected trajectories.
    """
    import pysindy as ps

    # Concatenate all trajectories
    Z_all = np.vstack(latent_trajectories)

    # PCA: top n_common_dims directions
    Z_centered = Z_all - Z_all.mean(axis=0)
    _, _, Vt = np.linalg.svd(Z_centered, full_matrices=False)
    V = Vt[:n_common_dims].T  # (latent_dim, n_common_dims)

    # Project each trajectory
    X_list = []
    Xd_list = []
    for traj in latent_trajectories:
        proj = traj @ V  # (T, n_common_dims)
        deriv = np.gradient(proj, dt, axis=0)
        # Trim endpoints to avoid boundary artifacts
        X_list.append(proj[1:-1])
        Xd_list.append(deriv[1:-1])

    X = np.vstack(X_list)
    Xdot = np.vstack(Xd_list)

    threshold = adaptive_threshold(Xdot, factor=0.1)

    # Use pysindy directly for API compatibility
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, alpha=0.05, max_iter=100),
        feature_library=ps.PolynomialLibrary(degree=2),
    )
    model.fit(X, t=dt, x_dot=Xdot)
    r2 = model.score(X, t=dt, x_dot=Xdot)

    print(f"  SINDy R²: {r2:.4f} (threshold={threshold:.5f}, dims={n_common_dims})")
    try:
        model.print()
    except Exception:
        pass

    return {"sindy_r2": r2, "sindy_threshold": threshold}


def eval_planning(model, n_tasks=50, n_plan_steps=50, device="cpu"):
    """Evaluate planning distance using PureLEWMPlanner.

    Creates a minimal LeWM-compatible wrapper so PureLEWMPlanner works.
    """
    # PureLEWMPlanner expects model with .encode() and .predict()
    # All our variants have these methods
    planner = PureLEWMPlanner(model, device=device)
    tasks = generate_task_suite(n_tasks=n_tasks, seed=42)

    distances = []
    for task in tasks:
        actions = planner.plan(task.start, task.target, n_steps=n_plan_steps)
        traj = execute_plan(actions, task.start)
        result = task.evaluate(traj)
        distances.append(result["distance"])

    avg_dist = float(np.mean(distances))
    print(f"  Planning distance: {avg_dist:.1f}")
    return avg_dist


def evaluate_variant(
    name: str,
    model: torch.nn.Module,
    data_path: str,
    device: str = "cpu",
    n_sindy_traj: int = 50,
    n_planning_tasks: int = 50,
) -> dict:
    """Run full evaluation pipeline on one variant."""
    print(f"\n{'='*50}")
    print(f"Evaluating: {name}")
    print(f"{'='*50}")

    # Load test data
    s_t, _, _ = load_tentacle_dataset(data_path, max_trajectories=100)
    idx = np.random.choice(len(s_t), min(10000, len(s_t)), replace=False)
    states = s_t[idx]

    # Collect latents
    Z = collect_latents_generic(model, states, device=device)

    results = {"name": name}

    # 1. Effective rank
    results["effective_rank"] = eval_effective_rank(Z)

    # 2. Linear probes
    results.update(eval_probes(Z, states))

    # 3. SINDy R²
    print("  Collecting trajectory latents for SINDy...")
    traj_latents = collect_trajectory_latents(model, data_path, n_traj=n_sindy_traj, device=device)
    sindy_res = eval_sindy(traj_latents)
    results.update(sindy_res)

    # 4. Planning distance
    print("  Running planning evaluation...")
    results["planning_distance"] = eval_planning(model, n_tasks=n_planning_tasks, device=device)

    # Store latents for CKA
    results["Z"] = Z

    return results


def compute_cross_variant_cka(all_results: dict) -> np.ndarray:
    """Compute pairwise CKA matrix across all variants."""
    Z_dict = {name: res["Z"] for name, res in all_results.items()}
    M, names = cka_matrix(Z_dict)

    print(f"\n{'='*50}")
    print("Cross-Variant CKA Matrix")
    print(f"{'='*50}")
    print(f"{'':20}", end="")
    for n in names:
        print(f"{n:>15}", end="")
    print()
    for i, ni in enumerate(names):
        print(f"{ni:20}", end="")
        for j in range(len(names)):
            print(f"{M[i,j]:>15.3f}", end="")
        print()

    return M


def summarize_a1(all_results: dict):
    """Print summary table."""
    print(f"\n{'='*75}")
    print("A1 Experiment Summary")
    print(f"{'='*75}")
    print(
        f"{'Variant':20} {'Eff.Rank':>10} {'SINDy R²':>10} "
        f"{'Curv.R²':>10} {'Vel.R²':>10} {'Plan.Dist':>10}"
    )
    print("-" * 75)

    for name, res in all_results.items():
        print(
            f"{name:20} "
            f"{res['effective_rank']:>10.2f} "
            f"{res['sindy_r2']:>10.4f} "
            f"{res['r2_curvature']:>10.3f} "
            f"{res['r2_velocity']:>10.3f} "
            f"{res['planning_distance']:>10.1f}"
        )
    print("=" * 75)
