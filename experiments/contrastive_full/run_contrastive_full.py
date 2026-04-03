"""Contrastive Full System experiment.

Replaces Phase 4's Reconstruction encoder with A1's Contrastive encoder,
retrains Interface Layer, and compares full system planning performance.

Usage:
    python experiments/contrastive_full/run_contrastive_full.py
    python experiments/contrastive_full/run_contrastive_full.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.a1.variants import ReconstructionAE, TemporalContrastiveModel
from phase4.interface.probe_interface import InterfaceLayer
from phase4.interface.train_interface import compute_physics_labels
from phase4.interface.validate_interface import measure_collapse_rate
from phase4.data.generate_tentacle_data import load_tentacle_dataset
from phase4.planning.task import generate_task_suite, execute_plan
from phase4.planning.planners import FullSystemPlanner, PureLEWMPlanner
from phase4.planning.evaluate import _get_diagnosis
from src.neusym_bridge.analysis.representation import effective_rank, linear_cka

DATA_PATH = "phase4/data/tentacle_data.h5"
A1_CKPT_DIR = Path("experiments/a1/checkpoints")
RESULTS_DIR = Path("experiments/contrastive_full/results")


# =====================================================================
# Task 1: Load and verify
# =====================================================================

def load_models(device="cpu"):
    """Load both A1 checkpoints."""
    model_c = TemporalContrastiveModel(latent_dim=64)
    ckpt_c = sorted(A1_CKPT_DIR.glob("a1_contrastive_epoch*.pt"))[-1]
    model_c.load_state_dict(torch.load(ckpt_c, map_location=device, weights_only=True))
    model_c.eval().to(device)
    print(f"Contrastive loaded from {ckpt_c}")

    model_r = ReconstructionAE(latent_dim=64)
    ckpt_r = sorted(A1_CKPT_DIR.glob("a1_reconstruction_epoch*.pt"))[-1]
    model_r.load_state_dict(torch.load(ckpt_r, map_location=device, weights_only=True))
    model_r.eval().to(device)
    print(f"Reconstruction loaded from {ckpt_r}")

    return model_c, model_r


def verify_encoders(model_c, model_r, states, device="cpu"):
    """Sanity check: confirm latent stats match A1 report."""
    print("\n" + "=" * 50)
    print("Task 1: Encoder Verification")
    print("=" * 50)

    s_tensor = torch.tensor(states[:5000], dtype=torch.float32).to(device)

    with torch.no_grad():
        Z_c = model_c.encode(s_tensor).cpu().numpy()
        Z_r = model_r.encode(s_tensor).cpu().numpy()

    er_c = effective_rank(Z_c)
    er_r = effective_rank(Z_r)
    cka = linear_cka(Z_c, Z_r)

    print(f"  Contrastive ER:  {er_c:.2f}  (A1: 27.3)")
    print(f"  Reconstruction ER: {er_r:.2f}  (A1: 22.2)")
    print(f"  Cross-encoder CKA: {cka:.4f}  (A1: ~0.001)")

    ok = 20 < er_c < 35 and cka < 0.05
    print(f"  Sanity check: {'PASS' if ok else 'FAIL'}")
    return {"er_contrastive": er_c, "er_reconstruction": er_r, "cka": cka}


# =====================================================================
# Task 2: Train Interface Layer for Contrastive encoder
# =====================================================================

def train_interface_for_encoder(
    model,
    s_t, a_t, s_t1,
    warmup_epochs=10,
    finetune_epochs=20,
    batch_size=256,
    lr=1e-3,
    lambda_sparse=0.01,
    device="cpu",
):
    """Two-stage interface training, same as Phase 4 but for any encoder."""
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    # Compute physics labels (same calibrated thresholds as Phase 4)
    labels = compute_physics_labels(s_t)

    s_t_t = torch.tensor(s_t, dtype=torch.float32)
    a_t_t = torch.tensor(a_t, dtype=torch.float32)
    s_t1_t = torch.tensor(s_t1, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    del s_t, a_t, s_t1, labels

    supervised_loader = DataLoader(
        TensorDataset(s_t_t, labels_t), batch_size=batch_size, shuffle=True,
    )
    consistency_loader = DataLoader(
        TensorDataset(s_t_t, a_t_t, s_t1_t), batch_size=batch_size, shuffle=True,
    )

    interface = InterfaceLayer(latent_dim=64).to(device)
    optimizer = torch.optim.Adam(interface.parameters(), lr=lr)
    bce = nn.BCELoss()

    # Stage 1: Supervised warmup
    print("  Stage 1: Supervised warmup")
    for epoch in range(warmup_epochs):
        interface.train()
        epoch_loss = 0.0
        for batch_s, batch_labels in supervised_loader:
            batch_s = batch_s.to(device)
            batch_labels = batch_labels.to(device)
            with torch.no_grad():
                z = model.encode(batch_s)
            confs = interface(z)
            loss = bce(confs, batch_labels) + lambda_sparse * confs.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_s)
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: loss={epoch_loss / len(s_t_t):.4f}")

    # Stage 2: Consistency fine-tuning with supervised anchor
    print("  Stage 2: Consistency fine-tuning")
    supervised_iter = iter(supervised_loader)
    for epoch in range(finetune_epochs):
        interface.train()
        supervised_iter = iter(supervised_loader)
        epoch_loss = 0.0
        for batch_s, batch_a, batch_s1 in consistency_loader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            batch_s1 = batch_s1.to(device)
            with torch.no_grad():
                z_t = model.encode(batch_s)
                z_t1_pred = model.predict(z_t, batch_a)
                z_t1_true = model.encode(batch_s1)
            facts_pred = interface(z_t1_pred)
            facts_true = interface(z_t1_true)
            loss_cons = F.mse_loss(facts_pred, facts_true)

            # Supervised anchor
            try:
                sup_s, sup_labels = next(supervised_iter)
            except StopIteration:
                supervised_iter = iter(supervised_loader)
                sup_s, sup_labels = next(supervised_iter)
            sup_s = sup_s.to(device)
            sup_labels = sup_labels.to(device)
            with torch.no_grad():
                z_sup = model.encode(sup_s)
            loss_anchor = 0.5 * bce(interface(z_sup), sup_labels)

            loss_sparse = lambda_sparse * interface(z_t).abs().mean()
            loss = loss_cons + loss_anchor + loss_sparse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_s)
        if epoch % 5 == 0:
            print(f"    Epoch {epoch}: loss={epoch_loss / len(s_t_t):.4f}")

    return interface


# =====================================================================
# Task 3: Validate Interface
# =====================================================================

def validate_interface(interface, model, states, device="cpu"):
    """Validate AUC, sparsity, collapse rate."""
    interface.eval().to(device)
    model.eval().to(device)

    states_sub = states[:5000]
    labels = compute_physics_labels(states_sub)

    loader = DataLoader(
        TensorDataset(torch.tensor(states_sub, dtype=torch.float32)),
        batch_size=512,
    )
    all_confs = []
    with torch.no_grad():
        for (batch,) in loader:
            z = model.encode(batch.to(device))
            all_confs.append(interface(z).cpu())
    confs = torch.cat(all_confs).numpy()

    aucs = []
    for i, name in enumerate(interface.predicate_names):
        if len(np.unique(labels[:, i])) < 2:
            aucs.append(0.5)
            print(f"    {name}: single class, AUC=0.5")
        else:
            auc = roc_auc_score(labels[:, i], confs[:, i])
            aucs.append(auc)
            print(f"    {name}: AUC={auc:.3f}")

    avg_active = (confs > 0.5).astype(float).sum(axis=1).mean()
    print(f"    Avg active predicates: {avg_active:.2f}/3")

    collapse = measure_collapse_rate(interface, model, states_sub, device)
    print(f"    Collapse rate: {collapse:.3f}")

    n_pass = sum(a > 0.65 for a in aucs)
    print(f"    AUC check: {n_pass}/3 > 0.65")

    return {"aucs": aucs, "avg_active": avg_active, "collapse_rate": collapse}


# =====================================================================
# Task 4: Full System Evaluation
# =====================================================================

def run_planning_comparison(
    model_c, interface_c,
    model_r, interface_r,
    n_tasks=100,
    n_plan_steps=50,
    device="cpu",
):
    """Compare Contrastive Full vs Reconstruction Full vs ablations."""
    print(f"\nGenerating {n_tasks} planning tasks...")
    tasks = generate_task_suite(n_tasks=n_tasks, seed=42)

    configs = {
        "recon_full": FullSystemPlanner(model_r, interface_r, device=device),
        "recon_pure": PureLEWMPlanner(model_r, device=device),
        "contrastive_full": FullSystemPlanner(model_c, interface_c, device=device),
        "contrastive_pure": PureLEWMPlanner(model_c, device=device),
    }

    results = {name: [] for name in configs}

    for i, task in enumerate(tasks):
        if i % 10 == 0:
            print(f"  Task {i}/{n_tasks}")
        for name, planner in configs.items():
            actions = planner.plan(task.start, task.target, n_steps=n_plan_steps)
            traj = execute_plan(actions, task.start)
            metrics = task.evaluate(traj)

            # Capture Relatum diagnosis for full system variants
            if "full" in name:
                encoder = model_c if "contrastive" in name else model_r
                iface = interface_c if "contrastive" in name else interface_r
                metrics["relatum_diagnosis"] = _get_diagnosis(
                    encoder, iface, traj, device
                )

            results[name].append(metrics)

    return results, tasks


def summarize_comparison(results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("Contrastive Full System Comparison")
    print(f"{'='*70}")
    print(f"{'Config':25} {'Success%':>10} {'Avg Dist':>10} {'Expl Rate':>10}")
    print("-" * 70)

    summaries = {}
    for name, rlist in results.items():
        avg_dist = float(np.mean([r["distance"] for r in rlist]))
        success = float(np.mean([r["success"] for r in rlist]))

        # Explanation rate for full system variants
        if "full" in name:
            failed = [r for r in rlist if not r["success"]]
            if failed:
                expl = sum(1 for r in failed
                           if r.get("relatum_diagnosis") and
                           r["relatum_diagnosis"].get("proof_steps", 0) > 0) / len(failed)
            else:
                expl = 1.0
        else:
            expl = float("nan")

        expl_str = f"{expl:.3f}" if not np.isnan(expl) else "N/A"
        print(f"{name:25} {success:>10.3f} {avg_dist:>10.1f} {expl_str:>10}")

        summaries[name] = {
            "success_rate": success,
            "avg_distance": avg_dist,
            "explanation_rate": expl if not np.isnan(expl) else None,
            "n_tasks": len(rlist),
        }

    print("=" * 70)

    # Determine outcome
    d_recon = summaries["recon_full"]["avg_distance"]
    d_contr = summaries["contrastive_full"]["avg_distance"]
    delta = d_contr - d_recon
    pct = delta / d_recon * 100

    if delta < -d_recon * 0.05:
        outcome = "A"
        desc = "Contrastive Full is better"
    elif abs(delta) < d_recon * 0.05:
        outcome = "B"
        desc = "Roughly equivalent"
    else:
        outcome = "C"
        desc = "Reconstruction Full is better"

    print(f"\nOutcome: Scenario {outcome} ({desc})")
    print(f"  Recon Full:       {d_recon:.1f}")
    print(f"  Contrastive Full: {d_contr:.1f}  ({'+' if delta > 0 else ''}{delta:.1f}, {'+' if pct > 0 else ''}{pct:.1f}%)")

    # Symbolic layer contribution for contrastive
    d_pure = summaries["contrastive_pure"]["avg_distance"]
    symb_delta = d_contr - d_pure
    print(f"\n  Symbolic layer contribution (contrastive):")
    print(f"    Pure:  {d_pure:.1f}")
    print(f"    Full:  {d_contr:.1f}  ({'+' if symb_delta > 0 else ''}{symb_delta:.1f})")

    return summaries, outcome


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Contrastive Full System Experiment")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_tasks = 20 if args.quick else 100

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Task 1: Load and verify
    model_c, model_r = load_models(device)
    s_t, a_t, s_t1 = load_tentacle_dataset(DATA_PATH, max_trajectories=500)
    verify_results = verify_encoders(model_c, model_r, s_t, device)

    # Task 2: Train interface layers
    print("\n" + "=" * 50)
    print("Task 2: Train Interface for Contrastive encoder")
    print("=" * 50)
    interface_c = train_interface_for_encoder(
        model_c, s_t.copy(), a_t.copy(), s_t1.copy(), device=device,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("Task 2b: Train Interface for Reconstruction encoder")
    print("=" * 50)
    interface_r = train_interface_for_encoder(
        model_r, s_t.copy(), a_t.copy(), s_t1.copy(), device=device,
    )
    if device == "cuda":
        torch.cuda.empty_cache()

    del s_t, a_t, s_t1
    gc.collect()

    # Task 3: Validate interfaces
    print("\n" + "=" * 50)
    print("Task 3: Validate Interfaces")
    print("=" * 50)
    s_test, _, _ = load_tentacle_dataset(DATA_PATH, max_trajectories=100)

    print("\n  Contrastive Interface:")
    val_c = validate_interface(interface_c, model_c, s_test, device)

    print("\n  Reconstruction Interface:")
    val_r = validate_interface(interface_r, model_r, s_test, device)
    del s_test

    # Task 4: Full system planning comparison
    print("\n" + "=" * 50)
    print("Task 4: Full System Planning Comparison")
    print("=" * 50)
    results, tasks = run_planning_comparison(
        model_c, interface_c,
        model_r, interface_r,
        n_tasks=n_tasks,
        device=device,
    )
    summaries, outcome = summarize_comparison(results)

    # Save results
    output = {
        "verify": {k: float(v) for k, v in verify_results.items()},
        "interface_contrastive": {k: (v if not isinstance(v, list) else [float(x) for x in v])
                                  for k, v in val_c.items()},
        "interface_reconstruction": {k: (v if not isinstance(v, list) else [float(x) for x in v])
                                     for k, v in val_r.items()},
        "planning": summaries,
        "outcome": outcome,
    }
    out_path = RESULTS_DIR / "contrastive_full_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
