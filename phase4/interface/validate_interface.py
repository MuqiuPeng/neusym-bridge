"""Validate the interface layer.

Checks:
1. Confidence distribution is reasonable (not all 0 or all 1)
2. AUC against physics ground truth > 0.65 for at least 2/3 predicates
3. Sparsity: average active predicates in [0.5, 2.5]
4. Collapse rate with Relatum in [0.1, 0.9]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.interface.probe_interface import InterfaceLayer
from phase4.interface.train_interface import compute_physics_labels
from phase4.models.lewm_tentacle import LeWMTentacle
from phase4.data.generate_tentacle_data import load_tentacle_dataset
from src.neusym_bridge.relatum.interface import RelatumInterface


TENTACLE_RULES = """
structural_risk(N) :- curvature_high(N), tension_saturated(N), tip_deviation(N).
"""


def validate_interface(
    interface: InterfaceLayer,
    lewm_model: LeWMTentacle,
    data_path: str | Path = "phase4/data/tentacle_data.h5",
    max_samples: int = 5000,
    device: str = "cpu",
) -> dict:
    """Full interface layer validation.

    Args:
        interface: Trained interface layer.
        lewm_model: Trained LeWM model (frozen).
        data_path: Path to dataset.
        max_samples: Max samples for validation.
        device: Device for inference.

    Returns:
        Dict with validation results.
    """
    lewm_model.eval().to(device)
    interface.eval().to(device)

    # Load test data
    s_t, _, _ = load_tentacle_dataset(data_path, max_trajectories=100)
    if len(s_t) > max_samples:
        idx = np.random.choice(len(s_t), max_samples, replace=False)
        s_t = s_t[idx]

    labels = compute_physics_labels(s_t)

    # Collect confidences
    loader = DataLoader(
        TensorDataset(torch.tensor(s_t, dtype=torch.float32)),
        batch_size=512,
    )

    all_confs = []
    with torch.no_grad():
        for (batch,) in loader:
            z = lewm_model.encode(batch.to(device))
            confs = interface(z)
            all_confs.append(confs.cpu())

    confs = torch.cat(all_confs).numpy()

    # 1. Confidence distribution
    print("Confidence statistics:")
    for i, name in enumerate(interface.predicate_names):
        print(f"  {name}: mean={confs[:, i].mean():.3f} std={confs[:, i].std():.3f}")

    # 2. AUC per predicate
    aucs = []
    for i, name in enumerate(interface.predicate_names):
        # Need both classes present for AUC
        if len(np.unique(labels[:, i])) < 2:
            print(f"  {name}: only one class present, AUC=N/A")
            aucs.append(0.5)
            continue
        auc = roc_auc_score(labels[:, i], confs[:, i])
        print(f"  {name} AUC: {auc:.3f}")
        aucs.append(auc)

    # 3. Sparsity
    active = (confs > 0.5).astype(float)
    avg_active = active.sum(axis=1).mean()
    print(f"Average active predicates: {avg_active:.2f}/{len(interface.predicate_names)}")

    # 4. Collapse rate with Relatum
    collapse_rate = measure_collapse_rate(interface, lewm_model, s_t, device)
    print(f"Collapse rate: {collapse_rate:.3f}")

    results = {
        "aucs": aucs,
        "avg_active_predicates": float(avg_active),
        "collapse_rate": collapse_rate,
        "confidence_means": confs.mean(axis=0).tolist(),
        "confidence_stds": confs.std(axis=0).tolist(),
        "auc_pass": sum(a > 0.65 for a in aucs) >= 2,
        "sparsity_pass": 0.5 <= avg_active <= 2.5,
        "collapse_pass": 0.1 <= collapse_rate <= 0.9,
    }

    passed = sum([results["auc_pass"], results["sparsity_pass"], results["collapse_pass"]])
    print(f"\nInterface validation: {passed}/3 checks passed")

    return results


def measure_collapse_rate(
    interface: InterfaceLayer,
    lewm_model: LeWMTentacle,
    states: np.ndarray,
    device: str = "cpu",
    n_samples: int = 200,
) -> float:
    """Measure what fraction of states lead to Relatum collapse.

    Feeds interface outputs into a fresh RelatumInterface with
    tentacle rules and checks how many trigger collapse.

    Returns:
        Fraction of samples where structural_risk collapses.
    """
    if len(states) > n_samples:
        idx = np.random.choice(len(states), n_samples, replace=False)
        states = states[idx]

    n_collapsed = 0

    for i in range(len(states)):
        ri = RelatumInterface()
        ri.load_rules_from_text(TENTACLE_RULES)
        ri.set_collapse_threshold("structural_risk", 0.6)

        s = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            z = lewm_model.encode(s)

        # Assert facts via interface
        interface.to_relatum_assertions(z.squeeze(0), f"node_{i}", ri)

        # Try to derive conclusions
        ri.update_closure([])

        if ri.is_collapsed(f"structural_risk(node_{i})"):
            n_collapsed += 1

    return n_collapsed / len(states)
