"""Four planner variants for ablation study.

1. FullSystemPlanner:    LeWM + InterfaceLayer + Relatum (complete system)
2. PureLEWMPlanner:      LeWM only (no symbolic layer)
3. PureRelatumPlanner:   Relatum with hand-coded predicates (no LeWM)
4. HardThresholdPlanner: LeWM + hard-threshold interface (no Noisy-OR/collapse)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from ..envs.tentacle_env import (
    ACTION_DIM,
    STATE_DIM,
    MAX_TENSION,
)
from ..models.lewm_tentacle import LeWMTentacle
from ..interface.probe_interface import InterfaceLayer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.neusym_bridge.relatum.interface import RelatumInterface


TENTACLE_RULES = """
structural_risk(N) :- curvature_high(N), tension_saturated(N), tip_deviation(N).
"""


class BasePlanner(ABC):
    """Abstract base for all planners."""

    @abstractmethod
    def plan(
        self,
        start: np.ndarray,
        target: np.ndarray,
        n_steps: int = 50,
    ) -> list[np.ndarray]:
        """Generate a sequence of actions to reach target from start.

        Args:
            start: (140,) start state.
            target: (140,) target state.
            n_steps: Number of planning steps.

        Returns:
            List of (80,) action arrays.
        """
        ...


class FullSystemPlanner(BasePlanner):
    """Complete system: LeWM + Interface + Relatum.

    Uses Relatum's active query and collapse mechanism to guide
    energy-optimal planning with safety constraints.
    """

    def __init__(
        self,
        lewm: LeWMTentacle,
        interface: InterfaceLayer,
        device: str = "cpu",
    ):
        self.lewm = lewm.eval().to(device)
        self.interface = interface.eval().to(device)
        self.device = device

    def plan(self, start, target, n_steps=50):
        with torch.no_grad():
            z_start = self.lewm.encode(
                torch.tensor(start, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)
            z_target = self.lewm.encode(
                torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)

        actions = []
        z_current = z_start
        ri = RelatumInterface()
        ri.load_rules_from_text(TENTACLE_RULES)
        ri.set_collapse_threshold("structural_risk", 0.6)

        for t in range(n_steps):
            # 1. Interface: latent -> probabilistic facts
            node_id = f"step_{t}"
            self.interface.to_relatum_assertions(z_current, node_id, ri)

            # 2. Active query: check for missing premises
            missing = ri.find_missing_premises()

            # 3. Update closure
            ri.update_closure([])

            # 4. Check for structural risk
            risk_collapsed = ri.is_collapsed(f"structural_risk({node_id})")

            if risk_collapsed:
                # Safety mode: reduce tensions, move conservatively
                action = self._safe_action(z_current, z_target)
            else:
                # Normal mode: energy-optimal action
                action = self._energy_optimal_action(z_current, z_target)

            actions.append(action)

            # Predict next latent
            with torch.no_grad():
                a_tensor = torch.tensor(
                    action, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                z_current = self.lewm.predict(
                    z_current.unsqueeze(0), a_tensor
                ).squeeze(0)

        return actions

    def _energy_optimal_action(self, z_current, z_target):
        """Latent-gradient-biased action toward target with energy minimization."""
        delta = (z_target - z_current).cpu().numpy()
        error_norm = np.linalg.norm(delta) + 1e-8
        scale = min(2.0, error_norm) * MAX_TENSION * 0.3

        # Use latent delta to bias cable activations:
        # project latent dims onto action space segments
        delta_norm = delta / error_norm
        # Tile latent direction to action space and use as bias
        bias = np.tile(delta_norm, ACTION_DIM // len(delta_norm) + 1)[:ACTION_DIM]
        bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)  # normalize to [0,1]

        rng = np.random.RandomState(hash(tuple(delta[:5].tolist())) % 2**31)
        noise = rng.exponential(0.3, size=ACTION_DIM)
        action = scale * (0.6 * bias + 0.4 * noise)
        return np.clip(action, 0.0, MAX_TENSION)

    def _safe_action(self, z_current, z_target):
        """Conservative action: lower tensions to reduce structural risk."""
        action = self._energy_optimal_action(z_current, z_target)
        return action * 0.3  # Reduce to 30% of normal


class PureLEWMPlanner(BasePlanner):
    """LeWM only, no symbolic layer.

    Uses latent-space distance as the sole planning signal.
    Greedy gradient descent toward target latent.
    """

    def __init__(self, lewm: LeWMTentacle, device: str = "cpu"):
        self.lewm = lewm.eval().to(device)
        self.device = device

    def plan(self, start, target, n_steps=50):
        with torch.no_grad():
            z_current = self.lewm.encode(
                torch.tensor(start, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)
            z_target = self.lewm.encode(
                torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)

        actions = []
        for t in range(n_steps):
            delta = (z_target - z_current).cpu().numpy()
            error_norm = np.linalg.norm(delta) + 1e-8
            scale = min(2.0, error_norm) * MAX_TENSION * 0.3

            # Latent-gradient-biased action
            delta_norm = delta / error_norm
            bias = np.tile(delta_norm, ACTION_DIM // len(delta_norm) + 1)[:ACTION_DIM]
            bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)

            rng = np.random.RandomState(hash(tuple(delta[:5].tolist())) % 2**31)
            noise = rng.exponential(0.3, size=ACTION_DIM)
            action = scale * (0.6 * bias + 0.4 * noise)
            action = np.clip(action, 0.0, MAX_TENSION)
            actions.append(action)

            with torch.no_grad():
                a_tensor = torch.tensor(
                    action, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                z_current = self.lewm.predict(
                    z_current.unsqueeze(0), a_tensor
                ).squeeze(0)

        return actions


class PureRelatumPlanner(BasePlanner):
    """Relatum with hand-coded predicates from simulator (no LeWM).

    Serves as an upper bound for the symbolic layer's planning
    ability when given perfect state information.
    """

    def __init__(self):
        pass

    def plan(self, start, target, n_steps=50):
        actions = []
        state = start.copy()

        for t in range(n_steps):
            # Hand-coded heuristic: move toward target
            delta = target - state
            error = np.linalg.norm(delta)

            # Distribute tension based on state error
            scale = min(1.0, error) * MAX_TENSION * 0.3

            rng = np.random.RandomState(t + hash(tuple(state[:5].tolist())) % 2**31)
            action = rng.exponential(scale, size=ACTION_DIM)

            # Hand-coded safety: reduce if curvature high
            # Threshold calibrated to normalized curvature (curvature/10)
            curvatures = [abs(state[seg * 7 + 6]) for seg in range(20)]
            if max(curvatures) > 0.095:
                action *= 0.3

            action = np.clip(action, 0.0, MAX_TENSION)
            actions.append(action)

            # Simple forward prediction (no world model)
            state = state + delta * 0.02 + rng.normal(0, 0.01, STATE_DIM)

        return actions


class HardThresholdPlanner(BasePlanner):
    """LeWM + interface with hard 0.5 threshold (no Noisy-OR/collapse).

    Tests whether the probabilistic collapse mechanism adds value
    over a simple binary classification approach.
    """

    def __init__(
        self,
        lewm: LeWMTentacle,
        interface: InterfaceLayer,
        device: str = "cpu",
    ):
        self.lewm = lewm.eval().to(device)
        self.interface = interface.eval().to(device)
        self.device = device

    def plan(self, start, target, n_steps=50):
        with torch.no_grad():
            z_current = self.lewm.encode(
                torch.tensor(start, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)
            z_target = self.lewm.encode(
                torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)
            ).squeeze(0)

        actions = []
        for t in range(n_steps):
            # Hard threshold: no Noisy-OR, just binary
            with torch.no_grad():
                confs = self.interface(z_current.unsqueeze(0)).squeeze(0)

            # Binary decision: all predicates > 0.5 means risk
            all_active = (confs > 0.5).all().item()

            delta = (z_target - z_current).cpu().numpy()
            error_norm = np.linalg.norm(delta) + 1e-8
            scale = min(2.0, error_norm) * MAX_TENSION * 0.3

            delta_norm = delta / error_norm
            bias = np.tile(delta_norm, ACTION_DIM // len(delta_norm) + 1)[:ACTION_DIM]
            bias = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)

            rng = np.random.RandomState(hash(tuple(delta[:5].tolist())) % 2**31)
            noise = rng.exponential(0.3, size=ACTION_DIM)
            action = scale * (0.6 * bias + 0.4 * noise)

            if all_active:
                action *= 0.3  # Safety reduction

            action = np.clip(action, 0.0, MAX_TENSION)
            actions.append(action)

            with torch.no_grad():
                a_tensor = torch.tensor(
                    action, dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                z_current = self.lewm.predict(
                    z_current.unsqueeze(0), a_tensor
                ).squeeze(0)

        return actions
