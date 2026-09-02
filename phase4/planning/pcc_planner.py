"""PCC-space planners for continuum robot control.

Three variants for ablation:
  PCCGreedyPlanner  -- move directly toward target in PCC space (no energy opt)
  PCCEnergyPlanner  -- joint energy + progress scoring with lambda trade-off
  PCCRandomPlanner  -- random PCC actions (baseline / lower bound)

No LeWM encoder required: planning in 16-dim PCC space.

Design note
-----------
All planners track the *applied* PCC state internally (not extracted from rod).
This avoids roundtrip errors in fit_pcc_module, which can exceed 20% for
large bend angles (sagitta approximation breaks down).  fit_pcc_module is
retained for diagnostics but NOT used in the planning loop.
"""

from __future__ import annotations

import numpy as np

from ..envs.pcc_env   import step_pcc, PCC_DIM, THETA_MAX, N_MODULES
from ..envs.pcc_state import (
    pcc_distance, pcc_to_node_positions,
    generate_pcc_tasks, set_pcc_state,
)

# Energy normalisation: max Sigma|theta_i| per step over all modules
_MAX_PCC_ENERGY = float(N_MODULES * THETA_MAX)   # 8 * 0.8*pi ~ 20.1


# ── helpers -------------------------------------------------------------------

def _clip_pcc(curvatures: np.ndarray) -> np.ndarray:
    """Clip PCC parameters to valid ranges."""
    out = curvatures.copy()
    out[0::2] = np.clip(out[0::2], 0.0, THETA_MAX)   # theta in [0, theta_max]
    out[1::2] = out[1::2] % (2.0 * np.pi)             # phi  in [0, 2*pi)
    return out


def _tip_dist(pcc_a: np.ndarray, pcc_b: np.ndarray) -> float:
    """Euclidean tip distance between two PCC configurations [m]."""
    pos_a = pcc_to_node_positions(pcc_a)
    pos_b = pcc_to_node_positions(pcc_b)
    return float(np.linalg.norm(pos_a[-1] - pos_b[-1]))


# ── base class ----------------------------------------------------------------

class PCCPlannerBase:
    """Common interface for PCC planners."""

    def plan(self, rod, target_pcc: np.ndarray,
             max_steps: int = 20) -> tuple[list[dict], float]:
        """Run planner until success or max_steps.

        Returns:
            (trajectory, total_energy)
            trajectory: list of dicts with 'applied_pcc', 'dist', 'tip_dist_m'
            total_energy: cumulative Sigma|theta| over all steps
        """
        raise NotImplementedError


# ── Greedy planner ------------------------------------------------------------

class PCCGreedyPlanner(PCCPlannerBase):
    """Greedily step toward target in PCC space.

    Uses a fixed fraction step each iteration.  Tracks applied PCC directly
    (no extract_pcc_state) to avoid roundtrip fitting errors.
    """

    def __init__(self, step_fraction: float = 0.30, success_tol: float = 0.05):
        self.step_fraction = step_fraction
        self.success_tol   = success_tol

    def plan(self, rod, target_pcc, max_steps=20):
        current_pcc  = np.zeros(PCC_DIM)   # starts straight
        total_energy = 0.0
        trajectory   = []

        for _ in range(max_steps):
            dist   = pcc_distance(current_pcc, target_pcc)
            tdist  = _tip_dist(current_pcc, target_pcc)
            trajectory.append({"applied_pcc": current_pcc.copy(),
                                "dist": dist, "tip_dist_m": tdist})

            if dist < self.success_tol:
                break

            action = _clip_pcc(current_pcc + self.step_fraction * (target_pcc - current_pcc))
            _, energy, _ = step_pcc(rod, action)
            total_energy += energy
            current_pcc  = action   # track applied, NOT extract_pcc_state

        return trajectory, total_energy


# ── Energy-optimal planner ---------------------------------------------------

class PCCEnergyPlanner(PCCPlannerBase):
    """Minimum-energy planner via candidate scoring.

    Each step samples N candidate PCC actions and selects the one minimising:

        score = energy_norm + lambda * dist_ratio

    where:
        energy_norm = Sigma|theta_cand| / _MAX_PCC_ENERGY   in [0, 1]
        dist_ratio  = pcc_distance(cand, target) / pcc_distance(current, target)

    Since dist_ratio is computed directly from the candidate PCC params
    (not from rod positions), there are no roundtrip extraction errors.
    """

    def __init__(self, n_candidates: int = 10,
                 lambda_energy: float = 1.0,
                 success_tol: float = 0.05):
        self.n_candidates  = n_candidates
        self.lambda_energy = lambda_energy
        self.success_tol   = success_tol

    def plan(self, rod, target_pcc, max_steps=20):
        rng          = np.random.default_rng(0)
        current_pcc  = np.zeros(PCC_DIM)   # starts straight
        total_energy = 0.0
        trajectory   = []

        for _ in range(max_steps):
            dist  = pcc_distance(current_pcc, target_pcc)
            tdist = _tip_dist(current_pcc, target_pcc)
            trajectory.append({"applied_pcc": current_pcc.copy(),
                                "dist": dist, "tip_dist_m": tdist})

            if dist < self.success_tol:
                break

            direction  = target_pcc - current_pcc
            candidates = self._sample_candidates(current_pcc, direction, rng)
            best_action, _ = self._select_best(current_pcc, target_pcc,
                                               candidates, dist)

            _, energy, _ = step_pcc(rod, best_action)
            total_energy += energy
            current_pcc   = best_action   # track applied

        return trajectory, total_energy

    def _sample_candidates(self, current: np.ndarray,
                            direction: np.ndarray,
                            rng: np.random.Generator) -> list[np.ndarray]:
        candidates = []
        for _ in range(self.n_candidates):
            scale = rng.uniform(0.05, 1.0)
            noise = rng.normal(0, 0.06, PCC_DIM)
            cand  = _clip_pcc(current + scale * direction + noise)
            candidates.append(cand)
        return candidates

    def _select_best(self, current: np.ndarray, target: np.ndarray,
                     candidates: list[np.ndarray],
                     current_dist: float) -> tuple[np.ndarray, float]:
        best_action = candidates[0]
        best_score  = float("inf")

        for cand in candidates:
            dist_after  = pcc_distance(cand, target)
            energy_norm = float(np.sum(np.abs(cand[0::2]))) / (_MAX_PCC_ENERGY + 1e-9)
            dist_ratio  = dist_after / (current_dist + 1e-9)
            score       = energy_norm + self.lambda_energy * dist_ratio

            if score < best_score:
                best_score  = score
                best_action = cand

        return best_action, best_score


# ── Random planner (baseline) ------------------------------------------------

class PCCRandomPlanner(PCCPlannerBase):
    """Random PCC actions -- lower-bound baseline."""

    def __init__(self, success_tol: float = 0.05):
        self.success_tol = success_tol

    def plan(self, rod, target_pcc, max_steps=20):
        rng          = np.random.default_rng(99)
        current_pcc  = np.zeros(PCC_DIM)
        total_energy = 0.0
        trajectory   = []

        for _ in range(max_steps):
            dist  = pcc_distance(current_pcc, target_pcc)
            tdist = _tip_dist(current_pcc, target_pcc)
            trajectory.append({"applied_pcc": current_pcc.copy(),
                                "dist": dist, "tip_dist_m": tdist})

            if dist < self.success_tol:
                break

            action = np.zeros(PCC_DIM)
            action[0::2] = rng.uniform(0.0, THETA_MAX, N_MODULES)
            action[1::2] = rng.uniform(0.0, 2.0 * np.pi, N_MODULES)

            _, energy, _ = step_pcc(rod, action)
            total_energy += energy
            current_pcc   = action

        return trajectory, total_energy
