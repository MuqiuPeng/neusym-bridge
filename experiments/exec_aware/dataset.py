"""Dataset for execution-aware training.

Combines temporal pairs (for InfoNCE) with lever execution pairs
(for execution consistency loss).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from phase4.envs.tentacle_env import make_tentacle, set_state, random_valid_state
from experiments.lever_control.levers import LEVERS, execute_lever

LEVER_TO_IDX = {l: i for i, l in enumerate(LEVERS)}


def collect_exec_pairs(
    n_episodes: int = 200,
    levers_per_episode: int = 20,
    seed: int = 10000,
) -> list[dict]:
    """Collect (state_before, lever, state_after, energy) pairs.

    Ensures all 8 levers are covered per episode.
    """
    rng = np.random.RandomState(seed)
    records = []

    for ep in range(n_episodes):
        env, rod = make_tentacle()
        state = random_valid_state(seed=seed + ep)
        set_state(rod, state)

        # Round-robin levers for coverage, then random fill
        levers_ep = list(LEVERS) + [
            rng.choice(LEVERS) for _ in range(levers_per_episode - len(LEVERS))
        ]
        rng.shuffle(levers_ep)

        for lever in levers_ep[:levers_per_episode]:
            new_state, energy = execute_lever(env, rod, lever)
            records.append({
                "state_before": state.copy(),
                "lever": lever,
                "state_after": new_state.copy(),
                "energy": energy,
            })
            state = new_state

        if (ep + 1) % 50 == 0:
            print(f"  Exec pairs: {ep + 1}/{n_episodes} episodes "
                  f"({len(records)} pairs)", flush=True)

    return records


def collect_temporal_pairs(
    n_episodes: int = 200,
    steps_per_episode: int = 20,
    seed: int = 20000,
) -> list[dict]:
    """Collect (s_t, s_{t+1}) temporal pairs for InfoNCE."""
    rng = np.random.RandomState(seed)
    records = []

    for ep in range(n_episodes):
        env, rod = make_tentacle()
        state = random_valid_state(seed=seed + ep)
        set_state(rod, state)

        for _ in range(steps_per_episode):
            lever = rng.choice(LEVERS)
            new_state, _ = execute_lever(env, rod, lever)
            records.append({
                "s_t": state.copy(),
                "s_t1": new_state.copy(),
            })
            state = new_state

        if (ep + 1) % 50 == 0:
            print(f"  Temporal pairs: {ep + 1}/{n_episodes} episodes "
                  f"({len(records)} pairs)", flush=True)

    return records


class ExecAwareDataset(Dataset):
    """Combined dataset for InfoNCE + execution consistency training."""

    def __init__(
        self,
        temporal_records: list[dict],
        exec_records: list[dict],
    ):
        self.temporal = temporal_records
        self.exec = exec_records
        self.n_temporal = len(temporal_records)
        self.n_exec = len(exec_records)
        self.rng = np.random.RandomState(42)

    def __len__(self):
        return max(self.n_temporal, self.n_exec)

    def __getitem__(self, idx):
        # Temporal pair
        t_idx = idx % self.n_temporal
        s_t = torch.tensor(self.temporal[t_idx]["s_t"], dtype=torch.float32)
        s_t1 = torch.tensor(self.temporal[t_idx]["s_t1"], dtype=torch.float32)

        # Random negative
        neg_idx = self.rng.randint(0, self.n_temporal)
        s_neg = torch.tensor(self.temporal[neg_idx]["s_t"], dtype=torch.float32)

        # Execution pair
        e_idx = idx % self.n_exec
        s_before = torch.tensor(self.exec[e_idx]["state_before"], dtype=torch.float32)
        lever_idx = torch.tensor(LEVER_TO_IDX[self.exec[e_idx]["lever"]], dtype=torch.long)
        s_after = torch.tensor(self.exec[e_idx]["state_after"], dtype=torch.float32)

        return s_t, s_t1, s_neg, s_before, lever_idx, s_after
