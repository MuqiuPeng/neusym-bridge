"""PyElastica 500-episode lever control experiment.

Runs with real Cosserat rod physics to verify that SimplifiedRod
fidelity was the bottleneck. Prints progress every 25 episodes.

Usage:
    python -m experiments.exec_aware.run_pyela_500
"""

from __future__ import annotations

import io
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Suppress PyElastica tqdm noise
import elastica
_orig_integrate = elastica.integrate
def _quiet_integrate(*args, **kwargs):
    import contextlib
    with contextlib.redirect_stderr(io.StringIO()):
        return _orig_integrate(*args, **kwargs)
elastica.integrate = _quiet_integrate

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.exec_aware.model import ExecutionAwareEncoder
from experiments.exec_aware.dataset import (
    collect_exec_pairs, collect_temporal_pairs, ExecAwareDataset,
)
from experiments.exec_aware.train import train_exec_aware
from experiments.exec_aware.sweep_k import sweep_k
from experiments.lever_control.collect_data import collect_transitions

OUTPUT_DIR = Path("experiments/exec_aware/outputs")
N_EPISODES = 500
LEVERS_PER_EP = 15


class EncoderWrapper:
    def __init__(self, m):
        self.m = m
    def encode(self, x):
        return self.m.encode(x)
    def eval(self):
        self.m.eval(); return self
    def to(self, d):
        self.m.to(d); return self


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 60, flush=True)
    print("PyElastica 500-Episode Lever Control", flush=True)
    print(f"Episodes: {N_EPISODES}, Levers/ep: {LEVERS_PER_EP}", flush=True)
    print(f"Estimated transitions: {N_EPISODES * LEVERS_PER_EP}", flush=True)
    print("=" * 60, flush=True)

    # ── Step 1: Collect temporal pairs ──
    print(f"\n[{_elapsed(t0)}] Collecting temporal pairs (InfoNCE)...", flush=True)
    temporal = collect_temporal_pairs(
        n_episodes=N_EPISODES, steps_per_episode=LEVERS_PER_EP,
    )
    print(f"[{_elapsed(t0)}] {len(temporal)} temporal pairs collected", flush=True)

    # ── Step 2: Collect execution pairs ──
    print(f"\n[{_elapsed(t0)}] Collecting execution pairs (exec consistency)...", flush=True)
    exec_pairs = collect_exec_pairs(
        n_episodes=N_EPISODES, levers_per_episode=LEVERS_PER_EP,
    )
    print(f"[{_elapsed(t0)}] {len(exec_pairs)} execution pairs collected", flush=True)

    # ── Step 3: Train encoder ──
    print(f"\n[{_elapsed(t0)}] Training Exec-Aware encoder...", flush=True)
    dataset = ExecAwareDataset(temporal, exec_pairs)
    model, hist = train_exec_aware(
        dataset, lambda_exec=0.5, n_epochs=50, device="cpu",
    )
    encoder = EncoderWrapper(model)
    print(f"[{_elapsed(t0)}] Training complete. "
          f"Final loss: nce={hist['nce'][-1]:.4f}, exec={hist['exec'][-1]:.4f}",
          flush=True)

    # ── Step 4: Collect lever transitions for graph ──
    print(f"\n[{_elapsed(t0)}] Collecting lever transitions for graph...", flush=True)
    lever_records = collect_transitions(
        n_episodes=N_EPISODES, levers_per_episode=LEVERS_PER_EP,
    )
    print(f"[{_elapsed(t0)}] {len(lever_records)} lever transitions collected", flush=True)

    # ── Step 5: k-sweep ──
    print(f"\n[{_elapsed(t0)}] Running k-sweep [5, 8, 10, 15, 20]...", flush=True)
    results = sweep_k(
        encoder, lever_records,
        k_values=[5, 8, 10, 15, 20],
        n_tasks=50, device="cpu",
    )

    # ── Save ──
    out_path = OUTPUT_DIR / "pyela_500ep_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {str(k): v for k, v in results.items()},
            f, indent=2,
        )

    print(f"\n[{_elapsed(t0)}] Results saved to {out_path}", flush=True)
    print(f"[{_elapsed(t0)}] Done.", flush=True)


def _elapsed(t0):
    m, s = divmod(int(time.time() - t0), 60)
    return f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
