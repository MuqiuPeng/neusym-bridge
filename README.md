# neusym-bridge

neusym-bridge extracts symbolic causal structure from the common representations
of neural networks. Independently trained networks converge on the same latent
structure; the question here is whether that structure carries causal
information that can be lifted into symbolic rules — probabilistic facts
collapsed into deterministic knowledge, and retracted along a provenance chain
when a premise goes — and whether a planner driven by those rules does better
than the network alone. The test beds are 2D heat diffusion (phases 1–3) and
tentacle control (phase 4 and the experiments after it).

The rule layer is [Relatum](https://github.com/MuqiuPeng/Relatum); the rule
files live in [`rules/`](rules/).

## Validation phases

| Phase | Question | Status | Key result |
|---|---|---|---|
| 1 Common structure | Do several models learn the same structure? | PASS 4/5 | CKA = 0.944 |
| 2 Physics correspondence | Does that structure correspond to the physics? | conditional pass | SVCCA = 0.999, 4 causal directions |
| 3 Collapse mechanism | Is probabilistic → deterministic logic correct? | PASS after fixes | 20 tests, 16/20 then 20/20 |
| 4 End to end | Is the whole system better than its parts? | PASS 6/6 | planning distance −62%, explanation rate 100% |

Follow-up experiments in [`experiments/`](experiments/) take the findings apart:
training-objective ablation (A1), cross-architecture replication (A4 —
cross-architecture CKA 0.635 against a 0.389 random baseline), intervention
robustness (B3 — 4/4 directions causal, n=50, p<0.0001), interface calibration
(B4 — mean ECE 0.021), component ablation (the interface layer drives −73.1% of
the gain; the Relatum reasoning layer costs +4.8% on top of a strong encoder),
multi-seed reproduction, maze and lever-control domains, and encoder
comparisons.

Each phase has a report in [`docs/results/`](docs/results/), and
[`docs/results/conclusion.md`](docs/results/conclusion.md) collects every claim
with its numbers (in Chinese).

## Layout

```
neusym-bridge/
├── configs/                    # experiment configuration (phase1, phase2, phase4)
├── checkpoints/                # trained phase 0-3 models
├── docs/results/              # reports, figures, archived results
├── rules/                      # Relatum rule files
├── scripts/                    # phase orchestration
│   ├── run_phase0.py           # data generation + model training
│   ├── run_phase1.py           # common structure detection (CKA, Procrustes)
│   ├── run_phase2.py           # physics correspondence (SVCCA, SINDy)
│   ├── run_phase3.py           # collapse mechanism verification
│   └── run_phase4.py           # end-to-end tentacle control
├── experiments/                # follow-up experiments, each with its own results
├── src/neusym_bridge/
│   ├── data/                   # PDE solvers and data generation
│   ├── models/                 # CNN world model + training
│   ├── analysis/               # representation analysis + structure extraction
│   └── relatum/                # probabilistic collapse mechanism
└── tests/                      # 46 tests
```

## Running it

Python 3.10 or later.

```bash
pip install -e ".[dev]"
pytest
```

```bash
python scripts/run_phase0.py     # generate data, train the three models
python scripts/run_phase1.py     # and so on, phase by phase
```

Phase 4 simulates an elastic rod, which needs one more dependency:

```bash
pip install -e ".[phase4]"
python scripts/run_phase4.py
```
