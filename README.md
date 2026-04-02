# neusym-bridge

Neurosymbolic bridge: extracting symbolic causal structure from neural network common representations.

## Project Structure

```
neusym-bridge/
├── configs/                    # Experiment configurations
├── docs/results/               # Experiment reports and archived results
├── rules/                      # Relatum rule files
├── scripts/                    # Phase orchestration scripts
│   ├── run_phase0.py           # Data generation + model training
│   ├── run_phase1.py           # Common structure detection (CKA, Procrustes)
│   ├── run_phase2.py           # Physics correspondence (SVCCA, SINDy)
│   └── run_phase3.py           # Collapse mechanism verification
├── src/neusym_bridge/
│   ├── data/                   # PDE solvers and data generation
│   ├── models/                 # CNN world model + training
│   ├── analysis/               # Representation analysis + structure extraction
│   └── relatum/                # Probabilistic collapse mechanism
└── tests/                      # 46 tests
```

## Validation Phases

| Phase | Status | Key Result |
|-------|--------|------------|
| 0: Environment | PASS | 3 models converge (ratio=1.006) |
| 1: Common Structure | PASS 4/5 | CKA=0.944 across 3 CNN models |
| 2: Physics Correspondence | Conditional | SVCCA=0.999, 3 causal directions found |
| 3: Collapse Mechanism | PASS 6/6 | Collapse, retraction, active query verified |
