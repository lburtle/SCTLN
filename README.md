# SCTLNs: Learning Graph Structure and Dynamics in Combinatorial Threshold-Linear Networks via Evolution Strategies — Code Supplementary

Code accompanying our NeurReps 2026 Proceedings Track submission. This repository is provided for reviewer reproducibility and has been anonymized for double-blind review.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

Tested with Python 3.12 on a single CPU node (8 cores, 16 GB). Approximate runtime for 150 seed sweeps is 9 to 10 hours at current hyperparameter settings.

## Repository structure

```
.
├── examples
│   ├── gradient_plots
│   ├── hyperparameters
│   ├── images
│   │   └── phase_portraits
│   └── results
├── logs
├── scripts
└── src
    └── sctln
        └── train
```

## Reproducing paper results

| Paper result | Script | Command | Output |
|---|---|---|---|
| Full SCTLN Oscillation Run & Sweep | `scripts/n_cycle.py` | `python scripts/n_cycle.py` | stdout |
| Fixed Point Run & Sweep | `scripts/fixed_point_test.py` | `python scripts/fixed_point_test.py` | stdout |
| Fixed-Topology Ablation | `scripts/soft_top_ablation.py` | `python scripts/soft_top_ablation.py` | stdout |
| Gradient Baseline | `scripts/gradient_baseline.py` | `python scripts/gradient_baseline.py` | stdout |
| Hardening Analysis | `scripts/hardening_analysis.py` | `python scripts/hardening_analysis.py examples/results/ablation_soft.jsonl` | stdout |
| Topology Recovery from Runs | `scripts/recover_topology.py` | `python scripts/recover_topology.py examples/results/ablation_direct_u.npz` | stdout |

We also have results output to json/jsonl files for previous runs, and log files to reference.

## Worked example

To reproduce Figure 1 (a) and (b), run:

```bash
python scripts/n_cycle.py
```

This writes output to `examples/images/learned_graph_N5.png` and `examples/images/scale_N5.png`, which correspond to Figures 1a and 1b.

## Core implementation

Model and metric implementations live in `src/`, imported by the scripts above:

- `train/es.py` — Implements OpenAI-ES as a usable optimizer for scripts requiring it.
- `dynamics.py` — Simulation roll out for network dynamics.
- `graph.py` — Graph initialization and visualization.
- `model.py` — Fitness and training pipeline/code.
- `readout.py` — Linear readout function definition and application.
- `topology.py` — Graph construction and parameter extraction.

## Provided results

`examples/results/` contains summary tables and figure-generation data for the numbers reported in the paper. Raw checkpoints and full per-seed logs are not included due to size constraints, but will be made available at the de-anonymized repository upon acceptance.

## Known caveats

- Hardening divergence is reported as-is and is not near-zero; this is expected and discussed in Section 4.3, not a bug in hardened_ctln_check.py.
- Currently takes roughly 9 hours to complete a 150 seed sweep for 5 node SCTLN training, which is longer than ideal, but should be expected.
- Raw checkpoints not included due to size; summary files in results/ are sufficient to verify reported numbers.

## Dependencies

See `requirements.txt`.
