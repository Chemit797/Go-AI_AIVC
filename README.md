# GOAI Virtual Cell Baseline

This repository implements the document baseline for the GOAI AI for Research virtual-cell direction: prediction of log2 yeast proteome responses from strain, perturbation, and experimental conditions.

The repository deliberately starts with a fully auditable baseline. It does not include residual decomposition, external molecular/genome embeddings, protein-prior networks, calibration branches, or ensemble methods. Those belong to post-reproduction experiments and must be compared against this reference.

## What is implemented

- Train-only `<80%` protein missingness filtering, `log2` targets, and an observation mask.
- Protein-wise training-mean baseline.
- Exact Matched Control diagnostic baseline using data source, instrument, plate, strain, medium, temperature, time, and time unit.
- Fixed two-hidden-layer condition MLP with mask-aware MSE.
- The document feature ladder: P0 one-hot, P1 training-only statistical priors, P2 condition crosses, P3 time sin/cos, and P4 deterministic chemical-name hash features.
- Independent frozen-split reporting for `val_chem_only`, `val_strain_only`, `val_both`, and `val_time`.
- Metadata-only test inference with a strict submission contract check.

## Data policy

Only the official train/validation metadata, train/validation proteome matrix, and metadata-only test file are used. The test metadata is used only to create condition features and preserve submission order. No test protein targets or target-derived artifacts are read.

Raw competition files, checkpoints, predictions, and runs are ignored by Git. Place the three official CSV files at the repository root, or change their local paths in `configs/baseline.yaml`.

## Installation

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the baseline

```powershell
python -m goai_baseline.audit --config configs/baseline.yaml
python -m goai_baseline.preprocess --config configs/baseline.yaml --output runs/preprocess/feature_contract.json
python -m goai_baseline.evaluate --config configs/baseline.yaml --variant b0_mean --output-dir runs/b0_mean
python -m goai_baseline.evaluate --config configs/baseline.yaml --variant b1_matched_control --output-dir runs/b1_matched_control
python -m goai_baseline.train --config configs/baseline.yaml --variant p0_onehot
```

Run the complete document feature ladder when P0 has been inspected:

```powershell
.\scripts\run_baseline_ladder.ps1 -IncludeP1ToP4
```

Each MLP run writes its configuration, input hashes, feature contract, features summary, loss history, checkpoint, split metrics, protein-level R2, and environment manifest below `runs/`.

## Generate a test prediction

Use a completed MLP run directory. The command never opens a test protein matrix.

```powershell
python -m goai_baseline.predict --config configs/baseline.yaml --run-dir runs\p0_onehot-YYYYMMDD-HHMMSS
python -m goai_baseline.submission runs\p0_onehot-YYYYMMDD-HHMMSS\prediction.csv --config configs/baseline.yaml --feature-contract runs\p0_onehot-YYYYMMDD-HHMMSS\feature_contract.json
```

The prediction is in log2 intensity scale. The contract checker verifies sample order, protein order, duplicate IDs, finite values, and no extra index column.

## Reproduction notes

The PDF examples use placeholder field names and a hard-coded protein count. The code maps them to the released schema and derives the retained protein contract from training rows. See [the method note](docs/baseline_method.md) and [the errata](docs/reproduction_errata.md) for the exact compatibility decisions.

## Tests

```powershell
pytest
```

The tests cover sample alignment, train-only filtering, exact-control keys, no-gradient missing values, unseen-entity feature fallback, metrics, and a complete tiny P0 train-to-submission cycle.
