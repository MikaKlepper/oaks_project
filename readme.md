# OAKS Toxicology Benchmark Pipeline

This repository contains a modular benchmarking pipeline for toxicology prediction
using pre-extracted histopathology features at slide or animal level.
It supports pooled feature probes and Multiple Instance Learning (MIL) methods.

The pipeline is designed to:
- Train and validate on **TG-GATES**
- Perform **external out-of-distribution (OOD) evaluation** on **UCB**
- Optional **calibration/fine-tuning** on UCB samples
- Support few-shot learning
- Benchmark multiple encoders and probes at scale
- Produce reproducible metrics and plots

---

## Supported Datasets

### TG-GATES (internal dataset)

Used for training, validation, and optional internal testing.

Required directory structure:

splitting_data/
TG-GATES/
- metadata.csv
- Splits/
  - train.csv
  - val.csv
  - test.csv
- Subsets/
  - val_balanced_subset.csv
- FewShotCompoundBalanced/
  - train_fewshot_k{K}.csv

---

### UCB (external dataset)

Used for external out-of-distribution testing and optional calibration.

Required directory structure:

splitting_data/
UCB/
- metadata.csv
- Subsets/
  - ucb_test.csv

Notes:
- You do **not** need train/val/test splits for UCB
- `ucb_test.csv` contains **all UCB samples**
- UCB is used only for **test** and **calibration** (never for TG-GATES validation)

---

## Feature Bank

This pipeline uses a **feature bank registry** (SQLite) to resolve features on disk.
Configure paths in `configs/base_config.yaml`:

- `features.bank_root`
- `features.registry_path`
- `features.local_bank_root` (optional)

---

## Dataset Selection Logic (Plain Language)

Dataset selection is explicit and reproducible:

- `--dataset tggates` means: train/validate/test on TG-GATES splits.
- `--dataset ucb` means: evaluate on UCB only (test stage).
- Calibration always **samples from UCB** and **fine-tunes the TG-GATES-trained probe**.

Short version:
- TG-GATES is the training backbone.
- UCB is the external OOD dataset (test + calibration only).

---

## Pipeline Stages (What Actually Happens)

### Training (stage=train)
- Uses TG-GATES **by default**
- Uses full training split, a few-shot subset, or an explicit `--subset_csv`
- If calibration is enabled, the calibration **train subset is drawn from UCB**

### Validation (stage=eval)
- Uses TG-GATES
- Liver hypertrophy defaults to `TG-GATES/Splits/val.csv`
- Any abnormality uses `Splits/latest/val.csv`

### Testing (stage=test)
- Uses the configured dataset and subset CSV
- TG-GATES: internal test split (or a subset CSV)
- UCB: external OOD test split (`ucb_test.csv`)
- Loads TG-GATES-trained weights and evaluates without retraining

### Calibration (when `--calibrate` is enabled)
- Train base model on TG-GATES (k-shot or full training split)
- Sample N **UCB** cases for calibration
- Warm-start from base checkpoint and fine-tune on that UCB subset
- Evaluate on TG-GATES val/test (or UCB test if dataset is UCB)

---

## Running the Pipeline (Quick Start)

Move into the pipeline directory:

cd pipeline

Train a single model:

python main.py \
  --config configs/base_config.yaml \
  --dataset tggates \
  --model UNI \
  --probe linear \
  --stage train

Train and validate:

python main.py \
  --config configs/base_config.yaml \
  --dataset tggates \
  --model UNI \
  --probe linear \
  --stage all

External OOD evaluation on UCB:

python main.py \
  --config configs/base_config.yaml \
  --dataset ucb \
  --model UNI \
  --probe abmil \
  --stage test \
  --test_subset_csv splitting_data/UCB/Subsets/ucb_test.csv

## Common Recipes (Copy/Paste)

Train on TG-GATES (full train) + validate:

python main.py \
  --config configs/base_config.yaml \
  --dataset tggates \
  --model H_OPTIMUS_1 \
  --probe linear \
  --stage all

Train on TG-GATES (k-shot) + validate:

python main.py \
  --config configs/base_config.yaml \
  --dataset tggates \
  --model H_OPTIMUS_1 \
  --probe linear \
  --k 100 \
  --stage all

Calibrate on UCB after TG-GATES training:

python main.py \
  --config configs/base_config.yaml \
  --dataset tggates \
  --model H_OPTIMUS_1 \
  --probe linear \
  --k 100 \
  --calibrate \
  --calibration_samples 25 \
  --calibration_seed 42 \
  --stage all

---

## Pipeline Flow (Diagram)

```mermaid
flowchart TD
  A[benchmark.py] --> B[main.py]
  B --> C[config_loader.py]
  C --> D[split_resolver.py]
  C --> E[prepare_dataset.py]
  E --> F[feature_bank_resolver.py]

  D -->|calibration ON| D1[UCB calibration subset]
  D -->|calibration OFF| D2[TG-GATES / latest splits]
  D -->|few-shot k| D3[k-subset CSVs]

  B --> G[train.py]
  G --> G1[checkpoint saved]
  G1 --> H[eval.py]

  H --> H1[metrics + outputs]
  H1 --> I[log_benchmark.py]
  I --> J[plot_benchmarks.py]

  H -->|validation| V[outputs/.../validation/]
  H -->|testing| T[outputs/.../testing/<dataset>/]
  I --> VB[outputs/validation/<dataset>/*.csv]
  I --> TB[outputs/testing/<dataset>/*.csv]
  G1 --> CKPT[train/probe_<probe>.pt|joblib]
```

---

## Running the Full Benchmark

From the repository root:

python benchmark.py

This will:
- Train on TG-GATES
- Validate on TG-GATES
- Perform external OOD evaluation on UCB (if configured)
- Run both **full** training and **k-shot** when configured in `benchmark.py`
- Skip completed experiments automatically
- Generate benchmark plots per dataset and stage

---

## Outputs (Where Things Land)

Outputs are written under the experiment root defined in the config, e.g.:

outputs/experiments_benchmark_final/<dataset>/<aggregation>/<encoder>/<probe>/k<k>/

Stage outputs are stored inside the experiment root:

<experiment_root>/
- train/
- validation/
- testing/<dataset>/

Notes:
- TG-GATES results live under TG-GATES experiment roots
- UCB results live under UCB experiment roots (test-only)
- Evaluation and test plots are generated separately

Benchmark summaries and plots are written to:

- `outputs/validation/<dataset>/...`
- `outputs/testing/<dataset>/...`

---

## Design Guarantees

- External datasets never affect TG-GATES validation
- Subset CSVs fully control evaluation samples
- No dataset leakage into model selection
- Reproducible experiments and logging
- Easily extensible to new datasets

---

## Final Notes

- “Subset CSV” defines which samples are evaluated
- UCB is an external out-of-distribution test set by design
- Severity distributions differ across datasets and are reported separately
