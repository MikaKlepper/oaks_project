# OAKS Toxicology Benchmark Pipeline

This repository contains a modular benchmarking pipeline for toxicology prediction
using pre-extracted histopathology features at slide or animal level.
It supports pooled feature probes and Multiple Instance Learning (MIL) methods.

The pipeline is designed to:
- Train and validate on TG-GATES
- Perform external evaluation on UCB
- Support few-shot learning
- Benchmark multiple encoders and probes at scale
- Produce reproducible metrics and plots

---

## Supported Datasets

### TG-GATES (internal dataset)

Used for training, validation, and optional internal testing.

Required structure:

splitting_data/
- TG-GATES/
  - metadata.csv
  - Splits/
    - train.csv
    - val.csv
    - test.csv
  - Subsets/
    - val_balanced_subset.csv
  - FewShotCompoundBalanced/
    - train_fewshot_k{K}.csv

### UCB (external dataset)

Used **only for external testing**.

Required structure:

splitting_data/
- UCB/
  - metadata.csv
  - Subsets/
    - ucb_test.csv

You do NOT need to create train/val/test split CSVs for UCB.
The file `ucb_test.csv` contains all UCB samples and is treated as the full test set.

---

## Feature Directory Structure

### TG-GATES

/data/temporary/toxicology/
- TG-GATES/
  - liver/
    - Trainings_FM/
      - <ENCODER>/
        - features/
    - Validations_FM/
      - <ENCODER>/
        - features/
    - Tests_FM/
      - <ENCODER>/
        - features/

### UCB

/data/temporary/toxicology/
- UCB/
  - liver/
    - Tests_FM/
      - <ENCODER>/
        - features/

Notes:
- <ENCODER> refers to the encoder name (e.g. UNI, CONCH, VIRCHOW2)
- UCB only requires Tests_FM
- The pipeline automatically falls back to Tests_FM if other splits do not exist

---

## Dataset Selection Logic

Dataset switching is automatic.

Rules:
- TG-GATES is the default dataset
- UCB is selected when `--test_subset_csv` contains "ucb"

Example:
--test_subset_csv splitting_data/UCB/Subsets/ucb_test.csv

This automatically triggers:
- UCB metadata loading
- UCB feature directory resolution
- Dataset-aware caching
- Dataset-aware logging and plots

No manual dataset flag is required.

---

## Pipeline Stages

### Training (stage=train)
- Always uses TG-GATES
- Uses full training split OR few-shot subsets OR explicit training subset CSV

### Validation (stage=eval)
- Always uses TG-GATES
- Defaults to TG-GATES/Subsets/val_balanced_subset.csv

### External Testing (stage=test)
- Uses a subset CSV
- For UCB: ucb_test.csv (full dataset)

---

## Running the Pipeline

Move into the pipeline directory:

cd pipeline

Train a single model:

python main.py \
  --config configs/base_config.yaml \
  --model UNI \
  --probe linear \
  --stage train

Train and validate:

python main.py \
  --config configs/base_config.yaml \
  --model UNI \
  --probe linear \
  --stage eval

External evaluation on UCB:

python main.py \
  --config configs/base_config.yaml \
  --model UNI \
  --probe abmil \
  --stage test \
  --test_subset_csv splitting_data/UCB/Subsets/ucb_test.csv

---

## Running the Full Benchmark

From the repository root:

python benchmark.py

This will:
- Train on TG-GATES
- Validate on TG-GATES
- Perform external evaluation on UCB
- Skip completed experiments automatically
- Generate benchmark plots per dataset

---

## Outputs

Metrics are stored per dataset:

outputs/
- eval/
  - tggates/
  - ucb/
- test/

Plots are generated automatically:
- Encoder vs performance
- MIL vs pooled comparisons
- Combined benchmark summaries

---

## Design Guarantees

- No dataset-specific branching in training code
- Subset CSVs fully control which samples are evaluated
- External datasets never affect training or validation
- Safe defaults and reproducible benchmarks
- Easily extensible to new datasets

---

## Final Notes

- Subset CSV does NOT mean partial dataset
- Subset CSV defines which samples are evaluated
- UCB subset contains all UCB samples by design
- This behavior is intentional and correct
