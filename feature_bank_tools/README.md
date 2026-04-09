# Feature Bank Tools

Standalone utilities for building a split-agnostic raw feature bank from the
legacy `Trainings_FM` / `Validations_FM` / `Tests_FM` layout.

These tools do not modify the current pipeline. They create a new raw feature
bank in HDF5 format plus a SQLite lookup registry that you can review before
integrating into the pipeline.

The folder is structured as a small internal API plus thin CLI wrappers:

```text
feature_bank_tools/
  README.md
  __init__.py

  feature_bank/
    __init__.py
    api.py
    common.py
    inventory.py
    raw_bank.py
    derived_bank.py
    registry.py
    sync.py
    validator.py

  cli/
    __init__.py
    inventory_legacy_features.py
    build_raw_feature_bank.py
    build_derived_feature_bank.py
    build_registry_sqlite.py
    validate_raw_feature_bank.py
    sync_feature_bank_to_local.py
```

## Scope

This tooling currently covers:

- inventory the old raw feature files
- build a unified raw HDF5 feature bank
- build unified derived HDF5 feature banks
- build a SQLite feature registry
- validate the new bank
- optionally mirror the bank to `/local/feature_bank`

## Target Layout

```text
feature_bank/
  raw/
    tggates/
      H_OPTIMUS_1.h5
      UNI.h5
    ucb/
      H_OPTIMUS_1.h5
  derived/
    tggates/
      mean/
        H_OPTIMUS_1.h5
      max/
        H_OPTIMUS_1.h5
    ucb/
      mean/
        H_OPTIMUS_1.h5
  registry/
    features.sqlite
  registries/
    raw_feature_index.parquet
```

Inside one raw HDF5 file:

```text
/slide/<slide_id>
```

Inside one derived HDF5 file:

```text
/slide/<slide_id>
/animal/<animal_id>
```

## Scripts

The CLI scripts are convenience wrappers around the internal API.

### 1. Inventory legacy raw features

```bash
python feature_bank_tools/cli/inventory_legacy_features.py \
  --legacy-root /data/temporary/toxicology \
  --bank-root /path/to/feature_bank
```

Writes:

- `feature_bank/registries/raw_feature_index.parquet`

### 2. Build unified raw feature bank

```bash
python feature_bank_tools/cli/build_raw_feature_bank.py \
  --bank-root /path/to/feature_bank
```

Writes:

- `feature_bank/raw/<dataset>/<encoder>.h5`

### 3. Build SQLite registry

```bash
python feature_bank_tools/cli/build_registry_sqlite.py \
  --bank-root /path/to/feature_bank
```

Writes:

- `feature_bank/registry/features.sqlite`

### 4. Build derived feature banks

```bash
python feature_bank_tools/cli/build_derived_feature_bank.py \
  --bank-root /path/to/feature_bank \
  --metadata-csv /path/to/metadata.csv \
  --dataset tggates \
  --all-encoders \
  --aggregations mean max min
```

Writes:

- `feature_bank/derived/<dataset>/<aggregation>/<encoder>.h5`

### 5. Validate feature bank

```bash
python feature_bank_tools/cli/validate_raw_feature_bank.py \
  --bank-root /path/to/feature_bank
```

Optional metadata validation:

```bash
python feature_bank_tools/cli/validate_raw_feature_bank.py \
  --bank-root /path/to/feature_bank \
  --metadata-csv /path/to/metadata.csv \
  --metadata-slide-id-col slide_id
```

### 6. Mirror shared bank to local node cache

This mirrors the same internal structure under `/local/feature_bank`.

```bash
python feature_bank_tools/cli/sync_feature_bank_to_local.py \
  --shared-bank-root /path/to/feature_bank \
  --local-bank-root /local/feature_bank
```

## Notes

- The registry stores:
  - `hdf5_relative_path`
  - `hdf5_key`
- That lets the same artifact resolve under both:
  - shared root: `feature_bank/<hdf5_relative_path>`
  - local root: `/local/feature_bank/<hdf5_relative_path>`
- Derived slide and animal features are also materialized under:
  - `feature_bank/derived/<dataset>/<aggregation>/<encoder>.h5`

## Python dependencies

These scripts require:

- `pandas`
- `pyarrow` for Parquet inventory files
- `torch`
- `numpy`
- `h5py`

Example install:

```bash
pip install pandas pyarrow torch numpy h5py
```
