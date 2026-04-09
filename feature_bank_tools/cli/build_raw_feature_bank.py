from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_bank_tools.feature_bank.raw_bank import build_raw_bank
else:
    from ..feature_bank.raw_bank import build_raw_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified raw HDF5 feature bank from a legacy inventory."
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=Path("feature_bank"),
        help="Unified feature bank root.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Path to legacy raw inventory table.",
    )
    parser.add_argument(
        "--overwrite-existing-datasets",
        action="store_true",
        help="Overwrite existing HDF5 datasets inside the raw bank.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory_path = (
        args.inventory
        if args.inventory is not None
        else args.bank_root / "registries" / "raw_feature_index.parquet"
    )

    summary = build_raw_bank(
        bank_root=args.bank_root,
        inventory_path=inventory_path,
        overwrite_existing_datasets=args.overwrite_existing_datasets,
    )

    print(f"[OK] Raw HDF5 bank updated under: {args.bank_root / 'raw'}")
    print(f"[INFO] Canonical raw artifacts considered: {summary['canonical_rows']}")
    print(f"[INFO] Raw datasets written: {summary['written']}")
    print(f"[INFO] Existing datasets skipped: {summary['skipped']}")
    print(f"[INFO] Missing source files skipped: {summary['missing_sources']}")


if __name__ == "__main__":
    main()
