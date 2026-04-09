from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_bank_tools.feature_bank.registry import build_feature_registry
else:
    from ..feature_bank.registry import build_feature_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the SQLite feature registry for the HDF5 feature bank."
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
        "--db-path",
        type=Path,
        default=None,
        help="Optional explicit SQLite registry path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory_path = (
        args.inventory
        if args.inventory is not None
        else args.bank_root / "registries" / "raw_feature_index.parquet"
    )
    db_path = (
        args.db_path
        if args.db_path is not None
        else args.bank_root / "registry" / "features.sqlite"
    )

    registry_df = build_feature_registry(
        bank_root=args.bank_root,
        inventory_path=inventory_path,
        db_path=db_path,
    )

    print(f"[OK] Wrote SQLite registry: {db_path}")
    print(f"[OK] Registry rows: {len(registry_df)}")


if __name__ == "__main__":
    main()
