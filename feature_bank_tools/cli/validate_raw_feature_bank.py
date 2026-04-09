from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_bank_tools.feature_bank.validator import validate_feature_bank
else:
    from ..feature_bank.validator import validate_feature_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the unified HDF5 feature bank and SQLite registry."
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=Path("feature_bank"),
        help="Unified feature bank root.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional explicit SQLite registry path.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional metadata CSV for slide coverage validation.",
    )
    parser.add_argument(
        "--metadata-slide-id-col",
        type=str,
        default="slide_id",
        help="Slide-id column in the metadata CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = (
        args.db_path
        if args.db_path is not None
        else args.bank_root / "registry" / "features.sqlite"
    )

    result = validate_feature_bank(
        bank_root=args.bank_root,
        db_path=db_path,
        metadata_csv=args.metadata_csv,
        metadata_slide_id_col=args.metadata_slide_id_col,
    )

    print(f"[INFO] Registry rows: {result['rows']}")
    print(f"[INFO] Duplicate artifact keys: {result['duplicate_keys']}")
    print(f"[INFO] Missing HDF5 files: {len(result['missing_files'])}")
    for path in result["missing_files"][:10]:
        print(f"  {path}")
    print(f"[INFO] Missing HDF5 keys: {len(result['missing_keys'])}")
    for key in result["missing_keys"][:10]:
        print(f"  {key}")

    if args.metadata_csv is not None:
        print(f"[INFO] Metadata slide ids: {result['metadata_slide_ids']}")
        print(f"[INFO] Registry slide ids: {result['registry_slide_ids']}")
        print(f"[INFO] Metadata ids missing in registry: {len(result['missing_in_registry'])}")
        print(f"[INFO] Registry ids missing in metadata: {len(result['extra_in_registry'])}")
        if result["missing_in_registry"]:
            print("[INFO] First metadata ids missing in registry:")
            for value in result["missing_in_registry"][:20]:
                print(f"  {value}")

    if (
        result["duplicate_keys"] > 0
        or result["missing_files"]
        or result["missing_keys"]
    ):
        raise SystemExit(1)

    print("[OK] Feature bank validation passed.")


if __name__ == "__main__":
    main()
