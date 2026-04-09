from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_bank_tools.feature_bank.sync import sync_bank_to_local
else:
    from ..feature_bank.sync import sync_bank_to_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror the unified HDF5 feature bank structure to a node-local root."
    )
    parser.add_argument(
        "--shared-bank-root",
        type=Path,
        default=Path("feature_bank"),
        help="Shared canonical feature bank root.",
    )
    parser.add_argument(
        "--local-bank-root",
        type=Path,
        default=Path("/local/feature_bank"),
        help="Local mirrored feature bank root.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional explicit SQLite registry path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = (
        args.db_path
        if args.db_path is not None
        else args.shared_bank_root / "registry" / "features.sqlite"
    )

    summary = sync_bank_to_local(
        shared_bank_root=args.shared_bank_root,
        local_bank_root=args.local_bank_root,
        db_path=db_path,
        overwrite=args.overwrite,
    )

    print(f"[OK] Sync complete -> {args.local_bank_root}")
    print(f"[INFO] HDF5 files copied: {summary['copied']}")
    print(f"[INFO] Skipped existing: {summary['skipped']}")
    print(f"[INFO] Missing shared HDF5 files: {summary['missing']}")


if __name__ == "__main__":
    main()
