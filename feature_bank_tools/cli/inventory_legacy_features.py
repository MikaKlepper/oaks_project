from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_bank_tools.feature_bank.common import write_dataframe
    from feature_bank_tools.feature_bank.inventory import build_legacy_inventory
else:
    from ..feature_bank.common import write_dataframe
    from ..feature_bank.inventory import build_legacy_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory raw slide2vec features from the legacy split-based layout."
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("/data/temporary/toxicology"),
        help="Root of the old split-based feature layout.",
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=Path("feature_bank"),
        help="Unified feature bank root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit inventory output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output
        if args.output is not None
        else args.bank_root / "registries" / "raw_feature_index.parquet"
    )

    df = build_legacy_inventory(args.legacy_root)
    if df.empty:
        raise SystemExit("No legacy raw feature files were found.")

    write_dataframe(df, output)
    print(f"[OK] Wrote inventory: {output}")
    print(f"[OK] Rows: {len(df)}")


if __name__ == "__main__":
    main()
