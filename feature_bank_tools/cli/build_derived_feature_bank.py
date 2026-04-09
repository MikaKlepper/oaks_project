from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from feature_bank_tools.feature_bank.derived_bank import build_derived_bank
else:
    from ..feature_bank.derived_bank import build_derived_bank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build derived pooled slide and animal HDF5 feature banks from the raw HDF5 bank."
    )
    parser.add_argument(
        "--bank-root",
        type=Path,
        default=Path("feature_bank"),
        help="Unified feature bank root.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="Metadata CSV used to map slide ids to animals.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset key, e.g. tggates or ucb.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Encoder name, e.g. H_OPTIMUS_1.",
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        default=None,
        help="Multiple encoder names, e.g. H_OPTIMUS_1 UNI PHIKON.",
    )
    parser.add_argument(
        "--all-encoders",
        action="store_true",
        help="Use all available encoders discovered under feature_bank/raw/<dataset>/.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default=None,
        choices=["mean", "max", "min"],
        help="Aggregation to use for slide and animal pooling.",
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        default=None,
        help="Multiple aggregations, e.g. mean max min.",
    )
    parser.add_argument(
        "--overwrite-existing-datasets",
        action="store_true",
        help="Overwrite existing derived datasets if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = build_derived_bank(
        bank_root=args.bank_root,
        metadata_csv=args.metadata_csv,
        dataset=args.dataset,
        encoder=args.encoder,
        encoders=args.encoders,
        all_encoders=args.all_encoders,
        aggregation=args.aggregation,
        aggregations=args.aggregations,
        overwrite_existing_datasets=args.overwrite_existing_datasets,
    )

    total_slide_written = sum(int(item["written_slides"]) for item in summaries)
    total_slide_skipped = sum(int(item["skipped_slides"]) for item in summaries)
    total_animal_written = sum(int(item["written_animals"]) for item in summaries)
    total_animal_skipped = sum(int(item["skipped_animals"]) for item in summaries)

    print(f"[OK] Derived HDF5 banks updated for dataset={args.dataset.lower()}")
    print(f"[INFO] Outputs created/updated: {len(summaries)}")
    print(f"[INFO] Slide features written: {total_slide_written}")
    print(f"[INFO] Slide features skipped: {total_slide_skipped}")
    print(f"[INFO] Animal features written: {total_animal_written}")
    print(f"[INFO] Animal features skipped: {total_animal_skipped}")
    for summary in summaries:
        print(
            "[DONE] "
            f"{summary['encoder']} | {summary['aggregation']} -> {summary['derived_path']}"
        )


if __name__ == "__main__":
    main()
