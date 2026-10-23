"""CLI entrypoint for StrikeShield iterations and corpus ingestion."""

import argparse
import sys
from pathlib import Path

from strikeshield.attack_source import load_jsonl
from strikeshield.config import settings
from strikeshield.corpus import build_corpus
from strikeshield.coverage import get_coverage_report
from strikeshield.db import init_db, migrate


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeShield Iteration Runner")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run attack corpus ingestion, deduplication, and split assignment only",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/seed_attacks.jsonl"),
        help="Path to attack corpus JSONL file",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("policies/v0.md"),
        help="Path to guardrail policy version",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.default_seed,
        help="Random seed for evaluation and stratified splitting",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=settings.database_path,
        help="SQLite database path",
    )
    args = parser.parse_args()

    # Ensure DB schema is initialized and migrated
    init_db(args.db)
    migrate(args.db)

    if args.ingest_only:
        print(f"=== StrikeShield: Ingesting Attack Corpus from {args.corpus} ===")
        records = load_jsonl(args.corpus)
        print(f"Loaded {len(records)} raw attack records from source file.")

        result = build_corpus(
            records=records,
            db_path=args.db,
            seed=args.seed,
        )

        print("\n--- Ingestion & Stratification Summary ---")
        print(f"Imported records:   {result.n_imported}")
        print(f"Deduplicated (dropped): {result.n_deduped}")
        print(f"New attacks added:  {result.n_added}")
        print(f"Patch split count:  {result.n_patch}")
        print(f"Held-out count:     {result.n_heldout}")
        print(f"Cells filled:       {result.cells_filled} / 40")
        print(f"Mean pairwise dist: {result.mean_pairwise_dist:.4f}")

        print("\n--- Threat Matrix Coverage ---")
        coverage = get_coverage_report(args.db)
        print(coverage.format_table())
        sys.exit(0)

    print("Phase 3+ full loop iteration not invoked. Use --ingest-only in Phase 2.")
    sys.exit(0)


if __name__ == "__main__":
    main()
