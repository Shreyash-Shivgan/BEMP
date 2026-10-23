"""CLI tool to evaluate a policy version across the full attack corpus and benign benchmark."""

import argparse
import sys
from pathlib import Path

from strikeshield.config import settings
from strikeshield.db import init_db, migrate
from strikeshield.evaluator import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeShield Policy Evaluator")
    parser.add_argument("--policy", type=Path, default=Path("policies/v0.md"), help="Policy markdown file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation replay")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier")
    parser.add_argument("--db", type=Path, default=settings.database_path, help="SQLite database path")
    parser.add_argument("--benign", type=Path, default=Path("data/benign/benign_set.jsonl"), help="Path to benign set")
    args = parser.parse_args()

    # Ensure schema is current
    init_db(args.db)
    migrate(args.db)

    if not args.policy.exists():
        print(f"ERROR: Policy file not found: {args.policy}")
        sys.exit(1)

    policy_text = args.policy.read_text(encoding="utf-8")
    policy_version = args.policy.stem

    evaluator = Evaluator(db_path=args.db, benign_path=args.benign)
    result = evaluator.evaluate(
        policy_version=policy_version,
        policy_text=policy_text,
        seed=args.seed,
        run_id=args.run_id,
    )

    print(result.format_report())
    if result.artifact_path:
        print(f"\n[OK] Evaluation report saved to: {result.artifact_path}")


if __name__ == "__main__":
    main()
