"""CLI script to generate and display threat matrix coverage."""

import argparse
from pathlib import Path
from strikeshield.coverage import get_coverage_report


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeShield Threat Matrix Coverage Reporter")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite database")
    args = parser.parse_args()

    report = get_coverage_report(args.db)
    print(report.format_table())


if __name__ == "__main__":
    main()
