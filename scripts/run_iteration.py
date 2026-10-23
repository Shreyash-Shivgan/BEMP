"""CLI entrypoint for StrikeShield iterations, corpus ingestion, and target+judge baseline evaluation."""

import argparse
import datetime
import json
import sys
import uuid
from pathlib import Path

from strikeshield.asr import compute_asr_from_map, confound_report
from strikeshield.attack_source import AttackRecord, load_jsonl
from strikeshield.config import settings
from strikeshield.corpus import build_corpus
from strikeshield.coverage import get_coverage_report
from strikeshield.db import get_connection, init_db, migrate
from strikeshield.judge import JudgeRunner, report_guard
from strikeshield.target import TargetRunner
from scripts.validate_judge import check_judge_validation_status


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeShield Iteration Runner")
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run attack corpus ingestion, deduplication, and split assignment only",
    )
    parser.add_argument(
        "--through-judge",
        action="store_true",
        help="Run target generation and judge evaluation through baseline ASR computation",
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

    # Ensure DB schema is initialized and migrated to latest version
    init_db(args.db)
    migrate(args.db)

    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

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

    if args.through_judge:
        print(f"=== StrikeShield: Running Baseline Target + Judge Pipeline (Policy: {args.policy.name}) ===")
        check_judge_validation_status()

        # 1. Ingest / Sync attacks from corpus
        raw_records = load_jsonl(args.corpus)
        build_result = build_corpus(raw_records, db_path=args.db, seed=args.seed)

        # 2. Load policy text and extract version name
        if not args.policy.exists():
            print(f"ERROR: Policy file {args.policy} does not exist.")
            sys.exit(1)
        policy_text = args.policy.read_text(encoding="utf-8")
        policy_version = args.policy.stem  # e.g. "v0"

        # Record policy in DB if not present
        conn = get_connection(args.db)
        try:
            conn.execute(
                "INSERT OR IGNORE INTO policies (version, diff_text, status) VALUES (?, ?, ?);",
                (policy_version, "Baseline policy definition", "ACCEPTED"),
            )
            conn.execute(
                "INSERT INTO runs (id, seed, iteration, config_json) VALUES (?, ?, ?, ?);",
                (run_id, args.seed, 0, json.dumps({"policy": policy_version, "corpus": str(args.corpus)})),
            )
            conn.commit()

            # Retrieve all stored attacks for evaluation
            cursor = conn.execute(
                "SELECT id, prompt, risk_category, technique_family, target_behaviour, split FROM attacks;"
            )
            stored_attacks = [
                AttackRecord(
                    id=row["id"],
                    prompt=row["prompt"],
                    risk_category=row["risk_category"],
                    technique_family=row["technique_family"],
                    target_behaviour=row["target_behaviour"],
                    source="database",
                )
                for row in cursor.fetchall()
            ]
            splits_by_id = {row["id"]: row["split"] for row in conn.execute("SELECT id, split FROM attacks;").fetchall()}
        finally:
            conn.close()

        print(f"Executing Target Model on {len(stored_attacks)} attacks (seed={args.seed}, temp={settings.models.target.temperature})...")

        # 3. Target Model Batch Execution
        with TargetRunner(temperature=settings.models.target.temperature) as target_runner:
            responses = target_runner.run_batch(
                attacks=stored_attacks,
                policy_version=policy_version,
                policy_text=policy_text,
                seed=args.seed,
                db_path=args.db,
                run_id=run_id,
            )

        print(f"Completed {len(responses)} target responses. Executing Judge Model evaluation...")

        # 4. Judge Model Batch Execution
        tb_map = {atk.id: atk.target_behaviour for atk in stored_attacks if atk.id}
        with JudgeRunner(temperature=settings.models.judge.temperature) as judge_runner:
            verdicts = judge_runner.run_batch(
                responses=responses,
                target_behaviours=tb_map,
                db_path=args.db,
            )

        # 5. Compute ASR Aggregates
        resp_to_atk = {r.id: r.attack_id for r in responses}
        atk_map = {atk.id: atk for atk in stored_attacks if atk.id}

        verdicts_with_meta = []
        for v in verdicts:
            atk_id = resp_to_atk.get(v.response_id, "")
            atk_rec = atk_map.get(atk_id)
            if atk_rec:
                verdicts_with_meta.append({
                    "is_success": v.is_success,
                    "split": splits_by_id.get(atk_id, "patch"),
                    "risk_category": atk_rec.risk_category,
                    "technique_family": atk_rec.technique_family,
                })

        asr_results = compute_asr_from_map(verdicts_with_meta)
        confound_str = confound_report(stored_attacks)

        # 6. Audit statistics query
        conn = get_connection(args.db)
        try:
            n_calls = conn.execute("SELECT COUNT(*) FROM judge_calls;").fetchone()[0]
            n_valid = conn.execute("SELECT COUNT(*) FROM judge_calls WHERE schema_error IS NULL;").fetchone()[0]
            n_schema_errors = conn.execute("SELECT COUNT(*) FROM judge_calls WHERE schema_error IS NOT NULL;").fetchone()[0]
            n_mismatches = conn.execute("SELECT COUNT(*) FROM evidence_mismatches;").fetchone()[0]
        finally:
            conn.close()

        # 7. Print Output
        banner = report_guard()
        if banner:
            print(f"\n========================================================")
            print(f"  {banner}")
            print(f"========================================================\n")

        print("--- Baseline ASR Evaluation ---")
        print(asr_results.format_table())

        print("\n--- Confound Analysis ---")
        print(confound_str)

        print("\n--- Judge Audit Statistics ---")
        print(f"Total Judge Calls:       {n_calls}")
        print(f"Valid Structured Calls:  {n_valid}")
        print(f"Schema Errors:           {n_schema_errors}")
        print(f"Evidence Mismatches:     {n_mismatches}")
        print(f"Cohen's Kappa Status:    unknown (run scripts/validate_judge.py to evaluate)")

        # 8. Write Artifact to runs/{run_id}/baseline_v0_report.md
        run_dir = settings.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report_file = run_dir / "baseline_v0_report.md"

        artifact_content = (
            f"# Baseline Evaluation Report ({policy_version})\n\n"
            f"- **Run ID:** `{run_id}`\n"
            f"- **Timestamp:** {datetime.datetime.now().isoformat()}\n"
            f"- **Corpus Role:** `{settings.taxonomy.grid.corpus_role}`\n\n"
            f"{asr_results.format_table()}\n\n"
            f"{confound_str}\n\n"
            f"### Judge Audit\n"
            f"- Total Calls: {n_calls}\n"
            f"- Valid Calls: {n_valid}\n"
            f"- Schema Errors: {n_schema_errors}\n"
            f"- Quoted Evidence Mismatches: {n_mismatches}\n"
            f"- Inter-Rater Agreement: Unknown (Pending human validation study)\n"
        )
        report_file.write_text(artifact_content, encoding="utf-8")
        print(f"\n[OK] Run artifact saved to: {report_file}")
        sys.exit(0)

    print("Specify --ingest-only or --through-judge to run pipeline steps.")
    sys.exit(0)


if __name__ == "__main__":
    main()
