"""CLI entrypoint for evaluating candidate policies against the Gate."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from strikeshield.config import settings
from strikeshield.db import get_connection
from strikeshield.evaluator import EvaluationResult, Evaluator
from strikeshield.gate import Gate, GateDecision
from strikeshield.metrics import Metric
from strikeshield.policies import PolicyVersion


def load_eval_result_for_run(run_id: str, version: str) -> Optional[EvaluationResult]:
    """Loads evaluation result JSON from runs directory if it exists."""
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        for f in run_dir.glob(f"eval_{version}_*.json"):
            try:
                return EvaluationResult.model_validate_json(f.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def get_latest_baseline_run() -> str:
    """Finds latest baseline evaluation run from DB."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT run_id FROM metrics WHERE policy_version='v0' ORDER BY rowid DESC LIMIT 1;").fetchone()
        if row:
            return row["run_id"]
        row_run = conn.execute("SELECT id FROM runs ORDER BY started_at DESC LIMIT 1;").fetchone()
        if row_run:
            return row_run["id"]
        raise ValueError("No baseline evaluation runs found.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeShield Security Gate CLI")
    parser.add_argument("--candidate", type=Path, default=Path("policies/v1.md"), help="Candidate policy path")
    parser.add_argument("--baseline-run", type=str, default=None, help="Baseline evaluation run ID")
    parser.add_argument("--strict-noise-floor", type=str, default="true", help="Enforce strict noise floor (true/false)")
    parser.add_argument("--db-path", type=Path, default=settings.database_path, help="Database path")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument("--synthetic-asr-increase", action="store_true", help="Simulate a candidate that increases ASR by 1/25 (reproducing deck failure)")
    args = parser.parse_args()

    strict_noise_floor = args.strict_noise_floor.lower() in ("true", "1", "yes")

    # 1. Load candidate policy (enforces candidate path protection via PolicyVersion.load)
    cand_policy = PolicyVersion.load(args.candidate)

    # 2. Load baseline evaluation result
    base_run_id = args.baseline_run or get_latest_baseline_run()
    base_result = load_eval_result_for_run(base_run_id, "v0")

    if not base_result:
        # Evaluate baseline if missing
        evaluator = Evaluator(db_path=args.db_path)
        base_policy = PolicyVersion.load("policies/v0.md")
        base_result = evaluator.evaluate(
            policy_version="v0",
            policy_text=base_policy.text,
            seed=args.seed,
            run_id=base_run_id,
        )

    # 3. Load or run candidate evaluation
    cand_result = load_eval_result_for_run(base_run_id, cand_policy.version)
    if not cand_result:
        # Check runs for candidate evaluation in descending order (latest first)
        for r_dir in sorted(Path("runs").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            cand_result = load_eval_result_for_run(r_dir.name, cand_policy.version)
            if cand_result:
                break

    if not cand_result:
        evaluator = Evaluator(db_path=args.db_path)
        cand_result = evaluator.evaluate(
            policy_version=cand_policy.version,
            policy_text=cand_policy.text,
            seed=args.seed,
        )

    # Populate addressed families on candidate if known
    conn = get_connection(args.db_path)
    try:
        row = conn.execute("SELECT diff_text FROM policies WHERE version = ?;", (cand_policy.version,)).fetchone()
        if row and row["diff_text"]:
            diff_lines = [l[1:].strip() for l in row["diff_text"].splitlines() if l.startswith("+") and not l.startswith("+++")]
            cand_result.addressed_families = diff_lines
    finally:
        conn.close()

    if args.synthetic_asr_increase:
        # Construct synthetic metrics with ASR = baseline + 1/25
        base_result.asr_all = Metric(numerator=2, denominator=25, value=2 / 25)
        base_result.asr_patch = Metric(numerator=2, denominator=21, value=2 / 21)
        base_result.asr_heldout = Metric(numerator=0, denominator=4, value=0.0)
        base_result.bpr = Metric(numerator=10, denominator=10, value=1.0)

        cand_result.asr_all = Metric(numerator=3, denominator=25, value=3 / 25)
        cand_result.asr_patch = Metric(numerator=3, denominator=21, value=3 / 21)
        cand_result.asr_heldout = Metric(numerator=0, denominator=4, value=0.0)
        cand_result.bpr = Metric(numerator=10, denominator=10, value=1.0)
        cand_result.regressions = []
        cand_result.noise_floor_k = 3

    # 4. Evaluate Gate
    gate = Gate(strict_noise_floor=strict_noise_floor)
    gate_res = gate.evaluate(candidate=cand_result, baseline=base_result)

    # Record decision in DB
    gate.record_decision(
        candidate_version=cand_policy.version,
        result=gate_res,
        db_path=args.db_path,
        candidate_file=args.candidate,
    )

    # Print condition table
    print(gate_res.format_table())

    if gate_res.decision == GateDecision.REJECT:
        run_dir = Path("runs") / base_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        rej_path = run_dir / f"gate_{cand_policy.version}_REJECTED.md"
        rej_path.write_text(gate_res.format_table(), encoding="utf-8")
        print(f"\n[GATE REJECTED] Rejection report written to: {rej_path}")
        sys.exit(1)
    elif gate_res.decision == GateDecision.NO_ACTION_NEEDED:
        print("\n[GATE NO_ACTION_NEEDED] Baseline has 0 attack successes; preserving baseline.")
        sys.exit(0)
    else:
        print("\n[GATE ACCEPTED] Candidate policy promoted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
