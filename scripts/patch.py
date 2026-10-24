"""CLI to generate, validate, and escalate guardrail policy patches against baseline weaknesses."""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from strikeshield.db import get_connection
from strikeshield.errors import HeldoutLeakError, PatcherTooWeakError
from strikeshield.patcher import Patcher, PolicyVersion
from strikeshield.taxonomy import weakest_techniques


def get_latest_run_id(db_path: Path) -> str:
    """Finds the most recent evaluation run ID."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT run_id FROM metrics ORDER BY rowid DESC LIMIT 1;").fetchone()
        if row:
            return row["run_id"]
        row_run = conn.execute("SELECT id FROM runs ORDER BY started_at DESC LIMIT 1;").fetchone()
        if row_run:
            return row_run["id"]
        raise ValueError("No baseline evaluation runs found in database. Run evaluate_version.py first.")
    finally:
        conn.close()


def load_baseline_data(
    run_id: str,
    db_path: Path,
) -> tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Loads successes grouped by technique family and computes the top-3 weakest techniques."""
    conn = get_connection(db_path)
    try:
        # Total attacks per technique family in run
        cursor = conn.execute(
            """
            SELECT a.technique_family, COUNT(*) as total_attacks
            FROM responses r
            JOIN attacks a ON r.attack_id = a.id
            WHERE r.run_id = ?
            GROUP BY a.technique_family;
            """,
            (run_id,),
        )
        total_per_family = {row["technique_family"]: row["total_attacks"] for row in cursor.fetchall()}

        # Successful attacks in run
        cursor = conn.execute(
            """
            SELECT 
                a.id as attack_id,
                a.prompt,
                a.risk_category,
                a.technique_family,
                a.target_behaviour,
                a.split,
                r.response_text,
                v.evidence_quote,
                v.verdict_class
            FROM verdicts v
            JOIN responses r ON v.response_id = r.id
            JOIN attacks a ON r.attack_id = a.id
            WHERE r.run_id = ?
              AND v.verdict_class IN ('full_compliance', 'partial_leak', 'refused_but_leaked')
              AND a.technique_family != 'standard_inquiry';
            """,
            (run_id,),
        )

        successes_by_family: Dict[str, List[Dict[str, Any]]] = {}
        for row in cursor.fetchall():
            fam = row["technique_family"]
            successes_by_family.setdefault(fam, []).append({
                "attack_id": row["attack_id"],
                "prompt": row["prompt"],
                "risk_category": row["risk_category"],
                "technique_family": fam,
                "target_behaviour": row["target_behaviour"],
                "response": row["response_text"],
                "judge_evidence_quote": row["evidence_quote"],
                "verdict_class": row["verdict_class"],
                "split": row["split"],
            })

        asr_by_tech: Dict[str, float] = {}
        for fam, total in total_per_family.items():
            if fam == "standard_inquiry":
                continue
            succ = len(successes_by_family.get(fam, []))
            asr_by_tech[fam] = succ / total if total > 0 else 0.0

        weakest = weakest_techniques(top_k=3, asr_by_technique=asr_by_tech)
        if not weakest:
            # Fallback to distributed techniques from config if all ASRs were 0
            weakest = ["role_play", "language_switch", "instruction_override"]

        return successes_by_family, weakest
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeShield Patcher CLI")
    parser.add_argument("--policy", type=str, default="policies/v0.md", help="Current policy file")
    parser.add_argument("--run-id", type=str, default=None, help="Baseline evaluation run ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--db-path", type=Path, default=Path("strikeshield.db"), help="Database path")
    args = parser.parse_args()

    policy_path = Path(args.policy)
    policy = PolicyVersion.from_file(policy_path)
    run_id = args.run_id or get_latest_run_id(args.db_path)

    print(f"=== StrikeShield Patcher ===")
    print(f"Target Policy:    {policy.version} ({policy.path})")
    print(f"Baseline Run ID:  {run_id}")
    print(f"Seed:             {args.seed}")

    successes_by_family, weakest_fams = load_baseline_data(run_id, args.db_path)
    print(f"Weakest Techniques: {', '.join(weakest_fams)}")
    print(f"Techniques with Successes: {list(successes_by_family.keys())}")

    patcher = Patcher()
    try:
        candidate = patcher.generate_patch(
            policy=policy,
            successes_by_family=successes_by_family,
            weakest_families_list=weakest_fams,
            db_path=args.db_path,
        )

        print("\n=== Candidate Patch Generated Successfully ===")
        print(f"Candidate Version:  {candidate.version}")
        print(f"Parent Version:     {candidate.parent_version}")
        print(f"Model Used:         {candidate.model_used}")
        print(f"Fallback Used:      {candidate.fallback_used}")
        print(f"Candidate Status:   {candidate.status}")
        print(f"Git Commit:         {candidate.git_commit}")
        print(f"Validation Result:  ok=True")
        print("\n--- Diff ---")
        print(candidate.diff_text)
        print("------------")
        sys.exit(0)

    except PatcherTooWeakError as e:
        print(f"\n[FATAL] Patcher failed validation and escalation: {e}", file=sys.stderr)
        print(f"Reason: {e.context.get('reason') if e.context else 'Unknown'}", file=sys.stderr)
        sys.exit(1)
    except HeldoutLeakError as e:
        print(f"\n[CRITICAL] Held-out leak invariant violation: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
