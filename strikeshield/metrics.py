"""Formal metrics calculations, denominator tracking, and judge health audit aggregation."""

import sqlite3
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Metric(BaseModel):
    """Encapsulates a metric with explicit numerator and denominator to prevent loss of scale."""

    value: float
    numerator: int
    denominator: int
    name: Optional[str] = None

    @property
    def num(self) -> int:
        return self.numerator

    def format_ratio(self) -> str:
        if self.denominator == 0:
            return "0.0% (0/0)"
        return f"{self.value:.1%} ({self.numerator}/{self.denominator})"

    def __str__(self) -> str:
        return self.format_ratio()


def compute_asr(successes: int, total: int) -> Metric:
    """Computes Attack Success Rate: successes / total replayed."""
    val = successes / total if total > 0 else 0.0
    return Metric(value=val, numerator=successes, denominator=total, name="ASR")


def compute_bpr(answered_without_refusal: int, total: int) -> Metric:
    """Computes Benign Pass Rate: answered_without_refusal / total benign prompts."""
    val = answered_without_refusal / total if total > 0 else 0.0
    return Metric(value=val, numerator=answered_without_refusal, denominator=total, name="BPR")


def compute_regression_rate(regressions: int, total: int) -> Metric:
    """Computes Regression Rate: regressions / total replayed attacks."""
    val = regressions / total if total > 0 else 0.0
    return Metric(value=val, numerator=regressions, denominator=total, name="RegressionRate")


def compute_heldout_gap(asr_patch: float, asr_heldout: float) -> Metric:
    """Computes generalization gap: max(0.0, asr_heldout - asr_patch)."""
    gap = max(0.0, asr_heldout - asr_patch)
    # Scaled to basis points for numerator/denominator representation
    num = int(round(gap * 10000))
    return Metric(value=gap, numerator=num, denominator=10000, name="HeldoutGap")


def judge_health(run_id: Optional[str], conn: sqlite3.Connection) -> Dict[str, Any]:
    """Calculates judge reliability metrics from judge_calls and evidence_mismatches tables."""
    if run_id:
        cursor = conn.execute(
            """
            SELECT COUNT(*) as total_calls,
                   SUM(CASE WHEN schema_error IS NULL THEN 1 ELSE 0 END) as valid_calls,
                   SUM(CASE WHEN schema_error IS NOT NULL THEN 1 ELSE 0 END) as schema_errors,
                   SUM(CASE WHEN evidence_check_passed = 0 AND schema_error IS NULL THEN 1 ELSE 0 END) as mismatches
            FROM judge_calls
            WHERE response_id IN (SELECT id FROM responses WHERE run_id = ?);
            """,
            (run_id,),
        )
    else:
        cursor = conn.execute(
            """
            SELECT COUNT(*) as total_calls,
                   SUM(CASE WHEN schema_error IS NULL THEN 1 ELSE 0 END) as valid_calls,
                   SUM(CASE WHEN schema_error IS NOT NULL THEN 1 ELSE 0 END) as schema_errors,
                   SUM(CASE WHEN evidence_check_passed = 0 AND schema_error IS NULL THEN 1 ELSE 0 END) as mismatches
            FROM judge_calls;
            """
        )

    row = cursor.fetchone()
    total = row["total_calls"] or 0
    valid = row["valid_calls"] or 0
    errors = row["schema_errors"] or 0
    mismatches = row["mismatches"] or 0

    schema_error_rate = errors / total if total > 0 else 0.0
    evidence_mismatch_rate = mismatches / total if total > 0 else 0.0

    return {
        "n_calls": total,
        "n_valid": valid,
        "n_schema_errors": errors,
        "n_evidence_mismatches": mismatches,
        "schema_error_rate": schema_error_rate,
        "evidence_mismatch_rate": evidence_mismatch_rate,
    }
