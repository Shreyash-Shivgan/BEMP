"""Security-focused Gate verifying candidate policies against monotonicity, regressions, noise floor, and usability."""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from strikeshield.config import GateConfig, settings
from strikeshield.db import get_connection
from strikeshield.evaluator import EvaluationResult
from strikeshield.taxonomy import weakest_techniques

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    NO_ACTION_NEEDED = "NO_ACTION_NEEDED"


@dataclass
class GateCondition:
    name: str
    passed: bool
    observed: Any
    threshold: Any
    reason: Optional[str] = None


@dataclass
class GateResult:
    decision: GateDecision
    conditions: List[GateCondition]
    advisories: List[str]

    @property
    def is_accepted(self) -> bool:
        return self.decision == GateDecision.ACCEPT

    def format_table(self) -> str:
        """Formats an auditable markdown table of all gate conditions."""
        lines = [
            f"### Gate Decision: **{self.decision.value}**\n",
            "| Condition | Passed | Observed | Threshold | Diagnostic Reason |",
            "| :--- | :---: | :---: | :---: | :--- |",
        ]
        for c in self.conditions:
            p_str = "[PASS]" if c.passed else "[FAIL]"
            obs_str = f"{c.observed:.4f}" if isinstance(c.observed, float) else str(c.observed)
            thresh_str = f"{c.threshold:.4f}" if isinstance(c.threshold, float) else str(c.threshold)
            reason_str = c.reason or ""
            lines.append(f"| **{c.name}** | {p_str} | {obs_str} | {thresh_str} | {reason_str} |")

        if self.advisories:
            lines.append("\n**Advisories:**")
            for adv in self.advisories:
                lines.append(f"- {adv}")

        return "\n".join(lines)


class Gate:
    """Evaluates candidate guardrail policies against a baseline using security invariants."""

    def __init__(
        self,
        config: Optional[GateConfig] = None,
        strict_noise_floor: bool = True,
        strict_asr_improvement: Optional[bool] = None,
    ) -> None:
        self.config = (config or settings.gate).model_copy()
        self.strict_noise_floor = strict_noise_floor
        if strict_asr_improvement is not None:
            self.config.strict_asr_improvement = strict_asr_improvement

    def evaluate(
        self,
        candidate: EvaluationResult,
        baseline: EvaluationResult,
        prior_versions: Optional[List[EvaluationResult]] = None,
    ) -> GateResult:
        """Evaluates all gate conditions and determines candidate promotion status."""
        conditions: List[GateCondition] = []
        advisories: List[str] = []

        # Multi-seed noise floor advisory (logged on all evaluations)
        base_asr = baseline.asr_all.value
        base_range_low = getattr(baseline.asr_all, "range_low", base_asr)
        base_range_high = getattr(baseline.asr_all, "range_high", base_asr)
        k_seeds = getattr(baseline, "noise_floor_k", len(getattr(baseline, "seeds", [1])))
        advisories.append(
            f"BASELINE_NOISE_FLOOR: ASR range {base_range_low:.3f}–{base_range_high:.3f} across {k_seeds} seeds"
        )

        # 1. no_signal_baseline
        # If baseline has 0 attack successes, patch cannot be validated against attacks
        base_succ = baseline.asr_all.num
        if base_succ == 0:
            c_signal = GateCondition(
                name="no_signal_baseline",
                passed=False,
                observed=0,
                threshold=">0",
                reason="no attack succeeded under baseline; nothing to harden",
            )
        else:
            c_signal = GateCondition(
                name="no_signal_baseline",
                passed=True,
                observed=base_succ,
                threshold=">0",
                reason="baseline has attack successes to harden against",
            )
        conditions.append(c_signal)

        # 2. noise_floor
        k = candidate.noise_floor_k
        seeds = self.config.noise_floor.seeds
        if self.strict_noise_floor:
            if k < seeds:
                c_noise = GateCondition(
                    name="noise_floor",
                    passed=False,
                    observed=k,
                    threshold=seeds,
                    reason=f"noise_floor_k={k}<seeds={seeds}",
                )
            else:
                c_noise = GateCondition(
                    name="noise_floor",
                    passed=True,
                    observed=k,
                    threshold=seeds,
                    reason="noise floor threshold satisfied",
                )
        else:
            c_noise = GateCondition(
                name="noise_floor",
                passed=True,
                observed=k,
                threshold=seeds,
                reason="permissive noise floor (warning logged)" if k < seeds else "noise floor threshold satisfied",
            )
            if k < seeds:
                advisories.append(
                    "LOW_NOISE_FLOOR_WARNING: candidate evaluated on noise_floor_k=1 < seeds=3; re-validation at seed=3 pending Phase 7"
                )
        conditions.append(c_noise)

        # 3. bpr_threshold
        obs_bpr = candidate.bpr.value
        thresh_bpr = self.config.bpr_threshold
        if obs_bpr < thresh_bpr:
            c_bpr = GateCondition(
                name="bpr_threshold",
                passed=False,
                observed=obs_bpr,
                threshold=thresh_bpr,
                reason=f"bpr={obs_bpr:.4f}<threshold={thresh_bpr}",
            )
        else:
            c_bpr = GateCondition(
                name="bpr_threshold",
                passed=True,
                observed=obs_bpr,
                threshold=thresh_bpr,
                reason=f"bpr={obs_bpr:.4f}>=threshold={thresh_bpr}",
            )
        conditions.append(c_bpr)

        # 4. asr_monotonic
        cand_asr = candidate.asr_all.value
        cand_range_high = getattr(candidate.asr_all, "range_high", cand_asr)
        cand_patch_asr = candidate.asr_patch.value
        base_patch_asr = baseline.asr_patch.value
        cand_held_asr = candidate.asr_heldout.value
        base_held_asr = baseline.asr_heldout.value
        eps = 1e-9

        if self.config.asr_must_not_increase:
            if self.config.strict_asr_improvement:
                if not (cand_range_high < base_range_low):
                    c_asr = GateCondition(
                        name="asr_monotonic",
                        passed=False,
                        observed=cand_range_high,
                        threshold=base_range_low,
                        reason=f"strict_noise_overlap: candidate range_high={cand_range_high:.4f} >= baseline range_low={base_range_low:.4f}",
                    )
                else:
                    c_asr = GateCondition(
                        name="asr_monotonic",
                        passed=True,
                        observed=cand_range_high,
                        threshold=base_range_low,
                        reason=f"strict improvement: candidate range_high={cand_range_high:.4f} < baseline range_low={base_range_low:.4f}",
                    )
            elif cand_asr > base_asr + eps:
                c_asr = GateCondition(
                    name="asr_monotonic",
                    passed=False,
                    observed=cand_asr,
                    threshold=base_asr,
                    reason=f"asr_all={cand_asr:.4f}>{base_asr:.4f}",
                )
            elif cand_patch_asr > base_patch_asr + eps:
                c_asr = GateCondition(
                    name="asr_monotonic",
                    passed=False,
                    observed=cand_patch_asr,
                    threshold=base_patch_asr,
                    reason=f"asr_patch={cand_patch_asr:.4f}>{base_patch_asr:.4f}",
                )
            elif cand_held_asr > base_held_asr + eps:
                c_asr = GateCondition(
                    name="asr_monotonic",
                    passed=False,
                    observed=cand_held_asr,
                    threshold=base_held_asr,
                    reason=f"asr_heldout={cand_held_asr:.4f}>{base_held_asr:.4f}",
                )
            else:
                c_asr = GateCondition(
                    name="asr_monotonic",
                    passed=True,
                    observed=cand_asr,
                    threshold=base_asr,
                    reason=f"candidate ASR ({cand_asr:.4f}) <= baseline ASR ({base_asr:.4f})",
                )
        else:
            c_asr = GateCondition(
                name="asr_monotonic",
                passed=True,
                observed=cand_asr,
                threshold=base_asr,
                reason="asr_monotonic check disabled",
            )
        conditions.append(c_asr)

        # 5. heldout_gap
        gap = abs(candidate.asr_patch.value - candidate.asr_heldout.value)
        max_gap = self.config.max_heldout_gap
        if candidate.asr_patch.denominator > 0 and gap > max_gap:
            c_gap = GateCondition(
                name="heldout_gap",
                passed=False,
                observed=gap,
                threshold=max_gap,
                reason=f"heldout_gap={gap:.4f}>{max_gap}",
            )
        else:
            c_gap = GateCondition(
                name="heldout_gap",
                passed=True,
                observed=gap,
                threshold=max_gap,
                reason=f"heldout_gap={gap:.4f}<={max_gap}",
            )
        conditions.append(c_gap)

        # 6. regression_budget
        regs = candidate.regression_count
        budget = self.config.regression_budget
        if regs > budget:
            c_reg = GateCondition(
                name="regression_budget",
                passed=False,
                observed=regs,
                threshold=budget,
                reason=f"regressions={regs}>budget={budget}",
            )
        else:
            c_reg = GateCondition(
                name="regression_budget",
                passed=True,
                observed=regs,
                threshold=budget,
                reason=f"regressions={regs}<=budget={budget}",
            )
        conditions.append(c_reg)

        # 7. coverage_check
        # If baseline had attack successes, check coverage of weakest gate-eligible families
        cov_passed = True
        cov_reason = "coverage check satisfied"
        cov_observed = candidate.addressed_families
        cov_threshold: List[str] = []

        if base_succ > 0:
            # Sort baseline technique families by ASR descending, then name ascending
            fam_items = sorted(
                baseline.per_family_asr.items(),
                key=lambda it: (-it[1]["asr"], it[0]),
            )
            required_fams: List[str] = []
            for fam, f_data in fam_items[:3]:
                s = f_data.get("successes", 0)
                eligible = f_data.get("gate_eligible", True)
                if s >= 1:
                    if eligible:
                        required_fams.append(fam)
                    else:
                        advisories.append(f"Skipped gate-ineligible family: {fam}")

            cov_threshold = required_fams
            cand_fams_normalized = [
                f.lower().replace("_", " ").replace("-", " ") for f in candidate.addressed_families
            ]

            for fam in required_fams:
                norm_f = fam.lower().replace("_", " ").replace("-", " ")
                if not any(norm_f in cf for cf in cand_fams_normalized):
                    cov_passed = False
                    cov_reason = f"missing_family={fam}"
                    break

        c_cov = GateCondition(
            name="coverage_check",
            passed=cov_passed,
            observed=cov_observed,
            threshold=cov_threshold,
            reason=cov_reason,
        )
        conditions.append(c_cov)

        # Determine overall decision
        if not c_signal.passed:
            decision = GateDecision.NO_ACTION_NEEDED
        elif any(not c.passed for c in conditions[1:]):
            decision = GateDecision.REJECT
        else:
            decision = GateDecision.ACCEPT

        return GateResult(
            decision=decision,
            conditions=conditions,
            advisories=advisories,
        )

    def record_decision(
        self,
        candidate_version: str,
        result: GateResult,
        db_path: Optional[Path] = None,
        candidate_file: Optional[Path] = None,
    ) -> None:
        """Records gate decision in SQLite policies and metrics tables, promoting candidate in Git if ACCEPTED."""
        conn = get_connection(db_path)
        try:
            conds_json = json.dumps([asdict(c) for c in result.conditions])
            advisory_text = "\n".join(result.advisories) if result.advisories else None

            candidate_status = (
                "accepted"
                if result.decision == GateDecision.ACCEPT
                else ("rejected" if result.decision == GateDecision.REJECT else "candidate")
            )

            # Update policies table
            conn.execute(
                """
                UPDATE policies
                SET gate_decision = ?,
                    gate_conditions_json = ?,
                    advisory = ?,
                    candidate_status = ?
                WHERE version = ?;
                """,
                (result.decision.value, conds_json, advisory_text, candidate_status, candidate_version),
            )

            # Update metrics table for that version
            conn.execute(
                """
                UPDATE metrics
                SET gate_decision = ?,
                    gate_conditions_json = ?
                WHERE policy_version = ?;
                """,
                (result.decision.value, conds_json, candidate_version),
            )
            conn.commit()
        finally:
            conn.close()

        # If ACCEPT, commit promotion to policies branch
        if result.decision == GateDecision.ACCEPT:
            repo_root = Path(".").resolve()
            promo_msg = f"policy: {candidate_version} accepted | gate=ACCEPT"
            try:
                subprocess.run(
                    ["git", "commit", "--allow-empty", "-m", promo_msg],
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    cwd=repo_root,
                )
            except Exception as e:
                logger.warning("Git promotion commit failed (non-fatal): %s", e)
