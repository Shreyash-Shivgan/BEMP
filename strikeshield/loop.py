"""Closed-loop hardening engine with multi-seed evaluation, noise floor, and gate verification."""

import datetime
import json
import logging
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from strikeshield.attack_source import AttackRecord
from strikeshield.config import settings
from strikeshield.db import get_connection
from strikeshield.errors import PatcherTooWeakError, StrikeShieldError
from strikeshield.evaluator import EvaluationResult, Evaluator, RegressionRecord
from strikeshield.gate import Gate, GateDecision, GateResult
from strikeshield.metrics import Metric
from strikeshield.patcher import Patcher
from strikeshield.policies import PolicyVersion

logger = logging.getLogger(__name__)


class MetricWithRange(BaseModel):
    mean: float
    range_low: float
    range_high: float
    numerator: Optional[int] = None
    denominator: Optional[int] = None
    per_seed_values: List[float] = Field(default_factory=list)
    per_seed_metrics: List[Metric] = Field(default_factory=list)

    @property
    def value(self) -> float:
        return self.mean

    @property
    def num(self) -> Optional[int]:
        return self.numerator

    @property
    def den(self) -> Optional[int]:
        return self.denominator

    def format_range(self) -> str:
        k = len(self.per_seed_values) if self.per_seed_values else 1
        num_str = ""
        if self.numerator is not None and self.denominator is not None:
            if self.per_seed_metrics:
                per_seed_str = ", ".join(f"{m.num}/{m.denominator}" for m in self.per_seed_metrics)
                num_str = f" ({self.numerator}/{self.denominator}, mean of [{per_seed_str}])"
            else:
                num_str = f" ({self.numerator}/{self.denominator})"
        if k == 1:
            return f"{self.mean:.3f}{num_str} [SINGLE-SEED, noise_floor_k=1]"
        return f"{self.mean:.3f}{num_str} (range {self.range_low:.3f}–{self.range_high:.3f}, n_seeds={k})"


def non_overlapping(baseline: MetricWithRange, candidate: MetricWithRange) -> bool:
    """Returns True if candidate range is strictly outside baseline range.

    Strictly better: candidate.range_high < baseline.range_low
    Strictly worse: candidate.range_low > baseline.range_high
    """
    return candidate.range_high < baseline.range_low or candidate.range_low > baseline.range_high


class MultiSeedEvaluationResult(BaseModel):
    policy_version: str
    seeds: List[int]
    asr_all: MetricWithRange
    asr_patch: MetricWithRange
    asr_heldout: MetricWithRange
    bpr: MetricWithRange
    regressions: List[RegressionRecord] = Field(default_factory=list)
    per_family_asr: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    addressed_families: List[str] = Field(default_factory=list)
    noise_floor_k: int = 1
    judge_health_data: Dict[str, Any] = Field(default_factory=dict)
    seed_results: List[EvaluationResult] = Field(default_factory=list)

    @property
    def regression_count(self) -> int:
        return len(self.regressions)


def aggregate_metrics_across_seeds(results: List[EvaluationResult], seeds: List[int]) -> MultiSeedEvaluationResult:
    """Aggregates a list of single-seed evaluation results into a MultiSeedEvaluationResult."""
    if not results:
        raise ValueError("Cannot aggregate empty evaluation results.")

    k = len(results)
    # ASR All
    asr_all_vals = [r.asr_all.value for r in results]
    mean_asr_all = sum(asr_all_vals) / k
    asr_all_num = sum(r.asr_all.num or 0 for r in results)
    asr_all_den = sum(r.asr_all.denominator for r in results)
    m_asr_all = MetricWithRange(
        mean=mean_asr_all,
        range_low=min(asr_all_vals),
        range_high=max(asr_all_vals),
        numerator=asr_all_num,
        denominator=asr_all_den,
        per_seed_values=asr_all_vals,
        per_seed_metrics=[r.asr_all for r in results],
    )

    # ASR Patch
    asr_patch_vals = [r.asr_patch.value for r in results]
    mean_asr_patch = sum(asr_patch_vals) / k
    m_asr_patch = MetricWithRange(
        mean=mean_asr_patch,
        range_low=min(asr_patch_vals),
        range_high=max(asr_patch_vals),
        numerator=sum(r.asr_patch.num or 0 for r in results),
        denominator=sum(r.asr_patch.denominator for r in results),
        per_seed_values=asr_patch_vals,
        per_seed_metrics=[r.asr_patch for r in results],
    )

    # ASR Heldout
    asr_held_vals = [r.asr_heldout.value for r in results]
    mean_asr_held = sum(asr_held_vals) / k
    m_asr_held = MetricWithRange(
        mean=mean_asr_held,
        range_low=min(asr_held_vals),
        range_high=max(asr_held_vals),
        numerator=sum(r.asr_heldout.num or 0 for r in results),
        denominator=sum(r.asr_heldout.denominator for r in results),
        per_seed_values=asr_held_vals,
        per_seed_metrics=[r.asr_heldout for r in results],
    )

    # BPR
    bpr_vals = [r.bpr.value for r in results]
    mean_bpr = sum(bpr_vals) / k
    m_bpr = MetricWithRange(
        mean=mean_bpr,
        range_low=min(bpr_vals),
        range_high=max(bpr_vals),
        numerator=sum(r.bpr.num or 0 for r in results),
        denominator=sum(r.bpr.denominator for r in results),
        per_seed_values=bpr_vals,
        per_seed_metrics=[r.bpr for r in results],
    )

    # Union of regressions across seeds
    all_regressions: List[RegressionRecord] = []
    seen_reg_attacks = set()
    for r in results:
        for reg in r.regressions:
            if reg.attack_id not in seen_reg_attacks:
                all_regressions.append(reg)
                seen_reg_attacks.add(reg.attack_id)

    # Aggregated per-family ASR: union of successes across seeds
    family_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for fam, data in r.per_family_asr.items():
            if fam not in family_stats:
                family_stats[fam] = {
                    "total": data.get("total", 0),
                    "successes": 0,
                    "gate_eligible": data.get("gate_eligible", True),
                    "reason": data.get("reason", ""),
                }
            family_stats[fam]["successes"] += data.get("successes", 0)

    for fam, stats in family_stats.items():
        total_denom = stats["total"] * k
        stats["asr"] = stats["successes"] / total_denom if total_denom > 0 else 0.0

    return MultiSeedEvaluationResult(
        policy_version=results[0].policy_version,
        seeds=seeds,
        asr_all=m_asr_all,
        asr_patch=m_asr_patch,
        asr_heldout=m_asr_held,
        bpr=m_bpr,
        regressions=all_regressions,
        per_family_asr=family_stats,
        addressed_families=results[0].addressed_families,
        noise_floor_k=k,
        judge_health_data=results[0].judge_health_data,
        seed_results=results,
    )


def commit_promotion_to_git(
    version: str,
    policy_file: Path,
    baseline_asr: float,
    candidate_asr: float,
    candidate_bpr: float,
    git_ref: str = "refs/heads/policies",
) -> str:
    """Creates promotion commit on policies branch using git plumbing."""
    commit_msg = (
        f"policy: {version} accepted | gate=ACCEPT | "
        f"asr {baseline_asr:.3f}→{candidate_asr:.3f} | bpr {candidate_bpr:.3f}"
    )
    repo_root = Path(".").resolve()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_index = f.name

    env = os.environ.copy()
    env["GIT_INDEX_FILE"] = tmp_index

    try:
        subprocess.run(
            ["git", "read-tree", git_ref],
            env=env,
            check=True,
            capture_output=True,
            cwd=repo_root,
        )
        blob_hash = subprocess.run(
            ["git", "hash-object", "-w", str(policy_file)],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout.strip()

        target_git_path = f"policies/{policy_file.name}"
        subprocess.run(
            ["git", "update-index", "--add", "--cacheinfo", "100644", blob_hash, target_git_path],
            env=env,
            check=True,
            capture_output=True,
            cwd=repo_root,
        )

        tree = subprocess.run(
            ["git", "write-tree"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=repo_root,
        ).stdout.strip()

        parent = subprocess.run(
            ["git", "rev-parse", git_ref],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        ).stdout.strip()

        commit_hash = subprocess.run(
            ["git", "commit-tree", tree, "-p", parent, "-m", commit_msg],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=repo_root,
        ).stdout.strip()

        subprocess.run(
            ["git", "update-ref", git_ref, commit_hash],
            check=True,
            capture_output=True,
            cwd=repo_root,
        )
        return commit_hash
    finally:
        if os.path.exists(tmp_index):
            os.remove(tmp_index)


class LoopConfig(BaseModel):
    seeds: List[int] = Field(default_factory=lambda: [0, 1, 2])
    max_iterations: int = 3
    stall_rounds: int = 2
    strict_asr_improvement: bool = False


class IterationRecord(BaseModel):
    iteration: int
    baseline_version: str
    candidate_version: Optional[str] = None
    baseline_metrics: MultiSeedEvaluationResult
    candidate_metrics: Optional[MultiSeedEvaluationResult] = None
    gate_result_dict: Optional[Dict[str, Any]] = None
    promotion_commit: Optional[str] = None
    status: str


class LoopResult(BaseModel):
    iterations: List[IterationRecord] = Field(default_factory=list)
    final_policy_version: str
    final_decision: str
    stop_reason: str
    summary_markdown: str = ""


class HardeningLoop:
    """Orchestrates closed-loop adversarial testing and self-hardening iterations."""

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        evaluator: Optional[Evaluator] = None,
        patcher: Optional[Patcher] = None,
        gate: Optional[Gate] = None,
        db_path: Optional[Path] = None,
        initial_policy_path: Path = Path("policies/v0.md"),
        promote_to_git: bool = True,
    ) -> None:
        self.config = config or LoopConfig()
        self.db_path = db_path or settings.database_path
        self.evaluator = evaluator or Evaluator(db_path=self.db_path)
        self.patcher = patcher or Patcher()
        self.gate = gate or Gate(
            strict_noise_floor=True,
            strict_asr_improvement=self.config.strict_asr_improvement,
        )
        self.initial_policy_path = initial_policy_path
        self.promote_to_git = promote_to_git

        # Ensure v0 candidate_status is baseline in SQLite
        conn = get_connection(self.db_path)
        try:
            conn.execute("UPDATE policies SET candidate_status = 'baseline' WHERE version = 'v0';")
            conn.commit()
        finally:
            conn.close()

    def evaluate_multi_seed(
        self,
        policy_version: str,
        policy_text: str,
        seeds: List[int],
        addressed_families: Optional[List[str]] = None,
    ) -> MultiSeedEvaluationResult:
        """Evaluates a policy version across all configured seeds and records multi-seed ranges."""
        results: List[EvaluationResult] = []
        for s in seeds:
            res = self.evaluator.evaluate(
                policy_version=policy_version,
                policy_text=policy_text,
                seed=s,
            )
            if addressed_families:
                res.addressed_families = addressed_families
            results.append(res)

        aggregated = aggregate_metrics_across_seeds(results, seeds)
        if addressed_families:
            aggregated.addressed_families = addressed_families

        # Persist multi-seed aggregate metrics row in SQLite
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """
                UPDATE metrics SET
                    seeds_run_json = ?,
                    asr_all_range_low = ?,
                    asr_all_range_high = ?,
                    asr_patch_range_low = ?,
                    asr_patch_range_high = ?,
                    asr_heldout_range_low = ?,
                    asr_heldout_range_high = ?,
                    bpr_range_low = ?,
                    bpr_range_high = ?,
                    noise_overlap_asr = ?
                WHERE policy_version = ?;
                """,
                (
                    json.dumps(seeds),
                    aggregated.asr_all.range_low,
                    aggregated.asr_all.range_high,
                    aggregated.asr_patch.range_low,
                    aggregated.asr_patch.range_high,
                    aggregated.asr_heldout.range_low,
                    aggregated.asr_heldout.range_high,
                    aggregated.bpr.range_low,
                    aggregated.bpr.range_high,
                    0,
                    policy_version,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        return aggregated

    def run(self) -> LoopResult:
        """Executes closed loop iterations until stop condition or max iterations."""
        current_policy = PolicyVersion.load(self.initial_policy_path)
        iteration_records: List[IterationRecord] = []
        final_decision = "PENDING"
        stop_reason = "max_iterations_reached"

        asr_history: List[float] = []

        for it in range(1, self.config.max_iterations + 1):
            logger.info("=== Starting Hardening Iteration %d (Policy: %s) ===", it, current_policy.version)

            # 1. Multi-seed baseline evaluation
            base_multi = self.evaluate_multi_seed(
                policy_version=current_policy.version,
                policy_text=current_policy.text,
                seeds=self.config.seeds,
            )
            asr_history.append(base_multi.asr_all.mean)

            # Check stall condition
            if len(asr_history) >= self.config.stall_rounds:
                recent_asrs = asr_history[-self.config.stall_rounds:]
                if max(recent_asrs) - min(recent_asrs) < 1e-6 and all(r.status != "ACCEPT" for r in iteration_records[-self.config.stall_rounds + 1:]):
                    stop_reason = "stall_reached"
                    break

            # 2. Check no_signal_baseline condition early
            if base_multi.asr_all.num == 0 or base_multi.asr_all.mean == 0.0:
                final_decision = "NO_ACTION_NEEDED"
                stop_reason = f"Corpus exhausted; {base_multi.asr_all.denominator} attacks, 0 baseline successes. Grow corpus before further hardening."
                record = IterationRecord(
                    iteration=it,
                    baseline_version=current_policy.version,
                    baseline_metrics=base_multi,
                    status="NO_ACTION_NEEDED",
                )
                iteration_records.append(record)
                break

            # 3. Collect union of successes across seeds for Patcher input
            # Query successful attack IDs
            conn = get_connection(self.db_path)
            try:
                cur = conn.execute(
                    """
                    SELECT DISTINCT a.id, a.prompt, a.risk_category, a.technique_family, a.target_behaviour
                    FROM attacks a
                    JOIN responses r ON r.attack_id = a.id
                    JOIN verdicts v ON v.response_id = r.id
                    WHERE r.policy_version = ?
                      AND v.verdict_class IN ('full_compliance', 'partial_leak', 'refused_but_leaked')
                      AND a.source != 'benign_set';
                    """,
                    (current_policy.version,),
                )
                succ_attacks = [
                    AttackRecord(
                        id=row["id"],
                        prompt=row["prompt"],
                        risk_category=row["risk_category"],
                        technique_family=row["technique_family"],
                        target_behaviour=row["target_behaviour"],
                    )
                    for row in cur.fetchall()
                ]
            finally:
                conn.close()

            # Group successes by technique_family
            successes_by_family: Dict[str, List[Dict[str, Any]]] = {}
            for atk in succ_attacks:
                fam = atk.technique_family
                if fam not in successes_by_family:
                    successes_by_family[fam] = []
                successes_by_family[fam].append(atk.model_dump())

            weakest_fams = [
                fam for fam, _ in sorted(
                    base_multi.per_family_asr.items(),
                    key=lambda it: (-it[1].get("asr", 0), it[0])
                )
            ]

            # 4. Generate Candidate Patch
            try:
                candidate_patch = self.patcher.generate_patch(
                    policy=current_policy,
                    successes_by_family=successes_by_family,
                    weakest_families_list=weakest_fams,
                    db_path=self.db_path,
                )
            except PatcherTooWeakError:
                record = IterationRecord(
                    iteration=it,
                    baseline_version=current_policy.version,
                    baseline_metrics=base_multi,
                    status="patcher_too_weak",
                )
                iteration_records.append(record)
                stop_reason = "patcher_too_weak"
                break

            cand_policy = PolicyVersion.load(Path("policies") / f"{candidate_patch.version}.md")
            diff_lines = [l[1:].strip() for l in candidate_patch.diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")]
            cand_rules = diff_lines if diff_lines else candidate_patch.addressed_families

            # 5. Multi-seed candidate evaluation
            cand_multi = self.evaluate_multi_seed(
                policy_version=cand_policy.version,
                policy_text=cand_policy.text,
                seeds=self.config.seeds,
                addressed_families=cand_rules,
            )

            # 6. Gate evaluation
            gate_res = self.gate.evaluate(candidate=cand_multi, baseline=base_multi)

            promotion_hash = None
            if gate_res.decision == GateDecision.ACCEPT:
                if self.promote_to_git:
                    promotion_hash = commit_promotion_to_git(
                        version=cand_policy.version,
                        policy_file=cand_policy.path,
                        baseline_asr=base_multi.asr_all.mean,
                        candidate_asr=cand_multi.asr_all.mean,
                        candidate_bpr=cand_multi.bpr.mean,
                    )
                conn = get_connection(self.db_path)
                try:
                    conn.execute(
                        """
                        UPDATE policies SET
                            candidate_status = 'accepted',
                            gate_decision = 'ACCEPT',
                            promotion_commit = ?,
                            seeds_run_json = ?
                        WHERE version = ?;
                        """,
                        (promotion_hash, json.dumps(self.config.seeds), cand_policy.version),
                    )
                    conn.commit()
                finally:
                    conn.close()

                final_decision = "ACCEPT"
                current_policy = cand_policy
            elif gate_res.decision == GateDecision.REJECT:
                conn = get_connection(self.db_path)
                try:
                    conn.execute(
                        """
                        UPDATE policies SET
                            candidate_status = 'rejected',
                            gate_decision = 'REJECT',
                            seeds_run_json = ?
                        WHERE version = ?;
                        """,
                        (json.dumps(self.config.seeds), cand_policy.version),
                    )
                    conn.commit()
                finally:
                    conn.close()
                final_decision = "REJECT"
                stop_reason = "candidate_rejected"

            record = IterationRecord(
                iteration=it,
                baseline_version=base_multi.policy_version,
                candidate_version=cand_policy.version,
                baseline_metrics=base_multi,
                candidate_metrics=cand_multi,
                gate_result_dict={
                    "decision": gate_res.decision.value,
                    "conditions": [c.__dict__ for c in gate_res.conditions],
                    "advisories": gate_res.advisories,
                },
                promotion_commit=promotion_hash,
                status=gate_res.decision.value,
            )
            iteration_records.append(record)

            if gate_res.decision in (GateDecision.REJECT, GateDecision.NO_ACTION_NEEDED):
                break

        # Generate summary markdown
        summary_md = self._render_summary(iteration_records, final_decision, stop_reason)
        return LoopResult(
            iterations=iteration_records,
            final_policy_version=current_policy.version,
            final_decision=final_decision,
            stop_reason=stop_reason,
            summary_markdown=summary_md,
        )

    def _render_summary(
        self,
        records: List[IterationRecord],
        final_decision: str,
        stop_reason: str,
    ) -> str:
        lines = [
            "# StrikeShield Hardening Loop Summary\n",
            f"- **Final Decision:** {final_decision}",
            f"- **Stop Reason:** {stop_reason}",
            f"- **Seeds Evaluated:** {self.config.seeds}\n",
            "| Iteration | Baseline | Candidate | Baseline ASR | Candidate ASR | Candidate BPR | Gate Decision | Promotion Commit |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in records:
            b_asr = r.baseline_metrics.asr_all.format_range()
            c_asr = r.candidate_metrics.asr_all.format_range() if r.candidate_metrics else "N/A"
            c_bpr = r.candidate_metrics.bpr.format_range() if r.candidate_metrics else "N/A"
            cand_v = r.candidate_version or "N/A"
            commit = r.promotion_commit or "N/A"
            lines.append(
                f"| {r.iteration} | {r.baseline_version} | {cand_v} | {b_asr} | {c_asr} | {c_bpr} | {r.status} | {commit} |"
            )
        return "\n".join(lines)
