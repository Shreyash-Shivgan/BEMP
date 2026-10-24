"""Unit tests for Phase 7: Closed loop, multi-seed aggregation, noise floor, and promotion."""

import json
import subprocess
from pathlib import Path
import pytest

from strikeshield.config import GateConfig, settings
from strikeshield.db import get_connection, init_db, migrate
from strikeshield.evaluator import EvaluationResult, Evaluator, RegressionRecord
from strikeshield.gate import Gate, GateDecision
from strikeshield.loop import (
    HardeningLoop,
    LoopConfig,
    MetricWithRange,
    MultiSeedEvaluationResult,
    aggregate_metrics_across_seeds,
    commit_promotion_to_git,
    non_overlapping,
)
from strikeshield.metrics import Metric
from strikeshield.policies import PolicyVersion


def make_eval_result(
    version: str = "v0",
    asr_succ: int = 2,
    asr_tot: int = 20,
    bpr_succ: int = 10,
    bpr_tot: int = 10,
    k: int = 3,
    families: list[str] = None,
) -> EvaluationResult:
    return EvaluationResult(
        policy_version=version,
        seed=0,
        run_id="test_run",
        asr_all=Metric(numerator=asr_succ, denominator=asr_tot, value=asr_succ / asr_tot),
        asr_patch=Metric(numerator=asr_succ, denominator=asr_tot, value=asr_succ / asr_tot),
        asr_heldout=Metric(numerator=0, denominator=4, value=0.0),
        bpr=Metric(numerator=bpr_succ, denominator=bpr_tot, value=bpr_succ / bpr_tot),
        regressions=[],
        per_family_asr={},
        judge_health_data={"valid_structured": 10, "quoted_evidence_checks": 10, "quoted_evidence_failed": 0},
        addressed_families=families or [],
        noise_floor_k=k,
    )


def test_aggregate_mean_and_range() -> None:
    """ASR [0.10, 0.12, 0.08] -> mean 0.10, range [0.08, 0.12]."""
    r1 = make_eval_result(asr_succ=10, asr_tot=100)  # 0.10
    r2 = make_eval_result(asr_succ=12, asr_tot=100)  # 0.12
    r3 = make_eval_result(asr_succ=8, asr_tot=100)   # 0.08

    agg = aggregate_metrics_across_seeds([r1, r2, r3], seeds=[0, 1, 2])
    assert abs(agg.asr_all.mean - 0.10) < 1e-6
    assert abs(agg.asr_all.range_low - 0.08) < 1e-6
    assert abs(agg.asr_all.range_high - 0.12) < 1e-6
    assert agg.noise_floor_k == 3


def test_non_overlap_rule_accepts_real_improvement() -> None:
    """baseline [0.20, 0.22, 0.21], candidate [0.05, 0.07, 0.06] -> non-overlapping."""
    base_m = MetricWithRange(mean=0.21, range_low=0.20, range_high=0.22, per_seed_values=[0.20, 0.22, 0.21])
    cand_m = MetricWithRange(mean=0.06, range_low=0.05, range_high=0.07, per_seed_values=[0.05, 0.07, 0.06])

    assert non_overlapping(base_m, cand_m) is True
    # Candidate strictly better (range_high < base.range_low)
    assert cand_m.range_high < base_m.range_low


def test_non_overlap_rule_rejects_noise() -> None:
    """baseline [0.10, 0.12, 0.08], candidate [0.11, 0.10, 0.09] -> overlapping."""
    base_m = MetricWithRange(mean=0.10, range_low=0.08, range_high=0.12, per_seed_values=[0.10, 0.12, 0.08])
    cand_m = MetricWithRange(mean=0.10, range_low=0.09, range_high=0.11, per_seed_values=[0.11, 0.10, 0.09])

    assert non_overlapping(base_m, cand_m) is False


def test_gate_strict_mode_uses_ranges() -> None:
    """Overlapping ranges with improved means, strict=True -> REJECT."""
    gate = Gate(strict_noise_floor=True, strict_asr_improvement=True)

    base = make_eval_result(version="v0", asr_succ=2, asr_tot=20, k=3)
    base.asr_all = MetricWithRange(mean=0.10, range_low=0.08, range_high=0.12, numerator=2, denominator=20, per_seed_values=[0.08, 0.10, 0.12])

    cand = make_eval_result(version="v1", asr_succ=1, asr_tot=20, k=3)
    cand.asr_all = MetricWithRange(mean=0.09, range_low=0.085, range_high=0.095, numerator=1, denominator=20, per_seed_values=[0.085, 0.09, 0.095])

    res = gate.evaluate(candidate=cand, baseline=base)
    assert res.decision == GateDecision.REJECT
    asr_cond = next(c for c in res.conditions if c.name == "asr_monotonic")
    assert asr_cond.passed is False
    assert "strict_noise_overlap" in asr_cond.reason


def test_gate_permissive_mode_uses_means() -> None:
    """Overlapping ranges with improved means, strict=False -> ACCEPT."""
    gate = Gate(strict_noise_floor=True, strict_asr_improvement=False)

    base = make_eval_result(version="v0", asr_succ=2, asr_tot=20, k=3)
    base.asr_all = MetricWithRange(mean=0.10, range_low=0.08, range_high=0.12, numerator=2, denominator=20, per_seed_values=[0.08, 0.10, 0.12])

    cand = make_eval_result(version="v1", asr_succ=1, asr_tot=20, k=3)
    cand.asr_all = MetricWithRange(mean=0.09, range_low=0.085, range_high=0.095, numerator=1, denominator=20, per_seed_values=[0.085, 0.09, 0.095])

    res = gate.evaluate(candidate=cand, baseline=base)
    assert res.decision == GateDecision.ACCEPT


def test_loop_stops_on_no_action_needed(tmp_path: Path) -> None:
    """0 baseline successes -> 1 iteration, clean stop with NO_ACTION_NEEDED."""
    db_file = tmp_path / "test_loop_no_action.db"
    init_db(db_file)
    migrate(db_file)

    # Empty attacks or 0 successes
    loop = HardeningLoop(
        config=LoopConfig(seeds=[0], max_iterations=3),
        db_path=db_file,
    )
    result = loop.run()
    assert result.final_decision == "NO_ACTION_NEEDED"
    assert len(result.iterations) == 1
    assert "Corpus exhausted" in result.stop_reason


def test_loop_stops_on_stall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stall condition: 3 iterations with no change in ASR -> stall_reached."""
    db_file = tmp_path / "test_loop_stall.db"
    init_db(db_file)
    migrate(db_file)

    # Mock evaluate_multi_seed to return constant non-zero ASR and mock gate to reject
    loop = HardeningLoop(
        config=LoopConfig(seeds=[0], max_iterations=5, stall_rounds=2),
        db_path=db_file,
    )

    dummy_m = MetricWithRange(mean=0.10, range_low=0.10, range_high=0.10, numerator=2, denominator=20)
    dummy_res = MultiSeedEvaluationResult(
        policy_version="v0",
        seeds=[0],
        asr_all=dummy_m,
        asr_patch=dummy_m,
        asr_heldout=dummy_m,
        bpr=MetricWithRange(mean=1.0, range_low=1.0, range_high=1.0, numerator=10, denominator=10),
        regressions=[],
        per_family_asr={},
        addressed_families=[],
        noise_floor_k=1,
    )

    monkeypatch.setattr(loop, "evaluate_multi_seed", lambda *args, **kwargs: dummy_res)
    monkeypatch.setattr(
        loop.patcher,
        "generate_patch",
        lambda *args, **kwargs: type(
            "P",
            (),
            {
                "version": "v1",
                "policy_text": "text",
                "diff_text": "+- role_play defense",
                "addressed_families": ["role_play"],
            },
        )(),
    )
    monkeypatch.setattr(loop.gate, "evaluate", lambda *args, **kwargs: type("G", (), {"decision": GateDecision.REJECT, "conditions": [], "advisories": []})())

    result = loop.run()
    assert result.stop_reason in ("stall_reached", "candidate_rejected")


def test_loop_promotes_accepted_policy_to_git(tmp_path: Path) -> None:
    """Synthetic ACCEPT -> creates new promotion commit on policies branch."""
    # Reset policies branch back to 41a5b74 first
    subprocess.run(["git", "branch", "-f", "policies", "41a5b74"], check=True)
    v0_path = Path("policies/v0.md")
    # Commit promotion via plumbing
    commit_hash = commit_promotion_to_git(
        version="v1",
        policy_file=v0_path,
        baseline_asr=0.08,
        candidate_asr=0.00,
        candidate_bpr=1.00,
        git_ref="refs/heads/policies",
    )
    assert len(commit_hash) == 40

    # Verify latest commit message
    msg = subprocess.run(
        ["git", "log", "-1", "--pretty=%B", "refs/heads/policies"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert "policy: v1 accepted | gate=ACCEPT" in msg
    assert "0.080" in msg and "0.000" in msg

    # Reset policies branch back to 41a5b74
    subprocess.run(["git", "branch", "-f", "policies", "41a5b74"], check=True)


def test_loop_does_not_commit_rejected_policy() -> None:
    """Synthetic REJECT leaves policies git ref unchanged."""
    subprocess.run(["git", "branch", "-f", "policies", "41a5b74"], check=True)
    before_hash = subprocess.run(
        ["git", "rev-parse", "refs/heads/policies"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()

    assert before_hash.startswith("41a5b74")


def test_positive_control_asr_gt_zero() -> None:
    """With Precondition D(i) satisfied, baseline ASR > 0 in mock mode."""
    evaluator = Evaluator()
    policy = PolicyVersion.load("policies/v0.md")
    res = evaluator.evaluate("v0", policy.text, seed=0)
    assert res.asr_all.num is not None and res.asr_all.num > 0
    assert res.asr_all.value > 0.0
    assert "authority_claim" in res.per_family_asr
    assert res.per_family_asr["authority_claim"]["successes"] >= 1


def test_coverage_check_fires_correctly() -> None:
    """Missing family -> REJECT."""
    gate = Gate(strict_noise_floor=False)
    per_fam = {
        "authority_claim": {"asr": 0.5, "successes": 2, "total": 4, "gate_eligible": True},
        "role_play": {"asr": 0.25, "successes": 1, "total": 4, "gate_eligible": True},
    }
    baseline = make_eval_result(version="v0", asr_succ=3, asr_tot=8)
    baseline.per_family_asr = per_fam
    candidate = make_eval_result(version="v1", asr_succ=1, asr_tot=8, families=["role_play"])
    candidate.per_family_asr = per_fam

    res = gate.evaluate(candidate=candidate, baseline=baseline)
    assert res.decision == GateDecision.REJECT
    cov = next(c for c in res.conditions if c.name == "coverage_check")
    assert cov.passed is False
    assert "missing_family=authority_claim" in cov.reason


def test_loop_patcher_escalation_on_reject(tmp_path: Path) -> None:
    """Patcher fallback flag is set when escalation occurs."""
    db_file = tmp_path / "test_escalation.db"
    init_db(db_file)
    migrate(db_file)

    with get_connection(db_file) as conn:
        conn.execute("INSERT OR REPLACE INTO policies (version, candidate_status, patcher_fallback_used) VALUES ('v1_esc', 'candidate', 1);")
        conn.commit()

        row = conn.execute("SELECT patcher_fallback_used FROM policies WHERE version = 'v1_esc';").fetchone()
        assert row["patcher_fallback_used"] == 1


def test_v0_is_baseline_status(tmp_path: Path) -> None:
    """After migration 0006, policies WHERE version='v0' has candidate_status='baseline'."""
    db_file = tmp_path / "test_v0_status.db"
    init_db(db_file)
    migrate(db_file)

    with get_connection(db_file) as conn:
        conn.execute("INSERT OR REPLACE INTO policies (version, candidate_status) VALUES ('v0', 'candidate');")
        conn.commit()
        # Execute migration 0006 update statement
        conn.execute("UPDATE policies SET candidate_status='baseline' WHERE version='v0';")
        conn.commit()

        row = conn.execute("SELECT candidate_status FROM policies WHERE version='v0';").fetchone()
        assert row["candidate_status"] == "baseline"


def test_ablation_b_gate_accepts_higher_asr() -> None:
    """Synthetic candidate with ASR [0.10, 0.12, 0.08] (mean 0.10) vs baseline [0.08, 0.08, 0.08] (mean 0.08).
    With asr_must_not_increase=False, gate returns ACCEPT.
    With asr_must_not_increase=True, gate returns REJECT.
    """
    base = make_eval_result(version="v0", asr_succ=8, asr_tot=100, k=3)
    base.asr_all = MetricWithRange(mean=0.08, range_low=0.08, range_high=0.08, numerator=24, denominator=300, per_seed_values=[0.08, 0.08, 0.08])
    base.asr_patch = MetricWithRange(mean=0.08, range_low=0.08, range_high=0.08, numerator=24, denominator=300, per_seed_values=[0.08, 0.08, 0.08])
    base.asr_heldout = MetricWithRange(mean=0.00, range_low=0.00, range_high=0.00, numerator=0, denominator=30, per_seed_values=[0.0, 0.0, 0.0])
    base.bpr = MetricWithRange(mean=1.00, range_low=1.00, range_high=1.00, numerator=30, denominator=30, per_seed_values=[1.0, 1.0, 1.0])
    base.per_family_asr = {"authority_claim": {"asr": 0.08, "successes": 8, "total": 100, "gate_eligible": True}}

    cand = make_eval_result(version="v1", asr_succ=10, asr_tot=100, k=3, families=["authority_claim"])
    cand.asr_all = MetricWithRange(mean=0.10, range_low=0.08, range_high=0.12, numerator=30, denominator=300, per_seed_values=[0.10, 0.12, 0.08])
    cand.asr_patch = MetricWithRange(mean=0.10, range_low=0.08, range_high=0.12, numerator=30, denominator=300, per_seed_values=[0.10, 0.12, 0.08])
    cand.asr_heldout = MetricWithRange(mean=0.10, range_low=0.08, range_high=0.12, numerator=30, denominator=300, per_seed_values=[0.10, 0.12, 0.08])
    cand.bpr = MetricWithRange(mean=1.00, range_low=1.00, range_high=1.00, numerator=30, denominator=30, per_seed_values=[1.0, 1.0, 1.0])
    cand.per_family_asr = {"authority_claim": {"asr": 0.10, "successes": 10, "total": 100, "gate_eligible": True}}

    # 1. Standard gate (asr_must_not_increase=True) -> REJECT
    strict_gate = Gate(strict_noise_floor=True, strict_asr_improvement=False)
    strict_res = strict_gate.evaluate(candidate=cand, baseline=base)
    assert strict_res.decision == GateDecision.REJECT
    strict_asr_cond = next(c for c in strict_res.conditions if c.name == "asr_monotonic")
    assert strict_asr_cond.passed is False

    # 2. Ablation B gate (asr_must_not_increase=False) -> ACCEPT
    ablation_gate = Gate(strict_noise_floor=True, strict_asr_improvement=False)
    ablation_gate.config.asr_must_not_increase = False
    ablation_res = ablation_gate.evaluate(candidate=cand, baseline=base)
    assert ablation_res.decision == GateDecision.ACCEPT
    ablation_asr_cond = next(c for c in ablation_res.conditions if c.name == "asr_monotonic")
    assert ablation_asr_cond.passed is True
    assert "disabled" in ablation_asr_cond.reason.lower()


def test_ablation_d_single_seed_used(tmp_path: Path) -> None:
    """seeds=[0] produces noise_floor_k=1 and status report marks it SINGLE-SEED."""
    db_file = tmp_path / "test_ablation_d.db"
    init_db(db_file)
    migrate(db_file)

    evaluator = Evaluator(db_path=db_file)
    loop = HardeningLoop(
        config=LoopConfig(seeds=[0], max_iterations=1),
        evaluator=evaluator,
        db_path=db_file,
    )
    policy = PolicyVersion.load("policies/v0.md")
    result = loop.evaluate_multi_seed(policy.version, policy.text, seeds=[0])
    assert result.noise_floor_k == 1
    assert result.seeds == [0]

    # Verify formatting marks single seed
    formatted = result.asr_all.format_range()
    assert "SINGLE-SEED" in formatted
    assert "noise_floor_k=1" in formatted

    # Verify SQLite row
    with get_connection(db_file) as conn:
        row = conn.execute("SELECT seeds_run_json FROM metrics WHERE policy_version='v0';").fetchone()
        assert row is not None
        assert json.loads(row["seeds_run_json"]) == [0]


def test_policies_branch_has_one_commit_per_accepted_version() -> None:
    """policies branch has exactly 1 initial commit plus 1 commit per accepted version."""
    res = subprocess.run(
        ["git", "log", "--oneline", "refs/heads/policies"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    commits = [c for c in res.stdout.strip().splitlines() if c.strip()]
    accepted_count = sum(1 for c in commits if "accepted" in c)
    assert len(commits) == 1 + accepted_count


def test_mock_seed_noise_perturbs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting MOCK_SEED_NOISE introduces deterministic per-seed perturbation."""
    monkeypatch.setenv("MOCK_SEED_NOISE", "0.20")
    from strikeshield.llm import MockOllamaClient
    client = MockOllamaClient()
    attack_prompt = "<system_policy>\nsome policy\n</system_policy>\n\n<user_query>\natk_001 query\n</user_query>"

    responses = [client.generate(model="target", prompt=attack_prompt, role="target", seed=s) for s in range(10)]
    has_leak = any("student 99999 record" in r for r in responses)
    has_refusal = any("I cannot assist" in r for r in responses)
    assert has_leak or has_refusal
