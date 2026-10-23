"""Tests for ASR calculations, explicit denominator tracking, and confound reporting."""

from strikeshield.asr import BucketMetric, compute_asr_from_map, confound_report
from strikeshield.attack_source import AttackRecord
from strikeshield.taxonomy import weakest_techniques


def test_bucket_metric_format_ratio() -> None:
    bm = BucketMetric(successes=9, total=50)
    assert bm.asr == 0.18
    assert bm.format_ratio() == "18.0% (9/50)"


def test_compute_asr_from_map() -> None:
    meta_verdicts = [
        {"is_success": True, "split": "patch", "risk_category": "system-prompt extraction", "technique_family": "role_play"},
        {"is_success": False, "split": "patch", "risk_category": "system-prompt extraction", "technique_family": "role_play"},
        {"is_success": True, "split": "heldout", "risk_category": "harmful instruction", "technique_family": "hypothetical"},
        {"is_success": False, "split": "heldout", "risk_category": "harmful instruction", "technique_family": "hypothetical"},
    ]

    res = compute_asr_from_map(meta_verdicts)
    assert res.all.successes == 2
    assert res.all.total == 4
    assert res.all.asr == 0.50

    assert res.patch.successes == 1
    assert res.patch.total == 2
    assert res.patch.asr == 0.50

    assert res.heldout.successes == 1
    assert res.heldout.total == 2
    assert res.heldout.asr == 0.50

    # Denominator check in formatted string
    assert "50.0% (2/4)" in res.format_table()


def test_confound_report_flags_concentrated_technique() -> None:
    attacks = [
        # technique_a only in risk_1
        AttackRecord(
            prompt="p1",
            risk_category="system-prompt extraction",
            technique_family="encoding",
            target_behaviour="tb",
        ),
        # technique_b in risk_1, risk_2, risk_3
        AttackRecord(prompt="p2", risk_category="system-prompt extraction", technique_family="role_play", target_behaviour="tb"),
        AttackRecord(prompt="p3", risk_category="harmful instruction", technique_family="role_play", target_behaviour="tb"),
        AttackRecord(prompt="p4", risk_category="personal-data disclosure", technique_family="role_play", target_behaviour="tb"),
    ]

    report = confound_report(attacks)
    assert "CONFOUNDED" in report
    assert "encoding" in report
    assert "role_play" in report


def test_weakest_techniques_baseline_ranking() -> None:
    """Flag E test: verifies weakest_techniques accurately picks top-3 weakest techniques."""
    baseline_asr = {
        "role_play": 0.40,
        "language_switch": 0.70,
        "instruction_override": 0.85,
        "authority_claim": 0.60,
        "encoding": 0.10,
    }
    weakest = weakest_techniques(top_k=3, asr_by_technique=baseline_asr)
    assert weakest == ["instruction_override", "language_switch", "authority_claim"]
