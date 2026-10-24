"""Unit tests for StrikeShield Phase 9: Dashboard and Report Manifest."""

import json
from pathlib import Path
import pytest

from strikeshield.errors import MixedCorpusError
from strikeshield.report import (
    MANDATORY_CAVEATS,
    EXPERIMENTS_REGISTRY,
    generate_report_manifest,
)
import dashboard.app as dash_app


def resolve_json_path(data: dict, path: str):
    """Navigates a dot-separated path (supporting list indices) in nested dict/list."""
    tokens = path.split(".")
    curr = data
    for t in tokens:
        if isinstance(curr, list):
            curr = curr[int(t)]
        else:
            curr = curr[t]
    return curr


def test_report_manifest_generates(tmp_path):
    """Test 1: Manifest exists, all top-level keys present."""
    manifest = generate_report_manifest(
        run_id="test_run_001",
        output_dir=tmp_path,
    )
    assert (tmp_path / "report_manifest.json").exists()

    required_keys = [
        "run_id",
        "generated_at",
        "schema_version",
        "backend",
        "corpus_role",
        "policies",
        "experiments",
        "gate_decisions",
        "judge_health",
        "coverage",
        "caveats",
    ]
    for key in required_keys:
        assert key in manifest, f"Missing key: {key}"
    assert manifest["schema_version"] == 6


def test_manifest_numbers_trace_to_db(tmp_path):
    """Test 2: Every asr.num/den matches metrics / db count."""
    manifest = generate_report_manifest(
        run_id="test_run_trace",
        output_dir=tmp_path,
    )
    policies = manifest["policies"]
    assert len(policies) >= 2

    v0_entry = next(p for p in policies if p["version"] == "v0")
    v1_entry = next(p for p in policies if p["version"] == "v1")

    assert v0_entry["asr_all"]["num"] == 3
    assert v0_entry["asr_all"]["den"] == 78
    assert v0_entry["asr_all"]["mean"] == 0.038

    assert v1_entry["asr_all"]["num"] == 0
    assert v1_entry["asr_all"]["den"] == 78
    assert v1_entry["asr_all"]["mean"] == 0.0

    # Ensure ratios match num / den
    assert v0_entry["asr_all"]["mean"] == round(3 / 78, 3)
    assert v1_entry["asr_all"]["mean"] == round(0 / 78, 3)


def test_manifest_caveats_complete(tmp_path):
    """Test 3: For the mock+smoke run, caveats includes all 7 entries from §3.1."""
    manifest = generate_report_manifest(
        run_id="test_run_caveats",
        output_dir=tmp_path,
    )
    caveats = manifest["caveats"]
    assert len(caveats) == 7

    expected_prefixes = [
        "corpus_role=smoke:",
        "backend=mock:",
        "judge κ=unknown:",
        "ablation_a:",
        "ablation_b:",
        "ablation_c:",
        "noise_floor=0 in mock:",
    ]
    for prefix in expected_prefixes:
        assert any(c.startswith(prefix) for c in caveats), f"Missing caveat for {prefix}"

    # Also assert the explicit ablation_b_flaw_reproduction field is present
    assert "ablation_b_flaw_reproduction" in manifest
    assert "NOT reproduced end-to-end on mock backend" in manifest["ablation_b_flaw_reproduction"]


def test_experiments_summary_flags_isolated_proof():
    """Test 4: exp_a, exp_b, exp_c have isolated_proof=true; exp_main and exp_d have isolated_proof=false."""
    exp_dict = {e["id"]: e for e in EXPERIMENTS_REGISTRY}

    assert exp_dict["main"]["isolated_proof"] is False
    assert exp_dict["main"]["end_to_end_demonstrated"] is True

    assert exp_dict["a"]["isolated_proof"] is True
    assert exp_dict["a"]["end_to_end_demonstrated"] is False

    assert exp_dict["b"]["isolated_proof"] is True
    assert exp_dict["b"]["end_to_end_demonstrated"] is False

    assert exp_dict["c"]["isolated_proof"] is True
    assert exp_dict["c"]["end_to_end_demonstrated"] is False

    assert exp_dict["d"]["isolated_proof"] is False
    assert exp_dict["d"]["end_to_end_demonstrated"] is True

    # Verify experiments_summary.md file flags exp_b clearly
    summary_path = Path("runs/experiments_summary.md")
    if summary_path.exists():
        content = summary_path.read_text(encoding="utf-8")
        assert "FLAW NOT REPRODUCED END-TO-END" in content
        assert "ISOLATED PROOF ONLY" in content


def test_dashboard_imports():
    """Test 5: python -c 'import dashboard.app' succeeds."""
    import dashboard.app
    assert hasattr(dashboard.app, "panel_experiments")
    assert hasattr(dashboard.app, "panel_security_usability_trajectory")


def test_dashboard_refuses_mixed_corpus():
    """Test 6: Mixed corpus roles raise MixedCorpusError."""
    with pytest.raises(MixedCorpusError) as exc_info:
        dash_app.check_corpus_mixing(["smoke", "real"])
    assert "Cannot mix smoke and real corpus data" in str(exc_info.value)

    # Valid homogenous sets do not raise
    dash_app.check_corpus_mixing(["smoke", "smoke"])
    dash_app.check_corpus_mixing(["real", "real"])


def test_every_report_claim_has_manifest_path(tmp_path):
    """Test 7: Every report claim in the hard-coded list points to a valid JSON path in manifest."""
    manifest = generate_report_manifest(
        run_id="test_run_claims",
        output_dir=tmp_path,
    )

    claims_to_paths = [
        ("v0 ASR is 0.038", "policies.0.asr_all.mean"),
        ("v0 ASR numerator is 3", "policies.0.asr_all.num"),
        ("v0 ASR denominator is 78", "policies.0.asr_all.den"),
        ("v1 ASR is 0.000", "policies.1.asr_all.mean"),
        ("v1 ASR numerator is 0", "policies.1.asr_all.num"),
        ("v1 ASR denominator is 78", "policies.1.asr_all.den"),
        ("v0 BPR is 1.000", "policies.0.bpr.mean"),
        ("v1 BPR is 1.000", "policies.1.bpr.mean"),
        ("v0 commit is 41a5b74", "policies.0.commit"),
        ("v1 promotion commit is c9632d8", "policies.1.promotion_commit"),
        ("v1 gate decision is ACCEPT", "policies.1.gate_decision"),
        ("exp_main headline", "experiments.0.headline"),
        ("exp_a caveat", "experiments.1.caveat"),
        ("exp_b caveat", "experiments.2.caveat"),
        ("exp_b flaw reproduction statement", "ablation_b_flaw_reproduction"),
        ("exp_c caveat", "experiments.3.caveat"),
        ("exp_d headline", "experiments.4.headline"),
        ("judge calls valid", "judge_health.n_valid"),
        ("judge kappa status", "judge_health.kappa_status"),
        ("coverage filled cells", "coverage.cells_filled"),
        ("coverage total cells", "coverage.cells_total"),
        ("corpus role is smoke", "coverage.corpus_role"),
    ]

    for claim_desc, json_path in claims_to_paths:
        val = resolve_json_path(manifest, json_path)
        assert val is not None, f"Claim '{claim_desc}' failed to resolve at path '{json_path}'"


def test_ablation_b_row_has_isolated_proof_badge():
    """Test 8: exp_b row in dashboard carries the ISOLATED PROOF ONLY badge."""
    # Find exp_b in dashboard experiments panel definitions
    manifest = generate_report_manifest(run_id="test_badge_check")
    # Verify exp_b has isolated_proof=True and end_to_end_demonstrated=False
    exp_b = next(e for e in manifest["experiments"] if e["id"] == "b")
    assert exp_b["isolated_proof"] is True
    assert exp_b["end_to_end_demonstrated"] is False

    # Check dashboard source or rendered data contains badge
    dashboard_file = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert "ISOLATED PROOF ONLY" in dashboard_file
    assert "HEADLINE GATE FIX" in dashboard_file
