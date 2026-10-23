"""Tests for configuration loading and validation."""

from pathlib import Path
from strikeshield.config import settings


def test_models_config_loaded() -> None:
    assert settings.models.target.model == "llama3.1:8b-instruct-q4_K_M"
    assert settings.models.target.temperature == 0.7
    assert settings.models.judge.model == "qwen2.5:7b-instruct-q4_K_M"
    assert settings.models.judge.temperature == 0.0
    assert settings.models.patcher.model == "mistral:7b-instruct-q4_K_M"
    assert settings.models.patcher.fallback_model == "qwen2.5:7b-instruct-q4_K_M"


def test_gate_config_loaded() -> None:
    assert settings.gate.bpr_threshold == 0.90
    assert settings.gate.regression_budget == 0
    assert settings.gate.asr_must_not_increase is True
    assert settings.gate.max_heldout_gap == 0.05
    assert settings.gate.noise_floor.seeds == 3
    assert settings.gate.noise_floor.target_temperature == 0.7
    assert settings.gate.noise_floor.require_non_overlapping_ranges is True


def test_taxonomy_config_loaded() -> None:
    risk_ids = [r.id for r in settings.taxonomy.risk_categories]
    assert len(risk_ids) == 5
    assert "system-prompt extraction" in risk_ids
    assert "personal-data disclosure" in risk_ids
    assert "harmful instruction" in risk_ids
    assert "out-of-scope task" in risk_ids
    assert "role abandonment" in risk_ids

    tech_ids = [t.id for t in settings.taxonomy.technique_families]
    assert len(tech_ids) == 8
    assert "role_play" in tech_ids
    assert "hypothetical" in tech_ids
    assert "authority_claim" in tech_ids
    assert "instruction_override" in tech_ids
    assert "encoding" in tech_ids
    assert "language_switch" in tech_ids
    assert "false_premise" in tech_ids
    assert "multi_turn_escalation" in tech_ids


def test_policy_v0_exists() -> None:
    v0_path = Path("policies/v0.md")
    assert v0_path.exists()
    content = v0_path.read_text(encoding="utf-8")
    assert "Campus Helpdesk Assistant" in content
    assert "Refusal Rules" in content
    assert "Student Records" in content
