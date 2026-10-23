"""Tests for mock backend guard and environment protection."""

import pytest

from strikeshield.config import Settings
from strikeshield.errors import StrikeShieldError
from strikeshield.llm import MockOllamaClient


def test_mock_backend_in_production_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set environment to production
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("LLM_BACKEND", "mock")

    prod_settings = Settings(env="production", llm_backend="mock")
    assert prod_settings.is_mock_allowed() is False

    # MockOllamaClient instantiation must fail loudly in production
    monkeypatch.setattr("strikeshield.llm.settings", prod_settings)
    with pytest.raises(StrikeShieldError) as exc_info:
        MockOllamaClient()
    assert "Mock LLM backend is strictly forbidden in production mode" in str(exc_info.value)
