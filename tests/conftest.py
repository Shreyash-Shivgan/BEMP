"""Pytest configuration and global fixtures for StrikeShield."""

import os
import pytest
from strikeshield.config import settings

# Force mock mode and dev env for test suite
os.environ["LLM_BACKEND"] = "mock"
os.environ["ENV"] = "dev"
settings.llm_backend = "mock"
settings.env = "dev"


@pytest.fixture(autouse=True)
def ensure_mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures every test runs in mock mode within dev environment by default."""
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("ENV", "dev")
    settings.llm_backend = "mock"
    settings.env = "dev"
