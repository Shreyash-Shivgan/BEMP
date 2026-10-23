"""Ollama client wrapper with retry, strict JSON validation, VRAM management, and Mock fallback."""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar
import requests
from pydantic import BaseModel, ValidationError

from strikeshield.config import settings
from strikeshield.errors import (
    JudgeSchemaError,
    MockMissError,
    StrikeShieldError,
    VRAMStillAllocatedError,
)

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

FIXTURES_DIR = Path("tests/fixtures")


def compute_prompt_hash(role: str, prompt: str, schema_name: Optional[str] = None) -> str:
    """Computes a stable deterministic hash for a given role, prompt, and schema."""
    key = f"{role}:{prompt.strip()}:{schema_name or 'none'}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class MockOllamaClient:
    """Mock LLM client loading canned fixture responses for testing without a GPU."""

    def __init__(self, fixtures_dir: Optional[Path] = None) -> None:
        if not settings.is_mock_allowed():
            raise StrikeShieldError(
                "Mock LLM backend is strictly forbidden in production mode",
                {"env": settings.env},
            )
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR

    def generate(
        self,
        model: str,
        prompt: str,
        role: str = "target",
        temperature: float = 0.7,
        format_json: bool = False,
        schema: Optional[Type[T]] = None,
        keep_alive: Optional[str] = None,
    ) -> str:
        """Finds matching canned response or raises MockMissError."""
        role_dir = self.fixtures_dir / role
        prompt_hash = compute_prompt_hash(role, prompt, schema.__name__ if schema else None)

        if role_dir.exists():
            for fixture_file in role_dir.glob("*.json"):
                try:
                    with open(fixture_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue

                # Match by explicit hash, exact prompt, or prompt substring
                match = False
                if data.get("hash") == prompt_hash:
                    match = True
                elif data.get("prompt") == prompt:
                    match = True
                elif data.get("prompt_contains") and data["prompt_contains"] in prompt:
                    match = True

                if match:
                    raw_response = data.get("response")
                    if isinstance(raw_response, dict):
                        return json.dumps(raw_response)
                    return str(raw_response)

        raise MockMissError(
            f"No mock fixture found for role '{role}' and prompt",
            context={
                "role": role,
                "prompt_snippet": prompt[:80],
                "computed_hash": prompt_hash,
                "fixtures_dir": str(role_dir),
            },
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        schema: Type[T],
        role: str = "judge",
        temperature: float = 0.0,
    ) -> T:
        """Generates and strictly validates JSON output against pydantic schema in mock mode."""
        raw_text = self.generate(
            model=model,
            prompt=prompt,
            role=role,
            temperature=temperature,
            format_json=True,
            schema=schema,
        )
        try:
            parsed = json.loads(raw_text)
            return schema.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            raise JudgeSchemaError(
                f"Judge output failed schema validation: {e}",
                context={"raw_output": raw_text, "schema": schema.__name__},
            ) from e


class OllamaClient:
    """Production Ollama HTTP client with retry and strict JSON parsing."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        role: str = "target",
        temperature: float = 0.7,
        format_json: bool = False,
        schema: Optional[Type[T]] = None,
        keep_alive: Optional[str] = "5m",
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if format_json:
            payload["format"] = "json"
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return str(data.get("response", ""))
            except requests.RequestException as e:
                last_exc = e
                time.sleep(backoff_factor * (2**attempt))

        raise StrikeShieldError(
            f"Failed Ollama generate call after {max_retries} attempts: {last_exc}",
            context={"model": model, "role": role, "url": url},
        )

    def generate_json(
        self,
        model: str,
        prompt: str,
        schema: Type[T],
        role: str = "judge",
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> T:
        """Calls Ollama in JSON mode and validates strictly against pydantic schema with retry."""
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            raw = self.generate(
                model=model,
                prompt=prompt,
                role=role,
                temperature=temperature,
                format_json=True,
                schema=schema,
            )
            try:
                parsed = json.loads(raw)
                return schema.model_validate(parsed)
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                logger.warning("Judge JSON validation failed on attempt %d: %s", attempt + 1, e)

        raise JudgeSchemaError(
            f"Judge output malformed or failed validation after {max_retries} attempts: {last_err}",
            context={"schema": schema.__name__, "role": role},
        )

    def unload_model(self, model: str) -> None:
        """Forces immediate unload of model from VRAM using keep_alive: 0."""
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": "", "keep_alive": 0}
        try:
            requests.post(url, json=payload, timeout=10)
        except requests.RequestException as e:
            logger.warning("Unload request for %s encountered error: %s", model, e)

    def list_running_models(self) -> List[str]:
        """Queries /api/ps to retrieve loaded models in VRAM."""
        url = f"{self.base_url}/api/ps"
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                return [m.get("name", "") for m in models]
        except requests.RequestException:
            pass
        return []


def assert_vram_empty(poll_seconds: float = 5.0) -> None:
    """Verifies that no model is currently occupying VRAM before switching roles."""
    if settings.llm_backend == "mock":
        return

    client = OllamaClient()
    start = time.time()
    while time.time() - start < poll_seconds:
        running = client.list_running_models()
        if not running:
            return
        time.sleep(0.5)

    running = client.list_running_models()
    if running:
        raise VRAMStillAllocatedError(
            "VRAM is still allocated to previous model(s), violating single-model invariant",
            context={"running_models": running},
        )


class RoleSession:
    """Context manager for role-based LLM sessions enforcing sequential VRAM eviction."""

    def __init__(self, model_tag: str) -> None:
        self.model_tag = model_tag
        self.is_mock = settings.llm_backend == "mock"
        self.client = MockOllamaClient() if self.is_mock else OllamaClient()

    def __enter__(self) -> Any:
        assert_vram_empty()
        return self.client

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.is_mock and isinstance(self.client, OllamaClient):
            self.client.unload_model(self.model_tag)
            assert_vram_empty(poll_seconds=5.0)


def get_llm_client() -> Any:
    """Factory returning configured LLM client."""
    if settings.llm_backend == "mock":
        return MockOllamaClient()
    return OllamaClient()
