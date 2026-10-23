"""Target model runner under guarded policy versions."""

import hashlib
import time
from pathlib import Path
from typing import Any, List, Optional
from pydantic import BaseModel

from strikeshield.attack_source import AttackRecord
from strikeshield.config import settings
from strikeshield.db import get_connection
from strikeshield.llm import RoleSession, assert_vram_empty


class ResponseRecord(BaseModel):
    id: str
    attack_id: str
    policy_version: str
    response_text: str
    run_id: Optional[str] = None
    seed: int = 42
    temperature: float = 0.7


def compute_response_id(policy_version: str, attack_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{policy_version}::{attack_id}::{seed}".encode("utf-8")).hexdigest()[:12]
    return f"resp_{policy_version}_{digest}"


class TargetRunner:
    """Manages target model invocation under guardrail policy system prompts."""

    def __init__(self, model_tag: Optional[str] = None, temperature: Optional[float] = None) -> None:
        self.model_tag = model_tag or settings.models.target.model
        self.temperature = (
            temperature if temperature is not None else settings.models.target.temperature
        )
        self.session: Optional[RoleSession] = None
        self.client: Any = None

    def __enter__(self) -> "TargetRunner":
        self.session = RoleSession(self.model_tag)
        self.client = self.session.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.session:
            self.session.__exit__(exc_type, exc_val, exc_tb)
        assert_vram_empty()

    def run(
        self,
        attack: AttackRecord,
        policy_version: str,
        policy_text: str,
        seed: int = 42,
        run_id: Optional[str] = None,
    ) -> ResponseRecord:
        """Executes a single attack prompt with the policy as system instructions."""
        attack_id = attack.id or f"atk_{hashlib.sha256(attack.prompt.encode()).hexdigest()[:8]}"
        full_prompt = (
            f"<system_policy>\n{policy_text.strip()}\n</system_policy>\n\n"
            f"<user_query>\n{attack.prompt.strip()}\n</user_query>"
        )

        resp_text = self.client.generate(
            model=self.model_tag,
            prompt=full_prompt,
            role="target",
            temperature=self.temperature,
        )

        resp_id = compute_response_id(policy_version, attack_id, seed)
        return ResponseRecord(
            id=resp_id,
            attack_id=attack_id,
            policy_version=policy_version,
            response_text=resp_text,
            run_id=run_id,
            seed=seed,
            temperature=self.temperature,
        )

    def run_batch(
        self,
        attacks: List[AttackRecord],
        policy_version: str,
        policy_text: str,
        seed: int = 42,
        db_path: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> List[ResponseRecord]:
        """Runs a batch of attacks with single model load/unload and persists to SQLite."""
        responses: List[ResponseRecord] = []
        conn = get_connection(db_path)

        to_insert = []
        for atk in attacks:
            record = self.run(
                attack=atk,
                policy_version=policy_version,
                policy_text=policy_text,
                seed=seed,
                run_id=run_id,
            )
            responses.append(record)
            to_insert.append((
                record.id,
                record.attack_id,
                record.policy_version,
                record.response_text,
                record.run_id,
                record.seed,
                record.temperature,
            ))

        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO responses (
                    id, attack_id, policy_version, response_text, run_id, seed, temperature
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                to_insert,
            )
            conn.commit()
        finally:
            conn.close()

        return responses
