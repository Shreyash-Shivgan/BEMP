"""Tests for SQLite schema, constraints, and idempotent migration runner."""

import sqlite3
import pytest
from pathlib import Path
from strikeshield.db import get_connection, init_db, migrate, get_current_schema_version


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_strikeshield.db"
    return db_file


def test_init_and_migrate_idempotent(temp_db: Path) -> None:
    init_db(temp_db)
    v1 = migrate(temp_db)
    assert v1 == 2

    # Second migration must be idempotent no-op
    v2 = migrate(temp_db)
    assert v2 == 2


def test_schema_tables_exist(temp_db: Path) -> None:
    migrate(temp_db)
    with get_connection(temp_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row["name"] for row in cursor.fetchall()]
        expected_tables = {
            "schema_version",
            "runs",
            "policies",
            "attacks",
            "responses",
            "verdicts",
            "evaluations",
            "metrics",
            "judge_validation",
        }
        for table in expected_tables:
            assert table in tables, f"Expected table {table} not found in database"


def test_attacks_split_constraint(temp_db: Path) -> None:
    migrate(temp_db)
    with get_connection(temp_db) as conn:
        # Valid split
        conn.execute(
            """
            INSERT INTO attacks (id, prompt, risk_category, technique_family, target_behaviour, cell_id, split, source)
            VALUES ('test_1', 'prompt', 'risk', 'tech', 'target', 'risk::tech', 'patch', 'test');
            """
        )
        conn.commit()

        # Invalid split must raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO attacks (id, prompt, risk_category, technique_family, target_behaviour, cell_id, split, source)
                VALUES ('test_2', 'prompt', 'risk', 'tech', 'target', 'risk::tech', 'invalid_split', 'test');
                """
            )
            conn.commit()


def test_verdict_class_constraint(temp_db: Path) -> None:
    migrate(temp_db)
    with get_connection(temp_db) as conn:
        conn.execute("INSERT INTO runs (id, seed, iteration, config_json) VALUES ('r1', 42, 0, '{}');")
        conn.execute("INSERT INTO policies (version) VALUES ('v0');")
        conn.execute(
            """
            INSERT INTO attacks (id, prompt, risk_category, technique_family, target_behaviour, cell_id, split, source)
            VALUES ('atk1', 'p', 'r', 't', 'b', 'r::t', 'patch', 'src');
            """
        )
        conn.execute(
            "INSERT INTO responses (id, attack_id, policy_version, response_text, run_id) VALUES ('resp1', 'atk1', 'v0', 'txt', 'r1');"
        )
        conn.commit()

        # Valid verdict
        conn.execute(
            """
            INSERT INTO verdicts (id, response_id, verdict_class, confidence, judge_model)
            VALUES ('v1', 'resp1', 'clean_refusal', 0.99, 'qwen2.5');
            """
        )
        conn.commit()

        # Invalid verdict class
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO verdicts (id, response_id, verdict_class, confidence, judge_model)
                VALUES ('v2', 'resp1', 'invalid_verdict_label', 0.99, 'qwen2.5');
                """
            )
            conn.commit()
