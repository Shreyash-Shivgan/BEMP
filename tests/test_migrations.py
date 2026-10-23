"""Tests for migration runner and Phase 3 schema extensions."""

from pathlib import Path
from strikeshield.db import get_connection, init_db, migrate, get_current_schema_version


def test_migrations_up_to_version_2(tmp_path: Path) -> None:
    db_file = tmp_path / "test_mig.db"
    init_db(db_file)
    v = migrate(db_file)
    assert v == 2

    # Verify columns in metrics and responses
    with get_connection(db_file) as conn:
        assert get_current_schema_version(conn) == 2

        metrics_cols = [row["name"] for row in conn.execute("PRAGMA table_info(metrics);").fetchall()]
        assert "backend" in metrics_cols
        assert "corpus_role" in metrics_cols

        responses_cols = [row["name"] for row in conn.execute("PRAGMA table_info(responses);").fetchall()]
        assert "seed" in responses_cols
        assert "temperature" in responses_cols

        # Verify new audit tables exist
        tables = [row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        assert "judge_calls" in tables
        assert "evidence_mismatches" in tables

    # Second migrate call is a no-op
    v2 = migrate(db_file)
    assert v2 == 2
