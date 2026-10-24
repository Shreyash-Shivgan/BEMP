"""Tests for migration runner and Phase 3 schema extensions."""

from pathlib import Path
from strikeshield.db import get_connection, init_db, migrate, get_current_schema_version


def test_migrations_up_to_version_6(tmp_path: Path) -> None:
    db_file = tmp_path / "test_mig.db"
    init_db(db_file)
    v = migrate(db_file)
    assert v == 6

    # Second migration must be idempotent
    assert migrate(db_file) == 6

    # Verify columns in metrics and responses
    with get_connection(db_file) as conn:
        assert get_current_schema_version(conn) == 6

        metrics_cols = [row["name"] for row in conn.execute("PRAGMA table_info(metrics);").fetchall()]
        assert "backend" in metrics_cols
        assert "corpus_role" in metrics_cols
        assert "judge_health_json" in metrics_cols
        assert "noise_floor_k" in metrics_cols
        assert "regressed_attack_ids_json" in metrics_cols
        assert "seeds_run_json" in metrics_cols
        assert "asr_all_range_low" in metrics_cols
        assert "asr_all_range_high" in metrics_cols
        assert "noise_overlap_asr" in metrics_cols

        responses_cols = [row["name"] for row in conn.execute("PRAGMA table_info(responses);").fetchall()]
        assert "seed" in responses_cols
        assert "temperature" in responses_cols

        policies_cols = [row["name"] for row in conn.execute("PRAGMA table_info(policies);").fetchall()]
        assert "candidate_status" in policies_cols
        assert "patcher_model" in policies_cols
        assert "patcher_fallback_used" in policies_cols
        assert "rejection_reason" in policies_cols

        # Verify new audit tables exist
        tables = [row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        assert "judge_calls" in tables
        assert "evidence_mismatches" in tables

    # Second migrate call is a no-op
    v2 = migrate(db_file)
    assert v2 == 6
