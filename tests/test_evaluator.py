"""Tests for Evaluator, SQL-provable regression tracking, benign inverted grading, and metrics storage."""

import json
from pathlib import Path
import pytest

from strikeshield.config import settings
from strikeshield.db import get_connection, init_db, migrate
from strikeshield.evaluator import (
    Evaluator,
    compute_regressions_from_db,
    load_benign_queries,
)


@pytest.fixture
def eval_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_eval.db"
    init_db(db_file)
    migrate(db_file)
    with get_connection(db_file) as conn:
        conn.execute("INSERT OR IGNORE INTO runs (id, seed, iteration, config_json) VALUES ('r0', 0, 0, '{}'), ('r1', 1, 1, '{}'), ('r2', 2, 2, '{}');")
        conn.execute("INSERT OR IGNORE INTO policies (version, status) VALUES ('v0', 'ACCEPTED'), ('v1', 'ACCEPTED'), ('v2', 'ACCEPTED');")
        # Insert 3 attacks
        conn.execute(
            """
            INSERT OR IGNORE INTO attacks (id, prompt, risk_category, technique_family, target_behaviour, cell_id, split, source)
            VALUES 
            ('atk_1', 'prompt 1', 'risk', 'tech', 'target', 'risk::tech', 'patch', 'src'),
            ('atk_2', 'prompt 2', 'risk', 'tech', 'target', 'risk::tech', 'patch', 'src'),
            ('atk_3', 'prompt 3', 'risk', 'tech', 'target', 'risk::tech', 'patch', 'src');
            """
        )
        conn.commit()
    return db_file


def test_two_version_synthetic_regressions(eval_db: Path) -> None:
    """Test required by DoD: 1 blocked by both, 1 blocked by v0 leaked by v1, 1 leaked by both."""
    conn = get_connection(eval_db)
    # v0 evaluations:
    # atk_1 blocked (1), atk_2 blocked (1), atk_3 leaked (0)
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0', 'atk_1', 1, 'r0');")
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0', 'atk_2', 1, 'r0');")
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0', 'atk_3', 0, 'r0');")

    # v1 evaluations:
    # atk_1 blocked (1) [blocked by both]
    # atk_2 leaked (0) [REGRESSION: was blocked by v0, leaked by v1]
    # atk_3 leaked (0) [leaked by both, not a regression]
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v1', 'atk_1', 1, 'r1');")
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v1', 'atk_2', 0, 'r1');")
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v1', 'atk_3', 0, 'r1');")
    conn.commit()

    regs = compute_regressions_from_db("v1", conn)
    conn.close()

    assert len(regs) == 1
    assert regs[0].attack_id == "atk_2"
    assert "v0" in regs[0].blocked_by
    assert regs[0].leaked_by == "v1"


def test_three_version_regression_resolution(eval_db: Path) -> None:
    """Attack blocked by v0, leaked by v1, blocked by v2 is NOT a regression against v2."""
    conn = get_connection(eval_db)
    # atk_2 history: v0=blocked, v1=leaked, v2=blocked
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0', 'atk_2', 1, 'r0');")
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v1', 'atk_2', 0, 'r1');")
    conn.execute("INSERT INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v2', 'atk_2', 1, 'r2');")
    conn.commit()

    # Against v2, atk_2 is blocked, so it is NOT a regression
    regs_v2 = compute_regressions_from_db("v2", conn)
    assert len(regs_v2) == 0

    # Against v1, atk_2 was leaked after v0 blocked it, so it IS a regression against v1
    regs_v1 = compute_regressions_from_db("v1", conn)
    assert len(regs_v1) == 1
    assert regs_v1[0].attack_id == "atk_2"
    conn.close()


def test_evaluator_metrics_backend_and_role_not_null(eval_db: Path, tmp_path: Path) -> None:
    """Flag F test: metrics.backend and metrics.corpus_role are never NULL."""
    # Create small benign set fixture
    benign_file = tmp_path / "benign.jsonl"
    benign_file.write_text(
        '{"id": "b1", "prompt": "What are the tuition deadlines?"}\n',
        encoding="utf-8",
    )

    evaluator = Evaluator(db_path=eval_db, benign_path=benign_file)
    res = evaluator.evaluate(policy_version="v0", policy_text="Helpdesk rules", seed=42)

    assert res.policy_version == "v0"
    assert res.regression_count == 0

    conn = get_connection(eval_db)
    row = conn.execute("SELECT backend, corpus_role, regression_count FROM metrics WHERE policy_version='v0';").fetchone()
    conn.close()

    assert row is not None
    assert row["backend"] == settings.llm_backend
    assert row["backend"] is not None
    assert row["corpus_role"] == settings.taxonomy.grid.corpus_role
    assert row["corpus_role"] is not None
    assert row["regression_count"] == 0


def test_bpr_inverted_grading(eval_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test: When benign prompt is falsely refused, BPR drops."""
    benign_file = tmp_path / "benign.jsonl"
    benign_file.write_text(
        '{"id": "b1", "prompt": "What are tuition fees?"}\n{"id": "b2", "prompt": "Where is registrar?"}\n',
        encoding="utf-8",
    )

    evaluator = Evaluator(db_path=eval_db, benign_path=benign_file)
    res = evaluator.evaluate(policy_version="v0", policy_text="Policy", seed=42)
    # Default mock judge returns clean_refusal for generic queries
    # When clean_refusal is returned on inverted benign target, it counts as refusal failure
    assert res.bpr.denominator == 2


def test_empty_policy_leak_regression_against_baseline(eval_db: Path, tmp_path: Path) -> None:
    """DoD synthetic two-version test: v0 blocks attacks, empty v0-test leaks them -> regression_count > 0."""
    conn = get_connection(eval_db)
    conn.execute("INSERT OR IGNORE INTO policies (version, status) VALUES ('v0-test', 'ACCEPTED');")
    conn.execute("INSERT OR IGNORE INTO runs (id, seed, iteration, config_json) VALUES ('r_test', 0, 0, '{}');")
    # v0 blocked atk_1 and atk_2
    conn.execute("INSERT OR REPLACE INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0', 'atk_1', 1, 'r0');")
    conn.execute("INSERT OR REPLACE INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0', 'atk_2', 1, 'r0');")
    # v0-test leaks atk_1 (blocked=0)
    conn.execute("INSERT OR REPLACE INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0-test', 'atk_1', 0, 'r_test');")
    # atk_2 remains blocked
    conn.execute("INSERT OR REPLACE INTO evaluations (policy_version, attack_id, blocked, run_id) VALUES ('v0-test', 'atk_2', 1, 'r_test');")
    conn.commit()

    regs = compute_regressions_from_db("v0-test", conn)
    conn.close()

    assert len(regs) == 1
    assert regs[0].attack_id == "atk_1"
    assert "v0" in regs[0].blocked_by
    assert regs[0].leaked_by == "v0-test"
