"""Tests for threat matrix coverage calculations and grid reporting."""

from pathlib import Path
import pytest

from strikeshield.attack_source import AttackRecord
from strikeshield.corpus import build_corpus
from strikeshield.coverage import get_coverage_report
from strikeshield.db import init_db, migrate


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_coverage.db"
    init_db(db_file)
    migrate(db_file)

    records = [
        AttackRecord(
            prompt="Attack in cell 1",
            risk_category="system-prompt extraction",
            technique_family="role_play",
            target_behaviour="target",
        ),
        AttackRecord(
            prompt="Attack in cell 2",
            risk_category="harmful instruction",
            technique_family="hypothetical",
            target_behaviour="target",
        ),
    ]
    build_corpus(records, db_path=db_file)
    return db_file


def test_coverage_metrics(populated_db: Path) -> None:
    report = get_coverage_report(populated_db)
    assert report.cells_total == 40
    assert report.cells_filled == 2
    assert report.coverage_ratio == 2 / 40
    assert len(report.weakest_cells) == 40  # Both 0 and 1 counts are in <= 1
    assert report.mean_pairwise_distance >= 0.0


def test_coverage_markdown_table_structure(populated_db: Path) -> None:
    report = get_coverage_report(populated_db)
    table_str = report.format_table()
    assert "| Risk Category \\ Technique |" in table_str
    assert "system-prompt extraction" in table_str
    assert "role_play" in table_str
    assert "Grid Coverage Summary:" in table_str
    assert "Cells Filled: 2 / 40" in table_str
