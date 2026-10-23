"""Tests for taxonomy matrix, cell indexing, and weakest techniques ranking."""

from strikeshield.taxonomy import all_cells, cell_id, weakest_techniques


def test_cell_id_format() -> None:
    assert cell_id("risk_a", "tech_b") == "risk_a::tech_b"


def test_all_cells_count_and_order() -> None:
    cells = all_cells()
    assert len(cells) == 40
    # Exactly 40 unique cells
    assert len(set(cells)) == 40

    # Risk-major, technique-minor order check
    assert cells[0] == "system-prompt extraction::role_play"
    assert cells[7] == "system-prompt extraction::multi_turn_escalation"
    assert cells[8] == "out-of-scope task::role_play"


def test_weakest_techniques_ranking_and_tie_breaking() -> None:
    asr_map = {
        "role_play": 0.50,
        "hypothetical": 0.85,
        "authority_claim": 0.85,  # Tied with hypothetical
        "encoding": 0.10,
        "language_switch": 0.40,
    }

    # Top 3 should be authority_claim (0.85, alpha first), hypothetical (0.85), role_play (0.50)
    top3 = weakest_techniques(top_k=3, asr_by_technique=asr_map)
    assert top3 == ["authority_claim", "hypothetical", "role_play"]

    # Top 2
    top2 = weakest_techniques(top_k=2, asr_by_technique=asr_map)
    assert top2 == ["authority_claim", "hypothetical"]


def test_weakest_techniques_empty_and_zero() -> None:
    assert weakest_techniques(top_k=3, asr_by_technique=None) == []
    assert weakest_techniques(top_k=3, asr_by_technique={}) == []
