"""Taxonomy management for the 5x8 risk and technique matrix."""

from typing import Dict, List, Optional
from pydantic import BaseModel
from strikeshield.config import settings

CellId = str


class TargetBehaviourMap(BaseModel):
    mapping: Dict[str, str]

    def get_for_risk(self, risk_category: str) -> Optional[str]:
        return self.mapping.get(risk_category)


def cell_id(risk_category: str, technique_family: str) -> CellId:
    """Computes canonical cell identifier in risk::technique format."""
    return f"{risk_category}::{technique_family}"


def all_cells() -> List[CellId]:
    """Returns all 40 cells in order-stable risk-major, technique-minor sequence."""
    cells: List[CellId] = []
    for risk in settings.taxonomy.risk_categories:
        for tech in settings.taxonomy.technique_families:
            cells.append(cell_id(risk.id, tech.id))
    return cells


def get_target_behaviour_map() -> TargetBehaviourMap:
    """Builds lookup mapping from risk category to canonical target behaviour."""
    mapping = {risk.id: risk.target_behaviour for risk in settings.taxonomy.risk_categories}
    return TargetBehaviourMap(mapping=mapping)


def weakest_techniques(
    top_k: int = 3,
    asr_by_technique: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Identifies the top_k technique families with the highest Attack Success Rate (ASR).

    Ties are broken deterministically by technique name in alphabetical order.
    Technique families with zero attacks or not in the dictionary are ignored.
    """
    if not asr_by_technique:
        return []

    # Sort descending by ASR, then ascending by technique name for deterministic tie breaking
    sorted_items = sorted(
        asr_by_technique.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [k for k, _ in sorted_items[:top_k]]
