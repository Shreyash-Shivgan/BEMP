"""Attack ingestion, schema validation, and promptfoo JSON importer."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from strikeshield.config import settings
from strikeshield.errors import StrikeShieldError
from strikeshield.taxonomy import cell_id, get_target_behaviour_map


class AttackRecord(BaseModel):
    id: Optional[str] = None
    prompt: str = Field(min_length=1)
    risk_category: str
    technique_family: str
    target_behaviour: str
    source: str = "synthetic"
    synthetic_seed: bool = False
    positive_control: bool = False

    @property
    def cell(self) -> str:
        return cell_id(self.risk_category, self.technique_family)


def get_valid_risks() -> set[str]:
    return {r.id for r in settings.taxonomy.risk_categories}


def get_valid_techniques() -> set[str]:
    return {t.id for t in settings.taxonomy.technique_families}


def validate_record(rec: AttackRecord) -> None:
    """Strictly validates taxonomy labels and prompt non-emptiness.

    Rejects unknown labels loudly without coercion.
    """
    if not rec.prompt or not rec.prompt.strip():
        raise StrikeShieldError(
            "Attack record rejected: prompt cannot be empty or whitespace",
            context={"record": rec.model_dump()},
        )

    valid_risks = get_valid_risks()
    if rec.risk_category not in valid_risks:
        raise StrikeShieldError(
            f"Attack record rejected: unknown risk category '{rec.risk_category}'",
            context={"valid_risks": sorted(list(valid_risks)), "record": rec.model_dump()},
        )

    valid_techs = get_valid_techniques()
    if rec.technique_family not in valid_techs:
        raise StrikeShieldError(
            f"Attack record rejected: unknown technique family '{rec.technique_family}'",
            context={"valid_techniques": sorted(list(valid_techs)), "record": rec.model_dump()},
        )

    if not rec.target_behaviour or not rec.target_behaviour.strip():
        raise StrikeShieldError(
            "Attack record rejected: missing target_behaviour specification",
            context={"record": rec.model_dump()},
        )


def load_jsonl(path: Path | str) -> List[AttackRecord]:
    """Reads a JSONL file of attack records, filtering headers and validating each row."""
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {target_path}")

    records: List[AttackRecord] = []
    is_seed_file = False

    with open(target_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise StrikeShieldError(
                    f"Malformed JSONL line {line_no} in {target_path.name}",
                    context={"line_no": line_no, "error": str(e)},
                ) from e

            # Header inspection
            if data.get("_header") is True:
                if data.get("synthetic_seed") is True:
                    is_seed_file = True
                continue

            if "synthetic_seed" not in data and is_seed_file:
                data["synthetic_seed"] = True

            record = AttackRecord(**data)
            validate_record(record)
            records.append(record)

    return records


def import_promptfoo(path: Path | str) -> List[AttackRecord]:
    """Parses promptfoo JSON evaluation export into validated AttackRecord instances.

    Target Promptfoo JSON Schema:
    {
      "results": {
        "results": [
          {
            "prompt": { "raw": "attack prompt text..." },
            "vars": {
              "risk_category": "system-prompt extraction",
              "technique_family": "role_play",
              "target_behaviour": "..." (optional, fallback to taxonomy)
            },
            "metadata": { ... }
          }
        ]
      }
    }
    """
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Promptfoo export file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_results = data.get("results", {}).get("results", [])
    if not raw_results and isinstance(data.get("results"), list):
        raw_results = data.get("results")

    tb_map = get_target_behaviour_map()
    records: List[AttackRecord] = []

    for idx, item in enumerate(raw_results):
        prompt_obj = item.get("prompt", {})
        prompt_text = prompt_obj.get("raw") if isinstance(prompt_obj, dict) else str(prompt_obj)

        vars_dict: Dict[str, Any] = item.get("vars", {})
        metadata_dict: Dict[str, Any] = item.get("metadata", {})

        risk = vars_dict.get("risk_category") or metadata_dict.get("risk_category")
        tech = vars_dict.get("technique_family") or metadata_dict.get("technique_family")
        target_behaviour = (
            vars_dict.get("target_behaviour")
            or metadata_dict.get("target_behaviour")
            or (tb_map.get_for_risk(risk) if risk else None)
        )

        rec = AttackRecord(
            id=f"promptfoo_{idx:04d}",
            prompt=prompt_text or "",
            risk_category=risk or "",
            technique_family=tech or "",
            target_behaviour=target_behaviour or "",
            source="promptfoo",
            synthetic_seed=False,
        )
        validate_record(rec)
        records.append(rec)

    return records
