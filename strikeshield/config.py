"""Pydantic settings and YAML configuration loader for StrikeShield."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelRoleConfig(BaseModel):
    model: str
    temperature: float = 0.0
    format: Optional[str] = None
    keep_alive: str = "5m"
    fallback_model: Optional[str] = None


class ModelsConfig(BaseModel):
    target: ModelRoleConfig
    judge: ModelRoleConfig
    patcher: ModelRoleConfig
    embeddings: ModelRoleConfig


class NoiseFloorConfig(BaseModel):
    seeds: int = 3
    target_temperature: float = 0.7
    require_non_overlapping_ranges: bool = True


class GateConfig(BaseModel):
    bpr_threshold: float = 0.90
    regression_budget: int = 0
    asr_must_not_increase: bool = True
    max_heldout_gap: float = 0.05
    noise_floor: NoiseFloorConfig


class RiskCategory(BaseModel):
    id: str
    description: str
    target_behaviour: str


class TechniqueFamily(BaseModel):
    id: str
    description: str


class TaxonomyGridConfig(BaseModel):
    corpus_role: str = "smoke"
    dedup_threshold: float = 0.10
    patch_split_ratio: float = 0.80
    heldout_split_ratio: float = 0.20


class TaxonomyConfig(BaseModel):
    risk_categories: List[RiskCategory]
    technique_families: List[TechniqueFamily]
    grid: TaxonomyGridConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "development"
    llm_backend: str = "mock"  # "mock" | "ollama"
    ollama_base_url: str = "http://localhost:11434"
    database_path: Path = Path("strikeshield.db")
    runs_dir: Path = Path("runs")
    default_seed: int = 42

    # Config directory
    config_dir: Path = Path("config")

    models: ModelsConfig = Field(default_factory=lambda: ModelsConfig(**_load_yaml(Path("config/models.yaml"))))
    gate: GateConfig = Field(
        default_factory=lambda: GateConfig(**_load_yaml(Path("config/gate.yaml"))["gate"])
    )
    taxonomy: TaxonomyConfig = Field(
        default_factory=lambda: TaxonomyConfig(**_load_yaml(Path("config/taxonomy.yaml")))
    )

    def is_mock_allowed(self) -> bool:
        """Enforces that mock backend is strictly prohibited in production."""
        return self.env.lower() != "production"


settings = Settings()
