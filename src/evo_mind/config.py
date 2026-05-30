"""Configuration via Pydantic-settings, loaded from TOML and env vars."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseModel):
    path: Path = Field(default_factory=lambda: Path.home() / ".evo_mind" / "data" / "evo_mind.db")
    pool_size: int = 5
    busy_timeout_ms: int = 5000


class ChromaConfig(BaseModel):
    path: Path = Field(default_factory=lambda: Path.home() / ".evo_mind" / "data" / "chroma")
    collection_name: str = "evo_mind_memories"
    distance_metric: str = "cosine"


class EmbeddingConfig(BaseModel):
    provider: str = "local"
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    cache_size: int = 10000
    normalize: bool = True


class ConsolidationConfig(BaseModel):
    auto_trigger: bool = True
    min_candidates: int = 20
    max_candidates_per_run: int = 500
    similarity_threshold: float = 0.75
    dedup_threshold: float = 0.97
    max_total_memories: int = 100_000
    min_importance_to_keep: float = 0.05
    default_max_age_days: int = 90


class EvolutionConfig(BaseModel):
    auto_trigger: bool = True
    min_support: int = 3
    min_confidence: float = 0.6
    evaluation_interval_hours: int = 24


class CLIConfig(BaseModel):
    theme: str = "dark"
    default_limit: int = 20


class EvoMindConfig(BaseSettings):
    """Root configuration for evo-mind.

    Loads from config/default.toml then overrides with env vars
    prefixed with EVOMIND_ (e.g., EVOMIND_DATABASE__PATH=...).
    """

    model_config = SettingsConfigDict(
        env_prefix="EVOMIND_",
        env_nested_delimiter="__",
        toml_file=str(Path(__file__).parent.parent.parent / "config" / "default.toml"),
    )

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    plugin_paths: list[Path] = Field(default_factory=list)
