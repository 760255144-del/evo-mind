"""Data models: dataclasses for Memory, SearchQuery, EvolutionRule, etc."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from evo_mind.types import MemorySource, MemoryStatus, MemoryType, RelationType, RuleType


# ---- Memory ----

@dataclass(slots=True)
class MemoryCreate:
    """Input DTO for creating a memory. No ID—timestamps are server-generated."""

    memory_type: MemoryType
    content: dict[str, Any]
    importance: float = 0.5
    session_id: str | None = None
    source: str = "direct"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Memory:
    """Full memory object returned from the store."""

    id: str
    rowid: int  # SQLite rowid for FTS
    memory_type: MemoryType
    content: dict[str, Any]
    content_hash: str
    content_plain: str | None
    embedding_id: str | None
    importance: float
    access_count: int
    last_accessed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    session_id: str | None
    source: str
    status: MemoryStatus
    consolidation_version: int
    consolidation_run_id: str | None
    metadata: dict[str, Any]
    deleted_at: datetime | None = None


# ---- Search ----

@dataclass(slots=True)
class SearchQuery:
    """Multi-strategy search query with configurable weights."""

    query_text: str | None = None
    query_embedding: list[float] | None = None
    memory_types: list[MemoryType] | None = None
    tags: list[str] | None = None
    session_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    min_importance: float = 0.0
    max_results: int = 20
    semantic_weight: float = 1.0
    keyword_weight: float = 0.5
    temporal_weight: float = 0.3
    include_relationships: bool = False
    include_embedding: bool = False


@dataclass(slots=True)
class SearchResult:
    """A single search result with score breakdown."""

    memory: Memory
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    related_memories: list[Memory] = field(default_factory=list)


# ---- Consolidation ----

@dataclass(slots=True)
class ConsolidationResult:
    """Output of a consolidation run."""

    run_id: str
    groups_formed: int
    summaries_generated: int
    duplicates_merged: int
    memories_pruned: int
    patterns_extracted: int
    compression_ratio: float


# ---- Evolution ----

@dataclass(slots=True)
class EvolutionRule:
    """A learned rule from evolutionary pattern detection."""

    id: str
    rule_type: RuleType
    label: str | None
    condition: dict[str, Any]
    action: dict[str, Any]
    confidence: float
    support_count: int
    contradiction_count: int
    status: str
    last_fired_at: datetime | None = None
    last_evaluated_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    superseded_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvolutionMetrics:
    """Fitness metrics recorded over time."""

    id: str
    metric_name: str
    metric_value: float
    recorded_at: datetime
    session_id: str | None = None
    dimension: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- Session ----

@dataclass(slots=True)
class Session:
    """A recording session."""

    id: str
    started_at: datetime
    ended_at: datetime | None = None
    summary: str | None = None
    memory_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---- Errors ----

class EvoMindError(Exception):
    """Base exception for all evo-mind errors."""


class MemoryNotFoundError(EvoMindError):
    """The requested memory was not found."""


class DuplicateMemoryError(EvoMindError):
    """A memory with this content hash already exists."""


class EmbeddingError(EvoMindError):
    """Failed to generate embeddings."""


class ConsolidationError(EvoMindError):
    """Consolidation pipeline failed."""


class EvolutionError(EvoMindError):
    """Evolution pipeline failed."""


class PluginError(EvoMindError):
    """Plugin lifecycle or hook error."""
