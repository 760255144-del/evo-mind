"""Type system: enums and Protocols for swappable backends."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


# ---- Enums ----

class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    FEEDBACK = "feedback"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    ARCHIVED = "archived"
    PRUNED = "pruned"


class RelationType(StrEnum):
    SUPERSEDES = "supersedes"
    CONTRADICTS = "contradicts"
    DERIVES_FROM = "derives_from"
    REINFORCES = "reinforces"
    REFERENCES = "references"
    GENERALIZES = "generalizes"
    CORRECTS = "corrects"


class RuleType(StrEnum):
    CORRECTION_PATTERN = "correction_pattern"
    STRATEGY_HEURISTIC = "strategy_heuristic"
    INFERRED_KNOWLEDGE = "inferred_knowledge"
    PROCEDURAL_TEMPLATE = "procedural_template"


class ConsolidationTrigger(StrEnum):
    MANUAL = "manual"
    THRESHOLD = "threshold"
    SCHEDULE = "schedule"


class ConsolidationStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RuleStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"


class MemorySource(StrEnum):
    DIRECT = "direct"
    CONSOLIDATION = "consolidation"
    DEDUCTION = "deduction"
    PLUGIN = "plugin"
    IMPORT = "import"


# ---- Protocols ----

@runtime_checkable
class EmbeddingEngine(Protocol):
    """Protocol for embedding backends (local model, API, mock)."""

    @property
    def dimension(self) -> int: ...

    async def encode(self, text: str) -> list[float]: ...

    async def encode_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends (ChromaDB, FAISS, etc.)."""

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]] | None = None,
        documents: list[str] | None = None,
    ) -> None: ...

    async def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, object] | None = None,
    ) -> tuple[list[str], list[float]]: ...

    async def delete(self, ids: list[str]) -> None: ...

    async def count(self) -> int: ...

    async def get_embeddings(self, ids: list[str]) -> list[list[float]] | None: ...


@runtime_checkable
class Summarizer(Protocol):
    """Protocol for summarization backends (heuristic, LLM, etc.)."""

    async def summarize(
        self, texts: list[str], max_length: int = 200
    ) -> str: ...


@runtime_checkable
class Plugin(Protocol):
    """Protocol that all plugins must satisfy."""

    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...

    async def on_load(self) -> None: ...
    async def on_unload(self) -> None: ...
