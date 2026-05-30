"""Hook specifications for the evo-mind plugin system.

Plugins implement any subset of these hooks via pluggy.
Uses the same framework as pytest's plugin architecture.
"""

from __future__ import annotations

import pluggy

hookspec = pluggy.HookspecMarker("evo_mind")


class EvoMindHooks:
    """Hook specifications for evo-mind plugins."""

    # ---- Memory Lifecycle ----

    @hookspec
    def on_memory_created(self, memory) -> None:
        """Called after a new memory is persisted and embedded."""

    @hookspec(firstresult=True)
    def on_memory_pre_create(self, memory_create) -> object | None:
        """Called BEFORE a memory is persisted.
        Return a modified MemoryCreate, or None to proceed unchanged.
        """

    @hookspec
    def on_memory_accessed(self, memory_id: str) -> None:
        """Called when a memory's access_count is incremented."""

    @hookspec
    def on_memory_pruned(self, memory) -> None:
        """Called after a memory is hard-deleted."""

    # ---- Retrieval ----

    @hookspec(firstresult=True)
    def on_retrieval_pre_search(self, query) -> object | None:
        """Modify or replace the SearchQuery before execution."""

    @hookspec
    def on_retrieval_post_search(self, results, query) -> None:
        """Modify search results in-place before returning to caller."""

    # ---- Consolidation ----

    @hookspec
    def on_consolidation_complete(self, result) -> None:
        """Called after a consolidation run finishes."""

    @hookspec(firstresult=True)
    def on_get_summarizer(self) -> object | None:
        """Provide a custom Summarizer implementation."""

    # ---- Evolution ----

    @hookspec
    def on_evolution_step(self, rules) -> None:
        """Called after an evolution pass completes, with new/modified rules."""

    @hookspec
    def on_rule_deprecated(self, rule) -> None:
        """Called when a rule's confidence drops below threshold."""

    # ---- Session ----

    @hookspec
    def on_session_start(self, session_id: str) -> None: ...

    @hookspec
    def on_session_end(self, session_id: str, summary: str | None) -> None: ...

    # ---- Embedding ----

    @hookspec(firstresult=True)
    def on_get_embedding_provider(self) -> object | None:
        """Provide a custom EmbeddingEngine.
        If a plugin returns one, it replaces the default sentence-transformers backend.
        """
