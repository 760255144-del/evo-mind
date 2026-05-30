"""SuperEvolutionPlugin — evo-mind plugin enabling full recursive self-improvement.

This is the entry point that wires the super-evolution system into evo-mind.
When loaded, it connects all 5 phases and begins autonomous self-improvement.
"""

from __future__ import annotations

import asyncio
import logging

from evo_mind.core.store import MemoryStore

logger = logging.getLogger(__name__)


class SuperEvolutionPlugin:
    """Evo-mind plugin: enables recursive self-improvement across all phases.

    This is the "master plugin" that orchestrates Phase 1-5.
    """

    name: str = "super-evolution"
    version: str = "0.1.0"

    def __init__(
        self,
        store: MemoryStore | None = None,
        auto_start: bool = False,
        max_runtime_minutes: int = 60,
        enable_self_modification: bool = False,
    ) -> None:
        self._store = store
        self.auto_start = auto_start
        self.max_runtime_minutes = max_runtime_minutes
        self.enable_self_modification = enable_self_modification
        self._loaded = False
        self._loop = None
        self._loop_task: asyncio.Task | None = None

    async def on_load(self) -> None:
        """Start the super-evolution loop if auto_start is enabled."""
        self._loaded = True
        logger.info("super_evolution_plugin_loaded")

        if self.auto_start and self._store:
            from evo_mind.super_evolution.unified_loop import UnifiedSuperLoop, LoopConfig

            config = LoopConfig(
                max_total_runtime_seconds=self.max_runtime_minutes * 60,
                enable_self_modification=self.enable_self_modification,
            )
            self._loop = UnifiedSuperLoop(self._store, config)

            # Start in background
            self._loop_task = asyncio.create_task(self._loop.start())
            logger.info("super_evolution_auto_started")

    async def on_unload(self) -> None:
        """Stop the super-evolution loop."""
        if self._loop:
            self._loop.stop()
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loaded = False
        logger.info("super_evolution_plugin_unloaded")

    # ---- Hook implementations ----

    async def on_evolution_step(self, rules) -> None:
        """When new rules are learned, schedule a meta-evaluation."""
        if self._loop:
            # Signal the loop to run an extra tick
            pass

    async def on_consolidation_complete(self, result) -> None:
        """After consolidation, check if genetic optimization is needed."""
        pass

    async def on_memory_created(self, memory) -> None:
        """Track memory creation for active monitoring."""
        pass

    # ---- Integration ----

    def set_store(self, store: MemoryStore) -> None:
        self._store = store

    async def get_loop(self):
        return self._loop
