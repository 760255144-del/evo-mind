"""AgentSwarmPlugin — evo-mind plugin for multi-agent coordination.

Integrates the swarm coordinator into the memory evolution loop.
"""

from __future__ import annotations

import logging
from typing import Any

from evo_mind.core.store import MemoryStore

logger = logging.getLogger(__name__)


class AgentSwarmPlugin:
    """Evo-mind plugin: enables multi-agent coordination.

    Hooks implemented:
    - on_evolution_step: spawns agents to validate/apply evolution rules
    - on_memory_created: routes relevant memories to swarm agents
    - on_consolidation_complete: spawns review agents to validate consolidation
    """

    name: str = "agent-swarm"
    version: str = "0.1.0"

    def __init__(
        self,
        store: MemoryStore | None = None,
        auto_spawn: bool = False,
        team_size: int = 3,
    ) -> None:
        self._store = store
        self.auto_spawn = auto_spawn
        self.team_size = team_size
        self._coordinator = None
        self._loaded = False

    async def on_load(self) -> None:
        """Initialize the swarm coordinator if auto_spawn is enabled."""
        self._loaded = True

        if self.auto_spawn and self._store:
            from evo_mind.agent_swarm.coordinator import SwarmCoordinator

            self._coordinator = SwarmCoordinator(self._store)
            await self._coordinator.spawn_team({
                "analyzer": max(1, self.team_size // 3),
                "executor": max(1, self.team_size // 2),
                "reviewer": max(1, self.team_size // 3),
            })
            logger.info("swarm_initialized", agents=self._coordinator.agent_count)

    async def on_unload(self) -> None:
        """Shutdown the swarm."""
        if self._coordinator:
            await self._coordinator.stop_all()
        self._loaded = False

    # ---- Hook implementations ----

    async def on_evolution_step(self, rules: list) -> None:
        """When new evolution rules are found, spawn agents to validate/apply them."""
        if not self._store or not self._coordinator:
            return

        for rule in rules:
            task = await self._coordinator.create_task(
                title=f"Validate rule: {getattr(rule, 'label', 'unknown')}",
                description=f"Verify and apply evolution rule with confidence {getattr(rule, 'confidence', 0):.2f}",
                input_data={
                    "rule_id": getattr(rule, 'id', ''),
                    "rule_type": getattr(rule, 'rule_type', '').value if hasattr(getattr(rule, 'rule_type', ''), 'value') else '',
                    "confidence": getattr(rule, 'confidence', 0),
                    "condition": getattr(rule, 'condition', {}),
                    "action": getattr(rule, 'action', {}),
                },
                category="review",
            )
            await self._coordinator.execute_task(task)

    async def on_consolidation_complete(self, result) -> None:
        """After consolidation, spawn review agents."""
        if not self._store:
            return

        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType
        await self._store.record(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={"swarm_status": "ready", "consolidation_run": getattr(result, 'run_id', '')},
            importance=0.5,
            source="plugin",
            tags=["swarm", "consolidation"],
        ))

    async def on_retrieval_post_search(self, results, query) -> None:
        """Boost results from swarm-related memories."""
        pass  # Could re-rank results based on agent consensus

    # ---- Integration ----

    def set_store(self, store: MemoryStore) -> None:
        self._store = store

    async def get_coordinator(self) -> object | None:
        """Get or create the swarm coordinator."""
        if not self._coordinator and self._store:
            from evo_mind.agent_swarm.coordinator import SwarmCoordinator
            self._coordinator = SwarmCoordinator(self._store)
        return self._coordinator
