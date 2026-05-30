"""EvolutionaryPlugin — evo-mind plugin for genetic algorithm optimization.

Triggers evolution on consolidation completion to continuously
optimize strategy parameters.
"""

from __future__ import annotations

import logging

from evo_mind.core.store import MemoryStore

logger = logging.getLogger(__name__)


class EvolutionaryPlugin:
    """Evo-mind plugin: runs genetic algorithm to optimize strategy weights.

    Implements on_evolution_step to apply evolved parameters.
    """

    name: str = "evolutionary-optimizer"
    version: str = "0.1.0"

    def __init__(
        self,
        store: MemoryStore | None = None,
        population_size: int = 50,
        max_generations: int = 50,
        auto_evolve: bool = True,
    ) -> None:
        self._store = store
        self.population_size = population_size
        self.max_generations = max_generations
        self.auto_evolve = auto_evolve
        self._loaded = False
        self._best_params: dict[str, float] = {}

    async def on_load(self) -> None:
        self._loaded = True
        logger.info("evolutionary_plugin_loaded")

    async def on_unload(self) -> None:
        self._loaded = False
        logger.info("evolutionary_plugin_unloaded")

    async def on_consolidation_complete(self, result) -> None:
        """After consolidation, run evolution to optimize parameters."""
        if not self.auto_evolve or not self._store:
            return

        from evo_mind.evolutionary.engine import EvolutionaryEngine

        engine = EvolutionaryEngine(
            self._store,
            population_size=self.population_size,
            max_generations=self.max_generations,
        )

        state = await engine.evolve()
        self._best_params = engine.get_best_params()

        logger.info(
            "evolutionary_optimization_complete",
            generations=state.generation,
            best_fitness=f"{state.best_fitness:.4f}",
            best_params=self._best_params,
        )

    async def on_evolution_step(self, rules) -> None:
        """When new rules are learned, run a quick evolution to adapt."""
        if not self._store:
            return

        # Quick evolution with smaller population
        from evo_mind.evolutionary.engine import EvolutionaryEngine

        engine = EvolutionaryEngine(
            self._store,
            population_size=max(10, self.population_size // 5),
            max_generations=max(5, self.max_generations // 10),
        )

        state = await engine.evolve()
        self._best_params = engine.get_best_params()

    def set_store(self, store: MemoryStore) -> None:
        self._store = store

    @property
    def best_params(self) -> dict[str, float]:
        return self._best_params
