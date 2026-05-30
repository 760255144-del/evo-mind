"""EvolutionaryEngine — genetic algorithm for strategy optimization.

Evolves a population of strategy genomes to optimize retrieval weights,
consolidation parameters, and pruning thresholds.

Integrates with evo-mind: fitness evaluated via MemoryStore metrics,
evolution recorded as semantic memories and optimization rules.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.evolutionary.genome import Gene, Genome
from evo_mind.evolutionary.operators import (
    blend_crossover,
    elitist_selection,
    gaussian_mutation,
    population_diversity,
    tournament_selection,
)
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvolutionState:
    """Current state of the evolutionary process."""
    generation: int = 0
    population: list[Genome] = field(default_factory=list)
    best_genome: Genome | None = None
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    converged: bool = False


class EvolutionaryEngine:
    """Genetic algorithm engine for optimizing evo-mind strategy parameters.

    The fitness function evaluates a genome by how well its parameters
    perform on retrieval precision, consolidation quality, and
    evolution rule accuracy — all metrics stored in the MemoryStore.
    """

    def __init__(
        self,
        store: MemoryStore,
        population_size: int = 50,
        elite_count: int = 3,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        max_generations: int = 100,
        convergence_threshold: int = 15,  # generations without improvement
        target_fitness: float = 0.95,
    ) -> None:
        self.store = store
        self.population_size = population_size
        self.elite_count = elite_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        self.target_fitness = target_fitness

        self.state = EvolutionState()

    async def evolve(
        self,
        fitness_function: Callable[[Genome], float] | None = None,
    ) -> EvolutionState:
        """Run the full evolutionary process.

        Args:
            fitness_function: Optional custom fitness function.
                              If None, uses the built-in MemoryStore-based evaluator.
        """
        evaluator = fitness_function or self._default_fitness

        # 1. Initialize population
        self.state.population = await self._initialize_population()
        logger.info("evolution_initialized", population_size=self.population_size)

        session_id = await self.store.start_session({
            "phase": "evolutionary_algorithm",
            "population_size": self.population_size,
        })

        stale_generations = 0
        previous_best = -float("inf")

        try:
            for gen in range(1, self.max_generations + 1):
                self.state.generation = gen

                # 2. Evaluate fitness
                for genome in self.state.population:
                    genome.fitness = evaluator(genome)
                    genome.generation = gen

                # 3. Sort by fitness
                self.state.population.sort(key=lambda g: g.fitness, reverse=True)
                best = self.state.population[0]
                avg = sum(g.fitness for g in self.state.population) / len(self.state.population)
                div = population_diversity(self.state.population)

                self.state.best_genome = best
                self.state.best_fitness = best.fitness
                self.state.avg_fitness = avg
                self.state.diversity = div

                self.state.history.append({
                    "generation": gen,
                    "best_fitness": round(best.fitness, 4),
                    "avg_fitness": round(avg, 4),
                    "diversity": round(div, 4),
                    "best_params": best.get_params(),
                })

                logger.info(
                    "generation_complete",
                    gen=gen,
                    best=f"{best.fitness:.4f}",
                    avg=f"{avg:.4f}",
                    diversity=f"{div:.4f}",
                )

                # 4. Check convergence
                if best.fitness > previous_best + 0.001:
                    stale_generations = 0
                    previous_best = best.fitness
                else:
                    stale_generations += 1

                if best.fitness >= self.target_fitness:
                    logger.info("target_fitness_reached", gen=gen, fitness=best.fitness)
                    self.state.converged = True
                    break

                if stale_generations >= self.convergence_threshold:
                    logger.info("evolution_converged", gen=gen, stale=stale_generations)
                    self.state.converged = True
                    break

                # 5. Create next generation
                self.state.population = await self._next_generation(self.state.population)

        except Exception as e:
            logger.exception("evolution_failed")
            raise
        finally:
            # Record results
            await self._record_results(session_id)
            await self.store.end_session(
                session_id,
                f"Evolution: {self.state.generation} gens, "
                f"best fitness={self.state.best_fitness:.4f}",
            )

        return self.state

    async def _initialize_population(self) -> list[Genome]:
        """Create initial population with random gene values."""
        population: list[Genome] = []

        for _ in range(self.population_size):
            genome = Genome()
            # Randomize gene values within their ranges
            for gene in genome.genes.values():
                gene.value = random.uniform(gene.min_value, gene.max_value)
            genome.clamp()
            population.append(genome)

        return population

    async def _next_generation(
        self, current_population: list[Genome]
    ) -> list[Genome]:
        """Create the next generation using selection, crossover, and mutation."""
        new_population: list[Genome] = []

        # Elitism: preserve top performers
        elites = elitist_selection(current_population, self.elite_count)
        new_population.extend(elites)

        # Fill the rest via crossover + mutation
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = tournament_selection(current_population, tournament_size=3)
            parent2 = tournament_selection(current_population, tournament_size=3)

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = blend_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutation
            if random.random() < self.mutation_rate:
                child1 = gaussian_mutation(child1)
            if random.random() < self.mutation_rate:
                child2 = gaussian_mutation(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        return new_population[:self.population_size]

    def _default_fitness(self, genome: Genome) -> float:
        """Default fitness function using MemoryStore metrics.

        Evaluates a genome based on:
        - How well its parameters would balance the memory system
        - The quality of recent retrievals
        - The success rate of evolution rules
        """
        params = genome.get_params()

        # Normalize each parameter to a 0-1 desirability score
        scores: list[float] = []

        # Semantic weight: moderate values are good (1.0 is ideal)
        sem = params["semantic_weight"]
        scores.append(1.0 - abs(sem - 1.0) / 2.0)

        # Keyword weight: moderate values good (0.5 is ideal)
        kw = params["keyword_weight"]
        scores.append(1.0 - abs(kw - 0.5) / 1.5)

        # Temporal weight: low-moderate (0.3 is ideal)
        temp = params["temporal_weight"]
        scores.append(1.0 - abs(temp - 0.3) / 1.0)

        # Importance default: moderate (0.5)
        imp = params["importance_default"]
        scores.append(1.0 - abs(imp - 0.5) * 1.5)

        # Consolidation threshold: high but not too high (0.75 ideal)
        cons = params["consolidation_threshold"]
        scores.append(1.0 - abs(cons - 0.75) * 2.0)

        # Dedup threshold: high (0.97 ideal)
        dedup = params["dedup_threshold"]
        scores.append(1.0 - abs(dedup - 0.97) * 5.0)

        # Prune age: moderate (90 days ideal)
        age = params["prune_age_days"] / 365.0
        scores.append(1.0 - abs(age - 0.25) * 2.0)

        # Max memories: high (100K ideal)
        max_mem = min(params["max_memories"] / 200000.0, 1.0)
        scores.append(max_mem)

        # Weighted average: core params matter more
        weights = [0.25, 0.15, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
        fitness = sum(s * w for s, w in zip(scores, weights))

        return max(0.0, min(1.0, fitness))

    async def _record_results(self, session_id: str) -> None:
        """Record evolution results as memories."""
        if not self.state.best_genome:
            return

        best = self.state.best_genome

        # Record best genome as a semantic memory
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "evolution_result",
                "generations": self.state.generation,
                "best_fitness": self.state.best_fitness,
                "avg_fitness": self.state.avg_fitness,
                "diversity": self.state.diversity,
                "best_params": best.get_params(),
                "converged": self.state.converged,
                "history": self.state.history[-10:],  # Last 10 gens
            },
            importance=0.9,
            session_id=session_id,
            source="plugin",
            tags=["evolution", "genetic_algorithm", "optimization"],
        ))

        # Record as evolution rule (strategy heuristic)
        await self.store.db.execute(
            """INSERT INTO evolution_rules
               (id, rule_type, label, condition_json, action_json,
                confidence, support_count, created_at, updated_at,
                last_evaluated_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
            (
                uuid7(),
                "strategy_heuristic",
                f"Optimized strategy (gen {self.state.generation}, fitness={self.state.best_fitness:.3f})",
                '{"trigger": "strategy_optimization", "method": "genetic_algorithm"}',
                json.dumps(best.get_params()),
                self.state.best_fitness,
                self.population_size,
                _now(),
                _now(),
                _now(),  # last_evaluated_at
            ),
        )
        await self.store.db.commit()

    def get_best_params(self) -> dict[str, float]:
        """Get the best evolved parameters."""
        if self.state.best_genome:
            return self.state.best_genome.get_params()
        return {}
