"""Unified Super-Loop — all phases working together in continuous self-evolution.

This is the highest-level orchestrator that runs indefinitely, connecting:
  Phase 1: Memory System   ← stores everything, learns patterns
  Phase 2: Code Evolution   ← fixes code issues found by other phases
  Phase 3: Agent Swarm      ← spawns agents to validate, execute, review
  Phase 4: Evolutionary Alg ← optimizes parameters via genetic search
  Phase 5: Meta-Engine      ← introspects and decides what to improve

The loop: Monitor → Introspect → Plan → Execute → Evaluate → Learn → Repeat
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from evo_mind.code_evolution.engine import CodeEvolutionEngine
from evo_mind.code_evolution.loop import SelfImprovementLoop
from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.evolution.engine import EvolutionEngine
from evo_mind.evolutionary.engine import EvolutionaryEngine as GeneticEngine
from evo_mind.super_evolution.meta_engine import MetaEngine, SystemMetrics, SuperEvolutionState
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LoopConfig:
    """Configuration for the unified super-loop."""
    # Timing
    meta_interval_seconds: float = 5.0       # How often meta-engine runs
    consolidation_interval_seconds: float = 30.0  # Memory consolidation frequency
    evolution_interval_seconds: float = 60.0      # Rule learning frequency
    code_evolution_interval_seconds: float = 120.0  # Code optimization frequency
    genetic_interval_seconds: float = 300.0         # Genetic algorithm frequency

    # Limits
    max_total_runtime_seconds: float = 3600.0  # 1 hour default
    max_self_modifications: int = 5
    convergence_streak: int = 10  # Iterations without improvement to stop

    # Features
    enable_code_evolution: bool = True
    enable_agent_swarm: bool = True
    enable_genetic_algorithm: bool = True
    enable_self_modification: bool = False  # Off by default (safety)
    enable_auto_consolidation: bool = True


@dataclass
class LoopState:
    """Runtime state of the unified super-loop."""
    running: bool = False
    paused: bool = False
    start_time: float = 0.0
    iteration: int = 0
    total_improvements: int = 0
    last_consolidation: float = 0.0
    last_evolution: float = 0.0
    last_code_evolution: float = 0.0
    last_genetic: float = 0.0
    convergence_streak: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)


class UnifiedSuperLoop:
    """The top-level continuous self-evolution loop.

    Runs all 5 phases in a coordinated, non-blocking cycle. Each phase
    operates on its own schedule. The meta-engine orchestrates when
    additional actions are needed.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: LoopConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or LoopConfig()
        self.state = LoopState()
        self.meta = MetaEngine(store)

        # Background tasks
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> LoopState:
        """Start the unified super-loop. Runs until stopped or converged."""
        self.state.running = True
        self.state.start_time = time.monotonic()
        self.state.iteration = 0

        logger.info("unified_super_loop_started")
        session_id = await self.store.start_session({
            "phase": "unified_super_loop",
            "config": self.config.__dict__,
        })

        # Set up signal handlers for graceful shutdown
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: setattr(self.state, 'running', False))
        except NotImplementedError:
            pass  # Not available on all platforms

        try:
            while self.state.running:
                self.state.iteration += 1
                now = time.monotonic()
                elapsed = now - self.state.start_time

                # Check max runtime
                if elapsed > self.config.max_total_runtime_seconds:
                    logger.info("max_runtime_reached: elapsed=%s", elapsed)
                    break

                # Check convergence
                if self.state.convergence_streak >= self.config.convergence_streak:
                    logger.info("loop_converged: streak=%s", self.state.convergence_streak)
                    break

                try:
                    improved = await self._tick(now, elapsed)
                    if improved:
                        self.state.convergence_streak = 0
                        self.state.total_improvements += 1
                    else:
                        self.state.convergence_streak += 1
                except Exception as e:
                    self.state.errors.append({
                        "iteration": self.state.iteration,
                        "error": str(e),
                        "time": _now(),
                    })
                    logger.error("tick_failed: iteration=%s error=%s", self.state.iteration, e)

                await asyncio.sleep(self.config.meta_interval_seconds)

        finally:
            self.state.running = False
            await self._shutdown()
            await self.store.end_session(
                session_id,
                f"Super-loop: {self.state.iteration} iterations, "
                f"{self.state.total_improvements} improvements",
            )

        return self.state

    async def _tick(self, now: float, elapsed: float) -> bool:
        """One iteration of the super-loop. Returns True if improvement made."""
        improved = False

        # 1. Memory consolidation (frequent)
        if (self.config.enable_auto_consolidation and
                now - self.state.last_consolidation >= self.config.consolidation_interval_seconds):
            improved |= await self._run_consolidation()
            self.state.last_consolidation = now

        # 2. Rule evolution (periodic)
        if now - self.state.last_evolution >= self.config.evolution_interval_seconds:
            improved |= await self._run_evolution()
            self.state.last_evolution = now

        # 3. Code evolution (occasional)
        if (self.config.enable_code_evolution and
                now - self.state.last_code_evolution >= self.config.code_evolution_interval_seconds):
            improved |= await self._run_code_evolution()
            self.state.last_code_evolution = now

        # 4. Genetic optimization (rare)
        if (self.config.enable_genetic_algorithm and
                now - self.state.last_genetic >= self.config.genetic_interval_seconds):
            improved |= await self._run_genetic_algorithm()
            self.state.last_genetic = now

        # 5. Meta-decision (every tick)
        metrics = await self.meta._collect_metrics()
        weaknesses = await self.meta._introspect(metrics)

        if weaknesses:
            action = await self.meta._select_action(metrics, weaknesses)
            success = await self.meta._execute_action(action, metrics)
            improved |= success

        return improved

    async def _run_consolidation(self) -> bool:
        try:
            from evo_mind.consolidation.engine import ConsolidationEngine
            from evo_mind.types import ConsolidationTrigger
            engine = ConsolidationEngine(
                self.store,
                None,  # retrieval — not needed for basic consolidation
                self.store.embedding,
                self.store.vector_store,
                self.store.db,
                config={"min_candidates": 5},
            )
            result = await engine.run_if_needed()
            return result is not None
        except Exception as e:
            logger.debug("consolidation_tick_failed: %s", e)
            return False

    async def _run_evolution(self) -> bool:
        try:
            from evo_mind.evolution.engine import EvolutionEngine
            engine = EvolutionEngine(self.store, self.store.db)
            rules = await engine.evolve()
            return len(rules) > 0
        except Exception as e:
            logger.debug("evolution_tick_failed: %s", e)
            return False

    async def _run_code_evolution(self) -> bool:
        try:
            from pathlib import Path
            from evo_mind.code_evolution.engine import CodeEvolutionEngine

            project_root = Path(__file__).parent.parent
            engine = CodeEvolutionEngine(
                self.store, project_root,
                max_fixes_per_run=3,
                auto_apply=self.config.enable_self_modification,
            )
            result = await engine.optimize()
            return result.fixes_succeeded > 0
        except Exception as e:
            logger.debug("code_evolution_tick_failed: %s", e)
            return False

    async def _run_genetic_algorithm(self) -> bool:
        try:
            from evo_mind.evolutionary.engine import EvolutionaryEngine
            engine = EvolutionaryEngine(
                self.store,
                population_size=20,
                max_generations=10,
            )
            state = await engine.evolve()
            return state.best_fitness > 0.5
        except Exception as e:
            logger.debug("genetic_tick_failed: %s", e)
            return False

    async def _shutdown(self) -> None:
        """Graceful shutdown: cancel background tasks, record final state."""
        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "super_loop_shutdown",
                "iterations": self.state.iteration,
                "total_improvements": self.state.total_improvements,
                "errors": self.state.errors[-5:],
                "uptime_seconds": time.monotonic() - self.state.start_time,
            },
            importance=1.0,
            source="plugin",
            tags=["super-evolution", "shutdown"],
        ))

        logger.info("unified_super_loop_stopped",
                     iterations=self.state.iteration,
                     improvements=self.state.total_improvements)

    def pause(self) -> None:
        self.state.paused = True

    def resume(self) -> None:
        self.state.paused = False

    def stop(self) -> None:
        self.state.running = False

    def get_status(self) -> dict[str, Any]:
        """Get current loop status."""
        return {
            "running": self.state.running,
            "paused": self.state.paused,
            "iteration": self.state.iteration,
            "improvements": self.state.total_improvements,
            "uptime_s": round(time.monotonic() - self.state.start_time, 1) if self.state.start_time else 0,
            "convergence_streak": self.state.convergence_streak,
            "error_count": len(self.state.errors),
            "schedule": {
                "last_consolidation_s": round(time.monotonic() - self.state.last_consolidation, 1) if self.state.last_consolidation else None,
                "last_evolution_s": round(time.monotonic() - self.state.last_evolution, 1) if self.state.last_evolution else None,
            },
        }
