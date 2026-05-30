"""Self-Improvement Loop — autonomous analyze→fix→test→learn cycle."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evo_mind.code_evolution.engine import CodeEvolutionEngine, OptimizationResult
from evo_mind.core.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class ImprovementStats:
    """Cumulative improvement statistics over multiple iterations."""
    total_iterations: int = 0
    total_issues_found: int = 0
    total_fixes_applied: int = 0
    total_fixes_succeeded: int = 0
    total_fixes_failed: int = 0
    total_duration_seconds: float = 0.0
    improvements: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.improvements is None:
            self.improvements = []

    @property
    def success_rate(self) -> float:
        total = self.total_fixes_applied
        if total == 0:
            return 0.0
        return self.total_fixes_succeeded / total

    @property
    def avg_duration(self) -> float:
        if self.total_iterations == 0:
            return 0.0
        return self.total_duration_seconds / self.total_iterations


class SelfImprovementLoop:
    """Orchestrates repeated optimization cycles with convergence detection.

    The loop:
    1. Runs CodeEvolutionEngine.optimize()
    2. Checks if fixes improved things (tests pass, fewer issues)
    3. Records the improvement delta
    4. Repeats until convergence (no new issues found, or all fixes fail)
    5. Records cumulative learning as evolution rules
    """

    def __init__(
        self,
        engine: CodeEvolutionEngine,
        store: MemoryStore,
        max_iterations: int = 10,
        convergence_threshold: int = 2,  # stop after N iterations with no new fixes
        sleep_between_iterations: float = 1.0,
    ) -> None:
        self.engine = engine
        self.store = store
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.sleep_between = sleep_between_iterations
        self.stats = ImprovementStats()

    async def run(self) -> ImprovementStats:
        """Run the self-improvement loop until convergence or max iterations."""
        streak_no_improvement = 0
        previous_issue_count: int | None = None

        logger.info("loop_started", max_iterations=self.max_iterations)

        for iteration in range(1, self.max_iterations + 1):
            logger.info("iteration_start", iteration=iteration)

            try:
                result = await self.engine.optimize()
            except Exception as e:
                logger.error("iteration_failed", iteration=iteration, error=str(e))
                break

            # Update cumulative stats
            self.stats.total_iterations += 1
            self.stats.total_issues_found += result.issues_found
            self.stats.total_fixes_applied += result.fixes_applied
            self.stats.total_fixes_succeeded += result.fixes_succeeded
            self.stats.total_fixes_failed += result.fixes_failed
            self.stats.total_duration_seconds += result.duration_seconds

            self.stats.improvements.append({
                "iteration": iteration,
                "files_scanned": result.files_scanned,
                "issues_found": result.issues_found,
                "fixes_applied": result.fixes_applied,
                "fixes_succeeded": result.fixes_succeeded,
                "fixes_failed": result.fixes_failed,
                "duration_s": result.duration_seconds,
            })

            # Convergence detection
            if previous_issue_count is not None:
                if result.issues_found == 0 or result.fixes_applied == 0:
                    streak_no_improvement += 1
                elif result.fixes_succeeded == 0 and result.fixes_applied > 0:
                    streak_no_improvement += 1
                else:
                    streak_no_improvement = 0

            previous_issue_count = result.issues_found

            if streak_no_improvement >= self.convergence_threshold:
                logger.info("loop_converged", streak=streak_no_improvement)
                break

            # Brief pause between iterations
            if iteration < self.max_iterations:
                await asyncio.sleep(self.sleep_between)

        # Record final learning
        await self._record_loop_learning()

        logger.info(
            "loop_completed",
            iterations=self.stats.total_iterations,
            success_rate=f"{self.stats.success_rate:.1%}",
        )
        return self.stats

    async def _record_loop_learning(self) -> None:
        """Record cumulative learning from the loop as evolution rules."""
        from evo_mind.core.models import MemoryCreate
        from evo_mind.types import MemoryType

        # Record the overall improvement as a semantic memory
        session_id = await self.store.start_session({
            "phase": "self_improvement_loop",
            "type": "cumulative_learning",
        })

        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "improvement_summary",
                "iterations": self.stats.total_iterations,
                "total_issues_found": self.stats.total_issues_found,
                "total_fixes_applied": self.stats.total_fixes_applied,
                "total_fixes_succeeded": self.stats.total_fixes_succeeded,
                "total_fixes_failed": self.stats.total_fixes_failed,
                "success_rate": self.stats.success_rate,
                "improvements": self.stats.improvements,
            },
            importance=0.9,
            source="plugin",
            tags=["improvement-loop", "summary"],
        ))

        await self.store.end_session(session_id, f"Improvement loop: {self.stats.success_rate:.1%} success rate")
