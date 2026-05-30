"""Super-Evolution Meta-Engine — orchestrates all phases in a higher-order loop.

This is the "brain" that decides what to improve, when, and how. It monitors
system-wide metrics, introspects on weaknesses, and orchestrates the 4 phases
of evolution in a unified, self-improving cycle.

Key concept: The meta-engine itself can be improved by the code-evolution
phase, enabling true recursive self-improvement.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable

from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType, RuleType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---- Types ----

class SystemPhase(StrEnum):
    """The four phases + meta."""
    MEMORY = "memory"
    CODE_EVOLUTION = "code_evolution"
    AGENT_SWARM = "agent_swarm"
    EVOLUTIONARY = "evolutionary"
    META = "meta"


class ImprovementAction(StrEnum):
    """Types of improvement actions the meta-engine can take."""
    OPTIMIZE_PARAMS = "optimize_params"
    EVOLVE_CODE = "evolve_code"
    SPAWN_AGENTS = "spawn_agents"
    CONSOLIDATE_MEMORIES = "consolidate_memories"
    LEARN_RULES = "learn_rules"
    SELF_MODIFY = "self_modify"
    GENERATE_GOAL = "generate_goal"
    RUN_DIAGNOSTICS = "run_diagnostics"
    RESTRUCTURE = "restructure"  # Change system architecture


@dataclass
class SystemMetrics:
    """Snapshot of system-wide performance metrics."""
    timestamp: str = field(default_factory=_now)
    total_memories: int = 0
    active_memories: int = 0
    consolidated_memories: int = 0
    evolution_rules_active: int = 0
    avg_rule_confidence: float = 0.0
    code_fixes_applied: int = 0
    code_fixes_succeeded: int = 0
    agents_spawned: int = 0
    agents_active: int = 0
    retrieval_precision: float = 0.0  # Estimated
    genetic_fitness: float = 0.0
    system_health: float = 1.0
    improvement_rate: float = 0.0  # Rate of change in metrics

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SuperEvolutionState:
    """The meta-state of the super-evolution process."""
    iteration: int = 0
    current_phase: SystemPhase | None = None
    metrics_history: list[SystemMetrics] = field(default_factory=list)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    active_goals: list[str] = field(default_factory=list)
    completed_goals: list[str] = field(default_factory=list)
    self_modifications: int = 0
    recursive_depth: int = 0
    total_uptime_seconds: float = 0.0
    convergence_score: float = 0.0  # How close to optimal


# ---- Meta-Engine ----

class MetaEngine:
    """The super-evolution orchestrator.

    This is a higher-order controller that:
    1. Collects metrics from all phases
    2. Introspects on system weaknesses
    3. Generates improvement goals
    4. Selects and executes the best action
    5. Evaluates outcomes and learns
    6. Can modify its own decision logic
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.id = uuid7()
        self.state = SuperEvolutionState()

        # Pluggable phase executors
        self._phase_executors: dict[SystemPhase, Callable] = {}

        # Decision weights (themselves evolvable!)
        self._action_weights: dict[ImprovementAction, float] = {
            ImprovementAction.OPTIMIZE_PARAMS: 0.20,
            ImprovementAction.EVOLVE_CODE: 0.15,
            ImprovementAction.SPAWN_AGENTS: 0.10,
            ImprovementAction.CONSOLIDATE_MEMORIES: 0.15,
            ImprovementAction.LEARN_RULES: 0.20,
            ImprovementAction.SELF_MODIFY: 0.05,
            ImprovementAction.GENERATE_GOAL: 0.10,
            ImprovementAction.RUN_DIAGNOSTICS: 0.05,
        }

        # Success tracking per action type
        self._action_success: dict[ImprovementAction, list[float]] = {
            action: [] for action in ImprovementAction
        }

    def register_phase(self, phase: SystemPhase, executor: Callable) -> None:
        """Register a phase executor function."""
        self._phase_executors[phase] = executor

    async def run(self, max_iterations: int = 100, sleep_seconds: float = 5.0) -> SuperEvolutionState:
        """Run the super-evolution loop."""
        start_time = time.monotonic()
        logger.info("super_evolution_started", max_iterations=max_iterations)

        session_id = await self.store.start_session({
            "phase": "super_evolution",
            "component": "meta_engine",
        })

        try:
            for iteration in range(1, max_iterations + 1):
                self.state.iteration = iteration
                self.state.total_uptime_seconds = time.monotonic() - start_time
                self.state.recursive_depth = self._compute_recursive_depth()

                # 1. Collect metrics
                metrics = await self._collect_metrics()
                self.state.metrics_history.append(metrics)

                # 2. Introspect
                weaknesses = await self._introspect(metrics)
                logger.info("introspection", iteration=iteration,
                            health=f"{metrics.system_health:.2f}",
                            weaknesses=weaknesses[:3])

                # 3. Generate goals if needed
                if not self.state.active_goals or len(weaknesses) > 3:
                    new_goals = await self._generate_goals(metrics, weaknesses)
                    self.state.active_goals.extend(new_goals)

                # 4. Select best action
                action = await self._select_action(metrics, weaknesses)
                self.state.actions_taken.append({
                    "iteration": iteration,
                    "action": action.value,
                    "metrics": metrics.to_dict(),
                    "time": _now(),
                })

                # 5. Execute action
                success = await self._execute_action(action, metrics)

                # 6. Evaluate and update weights
                await self._evaluate_outcome(action, success, metrics)

                # 7. Check goals
                await self._check_goals(metrics)

                # 8. Convergence detection
                self.state.convergence_score = self._compute_convergence()
                if self.state.convergence_score > 0.95:
                    logger.info("super_evolution_converged", iteration=iteration)
                    break

                # 9. Self-modify if beneficial
                if self.state.self_modifications < 3 and success:
                    if self.state.convergence_score < 0.6:
                        await self._consider_self_modification()

                await asyncio.sleep(sleep_seconds)

        except Exception as e:
            logger.exception("super_evolution_failed")
            raise
        finally:
            await self._record_super_state(session_id)
            await self.store.end_session(session_id, f"Super-evolution: {self.state.iteration} iterations")

        return self.state

    # ---- Metrics Collection ----

    async def _collect_metrics(self) -> SystemMetrics:
        """Gather system-wide performance data from all phases."""
        metrics = SystemMetrics()

        try:
            # Phase 1: Memory metrics
            metrics.total_memories = await self.store.count_total()
            from evo_mind.types import MemoryStatus
            metrics.active_memories = await self.store.count_by_status(MemoryStatus.ACTIVE)
            metrics.consolidated_memories = await self.store.count_by_status(MemoryStatus.CONSOLIDATED)

            # Evolution rules
            rows = await self.store.db.fetch_all(
                "SELECT COUNT(*) as cnt, AVG(confidence) as avg_conf FROM evolution_rules WHERE status='active'"
            )
            if rows:
                metrics.evolution_rules_active = rows[0]["cnt"] or 0
                metrics.avg_rule_confidence = rows[0]["avg_conf"] or 0.0

            # Genetic fitness (from evolutionary phase)
            fit_rows = await self.store.db.fetch_all(
                "SELECT metric_value FROM evolution_metrics WHERE metric_name='genetic_fitness' ORDER BY recorded_at DESC LIMIT 1"
            )
            if fit_rows:
                metrics.genetic_fitness = fit_rows[0]["metric_value"]

            # System health computation
            health_factors: list[float] = []

            # Memory health: ideal ratio of consolidated to total
            if metrics.total_memories > 0:
                cons_ratio = metrics.consolidated_memories / metrics.total_memories
                health_factors.append(1.0 - abs(cons_ratio - 0.7))  # 70% ideal

            # Rule health
            health_factors.append(metrics.avg_rule_confidence)

            # Diversity: having multiple active phases is healthy
            active_phases = sum([
                1 if metrics.active_memories > 0 else 0,
                1 if metrics.evolution_rules_active > 0 else 0,
                1 if metrics.code_fixes_applied > 0 else 0,
                1 if metrics.agents_spawned > 0 else 0,
            ])
            health_factors.append(active_phases / 4.0)

            metrics.system_health = (
                sum(health_factors) / len(health_factors) if health_factors else 0.5
            )

            # Improvement rate
            if len(self.state.metrics_history) >= 2:
                prev = self.state.metrics_history[-1]
                delta = metrics.system_health - prev.system_health
                metrics.improvement_rate = delta

        except Exception as e:
            logger.warning("metrics_collection_partial", error=str(e))

        return metrics

    # ---- Introspection ----

    async def _introspect(self, metrics: SystemMetrics) -> list[str]:
        """Analyze system state and identify weaknesses.

        This is the system "thinking about itself" — meta-cognition.
        """
        weaknesses: list[str] = []

        # Memory issues
        if metrics.total_memories > 100 and metrics.consolidated_memories / max(metrics.total_memories, 1) < 0.3:
            weaknesses.append("low_consolidation_ratio")
        if metrics.active_memories > 500:
            weaknesses.append("too_many_unconsolidated_memories")

        # Rule issues
        if metrics.evolution_rules_active < 3:
            weaknesses.append("few_evolution_rules")
        if metrics.avg_rule_confidence < 0.5:
            weaknesses.append("low_rule_confidence")

        # Code evolution issues
        if self.state.actions_taken:
            recent_code_actions = sum(
                1 for a in self.state.actions_taken[-10:]
                if a["action"] == ImprovementAction.EVOLVE_CODE.value
            )
            if recent_code_actions == 0 and metrics.code_fixes_applied < 10:
                weaknesses.append("insufficient_code_evolution")

        # Agent issues
        if metrics.agents_spawned < 1:
            weaknesses.append("no_agents_spawned")

        # Genetic stagnation
        if len(self.state.metrics_history) >= 5:
            recent_health = [m.system_health for m in self.state.metrics_history[-5:]]
            if max(recent_health) - min(recent_health) < 0.01:
                weaknesses.append("fitness_stagnation")

        # Memory bloat
        if metrics.total_memories > 50000:
            weaknesses.append("memory_bloat_risk")

        # Self-modification deficit
        if self.state.self_modifications == 0 and self.state.iteration > 20:
            weaknesses.append("no_self_modification_yet")

        return weaknesses

    # ---- Goal Generation ----

    async def _generate_goals(
        self, metrics: SystemMetrics, weaknesses: list[str]
    ) -> list[str]:
        """Generate autonomous improvement goals based on introspection."""
        goals: list[str] = []

        goal_map = {
            "low_consolidation_ratio": "Achieve 70% memory consolidation ratio",
            "too_many_unconsolidated_memories": "Reduce active memories below 300 via consolidation",
            "few_evolution_rules": "Discover 5+ new evolution rules",
            "low_rule_confidence": "Increase average rule confidence above 0.7",
            "insufficient_code_evolution": "Run code evolution on project to find 3+ improvements",
            "no_agents_spawned": "Spawn a team of 3 agents to validate rules",
            "fitness_stagnation": "Run genetic algorithm with higher mutation rate to escape local optimum",
            "memory_bloat_risk": "Prune low-importance memories to stay below 50K",
            "no_self_modification_yet": "Attempt one safe self-modification",
        }

        for weakness in weaknesses:
            if weakness in goal_map and goal_map[weakness] not in self.state.completed_goals:
                goals.append(goal_map[weakness])

        # Record goals as memories
        for goal in goals:
            await self.store.record(MemoryCreate(
                memory_type=MemoryType.SEMANTIC,
                content={"type": "super_goal", "goal": goal, "source_weakness": weaknesses[0] if weaknesses else "introspection"},
                importance=0.8,
                source="plugin",
                tags=["super-evolution", "goal", "autonomous"],
            ))

        logger.info("goals_generated", count=len(goals))
        return goals

    # ---- Action Selection ----

    async def _select_action(
        self, metrics: SystemMetrics, weaknesses: list[str]
    ) -> ImprovementAction:
        """Select the best action using weighted scoring + weakness priority."""
        import random

        # Boost weights based on weaknesses
        boosted_weights = dict(self._action_weights)

        weakness_action_map = {
            "low_consolidation_ratio": ImprovementAction.CONSOLIDATE_MEMORIES,
            "too_many_unconsolidated_memories": ImprovementAction.CONSOLIDATE_MEMORIES,
            "few_evolution_rules": ImprovementAction.LEARN_RULES,
            "low_rule_confidence": ImprovementAction.OPTIMIZE_PARAMS,
            "insufficient_code_evolution": ImprovementAction.EVOLVE_CODE,
            "no_agents_spawned": ImprovementAction.SPAWN_AGENTS,
            "fitness_stagnation": ImprovementAction.OPTIMIZE_PARAMS,
            "memory_bloat_risk": ImprovementAction.CONSOLIDATE_MEMORIES,
            "no_self_modification_yet": ImprovementAction.SELF_MODIFY,
        }

        for weakness in weaknesses:
            action = weakness_action_map.get(weakness)
            if action:
                boosted_weights[action] *= 3.0  # Triple weight for needed actions

        # Exploration bonus: occasionally pick a random under-used action
        if random.random() < 0.1:
            rare_actions = [
                a for a in ImprovementAction
                if len(self._action_success.get(a, [])) < 3
            ]
            if rare_actions:
                return random.choice(rare_actions)

        # Weighted random selection
        actions = list(boosted_weights.keys())
        weights = list(boosted_weights.values())
        total = sum(weights)

        if total == 0:
            return random.choice(list(ImprovementAction))

        r = random.uniform(0, total)
        cumulative = 0.0
        for action, weight in zip(actions, weights):
            cumulative += weight
            if cumulative >= r:
                return action

        return actions[-1]

    # ---- Action Execution ----

    async def _execute_action(
        self, action: ImprovementAction, metrics: SystemMetrics
    ) -> bool:
        """Execute the selected action using the appropriate phase."""
        self.state.current_phase = self._action_to_phase(action)
        logger.info("executing_action", action=action.value, phase=self.state.current_phase.value)

        success = True

        try:
            if action == ImprovementAction.CONSOLIDATE_MEMORIES:
                success = await self._execute_consolidation()

            elif action == ImprovementAction.LEARN_RULES:
                success = await self._execute_rule_learning()

            elif action == ImprovementAction.OPTIMIZE_PARAMS:
                success = await self._execute_genetic_optimization()

            elif action == ImprovementAction.EVOLVE_CODE:
                success = await self._execute_code_evolution()

            elif action == ImprovementAction.SPAWN_AGENTS:
                success = await self._execute_agent_spawn()

            elif action == ImprovementAction.SELF_MODIFY:
                success = await self._execute_self_modification()

            elif action == ImprovementAction.GENERATE_GOAL:
                success = True  # Already done in _generate_goals

            elif action == ImprovementAction.RUN_DIAGNOSTICS:
                success = True  # Metrics already collected

        except Exception as e:
            logger.error("action_failed: %s error=%s", action.value, e)
            success = False

        return success

    async def _execute_consolidation(self) -> bool:
        from evo_mind.consolidation.engine import ConsolidationEngine
        from evo_mind.types import ConsolidationTrigger

        engine = ConsolidationEngine(
            self.store,
            None,  # retrieval — not needed for basic consolidation
            self.store.embedding,  # OK: embedding is available from store
            self.store.vector_store,  # OK: vector_store is available from store
            self.store.db,
            config={"min_candidates": 10},
        )
        pending = await engine.get_pending_count()
        if pending >= 5:
            result = await engine.consolidate(trigger=ConsolidationTrigger.THRESHOLD)
            return result.summaries_generated > 0 or result.memories_pruned > 0
        return False

    async def _execute_rule_learning(self) -> bool:
        from evo_mind.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(self.store, self.store.db)
        rules = await engine.evolve()
        return len(rules) > 0

    async def _execute_genetic_optimization(self) -> bool:
        from evo_mind.evolutionary.engine import EvolutionaryEngine
        engine = EvolutionaryEngine(self.store, population_size=20, max_generations=10)
        state = await engine.evolve()
        return state.best_fitness > 0.5

    async def _execute_code_evolution(self) -> bool:
        from evo_mind.code_evolution.engine import CodeEvolutionEngine
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent  # evo_mind directory
        engine = CodeEvolutionEngine(self.store, project_root, max_fixes_per_run=3)
        result = await engine.optimize()
        return result.fixes_succeeded > 0

    async def _execute_agent_spawn(self) -> bool:
        from evo_mind.agent_swarm.coordinator import SwarmCoordinator

        coordinator = SwarmCoordinator(self.store)
        agents = await coordinator.spawn_team({
            "analyzer": 1,
            "executor": 1,
            "reviewer": 1,
        })
        return len(agents) >= 2

    async def _execute_self_modification(self) -> bool:
        from evo_mind.super_evolution.recursive_improver import RecursiveImprover
        improver = RecursiveImprover(self.store)
        result = await improver.improve(max_changes=2)
        if result.get("changes_made", 0) > 0:
            self.state.self_modifications += 1
            return True
        return False

    # ---- Outcome Evaluation ----

    async def _evaluate_outcome(
        self, action: ImprovementAction, success: bool, pre_metrics: SystemMetrics
    ) -> None:
        """Evaluate action outcome and update decision weights."""
        # Record success/failure for this action type
        self._action_success[action].append(1.0 if success else 0.0)

        # Keep only last 10 outcomes per action
        if len(self._action_success[action]) > 10:
            self._action_success[action] = self._action_success[action][-10:]

        # Update weights using exponential moving average
        recent_success_rate = (
            sum(self._action_success[action]) / len(self._action_success[action])
            if self._action_success[action] else 0.5
        )

        # Adjust weight: successful actions get more weight
        learning_rate = 0.1
        current = self._action_weights[action]
        target = max(0.01, recent_success_rate)
        self._action_weights[action] = current + learning_rate * (target - current)

        # Normalize weights
        total = sum(self._action_weights.values())
        if total > 0:
            for a in self._action_weights:
                self._action_weights[a] /= total

    # ---- Helpers ----

    def _action_to_phase(self, action: ImprovementAction) -> SystemPhase:
        mapping = {
            ImprovementAction.CONSOLIDATE_MEMORIES: SystemPhase.MEMORY,
            ImprovementAction.LEARN_RULES: SystemPhase.MEMORY,
            ImprovementAction.EVOLVE_CODE: SystemPhase.CODE_EVOLUTION,
            ImprovementAction.SPAWN_AGENTS: SystemPhase.AGENT_SWARM,
            ImprovementAction.OPTIMIZE_PARAMS: SystemPhase.EVOLUTIONARY,
            ImprovementAction.SELF_MODIFY: SystemPhase.META,
            ImprovementAction.GENERATE_GOAL: SystemPhase.META,
            ImprovementAction.RUN_DIAGNOSTICS: SystemPhase.META,
            ImprovementAction.RESTRUCTURE: SystemPhase.META,
        }
        return mapping.get(action, SystemPhase.META)

    async def _check_goals(self, metrics: SystemMetrics) -> None:
        """Check if any active goals have been achieved."""
        completed = []
        for goal in self.state.active_goals:
            achieved = False
            if "consolidation ratio" in goal and metrics.consolidated_memories / max(metrics.total_memories, 1) >= 0.7:
                achieved = True
            elif "rule confidence" in goal and metrics.avg_rule_confidence >= 0.7:
                achieved = True
            elif "evolution rules" in goal and metrics.evolution_rules_active >= 5:
                achieved = True

            if achieved:
                completed.append(goal)

        for goal in completed:
            self.state.active_goals.remove(goal)
            self.state.completed_goals.append(goal)
            logger.info("goal_achieved", goal=goal[:80])

    def _compute_convergence(self) -> float:
        """Compute how close the system is to optimal."""
        if len(self.state.metrics_history) < 5:
            return 0.0

        recent = self.state.metrics_history[-5:]
        health_values = [m.system_health for m in recent]

        # Convergence = high health + low variance
        avg_health = sum(health_values) / len(health_values)
        variance = sum((h - avg_health) ** 2 for h in health_values) / len(health_values)

        return avg_health * (1.0 - min(variance, 1.0))

    def _compute_recursive_depth(self) -> int:
        """Estimate recursion depth of self-improvement."""
        return self.state.self_modifications

    async def _consider_self_modification(self) -> None:
        """Check if self-modification would be beneficial."""
        if self.state.self_modifications >= 5:
            return  # Limit recursive depth

        recent_success_rates = [
            sum(self._action_success.get(a, [0.5])) / max(len(self._action_success.get(a, [])), 1)
            for a in ImprovementAction
        ]
        avg_success = sum(recent_success_rates) / len(recent_success_rates)

        # Only self-modify if we're struggling
        if avg_success < 0.5:
            logger.info("considering_self_modification", avg_success=f"{avg_success:.2f}")

    async def _record_super_state(self, session_id: str) -> None:
        """Record the super-evolution state as a memory."""
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "super_evolution_state",
                "iterations": self.state.iteration,
                "self_modifications": self.state.self_modifications,
                "recursive_depth": self.state.recursive_depth,
                "convergence_score": self.state.convergence_score,
                "goals_completed": len(self.state.completed_goals),
                "action_weights": {k.value: round(v, 3) for k, v in self._action_weights.items()},
                "action_success_rates": {
                    k.value: round(sum(v) / max(len(v), 1), 3)
                    for k, v in self._action_success.items() if v
                },
            },
            importance=1.0,
            session_id=session_id,
            source="plugin",
            tags=["super-evolution", "meta", "state"],
        ))
