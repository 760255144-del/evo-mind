"""SwarmCoordinator — manages agent lifecycle, task distribution, and consensus."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Type

from evo_mind.agent_swarm.agent import BaseAgent, AnalyzerAgent, ExecutorAgent, ReviewerAgent
from evo_mind.agent_swarm.consensus import ConsensusEngine
from evo_mind.agent_swarm.task import (
    SubtaskDecomposition,
    Task,
    TaskPriority,
    TaskResult,
    TaskStatus,
)
from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType, RelationType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SwarmCoordinator:
    """Orchestrates a swarm of agents sharing a MemoryStore.

    Capabilities:
    - Agent lifecycle management (spawn, monitor, terminate)
    - Task decomposition (break complex tasks into subtasks)
    - Intelligent task assignment (match tasks to capable agents)
    - Result aggregation and consensus
    - Learning: records all coordination as memories
    """

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        self.id = uuid7()
        self._agents: dict[str, BaseAgent] = {}
        self._tasks: dict[str, Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._agent_classes: dict[str, Type[BaseAgent]] = {
            "analyzer": AnalyzerAgent,
            "executor": ExecutorAgent,
            "reviewer": ReviewerAgent,
        }
        self.consensus = ConsensusEngine()

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def pending_tasks(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)

    # ---- Agent Management ----

    async def spawn_agent(
        self, agent_type: str, name: str | None = None
    ) -> BaseAgent:
        """Spawn a new agent of the given type."""
        agent_cls = self._agent_classes.get(agent_type)
        if not agent_cls:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self._agent_classes)}")

        agent = agent_cls(self.store)
        if name:
            agent.name = name  # type: ignore[assignment]

        await agent.start()
        self._agents[agent.id] = agent

        await self.store.record(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={
                "type": "agent_spawned",
                "coordinator_id": self.id,
                "agent_id": agent.id,
                "agent_type": agent_type,
                "agent_name": agent.name,
            },
            source="plugin",
            tags=["swarm", "spawn", agent_type],
        ))

        return agent

    async def spawn_team(
        self,
        composition: dict[str, int],  # {"analyzer": 2, "executor": 3, "reviewer": 1}
    ) -> list[BaseAgent]:
        """Spawn a team of agents with the given composition."""
        agents: list[BaseAgent] = []
        for agent_type, count in composition.items():
            for i in range(count):
                agent = await self.spawn_agent(agent_type, f"{agent_type}-{i+1}")
                agents.append(agent)
        logger.info("team_spawned", composition=composition, total=len(agents))
        return agents

    async def stop_all(self) -> None:
        """Stop all managed agents."""
        for agent in list(self._agents.values()):
            await agent.stop()
        self._agents.clear()
        logger.info("all_agents_stopped")

    def get_agent(self, agent_id: str) -> BaseAgent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[dict[str, Any]]:
        return [agent.status_report() for agent in self._agents.values()]

    # ---- Task Management ----

    async def create_task(
        self,
        title: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        input_data: dict[str, Any] | None = None,
        category: str = "general",
    ) -> Task:
        """Create a new task in the swarm."""
        task = Task(
            title=title,
            description=description,
            priority=priority,
            input_data=input_data or {},
            metadata={"category": category},
        )
        self._tasks[task.id] = task

        await self.store.record(MemoryCreate(
            memory_type=MemoryType.EPISODIC,
            content={
                "type": "task_created",
                "task_id": task.id,
                "title": title,
                "priority": priority.value,
                "category": category,
            },
            source="plugin",
            tags=["swarm", "task", "created"],
        ))

        return task

    async def decompose_task(self, task: Task, strategy: str = "parallel") -> SubtaskDecomposition:
        """Break a complex task into subtasks."""
        subtasks: list[Task] = []

        description = task.description.lower()
        title = task.title.lower()

        # Heuristic decomposition based on task type
        if "analyze" in description or "analyze" in title:
            subtasks = [
                Task(title=f"Analyze scope: {task.title}",
                     description=f"Determine scope for: {task.description}",
                     parent_task_id=task.id, metadata={"category": "analysis"}),
                Task(title=f"Gather data: {task.title}",
                     description=f"Collect relevant data for: {task.description}",
                     parent_task_id=task.id, metadata={"category": "execution"}),
                Task(title=f"Review findings: {task.title}",
                     description=f"Validate and summarize analysis results",
                     parent_task_id=task.id, metadata={"category": "review"}),
            ]
            # Wire dependencies AFTER list construction
            if len(subtasks) >= 2:
                subtasks[1].dependencies = [subtasks[0].id]
            if len(subtasks) >= 3:
                subtasks[2].dependencies = [subtasks[1].id]
        elif "build" in description or "implement" in description or "build" in title:
            subtasks = [
                Task(title=f"Design approach: {task.title}",
                     description=f"Design solution for: {task.description}",
                     parent_task_id=task.id, metadata={"category": "analysis"}),
                Task(title=f"Implement: {task.title}",
                     description=f"Implement the designed solution",
                     parent_task_id=task.id, metadata={"category": "execution"}),
                Task(title=f"Test: {task.title}",
                     description=f"Run tests and verify implementation",
                     parent_task_id=task.id, metadata={"category": "execution"}),
                Task(title=f"Review: {task.title}",
                     description=f"Code review the implementation",
                     parent_task_id=task.id, metadata={"category": "review"}),
            ]
            # Wire dependencies AFTER list construction
            if len(subtasks) >= 2:
                subtasks[1].dependencies = [subtasks[0].id]
            if len(subtasks) >= 3:
                subtasks[2].dependencies = [subtasks[1].id]
            if len(subtasks) >= 4:
                subtasks[3].dependencies = [subtasks[2].id]
        else:
            # Default: single analysis + execution pair
            subtasks = [
                Task(
                    title=f"Process: {task.title}",
                    description=task.description,
                    parent_task_id=task.id,
                    metadata={"category": "general"},
                ),
            ]

        decomposition = SubtaskDecomposition(
            parent_task=task,
            subtasks=subtasks,
            strategy=strategy,
        )

        for st in subtasks:
            self._tasks[st.id] = st

        logger.info("task_decomposed", parent=task.id[:12], subtasks=len(subtasks))

        return decomposition

    async def assign_best_agent(self, task: Task) -> BaseAgent | None:
        """Find and assign the best agent for a task."""
        task_category = task.metadata.get("category", "general")

        # Find available agents with matching capability
        candidates = []
        for agent in self._agents.values():
            if not agent.is_running:
                continue
            if task_category in agent.capabilities or "general" in agent.capabilities:
                active = len([t for t in agent._tasks.values() if t.status == TaskStatus.RUNNING])
                if active < agent.max_concurrent_tasks:
                    candidates.append((agent, active))

        if not candidates:
            # Try to spawn a suitable agent
            for atype, acls in self._agent_classes.items():
                if task_category in acls.capabilities:
                    agent = await self.spawn_agent(atype)
                    candidates.append((agent, 0))
                    break

        if not candidates:
            logger.warning("no_capable_agent", task=task.title[:50])
            return None

        # Pick agent with fewest active tasks
        candidates.sort(key=lambda x: x[1])
        agent = candidates[0][0]

        success = await agent.assign_task(task)
        return agent if success else None

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task through a suitable agent."""
        agent = await self.assign_best_agent(task)
        if not agent:
            return TaskResult(
                task_id=task.id,
                agent_id="",
                success=False,
                output={},
                error="No capable agent available",
            )

        result = await agent.execute_task(task)
        self._results[task.id] = result
        return result

    async def execute_pipeline(
        self,
        tasks: list[Task],
        strategy: str = "parallel",
    ) -> list[TaskResult]:
        """Execute multiple tasks with the given strategy."""
        results: list[TaskResult] = []

        if strategy == "parallel":
            # Run all tasks concurrently
            coroutines = [self.execute_task(task) for task in tasks]
            results = list(await asyncio.gather(*coroutines))

        elif strategy == "sequential":
            # Run tasks in order, respecting dependencies
            completed_ids: set[str] = set()
            remaining = list(tasks)

            while remaining:
                for task in list(remaining):
                    deps_met = all(d in completed_ids for d in task.dependencies)
                    if deps_met and task.is_ready:
                        result = await self.execute_task(task)
                        results.append(result)
                        completed_ids.add(task.id)
                        remaining.remove(task)
                        break
                else:
                    # Deadlock or all remaining have unmet deps
                    logger.warning("task_pipeline_stalled", remaining=len(remaining))
                    break

        elif strategy == "pipeline":
            # Each stage waits for previous stage to complete
            for task in tasks:
                result = await self.execute_task(task)
                results.append(result)
                if not result.success:
                    logger.warning("pipeline_stopped", failed_task=task.title[:50])
                    break

        # Use consensus to evaluate overall results
        evaluation = self.consensus.evaluate(results)
        await self._record_evaluation(evaluation, results)

        return results

    async def _record_evaluation(
        self, evaluation: dict[str, Any], results: list[TaskResult]
    ) -> None:
        """Record the consensus evaluation as a memory."""
        await self.store.record(MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "type": "consensus_evaluation",
                "total_tasks": len(results),
                "successful": evaluation.get("successful", 0),
                "failed": evaluation.get("failed", 0),
                "agreement_level": evaluation.get("agreement_level", 0),
                "recommendation": evaluation.get("recommendation", ""),
            },
            importance=0.7,
            source="plugin",
            tags=["swarm", "consensus", "evaluation"],
        ))

    def status_report(self) -> dict[str, Any]:
        """Full swarm status report."""
        return {
            "coordinator_id": self.id,
            "agents": self.list_agents(),
            "total_agents": self.agent_count,
            "pending_tasks": self.pending_tasks,
            "completed_tasks": len(self._results),
            "agent_types": {
                atype: len([
                    a for a in self._agents.values()
                    if a.__class__.__name__.lower().startswith(atype)
                ])
                for atype in self._agent_classes
            },
        }
