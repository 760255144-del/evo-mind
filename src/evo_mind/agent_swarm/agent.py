"""BaseAgent — foundation for all swarm agents with memory integration."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, ClassVar

from evo_mind.agent_swarm.task import Task, TaskResult, TaskStatus
from evo_mind.core.models import MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.types import MemoryType
from evo_mind.utils import uuid7

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseAgent(ABC):
    """Abstract base for all swarm agents.

    Each agent has:
    - A unique ID and name
    - Capabilities (what task types it can handle)
    - A reference to the shared MemoryStore
    - Lifecycle management (start/stop/pause/resume)
    - Self-replication capability (spawn sub-agents)
    """

    name: ClassVar[str] = "base_agent"
    description: ClassVar[str] = "Base agent class"
    capabilities: ClassVar[list[str]] = []  # Task categories this agent can handle
    max_concurrent_tasks: ClassVar[int] = 3

    def __init__(self, store: MemoryStore, agent_id: str | None = None) -> None:
        self.id = agent_id or uuid7()
        self.store = store
        self._tasks: dict[str, Task] = {}
        self._running = False
        self._paused = False
        self._spawned_children: list[str] = []  # IDs of spawned sub-agents

        # Stats
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_runtime_seconds = 0.0

    # ---- Lifecycle ----

    async def start(self) -> None:
        """Start the agent's main loop."""
        self._running = True
        self._paused = False
        await self._record_memory(
            "agent_started",
            {"agent_id": self.id, "name": self.name, "capabilities": self.capabilities},
            MemoryType.EPISODIC,
            tags=["agent", "lifecycle", "start"],
        )
        logger.info("agent_started", id=self.id[:12], name=self.name)

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._running = False
        await self._record_memory(
            "agent_stopped",
            {
                "agent_id": self.id,
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "runtime_s": self.total_runtime_seconds,
            },
            MemoryType.EPISODIC,
            tags=["agent", "lifecycle", "stop"],
        )
        logger.info("agent_stopped", id=self.id[:12], completed=self.tasks_completed)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    # ---- Task Handling ----

    async def assign_task(self, task: Task) -> bool:
        """Assign a task to this agent. Returns True if accepted."""
        if not self._can_handle(task):
            return False

        task.status = TaskStatus.ASSIGNED
        task.assigned_to = self.id
        self._tasks[task.id] = task

        await self._record_memory(
            "task_assigned",
            {"task_id": task.id, "title": task.title, "agent_id": self.id},
            MemoryType.EPISODIC,
            tags=["task", "assigned"],
        )
        return True

    async def execute_task(self, task: Task) -> TaskResult:
        """Execute an assigned task and return the result."""
        start_time = time.monotonic()
        task.status = TaskStatus.RUNNING
        task.started_at = _now()

        logger.info("task_running", agent=self.name, task=task.title[:50])

        try:
            output = await self._execute(task)
            task.status = TaskStatus.COMPLETED
            task.output_data = output
            self.tasks_completed += 1
            success = True
            error = None
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.tasks_failed += 1
            output = {"error": str(e)}
            success = False
            error = str(e)
            logger.warning("task_failed", agent=self.name, task=task.title[:50], error=str(e))

        duration = time.monotonic() - start_time
        self.total_runtime_seconds += duration
        task.completed_at = _now()

        result = TaskResult(
            task_id=task.id,
            agent_id=self.id,
            success=success,
            output=output,
            error=error,
            duration_seconds=round(duration, 3),
        )

        # Record the result as a memory
        await self._record_memory(
            "task_completed" if success else "task_failed",
            {
                "task_id": task.id,
                "title": task.title,
                "success": success,
                "output": {k: str(v)[:200] for k, v in output.items()},
                "error": error,
                "duration_s": duration,
            },
            MemoryType.FEEDBACK if not success else MemoryType.PROCEDURAL,
            tags=["task", "success" if success else "failed"],
        )

        return result

    @abstractmethod
    async def _execute(self, task: Task) -> dict[str, Any]:
        """Execute the task — override in subclasses."""

    def _can_handle(self, task: Task) -> bool:
        """Check if this agent can handle the task."""
        if len(self._tasks) >= self.max_concurrent_tasks:
            return False
        if self.capabilities:
            task_category = task.metadata.get("category", "")
            return task_category in self.capabilities or "general" in self.capabilities
        return True

    # ---- Self-Replication ----

    async def spawn_child(
        self,
        agent_class: type[BaseAgent],
        name: str | None = None,
    ) -> BaseAgent:
        """Spawn a new child agent of the given class."""
        child = agent_class(self.store)
        if name:
            child.name = name  # type: ignore[assignment]

        await child.start()
        self._spawned_children.append(child.id)

        await self._record_memory(
            "agent_spawned",
            {
                "parent_id": self.id,
                "child_id": child.id,
                "child_name": child.name,
                "child_class": agent_class.__name__,
            },
            MemoryType.EPISODIC,
            tags=["agent", "spawn", "replication"],
        )

        logger.info("agent_spawned", parent=self.id[:12], child=child.id[:12])
        return child

    async def spawn_swarm(
        self,
        agent_class: type[BaseAgent],
        count: int,
        names: list[str] | None = None,
    ) -> list[BaseAgent]:
        """Spawn multiple instances of the same agent class in parallel."""
        children: list[BaseAgent] = []
        for i in range(count):
            name = names[i] if names and i < len(names) else None
            child = await self.spawn_child(agent_class, name)
            children.append(child)
        return children

    # ---- Communication ----

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to the shared memory (all agents can read)."""
        await self._record_memory(
            "agent_broadcast",
            {
                "from_agent": self.id,
                "message": message,
            },
            MemoryType.EPISODIC,
            tags=["agent", "broadcast", "communication"],
        )

    async def read_mailbox(self, limit: int = 20) -> list[dict[str, Any]]:
        """Read recent broadcast messages from other agents."""
        from evo_mind.core.models import SearchQuery

        results = await self._retrieve(
            query="agent broadcast communication",
            tags=["agent", "broadcast"],
            limit=limit,
        )
        return [
            {"agent_id": r.memory.metadata.get("from_agent", ""), "content": r.memory.content}
            for r in results
        ]

    # ---- Helpers ----

    async def _record_memory(
        self,
        title: str,
        content: dict[str, Any],
        memory_type: MemoryType,
        tags: list[str] | None = None,
    ) -> str | None:
        """Record an experience to the shared memory."""
        try:
            mem = await self.store.record(MemoryCreate(
                memory_type=memory_type,
                content={"title": title, **content},
                importance=0.5,
                source="plugin",
                tags=tags or [],
            ))
            return mem.id
        except Exception as e:
            logger.debug("memory_record_failed", error=str(e))
            return None

    async def _retrieve(
        self,
        query: str,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list:
        """Search the shared memory."""
        from evo_mind.core.models import SearchQuery
        from evo_mind.retrieval.engine import RetrievalEngine

        try:
            retrieval = RetrievalEngine(self.store.db, self.store.vector_store, self.store.embedding)
            return await retrieval.search(SearchQuery(
                query_text=query,
                tags=tags,
                max_results=limit,
            ))
        except Exception:
            return []

    def status_report(self) -> dict[str, Any]:
        """Generate a status report for this agent."""
        return {
            "id": self.id,
            "name": self.name,
            "running": self._running,
            "paused": self._paused,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "active_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.RUNNING]),
            "pending_tasks": len([t for t in self._tasks.values() if t.status == TaskStatus.ASSIGNED]),
            "spawned_children": len(self._spawned_children),
            "total_runtime_s": round(self.total_runtime_seconds, 1),
        }


# ---- Specialized Agent Types ----

class AnalyzerAgent(BaseAgent):
    """Agent that analyzes code/files and reports issues."""

    name = "analyzer"
    description = "Analyzes code and data for patterns and issues"
    capabilities = ["analysis", "code_review", "data_analysis", "general"]

    async def _execute(self, task: Task) -> dict[str, Any]:
        """Analyze the input and return findings."""
        input_data = task.input_data
        target = input_data.get("target", "")
        analysis_type = input_data.get("analysis_type", "general")

        # Perform analysis
        findings: list[dict] = []

        if analysis_type == "code_review" and target:
            findings = await self._review_code(target)
        elif analysis_type == "pattern_detection":
            findings = await self._detect_patterns(input_data)
        else:
            findings = [{"type": "general", "note": f"Analysis of {target or 'input'} complete"}]

        return {
            "findings": findings,
            "count": len(findings),
            "analysis_type": analysis_type,
        }

    async def _review_code(self, target: str) -> list[dict]:
        """Review code and return findings."""
        from pathlib import Path
        path = Path(target)
        if not path.exists():
            return [{"error": f"File not found: {target}"}]

        findings = []
        try:
            content = path.read_text()
            lines = content.split("\n")
            import re

            # Check for common issues
            for i, line in enumerate(lines, 1):
                if re.search(r"except\s*:", line):
                    findings.append({"line": i, "issue": "bare_except", "severity": "warning"})
                if "TODO" in line or "FIXME" in line:
                    findings.append({"line": i, "issue": "todo_marker", "severity": "info"})
                if "import pdb" in line:
                    findings.append({"line": i, "issue": "debugger_import", "severity": "warning"})
        except Exception as e:
            findings.append({"error": str(e)})

        return findings

    async def _detect_patterns(self, data: dict) -> list[dict]:
        """Detect patterns in data."""
        return [{"pattern": "general", "confidence": 0.5}]


class ExecutorAgent(BaseAgent):
    """Agent that executes tasks, runs tests, and reports outcomes."""

    name = "executor"
    description = "Executes commands and reports results"
    capabilities = ["execution", "testing", "deployment", "general"]

    # Safe command allowlist — only these commands can be executed
    ALLOWED_COMMANDS = {
        "pytest", "python", "python3", "echo", "ls", "cat", "head", "tail",
        "wc", "find", "grep", "git", "pip", "uv", "ruff", "mypy",
        "evo-mind",
    }

    async def _execute(self, task: Task) -> dict[str, Any]:
        """Execute a command or action — ONLY from the allowlist."""
        command = task.input_data.get("command", "")
        timeout = task.input_data.get("timeout", 30)

        if not command:
            return {"error": "No command specified"}

        # Security: validate command against allowlist
        cmd_base = command.strip().split()[0] if command.strip() else ""
        if cmd_base not in self.ALLOWED_COMMANDS:
            return {
                "success": False,
                "error": f"Command '{cmd_base}' not in allowlist. Allowed: {sorted(self.ALLOWED_COMMANDS)}",
            }

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout)
            )

            return {
                "success": proc.returncode == 0,
                "returncode": proc.returncode,
                "stdout": stdout.decode()[:1000],
                "stderr": stderr.decode()[:500],
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ReviewerAgent(BaseAgent):
    """Agent that reviews outputs, validates quality, and provides feedback."""

    name = "reviewer"
    description = "Reviews and validates outputs from other agents"
    capabilities = ["review", "validation", "quality_assurance", "general"]

    async def _execute(self, task: Task) -> dict[str, Any]:
        """Review the output of another task."""
        subject = task.input_data.get("subject", "")
        criteria = task.input_data.get("criteria", ["correctness", "completeness"])

        # Perform review
        score = 0.7  # Default moderate score
        notes: list[str] = []
        passed: list[str] = []
        failed: list[str] = []

        for criterion in criteria:
            if self._check_criterion(subject, criterion):
                passed.append(criterion)
            else:
                failed.append(criterion)
                notes.append(f"Failed: {criterion}")

        score = len(passed) / len(criteria) if criteria else 1.0

        return {
            "score": score,
            "passed": passed,
            "failed": failed,
            "notes": notes,
            "recommendation": "approve" if score >= 0.7 else "revise",
        }

    def _check_criterion(self, subject: str, criterion: str) -> bool:
        """Check a single review criterion."""
        # Simple heuristic — real implementation would use LLM
        return len(subject) > 10  # Placeholder
