"""Task models for the agent swarm system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from evo_mind.utils import uuid7


class TaskStatus(StrEnum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class Task:
    """A unit of work that can be assigned to an agent."""

    id: str = field(default_factory=uuid7)
    title: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None  # Agent ID
    parent_task_id: str | None = None  # For task decomposition
    dependencies: list[str] = field(default_factory=list)  # Task IDs that must complete first
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def is_ready(self) -> bool:
        """Task is ready when pending and has no incomplete dependencies."""
        return self.status == TaskStatus.PENDING


@dataclass(slots=True)
class TaskResult:
    """Result of a completed task."""
    task_id: str
    agent_id: str
    success: bool
    output: dict[str, Any]
    error: str | None = None
    duration_seconds: float = 0.0
    memories_recorded: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class SubtaskDecomposition:
    """A parent task broken down into subtasks."""
    parent_task: Task
    subtasks: list[Task]
    strategy: str = "parallel"  # "parallel" | "sequential" | "pipeline"
    metadata: dict[str, Any] = field(default_factory=dict)
