"""Agent Swarm — multi-agent coordination with shared evo-mind memory."""

from evo_mind.agent_swarm.agent import (
    BaseAgent,
    AnalyzerAgent,
    ExecutorAgent,
    ReviewerAgent,
)
from evo_mind.agent_swarm.coordinator import SwarmCoordinator
from evo_mind.agent_swarm.consensus import ConsensusEngine
from evo_mind.agent_swarm.plugin import AgentSwarmPlugin
from evo_mind.agent_swarm.task import (
    Task,
    TaskResult,
    TaskPriority,
    TaskStatus,
    SubtaskDecomposition,
)

__all__ = [
    "BaseAgent",
    "AnalyzerAgent",
    "ExecutorAgent",
    "ReviewerAgent",
    "SwarmCoordinator",
    "ConsensusEngine",
    "AgentSwarmPlugin",
    "Task",
    "TaskResult",
    "TaskPriority",
    "TaskStatus",
    "SubtaskDecomposition",
]
