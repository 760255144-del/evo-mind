"""evo-mind: Self-Evolution Memory System for AI agents.

A production-grade memory system that enables AI to evolve across sessions
by recording experiences, consolidating knowledge, retrieving relevant
context, and detecting patterns to generate adaptive rules.

Phases:
  1. Memory System (core)
  2. Code Self-Optimization (code_evolution)
  3. Agent Swarm (agent_swarm)
  4. Evolutionary Algorithms (evolutionary)

Usage:
    from evo_mind import EvoMindConfig, MemoryStore, RetrievalEngine
    from evo_mind.code_evolution import CodeEvolutionEngine
    from evo_mind.agent_swarm import SwarmCoordinator
    from evo_mind.evolutionary import EvolutionaryEngine
"""

from evo_mind.config import (
    ChromaConfig,
    ConsolidationConfig,
    DatabaseConfig,
    EmbeddingConfig,
    EvolutionConfig,
    EvoMindConfig,
)
from evo_mind.core.models import (
    EvolutionRule,
    Memory,
    MemoryCreate,
    SearchQuery,
    SearchResult,
)
from evo_mind.core.store import MemoryStore
from evo_mind.retrieval.engine import RetrievalEngine
from evo_mind.types import MemoryType, MemoryStatus, RelationType, RuleType

__version__ = "0.2.0"
__all__ = [
    # Config
    "EvoMindConfig",
    "ChromaConfig",
    "ConsolidationConfig",
    "DatabaseConfig",
    "EmbeddingConfig",
    "EvolutionConfig",
    # Phase 1: Core
    "MemoryStore",
    "RetrievalEngine",
    # Models
    "Memory",
    "MemoryCreate",
    "SearchQuery",
    "SearchResult",
    "EvolutionRule",
    # Types
    "MemoryType",
    "MemoryStatus",
    "RelationType",
    "RuleType",
]
