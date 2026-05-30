"""Super Evolution — recursive self-improvement across all phases.

Phase 5: Meta-level orchestration enabling the system to:
- Introspect on its own performance
- Generate autonomous improvement goals
- Safely modify its own source code
- Run all 4 phases in a unified continuous loop
- Recursively improve its improvement process
"""

from evo_mind.super_evolution.meta_engine import (
    MetaEngine,
    SuperEvolutionState,
    SystemMetrics,
    SystemPhase,
    ImprovementAction,
)
from evo_mind.super_evolution.recursive_improver import (
    RecursiveImprover,
    CodeChange,
    ModificationResult,
)
from evo_mind.super_evolution.unified_loop import (
    UnifiedSuperLoop,
    LoopConfig,
    LoopState,
)
from evo_mind.super_evolution.plugin import SuperEvolutionPlugin

__all__ = [
    # Meta Engine
    "MetaEngine",
    "SuperEvolutionState",
    "SystemMetrics",
    "SystemPhase",
    "ImprovementAction",
    # Recursive Improver
    "RecursiveImprover",
    "CodeChange",
    "ModificationResult",
    # Unified Loop
    "UnifiedSuperLoop",
    "LoopConfig",
    "LoopState",
    # Plugin
    "SuperEvolutionPlugin",
]
