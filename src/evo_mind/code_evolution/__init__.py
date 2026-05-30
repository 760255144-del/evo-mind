"""Code Evolution — self-optimizing code pipeline integrated with evo-mind."""

from evo_mind.code_evolution.engine import CodeEvolutionEngine, CodeIssue, CodeFix, OptimizationResult
from evo_mind.code_evolution.loop import SelfImprovementLoop, ImprovementStats
from evo_mind.code_evolution.plugin import CodeEvolutionPlugin

__all__ = [
    "CodeEvolutionEngine",
    "CodeIssue",
    "CodeFix",
    "OptimizationResult",
    "SelfImprovementLoop",
    "ImprovementStats",
    "CodeEvolutionPlugin",
]
