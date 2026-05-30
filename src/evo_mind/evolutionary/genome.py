"""Genome — the genetic representation of an evolvable strategy."""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from evo_mind.utils import uuid7


@dataclass
class Gene:
    """A single evolvable parameter."""
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    mutation_rate: float = 0.1  # Probability of mutation
    mutation_strength: float = 0.1  # Maximum change per mutation


@dataclass
class Genome:
    """A collection of evolvable genes forming a complete strategy.

    Represents one individual in the evolutionary population.
    """

    id: str = field(default_factory=uuid7)
    genes: dict[str, Gene] = field(default_factory=dict)
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.genes:
            self.genes = self._default_genes()

    @staticmethod
    def _default_genes() -> dict[str, Gene]:
        """Create the default strategy genes."""
        return {
            "semantic_weight": Gene("semantic_weight", 1.0, 0.0, 3.0, 0.1, 0.2),
            "keyword_weight": Gene("keyword_weight", 0.5, 0.0, 2.0, 0.1, 0.15),
            "temporal_weight": Gene("temporal_weight", 0.3, 0.0, 1.5, 0.1, 0.1),
            "importance_default": Gene("importance_default", 0.5, 0.0, 1.0, 0.05, 0.05),
            "consolidation_threshold": Gene("consolidation_threshold", 0.75, 0.5, 0.95, 0.05, 0.05),
            "dedup_threshold": Gene("dedup_threshold", 0.97, 0.8, 1.0, 0.03, 0.02),
            "prune_age_days": Gene("prune_age_days", 90.0, 7.0, 365.0, 0.1, 15.0),
            "max_memories": Gene("max_memories", 100000.0, 1000.0, 1000000.0, 0.05, 5000.0),
        }

    def get_params(self) -> dict[str, float]:
        """Extract current parameter values as a dict."""
        return {name: gene.value for name, gene in self.genes.items()}

    def clamp(self) -> None:
        """Clamp all gene values to their valid ranges."""
        for gene in self.genes.values():
            gene.value = max(gene.min_value, min(gene.max_value, gene.value))

    def copy(self) -> Genome:
        """Create a deep copy of this genome."""
        return deepcopy(self)
