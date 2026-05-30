"""Genetic operators: selection, crossover, mutation."""

from __future__ import annotations

import random
from typing import Callable

from evo_mind.evolutionary.genome import Genome


# ---- Selection ----

def tournament_selection(
    population: list[Genome],
    tournament_size: int = 3,
) -> Genome:
    """Select the fittest genome from a random tournament of N individuals.

    Higher tournament_size = stronger selection pressure.
    """
    if not population:
        raise ValueError("Cannot select from empty population")
    if len(population) <= tournament_size:
        return max(population, key=lambda g: g.fitness)

    candidates = random.sample(population, tournament_size)
    return max(candidates, key=lambda g: g.fitness)


def roulette_selection(population: list[Genome]) -> Genome:
    """Fitness-proportionate (roulette wheel) selection.

    Genomes with higher fitness have higher probability of being selected.
    """
    if not population:
        raise ValueError("Cannot select from empty population")

    # Shift fitnesses to ensure non-negative
    min_fitness = min(g.fitness for g in population)
    shifted = [g.fitness - min_fitness + 0.001 for g in population]
    total = sum(shifted)

    if total == 0:
        return random.choice(population)

    r = random.uniform(0, total)
    cumulative = 0.0
    for genome, fitness in zip(population, shifted):
        cumulative += fitness
        if cumulative >= r:
            return genome

    return population[-1]


def elitist_selection(
    population: list[Genome],
    elite_count: int = 2,
) -> list[Genome]:
    """Preserve the top N elite genomes unchanged for the next generation."""
    sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
    return [g.copy() for g in sorted_pop[:elite_count]]


# ---- Crossover ----

def uniform_crossover(
    parent1: Genome,
    parent2: Genome,
    crossover_rate: float = 0.5,
) -> tuple[Genome, Genome]:
    """Uniform crossover: each gene has `crossover_rate` chance of swapping.

    Returns two offspring genomes.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    child1.parent_ids = [parent1.id, parent2.id]
    child2.parent_ids = [parent1.id, parent2.id]

    for gene_name in parent1.genes:
        if random.random() < crossover_rate:
            # Swap genes
            child1.genes[gene_name].value = parent2.genes[gene_name].value
            child2.genes[gene_name].value = parent1.genes[gene_name].value

    child1.clamp()
    child2.clamp()
    return child1, child2


def blend_crossover(
    parent1: Genome,
    parent2: Genome,
    alpha: float = 0.5,
) -> tuple[Genome, Genome]:
    """BLX-alpha crossover: offspring genes are random blend between parents.

    For each gene: child = random between [p1 - alpha*(p2-p1), p2 + alpha*(p2-p1)]
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    child1.parent_ids = [parent1.id, parent2.id]
    child2.parent_ids = [parent1.id, parent2.id]

    for gene_name in parent1.genes:
        g1 = parent1.genes[gene_name]
        g2 = parent2.genes[gene_name]

        lo = min(g1.value, g2.value) - alpha * abs(g2.value - g1.value)
        hi = max(g1.value, g2.value) + alpha * abs(g2.value - g1.value)

        child1.genes[gene_name].value = random.uniform(lo, hi)
        child2.genes[gene_name].value = random.uniform(lo, hi)

    child1.clamp()
    child2.clamp()
    return child1, child2


# ---- Mutation ----

def gaussian_mutation(
    genome: Genome,
    mutation_strength: float | None = None,
) -> Genome:
    """Apply Gaussian noise to each gene based on its individual mutation rate.

    Each gene mutates with probability = gene.mutation_rate.
    Mutation amount ~ N(0, mutation_strength * gene.range).
    """
    mutant = genome.copy()

    for gene in mutant.genes.values():
        if random.random() < gene.mutation_rate:
            strength = mutation_strength or gene.mutation_strength
            delta = random.gauss(0, strength * (gene.max_value - gene.min_value))
            gene.value += delta

    mutant.clamp()
    return mutant


def resetting_mutation(
    genome: Genome,
    mutation_rate: float = 0.05,
) -> Genome:
    """With probability `mutation_rate`, reset a gene to a random value in its range."""
    mutant = genome.copy()

    for gene in mutant.genes.values():
        if random.random() < mutation_rate:
            gene.value = random.uniform(gene.min_value, gene.max_value)

    return mutant


# ---- Diversity ----

def population_diversity(population: list[Genome]) -> float:
    """Measure genetic diversity as average pairwise Euclidean distance of gene values."""
    if len(population) < 2:
        return 0.0

    gene_names = list(population[0].genes.keys())
    total_dist = 0.0
    pairs = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            dist = sum(
                (population[i].genes[g].value - population[j].genes[g].value) ** 2
                for g in gene_names
            ) ** 0.5
            total_dist += dist
            pairs += 1

    return total_dist / pairs if pairs > 0 else 0.0
