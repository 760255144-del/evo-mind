"""Semantic search via ChromaDB vector similarity."""

from __future__ import annotations

from typing import Any

from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo
from evo_mind.persistence.vector_store import ChromaVectorStore


async def semantic_search(
    db: Database,
    vector_store: ChromaVectorStore,
    query_embedding: list[float],
    n: int = 40,
    where: dict[str, object] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Find memories by vector similarity.

    Returns list of (memory_id, similarity_score, {breakdown}).
    """
    ids, distances = await vector_store.query(query_embedding, n_results=n, where=where)

    results: list[tuple[str, float, dict[str, float]]] = []
    repo = MemoryRepo(db)

    for vec_id, distance in zip(ids, distances):
        # vec_id format: "emb_{memory_id}"
        memory_id = vec_id.replace("emb_", "", 1) if vec_id.startswith("emb_") else vec_id
        memory = await repo.get(memory_id)
        if memory is not None:
            # Convert cosine distance [0,2] to similarity [0,1]
            similarity = 1.0 - (distance / 2.0)
            results.append((memory_id, similarity, {"semantic": similarity}))

    return results


async def find_contradictions(
    db: Database,
    vector_store: ChromaVectorStore,
    query_embedding: list[float],
    n: int = 10,
) -> list[tuple[str, float, dict[str, float]]]:
    """Find memories semantically distant (potential contradictions).

    Gets results with lowest similarity scores, representing maximum distance.
    """
    # Query for more results, then take the most distant
    ids, distances = await vector_store.query(query_embedding, n_results=n * 5)

    # Sort by distance descending (most contradictory first)
    repo = MemoryRepo(db)
    results: list[tuple[str, float, dict[str, float]]] = []

    # Process in reverse order (furthest first)
    sorted_pairs = sorted(
        [(i, d) for i, d in zip(ids, distances)], key=lambda x: x[1], reverse=True
    )

    for vec_id, distance in sorted_pairs[:n]:
        memory_id = vec_id.replace("emb_", "", 1) if vec_id.startswith("emb_") else vec_id
        memory = await repo.get(memory_id)
        if memory:
            results.append((memory_id, distance, {"semantic_contradiction": distance}))

    return results
