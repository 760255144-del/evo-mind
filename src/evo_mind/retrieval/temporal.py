"""Temporal search with recency decay scoring."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo


def _exponential_decay(
    created_at: datetime,
    half_life_hours: float = 168.0,  # Default: 7-day half-life
    reference_time: datetime | None = None,
) -> float:
    """Compute exponential decay score for a memory based on age.

    score = 2^(-age_hours / half_life_hours)
    """
    ref = reference_time or datetime.now(timezone.utc)
    age_hours = (ref - created_at).total_seconds() / 3600.0
    if age_hours < 0:
        age_hours = 0.0
    return math.pow(2, -age_hours / half_life_hours)


async def temporal_search(
    db: Database,
    n: int = 40,
    start: datetime | None = None,
    end: datetime | None = None,
    min_importance: float = 0.0,
    half_life_hours: float = 168.0,
) -> list[tuple[str, float, dict[str, float]]]:
    """Retrieve memories scored by recency (exponential decay) + importance."""
    repo = MemoryRepo(db)

    # Build query with optional time range
    conditions = ["deleted_at IS NULL"]
    params: list[object] = []

    if start:
        conditions.append("created_at >= ?")
        params.append(start.isoformat())
    if end:
        conditions.append("created_at <= ?")
        params.append(end.isoformat())
    if min_importance > 0:
        conditions.append("importance >= ?")
        params.append(min_importance)

    where = " AND ".join(conditions)
    query = f"""
        SELECT id, created_at, importance
        FROM memories
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT ?
    """
    params.append(n * 2)  # Get more candidates then re-score

    rows = await db.fetch_all(query, params)
    ref_time = end or datetime.now(timezone.utc)

    results: list[tuple[str, float, dict[str, float]]] = []
    for row in rows:
        created = datetime.fromisoformat(row["created_at"])
        decay = _exponential_decay(created, half_life_hours, ref_time)
        importance = row["importance"]
        # Combine decay and importance
        score = 0.7 * decay + 0.3 * importance
        results.append((row["id"], score, {"temporal": score, "decay": decay, "importance_bonus": importance * 0.3}))

    # Sort by combined score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]
