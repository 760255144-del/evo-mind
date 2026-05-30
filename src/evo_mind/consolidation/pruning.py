"""Forgetting policy — importance/age-based memory pruning."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo
from evo_mind.persistence.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class PruningPolicy:
    """Age and importance-based forgetting.

    Score = importance * e^(-lambda * age_days)
    Memories below threshold are pruned.
    """

    def __init__(
        self,
        max_age_days: int = 90,
        min_importance: float = 0.05,
        max_memories: int = 100_000,
    ) -> None:
        self.max_age_days = max_age_days
        self.min_importance = min_importance
        self.max_memories = max_memories

    async def apply(
        self,
        repo: MemoryRepo,
        vector_store: ChromaVectorStore,
        max_total: int | None = None,
    ) -> int:
        """Apply the pruning policy. Returns number of memories pruned."""
        max_mem = max_total or self.max_memories
        total = await repo.count_total()

        if total <= max_mem:
            return 0

        # Get candidates for pruning: fetch all active/consolidated memories
        # ordered by importance ascending, created_at ascending (oldest first)
        pruned = 0
        to_prune = total - max_mem
        now = datetime.now(timezone.utc)

        # Fetch lowest-importance, oldest memories
        rows = await repo.db.fetch_all(
            """SELECT id, importance, created_at, embedding_id
               FROM memories
               WHERE deleted_at IS NULL
                 AND status IN ('active', 'consolidated', 'archived')
               ORDER BY importance ASC, created_at ASC
               LIMIT ?""",
            (to_prune * 2,),  # Get more to account for scoring filter
        )

        if not rows:
            return 0

        # Score and sort
        scored: list[tuple[str, str | None, float]] = []
        for row in rows:
            created = datetime.fromisoformat(row["created_at"])
            age_days = (now - created).days
            importance = row["importance"]

            # Prune if below importance threshold and old enough
            if importance < self.min_importance and age_days > self.max_age_days:
                scored.append((row["id"], row["embedding_id"], -1.0))  # Force prune
            else:
                # Score: lower = more likely to prune
                # Combine low importance with high age
                decay = math.exp(-0.01 * age_days)
                score = importance * decay
                scored.append((row["id"], row["embedding_id"], score))

        # Take lowest-scoring first
        scored.sort(key=lambda x: x[2])
        to_remove = scored[:to_prune]

        for mem_id, embedding_id, _ in to_remove:
            try:
                # Remove from vector store
                if embedding_id:
                    await vector_store.delete([embedding_id])
                # Hard delete from SQLite
                await repo.prune(mem_id)
                pruned += 1
            except Exception:
                logger.warning("prune_failed", memory_id=mem_id, exc_info=True)

        if pruned > 0:
            logger.info("pruning_completed", pruned=pruned, remaining=total - pruned)

        return pruned


def _compute_prune_score(
    importance: float,
    created_at: datetime,
    access_count: int,
    last_accessed: datetime | None,
    now: datetime,
) -> float:
    """Compute a composite pruning score. Lower = more likely to be pruned."""
    age_days = max(0, (now - created_at).days)
    days_since_access = (
        max(0, (now - last_accessed).days) if last_accessed else age_days
    )

    # Components (all in [0, 1]):
    importance_factor = importance
    age_penalty = math.exp(-0.005 * age_days)  # Slow decay
    access_penalty = math.exp(-0.05 * days_since_access)  # Faster decay for unused

    return importance_factor * (0.5 * age_penalty + 0.5 * access_penalty)
