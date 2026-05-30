"""Near-duplicate detection via embedding distance."""

from __future__ import annotations

import logging
import numpy as np

from evo_mind.persistence.memory_repo import MemoryRepo
from evo_mind.persistence.vector_store import ChromaVectorStore
from evo_mind.types import MemoryStatus, RelationType

logger = logging.getLogger(__name__)


class Deduplicator:
    """Detects and merges near-duplicate memories using embedding cosine distance."""

    def __init__(self, vector_store: ChromaVectorStore, threshold: float = 0.97) -> None:
        self.vector_store = vector_store
        self.threshold = threshold

    async def deduplicate(self, repo: MemoryRepo) -> int:
        """Find and merge near-duplicate memory pairs.

        Returns the number of memories merged.
        """
        # Get recent active/consolidated memories
        active = await repo.list_recent(limit=500)
        if len(active) < 2:
            return 0

        # Get embeddings for all
        embed_ids = [f"emb_{m.id}" for m in active if m.embedding_id]
        if len(embed_ids) < 2:
            return 0

        try:
            embeddings = await self.vector_store.get_embeddings(embed_ids)
        except Exception:
            logger.warning("dedup_embedding_load_failed")
            return 0

        if not embeddings or len(embeddings) < 2:
            return 0

        # Compute pairwise cosine similarity
        emb_array = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = emb_array / norms
        sim_matrix = np.dot(normalized, normalized.T)

        # Find pairs above threshold
        memory_ids = [eid.replace("emb_", "", 1) for eid in embed_ids]
        id_to_memory = {m.id: m for m in active}  # O(1) lookup dict
        merged_count = 0
        merged: set[str] = set()

        for i in range(len(memory_ids)):
            if memory_ids[i] in merged:
                continue
            for j in range(i + 1, len(memory_ids)):
                if memory_ids[j] in merged:
                    continue
                if sim_matrix[i, j] >= self.threshold:
                    mem_i = id_to_memory.get(memory_ids[i])
                    mem_j = id_to_memory.get(memory_ids[j])

                    if mem_i and mem_j:
                        if mem_i.importance >= mem_j.importance:
                            keeper, victim = mem_i, mem_j
                        else:
                            keeper, victim = mem_j, mem_i

                        # Merge tags from victim to keeper
                        victim_tags = await repo.get_tags(victim.id)
                        if victim_tags:
                            await repo.add_tags(keeper.id, victim_tags)

                        # Create supersedes relationship
                        await repo.relate(
                            keeper.id, victim.id, RelationType.SUPERSEDES,
                            strength=sim_matrix[i, j],
                        )

                        # Clean up victim's vector before archiving
                        if victim.embedding_id:
                            try:
                                await self.vector_store.delete([victim.embedding_id])
                            except Exception:
                                logger.debug("victim_vector_cleanup_failed", id=victim.id[:12])
                        await repo.archive(victim.id)
                        merged.add(victim.id)
                        merged_count += 1

                        logger.debug(
                            "duplicates_merged",
                            keeper=keeper.id[:8],
                            victim=victim.id[:8],
                            similarity=round(sim_matrix[i, j], 4),
                        )

        if merged_count > 0:
            logger.info("dedup_completed", merged=merged_count)

        return merged_count
