"""RetrievalEngine: multi-strategy memory retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from evo_mind.core.models import Memory, SearchQuery, SearchResult
from evo_mind.embedding.local import LocalEmbeddingEngine
from evo_mind.persistence.database import Database
from evo_mind.persistence.vector_store import ChromaVectorStore
from evo_mind.retrieval.fusion import reciprocal_rank_fusion
from evo_mind.retrieval.keyword import keyword_search
from evo_mind.retrieval.semantic import semantic_search
from evo_mind.retrieval.temporal import temporal_search
from evo_mind.types import MemoryType, RelationType

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Multi-strategy memory retrieval with configurable fusion.

    Executes semantic + keyword + temporal search in parallel,
    fuses results via Reciprocal Rank Fusion with configurable weights.
    """

    def __init__(
        self,
        db: Database,
        vector_store: ChromaVectorStore,
        embedding: LocalEmbeddingEngine,
    ) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embedding = embedding

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Execute multi-strategy search and return fused, scored results."""
        # 1. Get query embedding if needed
        if query.query_text and not query.query_embedding:
            query_embedding = await self.embedding.encode(query.query_text)
        elif query.query_embedding:
            query_embedding = query.query_embedding
        else:
            query_embedding = None

        # 2. Build metadata filter for vector search
        where_clause: dict[str, object] | None = None
        if query.memory_types:
            where_clause = {
                "memory_type": {"$in": [mt.value for mt in query.memory_types]}
            }

        # 3. Run strategies in parallel
        expanded = max(query.max_results * 2, 40)
        tasks = []

        # Semantic search
        if query.semantic_weight > 0 and query_embedding is not None:
            tasks.append(
                semantic_search(
                    self.db,
                    self.vector_store,
                    query_embedding,
                    n=expanded,
                    where=where_clause,
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        # Keyword search
        if query.keyword_weight > 0 and query.query_text:
            tasks.append(
                keyword_search(self.db, query.query_text, n=expanded)
            )
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        # Temporal search
        if query.temporal_weight > 0:
            tasks.append(
                temporal_search(
                    self.db,
                    n=expanded,
                    start=query.start_time,
                    end=query.end_time,
                    min_importance=query.min_importance,
                )
            )
        else:
            tasks.append(asyncio.sleep(0, result=[]))

        results_groups = await asyncio.gather(*tasks)

        # 4. Fuse results
        strategy_weights = [
            query.semantic_weight,
            query.keyword_weight,
            query.temporal_weight,
        ]
        fused = reciprocal_rank_fusion(results_groups, strategy_weights, k=60)

        # 5. Limit and build SearchResults
        fused = fused[: query.max_results]

        search_results: list[SearchResult] = []
        for item in fused:
            memory_id = item.get("memory_id", "")
            if not memory_id:
                continue
            memory = item.get("memory")  # fusion may or may not hydrate
            if memory is None or not isinstance(memory, Memory):
                # Hydrate from DB if fusion only returned memory_id
                from evo_mind.persistence.memory_repo import MemoryRepo
                memory = await (MemoryRepo(self.db).get(memory_id))
                if memory is None:
                    continue
            score = item["score"]
            score_breakdown = {
                k: v for k, v in item["score_breakdown"].items() if v > 0
            }

            # Record access (tracked background task)
            asyncio.create_task(self._record_access(memory.id))

            # Enrich with relationships if requested
            related: list[Memory] = []
            if query.include_relationships:
                related = await self._get_related_memories(memory.id)

            search_results.append(
                SearchResult(
                    memory=memory,
                    score=score,
                    score_breakdown=score_breakdown,
                    related_memories=related,
                )
            )

        logger.info(
            "search_completed",
            query=query.query_text[:50] if query.query_text else None,
            results=len(search_results),
        )
        return search_results

    async def find_similar(self, memory_id: str, n: int = 10) -> list[SearchResult]:
        """Find memories similar to a given memory by embedding distance."""
        from evo_mind.persistence.memory_repo import MemoryRepo

        repo = MemoryRepo(self.db)
        memory = await repo.get(memory_id)
        if not memory or not memory.embedding_id:
            return []

        # Get the embedding
        vecs = await self.vector_store.get_embeddings([memory.embedding_id])
        if not vecs or not vecs[0]:
            return []

        query = SearchQuery(
            query_embedding=vecs[0],
            max_results=n,
            semantic_weight=1.0,
            keyword_weight=0.0,
            temporal_weight=0.0,
        )
        return await self.search(query)

    async def find_context(
        self, query_text: str, window: int = 10
    ) -> list[SearchResult]:
        """Retrieve memories plus their temporal neighbors for context."""
        result = await self.search(
            SearchQuery(
                query_text=query_text,
                max_results=5,
                include_relationships=True,
            )
        )
        return result

    async def _record_access(self, memory_id: str) -> None:
        """Record memory access in the background."""
        try:
            from evo_mind.persistence.memory_repo import MemoryRepo

            repo = MemoryRepo(self.db)
            await repo.record_access(memory_id)
        except Exception:
            logger.debug("access_record_failed", memory_id=memory_id, exc_info=True)

    async def _get_related_memories(self, memory_id: str) -> list[Memory]:
        """Fetch one-hop related memories."""
        try:
            from evo_mind.persistence.memory_repo import MemoryRepo

            repo = MemoryRepo(self.db)
            relations = await repo.get_related(memory_id)
            return [r[0] for r in relations[:10]]  # Limit to 10
        except Exception:
            return []
