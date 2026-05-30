"""ConsolidationEngine: transforms raw episodic memories into structured knowledge."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from evo_mind.utils import uuid7

from evo_mind.consolidation.deduplicator import Deduplicator
from evo_mind.consolidation.pruning import PruningPolicy
from evo_mind.consolidation.summarizer import HeuristicSummarizer
from evo_mind.core.models import ConsolidationResult, Memory, MemoryCreate
from evo_mind.core.store import MemoryStore
from evo_mind.embedding.local import LocalEmbeddingEngine
from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo
from evo_mind.persistence.vector_store import ChromaVectorStore
from evo_mind.retrieval.engine import RetrievalEngine
from evo_mind.types import (
    ConsolidationStatus,
    ConsolidationTrigger,
    MemorySource,
    MemoryStatus,
    MemoryType,
    RelationType,
)

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConsolidationEngine:
    """Transforms raw episodic memories into structured knowledge.

    Pipeline:
    1. Select candidate memories (unconsolidated, active)
    2. Group by embedding similarity
    3. Summarize each group into a new semantic memory
    4. Detect and merge near-duplicates
    5. Create relationships
    6. Apply pruning policy
    """

    def __init__(
        self,
        store: MemoryStore,
        retrieval: RetrievalEngine,
        embedding: LocalEmbeddingEngine,
        vector_store: ChromaVectorStore,
        db: Database,
        config: dict | None = None,
    ) -> None:
        self.store = store
        self.retrieval = retrieval
        self.embedding = embedding
        self.vector_store = vector_store
        self.db = db
        self.repo = MemoryRepo(db)

        # Configuration
        cfg = config or {}
        self.min_candidates = cfg.get("min_candidates", 20)
        self.max_candidates = cfg.get("max_candidates_per_run", 500)
        self.similarity_threshold = cfg.get("similarity_threshold", 0.75)
        self.dedup_threshold = cfg.get("dedup_threshold", 0.97)
        self.max_total_memories = cfg.get("max_total_memories", 100_000)
        self.min_importance = cfg.get("min_importance_to_keep", 0.05)
        self.max_age_days = cfg.get("default_max_age_days", 90)

        self.summarizer = HeuristicSummarizer()
        self.deduplicator = Deduplicator(vector_store, self.dedup_threshold)
        self.pruning = PruningPolicy(
            self.max_age_days, self.min_importance, self.max_total_memories
        )

    async def consolidate(
        self,
        *,
        trigger: ConsolidationTrigger = ConsolidationTrigger.MANUAL,
        min_candidates: int | None = None,
        max_candidates: int | None = None,
        dry_run: bool = False,
    ) -> ConsolidationResult:
        """Run the full consolidation pipeline."""
        run_id = str(uuid7())
        started_at = _now()

        min_cand = min_candidates or self.min_candidates
        max_cand = max_candidates or self.max_candidates

        logger.info("consolidation_started", run_id=run_id, trigger=trigger.value)

        # 1. Count and select candidates
        pending_count = await self.repo.count_by_status(MemoryStatus.ACTIVE)
        if pending_count < min_cand:
            logger.info("consolidation_skipped", reason="below_threshold", count=pending_count)
            return ConsolidationResult(
                run_id=run_id, groups_formed=0, summaries_generated=0,
                duplicates_merged=0, memories_pruned=0, patterns_extracted=0,
                compression_ratio=0.0,
            )

        candidates = await self.repo.list_by_status(MemoryStatus.ACTIVE, limit=max_cand)
        if not candidates:
            return ConsolidationResult(
                run_id=run_id, groups_formed=0, summaries_generated=0,
                duplicates_merged=0, memories_pruned=0, patterns_extracted=0,
                compression_ratio=0.0,
            )

        # Record the run
        await self.db.execute(
            """INSERT INTO consolidation_runs (id, started_at, trigger, candidates_count, status)
               VALUES (?, ?, ?, ?, 'running')""",
            (run_id, started_at, trigger.value, len(candidates)),
        )
        await self.db.commit()

        # Mark candidates as consolidating
        candidate_ids = [m.id for m in candidates]
        await self.repo.mark_consolidating(candidate_ids, run_id)

        try:
            if dry_run:
                return ConsolidationResult(
                    run_id=run_id, groups_formed=0, summaries_generated=0,
                    duplicates_merged=0, memories_pruned=0, patterns_extracted=0,
                    compression_ratio=0.0,
                )

            # 2. Group by embedding similarity
            groups = await self._cluster_candidates(candidates)
            groups_formed = len(groups)
            logger.info("groups_formed", count=groups_formed)

            # 3. Summarize each group
            summaries_generated = 0
            new_memory_ids: list[str] = []
            for group_ids in groups:
                if len(group_ids) < 2:
                    continue
                summary_mem = await self._summarize_group(group_ids, run_id)
                if summary_mem:
                    new_memory_ids.append(summary_mem.id)
                    summaries_generated += 1

            # 4. Deduplicate
            dup_merged = await self.deduplicator.deduplicate(self.repo)

            # 5. Prune
            total_count = await self.repo.count_total()
            pruned = 0
            if total_count > self.max_total_memories:
                pruned = await self.pruning.apply(
                    self.repo, self.vector_store, self.max_total_memories
                )

            # 6. Calculate metrics
            total_initial = len(candidates)
            total_final = summaries_generated
            compression_ratio = total_final / total_initial if total_initial > 0 else 0.0
            patterns_extracted = 0  # Will be done by EvolutionEngine

            # Mark run as completed
            end_time = _now()
            await self.db.execute(
                """UPDATE consolidation_runs
                   SET completed_at = ?, status = 'completed',
                       groups_formed = ?, summaries_generated = ?,
                       duplicates_merged = ?, memories_pruned = ?,
                       patterns_extracted = ?, compression_ratio = ?
                   WHERE id = ?""",
                (end_time, groups_formed, summaries_generated,
                 dup_merged, pruned, patterns_extracted, compression_ratio, run_id),
            )
            await self.db.commit()

            result = ConsolidationResult(
                run_id=run_id, groups_formed=groups_formed,
                summaries_generated=summaries_generated,
                duplicates_merged=dup_merged, memories_pruned=pruned,
                patterns_extracted=patterns_extracted,
                compression_ratio=round(compression_ratio, 4),
            )

            logger.info("consolidation_completed: groups=%s summaries=%s dedup=%s prune=%s ratio=%s",
                        result.groups_formed, result.summaries_generated,
                        result.duplicates_merged, result.memories_pruned,
                        result.compression_ratio)
            return result

        except Exception as e:
            logger.exception("consolidation_failed: run_id=%s error=%s", run_id, str(e))
            # Revert stuck 'consolidating' memories back to 'active'
            await self.db.execute(
                "UPDATE memories SET status = 'active' WHERE status = 'consolidating' AND consolidation_run_id = ?",
                (run_id,),
            )
            await self.db.execute(
                """UPDATE consolidation_runs
                   SET completed_at = ?, status = 'failed', error_message = ?
                   WHERE id = ?""",
                (_now(), str(e), run_id),
            )
            await self.db.commit()
            raise

    async def _cluster_candidates(self, candidates: list[Memory]) -> list[list[str]]:
        """Group memories by embedding similarity using simple threshold-based clustering."""
        # Load embeddings for all candidates
        embed_ids = [f"emb_{m.id}" for m in candidates if m.embedding_id]
        id_to_memory = {m.id: m for m in candidates}

        if len(embed_ids) < 2:
            return [[m.id] for m in candidates]

        # Get all embeddings at once
        try:
            all_embeddings = await self.vector_store.get_embeddings(embed_ids)
        except Exception:
            return [[m.id] for m in candidates]

        if not all_embeddings:
            return [[m.id] for m in candidates]

        # Simple greedy clustering by cosine similarity
        import numpy as np

        memory_ids = [eid.replace("emb_", "", 1) for eid in embed_ids]
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings_array / norms
        sim_matrix = np.dot(normalized, normalized.T)

        # Greedy clustering
        assigned: set[int] = set()
        groups: list[list[str]] = []

        for i in range(len(memory_ids)):
            if i in assigned:
                continue

            group = [memory_ids[i]]
            assigned.add(i)

            for j in range(i + 1, len(memory_ids)):
                if j in assigned:
                    continue
                if sim_matrix[i, j] >= self.similarity_threshold:
                    group.append(memory_ids[j])
                    assigned.add(j)

            groups.append(group)

        # Add any unprocessed candidates as singletons
        for mem_id in id_to_memory:
            if mem_id not in assigned and mem_id not in [m for g in groups for m in g]:
                groups.append([mem_id])

        return groups

    async def _summarize_group(
        self, memory_ids: list[str], run_id: str
    ) -> Memory | None:
        """Create a condensed semantic memory from a group of related memories."""
        memories = await self.store.get_by_ids(memory_ids)
        if len(memories) < 2:
            return None

        # Collect content for summarization
        texts: list[str] = []
        all_tags: set[str] = set()
        for mem in memories:
            if mem.content_plain:
                texts.append(mem.content_plain)
            else:
                texts.append(json.dumps(mem.content, ensure_ascii=False))
            tags = await self.repo.get_tags(mem.id)
            all_tags.update(tags)

        # Generate summary
        summary_text = await self.summarizer.summarize(texts)

        # Create the consolidated memory
        consolidated = MemoryCreate(
            memory_type=MemoryType.SEMANTIC,
            content={
                "text": summary_text,
                "source_count": len(memories),
                "source_ids": memory_ids[:20],  # Cap for size
                "source_types": list(set(m.memory_type.value for m in memories)),
            },
            importance=min(
                1.0,
                max(m.importance for m in memories) * 1.1,  # Slight boost
            ),
            source=MemorySource.CONSOLIDATION.value,
            tags=list(all_tags),
        )

        new_mem = await self.store.record(consolidated)

        # Create 'derives_from' relationships
        for source_id in memory_ids:
            await self.repo.relate(
                new_mem.id, source_id, RelationType.DERIVES_FROM,
                strength=0.9,
                evidence={"consolidation_run_id": run_id},
            )

        # Mark source memories as consolidated
        await self.repo.mark_consolidated(memory_ids, run_id)

        logger.debug("group_summarized", new_id=new_mem.id, source_count=len(memories))
        return new_mem

    async def get_pending_count(self) -> int:
        """Count of unconsolidated, active memories."""
        return await self.repo.count_by_status(MemoryStatus.ACTIVE)

    async def run_if_needed(self) -> ConsolidationResult | None:
        """Auto-trigger consolidation if pending count exceeds threshold."""
        count = await self.get_pending_count()
        if count >= self.min_candidates:
            return await self.consolidate(trigger=ConsolidationTrigger.THRESHOLD)
        return None
