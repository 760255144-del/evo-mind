"""MemoryStore: Central CRUD interface wiring repo + vector store + embedding."""

from __future__ import annotations

import logging
from typing import Any

from evo_mind.core.models import Memory, MemoryCreate
from evo_mind.embedding.local import LocalEmbeddingEngine
from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo
from evo_mind.persistence.vector_store import ChromaVectorStore
from evo_mind.types import MemoryStatus, MemoryType, RelationType

logger = logging.getLogger(__name__)


class MemoryStore:
    """Central CRUD interface for all memory types.

    Orchestrates the MemoryRepo (SQLite), ChromaVectorStore (embeddings),
    and LocalEmbeddingEngine (text → vector).
    """

    def __init__(
        self,
        db: Database,
        vector_store: ChromaVectorStore,
        embedding: LocalEmbeddingEngine,
        repo: MemoryRepo | None = None,
    ) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embedding = embedding
        self.repo = repo or MemoryRepo(db)

    # ---- Create ----

    async def record(
        self,
        memory: MemoryCreate,
        *,
        auto_embed: bool = True,
        relate_to: list[str] | None = None,
    ) -> Memory:
        """Record a new memory, embed it, and optionally create relationships."""
        # 1. Check for duplicate by hash
        import xxhash
        import json

        content_json = json.dumps(memory.content, ensure_ascii=False, sort_keys=True)
        content_hash = xxhash.xxh64(content_json).hexdigest()
        existing = await self.repo.get_by_hash(content_hash)
        if existing is not None:
            logger.debug("duplicate_memory_skipped", hash=content_hash)
            return existing

        # 2. Create in SQLite
        mem = await self.repo.create(memory)

        # 3. Generate embedding and store in vector DB
        if auto_embed:
            try:
                plain_text = self.repo._extract_plain_text(memory.content)
                vec = await self.embedding.encode(plain_text)
                embed_id = f"emb_{mem.id}"
                await self.vector_store.add(
                    ids=[embed_id],
                    embeddings=[vec],
                    metadatas=[{
                        "memory_id": mem.id,
                        "memory_type": mem.memory_type.value,
                        "importance": mem.importance,
                    }],
                    documents=[plain_text],
                )
                await self.repo.set_embedding_id(mem.id, embed_id)
                # Refresh the memory object
                refreshed = await self.repo.get(mem.id)
                if refreshed is None:
                    raise RuntimeError(f"Memory {mem.id} not found after creation")
                mem = refreshed
            except Exception:
                logger.exception("embedding_failed", memory_id=mem.id)
                # Memory is still usable without embedding

        # 4. Create relationships
        if relate_to:
            for target_id in relate_to:
                await self.repo.relate(
                    source_id=mem.id,
                    target_id=target_id,
                    relation_type=RelationType.REFERENCES,
                )

        logger.info("memory_recorded", id=mem.id, type=mem.memory_type.value)
        return mem

    async def record_batch(
        self, memories: list[MemoryCreate], *, auto_embed: bool = True
    ) -> list[Memory]:
        """Record multiple memories efficiently."""
        results: list[Memory] = []
        for mem in memories:
            result = await self.record(mem, auto_embed=auto_embed)
            results.append(result)
        return results

    # ---- Read ----

    async def get(self, memory_id: str) -> Memory | None:
        return await self.repo.get(memory_id)

    async def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        placeholders = ",".join(["?"] * len(memory_ids))
        rows = await self.db.fetch_all(
            f"SELECT rowid, * FROM memories WHERE id IN ({placeholders}) AND deleted_at IS NULL",
            memory_ids,
        )
        return [self.repo._row_to_memory(r) for r in rows]

    async def get_by_hash(self, content_hash: str) -> Memory | None:
        return await self.repo.get_by_hash(content_hash)

    async def list_by_session(self, session_id: str, limit: int = 100) -> list[Memory]:
        return await self.repo.list_by_session(session_id, limit)

    async def list_recent(
        self, limit: int = 50, memory_type: MemoryType | None = None
    ) -> list[Memory]:
        return await self.repo.list_recent(limit, memory_type)

    async def list_by_status(
        self, status: MemoryStatus, limit: int = 100
    ) -> list[Memory]:
        return await self.repo.list_by_status(status, limit)

    # ---- Update ----

    async def update(
        self,
        memory_id: str,
        *,
        content: dict[str, Any] | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
        re_embed: bool = False,
    ) -> Memory | None:
        if content is not None:
            if re_embed:
                # Save old embedding_id before content update clears it
                mem = await self.repo.get(memory_id)
                old_embed_id = mem.embedding_id if mem else None

            await self.repo.update_content(memory_id, content, re_embed=re_embed)

            if re_embed:
                mem = await self.repo.get(memory_id)
                if mem and mem.content_plain:
                    vec = await self.embedding.encode(mem.content_plain)
                    embed_id = f"emb_{memory_id}"
                    # Remove old embedding if it existed
                    if old_embed_id:
                        try:
                            await self.vector_store.delete([old_embed_id])
                        except Exception:
                            logger.debug("old_embedding_cleanup_failed", embed_id=old_embed_id)
                    await self.vector_store.add(
                        ids=[embed_id],
                        embeddings=[vec],
                        metadatas=[{"memory_id": mem.id, "memory_type": mem.memory_type.value}],
                        documents=[mem.content_plain],
                    )
                    await self.repo.set_embedding_id(memory_id, embed_id)

        if importance is not None:
            await self.repo.update_importance(memory_id, importance)

        if tags is not None:
            # Replace all tags
            current = await self.repo.get_tags(memory_id)
            if current:
                await self.repo.remove_tags(memory_id, current)
            if tags:
                await self.repo.add_tags(memory_id, tags)

        return await self.repo.get(memory_id)

    async def record_access(self, memory_id: str) -> None:
        await self.repo.record_access(memory_id)

    async def set_status(self, memory_id: str, status: MemoryStatus) -> None:
        await self.repo.set_status(memory_id, status)

    # ---- Delete ----

    async def archive(self, memory_id: str) -> None:
        """Soft-delete (status = archived)."""
        await self.repo.archive(memory_id)
        logger.info("memory_archived", id=memory_id)

    async def prune(self, memory_id: str) -> None:
        """Hard-delete from SQLite and vector store."""
        mem = await self.repo.get(memory_id)
        if mem and mem.embedding_id:
            try:
                await self.vector_store.delete([mem.embedding_id])
            except Exception:
                logger.warning("vector_delete_failed", embedding_id=mem.embedding_id)
        await self.repo.prune(memory_id)
        logger.info("memory_pruned", id=memory_id)

    # ---- Relationships ----

    async def relate(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        return await self.repo.relate(source_id, target_id, relation_type, strength, evidence)

    async def get_related(
        self,
        memory_id: str,
        relation_types: list[RelationType] | None = None,
    ) -> list[tuple[Memory, RelationType, float]]:
        return await self.repo.get_related(memory_id, relation_types)

    # ---- Tags ----

    async def add_tags(self, memory_id: str, tags: list[str]) -> None:
        await self.repo.add_tags(memory_id, tags)

    async def remove_tags(self, memory_id: str, tags: list[str]) -> None:
        await self.repo.remove_tags(memory_id, tags)

    async def get_tags(self, memory_id: str) -> list[str]:
        return await self.repo.get_tags(memory_id)

    # ---- Sessions ----

    async def start_session(self, metadata: dict[str, Any] | None = None) -> str:
        return await self.repo.start_session(metadata)

    async def end_session(self, session_id: str, summary: str | None = None) -> None:
        await self.repo.end_session(session_id, summary)

    async def get_session(self, session_id: str) -> dict | None:
        session = await self.repo.get_session(session_id)
        if session is None:
            return None
        return {
            "id": session.id,
            "started_at": session.started_at.isoformat(),
            "ended_at": session.ended_at.isoformat() if session.ended_at else None,
            "summary": session.summary,
            "memory_count": session.memory_count,
            "metadata": session.metadata,
        }

    # ---- Stats ----

    async def count_by_status(self, status: MemoryStatus) -> int:
        return await self.repo.count_by_status(status)

    async def count_total(self) -> int:
        return await self.repo.count_total()
