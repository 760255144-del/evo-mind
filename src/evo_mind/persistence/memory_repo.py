"""Repository: Memory CRUD operations against SQLite."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from evo_mind.utils import uuid7

from evo_mind.core.models import (
    Memory,
    MemoryCreate,
    Session,
)
from evo_mind.persistence.database import Database
from evo_mind.types import MemorySource, MemoryStatus, MemoryType, RelationType

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dt(val: str | None) -> datetime | None:
    if val is None:
        return None
    return datetime.fromisoformat(val)


# ---- SQL constants ----

SQL_INSERT_MEMORY = """
INSERT INTO memories (id, memory_type, content_json, content_hash, content_plain,
    embedding_id, importance, access_count, last_accessed_at, created_at, updated_at,
    session_id, source, status, consolidation_version, metadata_json)
VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?, ?, 'active', 0, ?)
"""

SQL_SELECT_MEMORY_BY_ID = """
SELECT rowid, * FROM memories WHERE id = ? AND deleted_at IS NULL
"""

SQL_SELECT_MEMORY_BY_HASH = """
SELECT rowid, * FROM memories WHERE content_hash = ? AND deleted_at IS NULL
"""

SQL_SELECT_MEMORIES_BY_SESSION = """
SELECT rowid, * FROM memories WHERE session_id = ? AND deleted_at IS NULL
ORDER BY created_at DESC LIMIT ?
"""

SQL_SELECT_RECENT_MEMORIES = """
SELECT rowid, * FROM memories WHERE deleted_at IS NULL
{type_filter}
ORDER BY created_at DESC LIMIT ?
"""

SQL_SELECT_MEMORIES_BY_STATUS = """
SELECT rowid, * FROM memories
WHERE status = ? AND deleted_at IS NULL
ORDER BY created_at LIMIT ?
"""

SQL_UPDATE_MEMORY_CONTENT = """
UPDATE memories SET content_json = ?, content_hash = ?, content_plain = ?,
    embedding_id = ?, updated_at = ?
WHERE id = ?
"""

SQL_UPDATE_MEMORY_IMPORTANCE = """
UPDATE memories SET importance = ?, updated_at = ? WHERE id = ?
"""

SQL_UPDATE_MEMORY_ACCESS = """
UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?
"""

SQL_UPDATE_MEMORY_STATUS = """
UPDATE memories SET status = ?, updated_at = ? WHERE id = ?
"""

SQL_UPDATE_MEMORY_CONSOLIDATION = """
UPDATE memories SET status = ?, consolidation_version = consolidation_version + 1,
    consolidation_run_id = ?, updated_at = ?
WHERE id = ?
"""

SQL_SOFT_DELETE_MEMORY = """
UPDATE memories SET deleted_at = ?, updated_at = ? WHERE id = ?
"""

SQL_HARD_DELETE_MEMORY = "DELETE FROM memories WHERE id = ?"

SQL_COUNT_BY_STATUS = "SELECT COUNT(*) as cnt FROM memories WHERE status = ? AND deleted_at IS NULL"

SQL_COUNT_TOTAL = "SELECT COUNT(*) as cnt FROM memories WHERE deleted_at IS NULL"

SQL_INSERT_RELATIONSHIP = """
INSERT INTO memory_relationships (id, source_id, target_id, relation_type, strength, evidence_json, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

SQL_SELECT_RELATIONSHIPS = """
SELECT m.rowid, m.*, mr.relation_type, mr.strength
FROM memory_relationships mr
JOIN memories m ON m.id = mr.target_id
WHERE mr.source_id = ?
"""

SQL_SELECT_RELATIONSHIPS_FILTERED = """
SELECT m.rowid, m.*, mr.relation_type, mr.strength
FROM memory_relationships mr
JOIN memories m ON m.id = mr.target_id
WHERE mr.source_id = ? AND mr.relation_type IN ({})
"""

# Tags
SQL_INSERT_TAG = "INSERT OR IGNORE INTO tags (id, name, created_at) VALUES (?, ?, ?)"
SQL_INSERT_MEMORY_TAG = "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)"
SQL_SELECT_TAGS_FOR_MEMORY = """
SELECT t.name FROM tags t JOIN memory_tags mt ON t.id = mt.tag_id WHERE mt.memory_id = ?
"""
SQL_DELETE_MEMORY_TAGS = "DELETE FROM memory_tags WHERE memory_id = ?"

# Sessions
SQL_INSERT_SESSION = "INSERT INTO sessions (id, started_at, metadata_json) VALUES (?, ?, ?)"
SQL_UPDATE_SESSION_END = "UPDATE sessions SET ended_at = ?, summary = ?, memory_count = ? WHERE id = ?"
SQL_SELECT_SESSION = "SELECT * FROM sessions WHERE id = ?"


class MemoryRepo:
    """Async repository for Memory and related entity CRUD."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- Memory CRUD ----

    async def create(self, mem: MemoryCreate) -> Memory:
        import xxhash

        memory_id = str(uuid7())
        now = _now()
        content_json = json.dumps(mem.content, ensure_ascii=False, sort_keys=True)
        content_hash = xxhash.xxh64(content_json).hexdigest()
        content_plain = self._extract_plain_text(mem.content)

        await self.db.execute(
            SQL_INSERT_MEMORY,
            (
                memory_id,
                mem.memory_type.value,
                content_json,
                content_hash,
                content_plain,
                None,  # embedding_id — set after embedding
                mem.importance,
                now,
                now,
                mem.session_id,
                mem.source,
                json.dumps(mem.metadata, ensure_ascii=False),
            ),
        )

        # Handle tags
        if mem.tags:
            await self._add_tags_internal(memory_id, mem.tags)

        await self.db.commit()

        return await self.get(memory_id)  # type: ignore[return-value]

    async def get(self, memory_id: str) -> Memory | None:
        row = await self.db.fetch_one(SQL_SELECT_MEMORY_BY_ID, (memory_id,))
        return self._row_to_memory(row) if row else None

    async def get_by_hash(self, content_hash: str) -> Memory | None:
        row = await self.db.fetch_one(SQL_SELECT_MEMORY_BY_HASH, (content_hash,))
        return self._row_to_memory(row) if row else None

    async def list_by_session(self, session_id: str, limit: int = 100) -> list[Memory]:
        rows = await self.db.fetch_all(SQL_SELECT_MEMORIES_BY_SESSION, (session_id, limit))
        return [self._row_to_memory(r) for r in rows]

    async def list_recent(
        self, limit: int = 50, memory_type: MemoryType | None = None
    ) -> list[Memory]:
        if memory_type:
            query = SQL_SELECT_RECENT_MEMORIES.format(
                type_filter="AND memory_type = ?"
            )
            rows = await self.db.fetch_all(query, (memory_type.value, limit))
        else:
            query = SQL_SELECT_RECENT_MEMORIES.format(type_filter="")
            rows = await self.db.fetch_all(query, (limit,))
        return [self._row_to_memory(r) for r in rows]

    async def list_by_status(
        self, status: MemoryStatus, limit: int = 100
    ) -> list[Memory]:
        rows = await self.db.fetch_all(
            SQL_SELECT_MEMORIES_BY_STATUS, (status.value, limit)
        )
        return [self._row_to_memory(r) for r in rows]

    async def update_content(
        self, memory_id: str, content: dict[str, Any], re_embed: bool = False
    ) -> Memory | None:
        import xxhash

        content_json = json.dumps(content, ensure_ascii=False, sort_keys=True)
        content_hash = xxhash.xxh64(content_json).hexdigest()
        content_plain = self._extract_plain_text(content)
        now = _now()

        # Preserve existing embedding_id unless re-embedding
        current = await self.get(memory_id)
        embed_id = None if re_embed else (current.embedding_id if current else None)

        await self.db.execute(
            SQL_UPDATE_MEMORY_CONTENT,
            (content_json, content_hash, content_plain, embed_id, now, memory_id),
        )
        await self.db.commit()
        return await self.get(memory_id)

    async def update_importance(self, memory_id: str, importance: float) -> None:
        await self.db.execute(
            SQL_UPDATE_MEMORY_IMPORTANCE, (importance, _now(), memory_id)
        )
        await self.db.commit()

    async def record_access(self, memory_id: str) -> None:
        await self.db.execute(SQL_UPDATE_MEMORY_ACCESS, (_now(), memory_id))
        await self.db.commit()

    async def set_status(self, memory_id: str, status: MemoryStatus) -> None:
        await self.db.execute(
            SQL_UPDATE_MEMORY_STATUS, (status.value, _now(), memory_id)
        )
        await self.db.commit()

    async def set_embedding_id(self, memory_id: str, embedding_id: str) -> None:
        await self.db.execute(
            "UPDATE memories SET embedding_id = ?, updated_at = ? WHERE id = ?",
            (embedding_id, _now(), memory_id),
        )
        await self.db.commit()

    async def mark_consolidating(
        self, memory_ids: list[str], run_id: str
    ) -> None:
        """Mark multiple memories as consolidating in one transaction."""
        if not memory_ids:
            return
        now = _now()
        placeholders = ",".join(["?"] * len(memory_ids))
        await self.db.execute(
            f"UPDATE memories SET status = 'consolidating', "
            f"consolidation_run_id = ?, updated_at = ? WHERE id IN ({placeholders})",
            (run_id, now, *memory_ids),
        )
        await self.db.commit()

    async def mark_consolidated(
        self, memory_ids: list[str], run_id: str
    ) -> None:
        """Mark memories as consolidated."""
        if not memory_ids:
            return
        now = _now()
        placeholders = ",".join(["?"] * len(memory_ids))
        await self.db.execute(
            f"UPDATE memories SET status = 'consolidated', "
            f"consolidation_version = consolidation_version + 1, "
            f"consolidation_run_id = ?, updated_at = ? WHERE id IN ({placeholders})",
            (run_id, now, *memory_ids),
        )
        await self.db.commit()

    async def archive(self, memory_id: str) -> None:
        now = _now()
        await self.db.execute(SQL_SOFT_DELETE_MEMORY, (now, now, memory_id))
        await self.db.commit()

    async def prune(self, memory_id: str) -> None:
        await self.db.execute(SQL_HARD_DELETE_MEMORY, (memory_id,))
        await self.db.commit()

    async def count_by_status(self, status: MemoryStatus) -> int:
        row = await self.db.fetch_one(SQL_COUNT_BY_STATUS, (status.value,))
        return row["cnt"] if row else 0

    async def count_total(self) -> int:
        row = await self.db.fetch_one(SQL_COUNT_TOTAL)
        return row["cnt"] if row else 0

    # ---- Relationships ----

    async def relate(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        rel_id = str(uuid7())
        evidence_json = json.dumps(evidence or {}, ensure_ascii=False)
        await self.db.execute(
            SQL_INSERT_RELATIONSHIP,
            (
                rel_id,
                source_id,
                target_id,
                relation_type.value,
                strength,
                evidence_json,
                _now(),
            ),
        )
        await self.db.commit()
        return rel_id

    async def get_related(
        self,
        memory_id: str,
        relation_types: list[RelationType] | None = None,
    ) -> list[tuple[Memory, RelationType, float]]:
        if relation_types:
            placeholders = ",".join(["?"] * len(relation_types))
            query = SQL_SELECT_RELATIONSHIPS_FILTERED.format(placeholders)
            params = [memory_id] + [rt.value for rt in relation_types]
        else:
            query = SQL_SELECT_RELATIONSHIPS
            params = [memory_id]

        rows = await self.db.fetch_all(query, params)
        results: list[tuple[Memory, RelationType, float]] = []
        for row in rows:
            memory = self._row_to_memory(row)
            rt = RelationType(row["relation_type"])
            results.append((memory, rt, row["strength"]))
        return results

    # ---- Tags ----

    async def add_tags(self, memory_id: str, tags: list[str]) -> None:
        await self._add_tags_internal(memory_id, tags)
        await self.db.commit()

    async def _add_tags_internal(self, memory_id: str, tags: list[str]) -> None:
        import uuid
        TAG_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
        now = _now()
        for tag_name in tags:
            tag_name_clean = tag_name.strip().lower()
            tag_id = str(uuid.uuid5(TAG_NS, tag_name_clean))
            await self.db.execute(SQL_INSERT_TAG, (tag_id, tag_name_clean, now))
            await self.db.execute(SQL_INSERT_MEMORY_TAG, (memory_id, tag_id))

    async def remove_tags(self, memory_id: str, tags: list[str]) -> None:
        for tag_name in tags:
            await self.db.execute(
                "DELETE FROM memory_tags WHERE memory_id = ? AND tag_id IN "
                "(SELECT id FROM tags WHERE name = ?)",
                (memory_id, tag_name.strip().lower()),
            )
        await self.db.commit()

    async def get_tags(self, memory_id: str) -> list[str]:
        rows = await self.db.fetch_all(SQL_SELECT_TAGS_FOR_MEMORY, (memory_id,))
        return [r["name"] for r in rows]

    # ---- Sessions ----

    async def start_session(self, metadata: dict[str, Any] | None = None) -> str:
        session_id = str(uuid7())
        await self.db.execute(
            SQL_INSERT_SESSION, (session_id, _now(), json.dumps(metadata or {}))
        )
        await self.db.commit()
        return session_id

    async def end_session(
        self, session_id: str, summary: str | None = None
    ) -> None:
        # Count memories in this session
        row = await self.db.fetch_one(
            "SELECT COUNT(*) as cnt FROM memories WHERE session_id = ? AND deleted_at IS NULL",
            (session_id,),
        )
        count = row["cnt"] if row else 0
        await self.db.execute(
            SQL_UPDATE_SESSION_END, (_now(), summary, count, session_id)
        )
        await self.db.commit()

    async def get_session(self, session_id: str) -> Session | None:
        row = await self.db.fetch_one(SQL_SELECT_SESSION, (session_id,))
        if not row:
            return None
        return Session(
            id=row["id"],
            started_at=datetime.fromisoformat(row["started_at"]),
            ended_at=_parse_dt(row["ended_at"]),
            summary=row["summary"],
            memory_count=row["memory_count"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    # ---- Helpers ----

    def _row_to_memory(self, row: aiosqlite.Row) -> Memory:
        return Memory(
            id=row["id"],
            rowid=row["rowid"],
            memory_type=MemoryType(row["memory_type"]),
            content=json.loads(row["content_json"]),
            content_hash=row["content_hash"],
            content_plain=row["content_plain"],
            embedding_id=row["embedding_id"],
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed_at=_parse_dt(row["last_accessed_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            session_id=row["session_id"],
            source=row["source"],
            status=MemoryStatus(row["status"]),
            consolidation_version=row["consolidation_version"],
            consolidation_run_id=row["consolidation_run_id"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            deleted_at=_parse_dt(row["deleted_at"]) if row["deleted_at"] else None,
        )

    @staticmethod
    def _extract_plain_text(content: dict[str, Any]) -> str:
        """Extract searchable plain text from structured content."""
        parts: list[str] = []
        for key in ("text", "description", "summary", "code", "query"):
            if key in content and isinstance(content[key], str):
                parts.append(content[key])
        if "messages" in content and isinstance(content["messages"], list):
            for msg in content["messages"]:
                if isinstance(msg, dict) and "content" in msg:
                    parts.append(str(msg["content"]))
        return "\n".join(parts) if parts else json.dumps(content, ensure_ascii=False)
