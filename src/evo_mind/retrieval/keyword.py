"""Keyword search via SQLite FTS5 full-text index."""

from __future__ import annotations

from evo_mind.persistence.database import Database
from evo_mind.persistence.memory_repo import MemoryRepo


async def keyword_search(
    db: Database,
    query_text: str,
    n: int = 40,
) -> list[tuple[str, float, dict[str, float]]]:
    """Full-text search using FTS5 with BM25 scoring."""
    repo = MemoryRepo(db)

    # Escape FTS5 special characters and build a safe query
    safe_query = _escape_fts5(query_text)
    if not safe_query:
        return []

    rows = await db.fetch_all(
        """
        SELECT m.id, m.rowid, fts.rank
        FROM memories_fts fts
        JOIN memories m ON m.rowid = fts.rowid
        WHERE fts MATCH ?
          AND m.deleted_at IS NULL
        ORDER BY fts.rank
        LIMIT ?
        """,
        (safe_query, n),
    )

    results: list[tuple[str, float, dict[str, float]]] = []
    for row in rows:
        # BM25 rank: lower is better, convert to a [0,1] score
        # FTS5 rank is negative, so negate and normalize
        raw_rank: float = float(row["rank"])
        # Normalize: typical FTS5 ranks are roughly [-10, 0]
        score = max(0.0, min(1.0, -raw_rank / 10.0))
        results.append((row["id"], score, {"keyword": score}))

    return results


def _escape_fts5(text: str) -> str:
    """Escape special FTS5 query characters and wildcard the query."""
    # Remove characters that break FTS5 syntax
    special = r'*()^"'
    cleaned = text
    for char in special:
        cleaned = cleaned.replace(char, " ")

    # Tokenize and OR them together
    tokens = cleaned.strip().split()
    if not tokens:
        return ""

    # Build a simple OR query with prefix matching on the last token
    parts = []
    for i, token in enumerate(tokens):
        token = token.strip()
        if not token:
            continue
        if i == len(tokens) - 1:
            # Add prefix wildcard to the last token
            parts.append(f'"{token}"*')
        else:
            parts.append(f'"{token}"')
    return " OR ".join(parts)
