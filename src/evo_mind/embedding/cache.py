"""LRU cache for embeddings — avoids re-encoding identical text."""

from __future__ import annotations

import threading


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors.

    Uses Python's dict ordering (insertion-ordered since 3.7) for O(1) LRU eviction.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._cache: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> list[float] | None:
        """Get a cached embedding, moving it to the end (most recently used)."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recent) by re-inserting
                value = self._cache.pop(key)
                self._cache[key] = value
                return value.copy()
        return None

    def put(self, key: str, embedding: list[float]) -> None:
        """Cache an embedding. Evicts least recently used if at capacity."""
        with self._lock:
            if key in self._cache:
                # Update existing — move to end
                self._cache.pop(key)
            elif len(self._cache) >= self._max_size:
                # Evict the first (least recently used) item
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
            self._cache[key] = embedding

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
