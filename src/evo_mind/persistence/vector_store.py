"""ChromaDB wrapper implementing the VectorStore Protocol."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import chromadb
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Maximum IDs per ChromaDB operation (batch size safety limit)
BATCH_SIZE = 100


class ChromaVectorStore:
    """ChromaDB-backed vector store wrapped for async usage.

    Uses PersistentClient (embedded) with no external server.
    All blocking calls are wrapped in asyncio.to_thread().
    Writes are serialized via a semaphore to avoid SQLite contention.
    """

    def __init__(
        self,
        path: Path,
        collection_name: str = "evo_mind_memories",
        distance_metric: str = "cosine",
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None
        self._write_semaphore = asyncio.Semaphore(1)

    async def initialize(self) -> None:
        """Create/load the ChromaDB client and collection."""
        self._client = await asyncio.to_thread(
            lambda: chromadb.PersistentClient(path=str(self.path))
        )

        # Get or create collection
        try:
            self._collection = await asyncio.to_thread(
                lambda: self._client.get_collection(  # type: ignore[union-attr]
                    name=self.collection_name
                )
            )
            logger.info("chroma_collection_loaded", name=self.collection_name)
        except Exception:
            self._collection = await asyncio.to_thread(
                lambda: self._client.create_collection(  # type: ignore[union-attr]
                    name=self.collection_name,
                    metadata={"hnsw:space": self.distance_metric},
                )
            )
            logger.info("chroma_collection_created", name=self.collection_name)

    async def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """Add vectors to the collection. Serialized via semaphore."""
        assert self._collection is not None, "Vector store not initialized"

        async with self._write_semaphore:
            for i in range(0, len(ids), BATCH_SIZE):
                batch_slice = slice(i, i + BATCH_SIZE)
                await asyncio.to_thread(
                    lambda: self._collection.add(  # type: ignore[union-attr]
                        ids=ids[batch_slice],
                        embeddings=embeddings[batch_slice],
                        metadatas=metadatas[batch_slice] if metadatas else None,
                        documents=documents[batch_slice] if documents else None,
                    )
                )
            logger.debug("chroma_added", count=len(ids))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
    async def query(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict[str, object] | None = None,
    ) -> tuple[list[str], list[float]]:
        """Query the collection for nearest neighbors."""
        assert self._collection is not None, "Vector store not initialized"

        result = await asyncio.to_thread(
            lambda: self._collection.query(  # type: ignore[union-attr]
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["distances"],
            )
        )

        ids_raw = result.get("ids", []) if result else []
        ids: list[str] = ids_raw[0] if ids_raw else []
        distances_raw = result.get("distances", []) if result else []
        distances: list[float] = distances_raw[0] if distances_raw else []

        return ids, distances

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors from the collection."""
        assert self._collection is not None, "Vector store not initialized"
        if not ids:
            return

        async with self._write_semaphore:
            await asyncio.to_thread(
                lambda: self._collection.delete(ids=ids)  # type: ignore[union-attr]
            )
            logger.debug("chroma_deleted", count=len(ids))

    async def count(self) -> int:
        """Return the number of vectors in the collection."""
        assert self._collection is not None, "Vector store not initialized"
        return await asyncio.to_thread(lambda: self._collection.count())  # type: ignore[union-attr]

    async def get_embeddings(self, ids: list[str]) -> list[list[float]] | None:
        """Retrieve embeddings by ID. Returns None for any IDs not found."""
        assert self._collection is not None, "Vector store not initialized"
        if not ids:
            return []

        result = await asyncio.to_thread(
            lambda: self._collection.get(  # type: ignore[union-attr]
                ids=ids, include=["embeddings"]
            )
        )

        if not result or not result.get("embeddings"):
            return None

        return result["embeddings"]  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Clean up resources."""
        self._collection = None
        self._client = None
        logger.info("chroma_closed")
