"""Local embedding engine using sentence-transformers (all-MiniLM-L6-v2)."""

from __future__ import annotations

import asyncio
import logging
from functools import partial

from evo_mind.embedding.cache import EmbeddingCache

logger = logging.getLogger(__name__)


class LocalEmbeddingEngine:
    """Sentence-transformers based embedding with async wrapper.

    Uses all-MiniLM-L6-v2 by default: 384-dimensional embeddings,
    ~80MB model, loads in ~2s, ~1K sentences/s on CPU.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        cache_size: int = 10000,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._normalize = normalize
        self._model = None
        self._dimension: int | None = None
        self._cache = EmbeddingCache(max_size=cache_size)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embedding engine not initialized. Call initialize() first.")
        return self._dimension

    async def initialize(self) -> None:
        """Load the sentence-transformers model (blocking, runs in thread)."""
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(
            None,
            partial(self._load_model),
        )
        self._dimension = self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
        logger.info(
            "embedding_model_loaded",
            model=self.model_name,
            dimension=self._dimension,
            device=self.device,
        )

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("sentence-transformers not installed. Run: pip install sentence-transformers")
        return SentenceTransformer(self.model_name, device=self.device)

    async def encode(self, text: str) -> list[float]:
        """Encode a single text to an embedding vector."""
        if not text.strip():
            return [0.0] * self.dimension

        # Check cache
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        assert self._model is not None

        embedding_list = await loop.run_in_executor(
            None,
            partial(
                self._model.encode,
                [text],
                batch_size=1,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            ),
        )

        vec: list[float] = embedding_list[0].tolist()  # type: ignore[union-attr]
        self._cache.put(text, vec)
        return vec

    async def encode_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Encode a batch of texts efficiently."""
        if not texts:
            return []

        bs = batch_size or self.batch_size
        loop = asyncio.get_running_loop()
        assert self._model is not None

        # Check cache for each text
        results: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if not text.strip():
                results[i] = [0.0] * self.dimension
                continue
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            embeddings = await loop.run_in_executor(
                None,
                partial(
                    self._model.encode,
                    uncached_texts,
                    batch_size=bs,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                ),
            )

            for j, embedding in enumerate(embeddings):
                vec: list[float] = embedding.tolist()  # type: ignore[union-attr]
                idx = uncached_indices[j]
                results[idx] = vec
                self._cache.put(uncached_texts[j], vec)

        return results  # type: ignore[return-value]

    async def close(self) -> None:
        """Clean up model resources."""
        self._model = None
        self._cache.clear()
        logger.info("embedding_engine_closed")
