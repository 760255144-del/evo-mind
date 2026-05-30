"""Heuristic summarizer — extracts key information from groups of memories."""

from __future__ import annotations

import re
from collections import Counter


class HeuristicSummarizer:
    """Basic extractive summarizer using frequency-based sentence scoring.

    This is a lightweight default. Plugins can provide LLM-based summarizers
    via the on_get_summarizer hook.
    """

    def __init__(self, max_summary_sentences: int = 3, max_length: int = 500) -> None:
        self.max_sentences = max_summary_sentences
        self.max_length = max_length

    async def summarize(self, texts: list[str], max_length: int = 200) -> str:
        """Generate an extractive summary from multiple related texts."""
        combined = " ".join(texts)

        # Split into sentences
        sentences = self._split_sentences(combined)
        if len(sentences) <= self.max_sentences:
            summary = " ".join(sentences)
            return summary[:max_length]

        # Score sentences by word frequency
        words = self._tokenize(combined)
        if not words:
            return combined[:max_length]

        word_freq = Counter(words)

        # Score each sentence with original index
        scored: list[tuple[str, float, int]] = []
        for idx, sent in enumerate(sentences):
            sent_words = self._tokenize(sent)
            if not sent_words:
                continue
            score = sum(word_freq[w] for w in sent_words) / len(sent_words)
            scored.append((sent, score, idx))

        # Select top sentences, restore original order
        scored.sort(key=lambda x: x[1], reverse=True)
        top_sentences = scored[: self.max_sentences]
        top_sentences.sort(key=lambda x: x[2])

        summary = " ".join(s[0] for s in top_sentences)
        return summary[:max_length]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitter."""
        # Split on sentence boundary markers
        raw = re.split(r'(?<=[.!?。！？\n])\s+', text)
        return [s.strip() for s in raw if s.strip() and len(s.strip()) > 3]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenizer with stopword removal."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "to", "of",
            "in", "on", "at", "for", "with", "by", "from", "and", "or",
            "but", "not", "no", "yes", "this", "that", "these", "those",
            "it", "its", "he", "she", "they", "we", "you", "i", "me", "my",
        }
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        return [w for w in words if w not in stopwords]
