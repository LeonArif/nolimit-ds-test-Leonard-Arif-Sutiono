from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    faiss = None
    HAS_FAISS = False

from .embedder import SentenceEmbedder
from .pdf_loader import PDFChunk


@dataclass(frozen=True)
class SearchResult:
    file_name: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class RagIndex:
    def __init__(self, chunks: Sequence[PDFChunk], embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        self.chunks = list(chunks)
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.dimension = int(self.embeddings.shape[1]) if self.embeddings.size else 0
        self._faiss_index = None

        if HAS_FAISS and len(self.embeddings) > 0:
            index = faiss.IndexFlatIP(self.dimension)
            index.add(self.embeddings)
            self._faiss_index = index

    @classmethod
    def build(cls, chunks: Sequence[PDFChunk], embedder: SentenceEmbedder) -> "RagIndex":
        embeddings = embedder.encode([chunk.text for chunk in chunks])
        return cls(chunks, embeddings)

    def search(self, query: str, embedder: SentenceEmbedder, top_k: int = 3) -> list[SearchResult]:
        if not self.chunks:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_embedding = embedder.encode([query])
        limit = min(top_k, len(self.chunks))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_embedding, limit)
            return [
                self._to_result(index, float(score))
                for score, index in zip(scores[0], indices[0])
                if index != -1
            ]

        scores = self.embeddings @ query_embedding[0]
        ranked_indices = np.argsort(scores)[::-1][:limit]
        return [self._to_result(int(index), float(scores[index])) for index in ranked_indices]

    def _to_result(self, index: int, score: float) -> SearchResult:
        chunk = self.chunks[index]
        return SearchResult(
            file_name=chunk.file_name,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            score=score,
        )