from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")

        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())