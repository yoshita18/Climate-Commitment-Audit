from __future__ import annotations
import logging
from functools import lru_cache
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps sentence-transformers for local, API-free embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: Union[str, list[str]], batch_size: int = 64) -> np.ndarray:
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0].tolist()

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
    return EmbeddingModel(model_name)
