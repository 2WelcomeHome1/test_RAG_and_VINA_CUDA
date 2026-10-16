"""Embedding providers: texts become vectors, checked at the trust boundary before they reach the database."""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike
from pgvector import Vector

from app.core.config import settings
from app.providers.base import ProviderError

logger = logging.getLogger("rag.providers")

EMBEDDING_DIMENSIONS = 1024
TOKEN = re.compile(r"[^\W_]+")


class EmbeddingProvider(ABC):
    """Calls the model and checks its output; subclasses implement only _encode.

    A wrong dimension or NaN would otherwise surface later as a database error (500) instead of 503.
    """

    def embed(self, text: str) -> Vector:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        try:
            matrix = np.asarray(self._encode(texts), dtype=np.float32)
        except Exception as error:
            raise ProviderError("Embedding provider unavailable") from error
        if matrix.shape != (len(texts), EMBEDDING_DIMENSIONS) or not np.isfinite(matrix).all():
            logger.warning(
                "Модель эмбеддингов вернула некорректный результат: форма %s вместо %s, NaN/inf: %s",
                matrix.shape, (len(texts), EMBEDDING_DIMENSIONS), not np.isfinite(matrix).all(),
            )
            raise ProviderError("Embedding provider returned an invalid vector")
        return [Vector(row) for row in matrix]

    @abstractmethod
    def _encode(self, texts: list[str]) -> ArrayLike:
        """One vector per text, in the same order."""


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Hashes tokens into a fixed-size vector: reproducible, but lexical rather than semantic."""

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return [self._hash(text) for text in texts]

    @staticmethod
    def _hash(text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        for token in TOKEN.findall(text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Загружаю модель эмбеддингов %s", settings.RETRIEVER_MODEL)
        self._model = SentenceTransformer(settings.RETRIEVER_MODEL, device=settings.EMBEDDING_DEVICE or None)
        if self._model.device.type == "cuda":
            self._model.half()  # fp16 on GPU: about twice as fast, retrieval quality is unchanged in practice
        self._batch_size = settings.EMBEDDING_BATCH_SIZE
        dimensions = self._model.get_sentence_embedding_dimension()
        if dimensions != EMBEDDING_DIMENSIONS:
            raise RuntimeError(f"RETRIEVER_MODEL gives {dimensions}-d vectors, the schema expects {EMBEDDING_DIMENSIONS}")
        if self._model.device.type == "cpu":
            logger.warning("Модель эмбеддингов работает на CPU: индексация и поиск будут медленными")
        logger.info(
            "Модель эмбеддингов загружена: устройство=%s, fp16=%s, размерность=%d, пакет=%d",
            self._model.device, self._model.device.type == "cuda", dimensions, self._batch_size,
        )

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


EMBEDDING_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "deterministic": DeterministicEmbeddingProvider,
    "sentence-transformers": SentenceTransformerEmbeddingProvider,
}


def is_zero(vector: Vector) -> bool:
    """A zero vector has no direction: cosine distance to it is undefined and HNSW does not index it."""
    return not vector.to_numpy().any()
