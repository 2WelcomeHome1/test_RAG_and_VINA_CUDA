from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from app.core.config import settings
from app.providers.base import ProviderError

logger = logging.getLogger("rag.providers")


class RerankerProvider(ABC):

    def rerank(self, query: str, texts: list[str], top_k: int) -> list[int]:
        if not texts:
            return []
        try:
            scores = np.asarray(self._score(query, texts), dtype=np.float32)
        except Exception as error:
            raise ProviderError("Reranker unavailable") from error
        if scores.shape != (len(texts),) or not np.isfinite(scores).all():
            logger.warning("Реранкер вернул некорректные оценки: форма %s вместо %s", scores.shape, (len(texts),))
            raise ProviderError("Reranker returned invalid scores")

        best = np.argsort(-scores, kind="stable")[:top_k].tolist()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Реранкинг: кандидатов %d, оценки отобранных %s, позиции в векторной выдаче %s",
                len(texts), [round(float(scores[index]), 3) for index in best], best,
            )
        return best

    @abstractmethod
    def _score(self, query: str, texts: list[str]) -> ArrayLike:
        """One relevance score per text, in the same order; higher is more relevant."""


class DeterministicRerankerProvider(RerankerProvider):
    """Keeps the vector search order: the first candidate scores highest."""

    def _score(self, query: str, texts: list[str]) -> np.ndarray:
        del query
        return -np.arange(len(texts), dtype=np.float32)


class CrossEncoderRerankerProvider(RerankerProvider):
    """A sentence-transformers cross-encoder: it reads the query and each chunk together, so it ranks
    more precisely than vector distance, at the cost of one model pass per candidate."""

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Загружаю модель реранкера %s", settings.RERANKER_MODEL)
        self._model = CrossEncoder(
            settings.RERANKER_MODEL,
            device=settings.RERANKER_DEVICE or None,
            max_length=settings.RERANKER_MAX_LENGTH,
        )
        device = self._model.model.device
        if device.type == "cuda":
            self._model.model.half()  # fp16
        self._batch_size = settings.RERANKER_BATCH_SIZE
        if device.type == "cpu":
            logger.warning("Реранкер работает на CPU: каждый запрос будет заметно медленнее")
        logger.info(
            "Модель реранкера загружена: устройство=%s, fp16=%s, max_length=%d, пакет=%d",
            device, device.type == "cuda", settings.RERANKER_MAX_LENGTH, self._batch_size,
        )

    def _score(self, query: str, texts: list[str]) -> np.ndarray:
        scores = self._model.predict(
            [(query, text) for text in texts],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if scores.ndim == 2 and scores.shape[1] == 2:
            return scores[:, 1]
        return scores


RERANKER_PROVIDERS: dict[str, type[RerankerProvider]] = {
    "deterministic": DeterministicRerankerProvider,
    "cross-encoder": CrossEncoderRerankerProvider,
}
