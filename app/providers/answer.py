from __future__ import annotations

import logging
import random
import threading
import time
from abc import ABC, abstractmethod

from app.core.config import settings
from app.prompt import build_messages
from app.providers.base import ProviderError

logger = logging.getLogger("rag.providers")


class AnswerProvider(ABC):
    """Calls the model and checks its output; subclasses implement only _generate."""

    def answer(self, query: str, contexts: list[str]) -> str:
        try:
            result = self._generate(query, contexts)
        except Exception as error:
            raise ProviderError("Answer provider unavailable") from error
        if not isinstance(result, str) or not result.strip():
            raise ProviderError("Answer provider returned an invalid answer")
        return result.strip()

    @abstractmethod
    def _generate(self, query: str, contexts: list[str]) -> str: ...


class DeterministicAnswerProvider(AnswerProvider):
    """Answers with the retrieved contexts themselves, so the answer is grounded by construction."""

    def _generate(self, query: str, contexts: list[str]) -> str:
        del query
        return "\n".join(contexts)


class OllamaAnswerProvider(AnswerProvider):
    MAX_ATTEMPTS = 2
    RETRY_DELAY_S = 0.5

    def __init__(self) -> None:
        from ollama import Client

        self._client = Client(
            host=settings.OLLAMA_BASE_URL, timeout=settings.OLLAMA_TIMEOUT
        )
        self._options = {
            "temperature": settings.LLM_TEMPERATURE,
            "top_k": settings.LLM_TOP_K,
            "top_p": settings.LLM_TOP_P,
            "repeat_penalty": settings.LLM_REPEAT_PENALTY,
            "seed": settings.LLM_SEED,
            "num_ctx": settings.LLM_NUM_CTX,
            "num_predict": settings.LLM_NUM_PREDICT,
            "num_batch": settings.LLM_NUM_BATCH,
        }

        self._slots = threading.BoundedSemaphore(settings.LLM_MAX_CONCURRENCY)
        logger.info("Загружаю LLM %s в Ollama (%s)", settings.LLM_MODEL, settings.OLLAMA_BASE_URL)
        started = time.perf_counter()
        try:
            self._client.chat(
                model=settings.LLM_MODEL, messages=[], options=self._options, keep_alive=settings.LLM_KEEP_ALIVE
            )
        except Exception as error:
            logger.error(
                "LLM %s недоступна: скачана ли модель (make pull-llm) и запущен ли Ollama? %s",
                settings.LLM_MODEL, error,
            )
            raise RuntimeError(f"Ollama model {settings.LLM_MODEL!r} is not available: {error}") from error
        logger.info(
            "LLM загружена за %.1f с: num_ctx=%d, num_predict=%d, temperature=%s, think=%s",
            time.perf_counter() - started,
            settings.LLM_NUM_CTX,
            settings.LLM_NUM_PREDICT,
            settings.LLM_TEMPERATURE,
            settings.LLM_THINK,
        )

    def _generate(self, query: str, contexts: list[str]) -> str:
        if not self._slots.acquire(timeout=settings.LLM_QUEUE_TIMEOUT):
            logger.warning(
                "Все %d слотов генерации заняты дольше %.0f с", settings.LLM_MAX_CONCURRENCY, settings.LLM_QUEUE_TIMEOUT
            )
            raise TimeoutError("no free generation slot")
        try:
            return self._chat(build_messages(query, contexts))
        finally:
            self._slots.release()

    def _chat(self, messages: list[dict[str, str]]) -> str:
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                response = self._client.chat(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    options=self._options,
                    think=settings.LLM_THINK,
                    keep_alive=settings.LLM_KEEP_ALIVE,
                )
                self._log_stats(response)
                return response.message.content
            except ConnectionError as error:
                if attempt == self.MAX_ATTEMPTS:
                    logger.error("Ollama недоступна после %d попыток: %s", self.MAX_ATTEMPTS, error)
                    raise
                delay = self.RETRY_DELAY_S * attempt + random.uniform(0, 0.1)
                logger.warning("Ollama недоступна (попытка %d), повтор через %.1f с: %s", attempt, delay, error)
                time.sleep(delay)
        raise AssertionError("unreachable")

    @staticmethod
    def _log_stats(response) -> None:
        if getattr(response, "done_reason", None) == "length":
            logger.warning(
                "Ответ LLM обрезан лимитом LLM_NUM_PREDICT=%d: увеличьте лимит или сократите ответ",
                settings.LLM_NUM_PREDICT,
            )
        prompt_tokens = getattr(response, "prompt_eval_count", None) or 0
        answer_tokens = getattr(response, "eval_count", None) or 0
        eval_ns = getattr(response, "eval_duration", None) or 0
        logger.debug(
            "LLM: токенов промпта %d, токенов ответа %d, скорость %.1f ток/с",
            prompt_tokens, answer_tokens, answer_tokens / (eval_ns / 1e9) if eval_ns else 0.0,
        )


ANSWER_PROVIDERS: dict[str, type[AnswerProvider]] = {
    "deterministic": DeterministicAnswerProvider,
    "ollama": OllamaAnswerProvider,
}
