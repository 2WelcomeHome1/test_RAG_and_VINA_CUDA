"""Checks of components that need no HTTP round trip: logging and the Ollama answer provider.

Ollama itself is replaced with a fake client, so these tests need no model and no GPU.
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import ollama
import pytest

from app.core.config import settings
from app.core.logger import InterceptHandler
from app.providers import OllamaAnswerProvider, ProviderError


def test_log_call_with_bad_arguments_does_not_fail_the_caller():
    # A wrong format argument in a log call must be reported by logging, not raised into a request.
    # The handler is called directly: pytest's own capture handler deliberately fails on such records.
    record = logging.LogRecord("rag.api", logging.INFO, __file__, 1, "chunks: %d", ("not a number",), None)
    InterceptHandler().handle(record)


class FakeOllama:
    """Records the warm-up and holds every chat until `release` is set, like a busy Ollama."""

    def __init__(self):
        self.warmups: list[dict | None] = []
        self.release = threading.Event()

    def chat(self, **kwargs):
        if not kwargs["messages"]:
            self.warmups.append(kwargs.get("options"))
            return None
        self.release.wait(10)
        return SimpleNamespace(message=SimpleNamespace(content="ответ [1]"), done_reason="stop")


@pytest.fixture
def fake_ollama(monkeypatch):
    fake = FakeOllama()
    monkeypatch.setattr(ollama, "Client", lambda **_kwargs: fake)
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(settings, "LLM_QUEUE_TIMEOUT", 0.2)
    yield fake
    fake.release.set()


def test_ollama_warm_up_uses_the_query_options(fake_ollama):
    # Ollama reloads the model when num_ctx or num_batch differ, so a warm-up with defaults would make
    # the first real query pay for a second load.
    provider = OllamaAnswerProvider()
    assert fake_ollama.warmups == [provider._options]
    assert provider._options["num_ctx"] == settings.LLM_NUM_CTX


def test_ollama_generation_over_the_cap_fails_fast(fake_ollama):
    # A busy slot must give a quick ProviderError (the API then returns the citations without a
    # summary), not park the worker thread in Ollama's queue.
    provider = OllamaAnswerProvider()
    holder = threading.Thread(target=provider.answer, args=("first", ["context"]))
    holder.start()
    time.sleep(0.05)
    # Without the cap the second call would wait for the first one, released here after 2 s.
    threading.Timer(2, fake_ollama.release.set).start()

    started = time.perf_counter()
    with pytest.raises(ProviderError):
        provider.answer("second", ["context"])
    assert time.perf_counter() - started < 1

    holder.join()
    assert provider.answer("third", ["context"]) == "ответ [1]"  # the slot is free again
