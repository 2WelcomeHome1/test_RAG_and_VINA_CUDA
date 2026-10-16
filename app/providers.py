"""Embedding and answer providers: interfaces, deterministic implementations, registry, output checks."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from pgvector import Vector

# Must match the vector(...) column in migrations/001_init.sql.
EMBEDDING_DIMENSIONS = 384
TOKEN = re.compile(r"[^\W_]+")


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> Sequence[float]: ...


class AnswerProvider(Protocol):
    def answer(self, query: str, contexts: list[str]) -> str: ...


class ProviderError(Exception):
    """A provider failed or returned unusable output; the message is safe to return to the client."""


class DeterministicEmbeddingProvider:
    """Hashes tokens into a fixed-size vector: reproducible, but lexical rather than semantic."""

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        for token in TOKEN.findall(text.casefold()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
            vector[index] += 1.0 if digest[4] & 1 else -1.0
        magnitude = sum(value * value for value in vector) ** 0.5
        if magnitude:
            vector = [value / magnitude for value in vector]
        return vector


class DeterministicAnswerProvider:
    """Answers with the retrieved contexts themselves, so the answer is grounded by construction."""

    def answer(self, query: str, contexts: list[str]) -> str:
        del query
        return "\n".join(contexts)


EMBEDDING_PROVIDERS: dict[str, type[EmbeddingProvider]] = {"deterministic": DeterministicEmbeddingProvider}
ANSWER_PROVIDERS: dict[str, type[AnswerProvider]] = {"deterministic": DeterministicAnswerProvider}


def create_provider(registry: dict[str, type], variable: str):
    """Instantiate the provider named by an environment variable; unknown names fail at startup."""
    name = os.environ.get(variable, "deterministic")
    if name not in registry:
        raise RuntimeError(f"Unknown {variable}={name!r}; supported: {', '.join(sorted(registry))}")
    return registry[name]()


def embed(provider: EmbeddingProvider, text: str) -> Vector:
    """Embed text and check the result at the trust boundary; returns a compact float32 vector.

    A wrong dimension or NaN would otherwise surface later as a database error (500) instead of 503.
    """
    try:
        vector = Vector(provider.embed(text))
    except Exception as error:
        raise ProviderError("Embedding provider unavailable") from error
    values = vector.to_numpy()
    if values.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(values).all():
        raise ProviderError("Embedding provider returned an invalid vector")
    return vector


def is_zero(vector: Vector) -> bool:
    """A zero vector has no direction: cosine distance to it is undefined and HNSW does not index it."""
    return not vector.to_numpy().any()


def answer(provider: AnswerProvider, query: str, contexts: list[str]) -> str:
    try:
        result = provider.answer(query, contexts)
    except Exception as error:
        raise ProviderError("Answer provider unavailable") from error
    if not isinstance(result, str):
        raise ProviderError("Answer provider returned an invalid answer")
    return result
