from __future__ import annotations

import logging
import time
from typing import TypeVar

from app.core.config import settings

logger = logging.getLogger("rag.providers")

T = TypeVar("T")


class ProviderError(Exception):
    """A provider failed or returned unusable output; the message is safe to return to the client."""


def create_provider(registry: dict[str, type[T]], variable: str) -> T:
    """Instantiate the provider named by a setting; unknown names fail at startup."""
    name = getattr(settings, variable)
    if name not in registry:
        raise RuntimeError(f"Unknown {variable}={name!r}; supported: {', '.join(sorted(registry))}")
    started = time.perf_counter()
    provider = registry[name]()
    logger.info("Провайдер %s=%s готов за %.1f с", variable, name, time.perf_counter() - started)
    return provider
