from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), case_sensitive=True, extra="ignore")

    # LOGGING
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_DIR: str = ""
    LOG_RETENTION_DAYS: int = Field(default=14, gt=0)

    # PROVIDERS
    EMBEDDING_PROVIDER: str = "deterministic"
    RERANKER_PROVIDER: str = "deterministic"
    ANSWER_PROVIDER: str = "deterministic"

    # EMBEDDING
    RETRIEVER_MODEL: str = "deepvk/USER-bge-m3"
    EMBEDDING_DEVICE: str = ""
    EMBEDDING_BATCH_SIZE: int = Field(default=128, gt=0)

    # RERANKER
    RERANKER_MODEL: str = "qilowoq/bge-reranker-v2-m3-en-ru"
    RERANKER_DEVICE: str = ""
    RERANKER_MAX_LENGTH: int = Field(default=512, gt=0)
    RERANKER_BATCH_SIZE: int = Field(default=64, gt=0)

    # OLLAMA
    OLLAMA_BASE_URL: str = Field(default="http://ollama:11434", env="OLLAMA_BASE_URL")
    OLLAMA_TIMEOUT: float = Field(default=120, gt=0)
    LLM_MODEL: str = "qwen3.8:27b-mtp-q4_K_M"
    LLM_KEEP_ALIVE: str = "1h"
    LLM_THINK: bool = False

    LLM_MAX_CONCURRENCY: int = Field(default=4, gt=0)
    LLM_QUEUE_TIMEOUT: float = Field(default=30, gt=0)

    # LLM GENERATION
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0)
    LLM_TOP_K: int = Field(default=20, gt=0)
    LLM_TOP_P: float = Field(default=0.85, gt=0, le=1)
    LLM_REPEAT_PENALTY: float = Field(default=1.0, gt=0)
    LLM_SEED: int = 42

    # LLM CONTEXT
    LLM_NUM_CTX: int = Field(default=16384, gt=0)
    LLM_NUM_PREDICT: int = Field(default=2048, gt=0)
    LLM_NUM_BATCH: int = Field(default=512, gt=0)


settings = Settings()
