from __future__ import annotations

import inspect
import logging
import sys
from pathlib import Path

from loguru import logger

from app.core.config import settings

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} [{level}] {extra[logger_name]}:{line} | {message}"
QUIET_LOGGERS = ("httpx", "httpcore", "urllib3", "botocore", "boto3", "s3transfer")
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")
HEALTHCHECK_PATH = "/openapi.json"


class InterceptHandler(logging.Handler):
    """Hands a standard logging record to loguru, keeping its level, logger name and caller line."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            try:
                level: str | int = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            # Skip the logging module's own frames, so {line} points at the code that logged.
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1
            logger.bind(logger_name=record.name).opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
        except Exception:
            self.handleError(record)


class HealthcheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            return not (args[2] == HEALTHCHECK_PATH and str(args[0]).startswith("127.0.0.1"))
        return True


def setup_logging() -> None:
    logger.remove()
    logger.configure(extra={"logger_name": "app"})
    logger.add(sys.stderr, level=settings.LOG_LEVEL, format=LOG_FORMAT, backtrace=False, diagnose=False)
    if settings.LOG_DIR:
        log_dir = Path(settings.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "rag_{time:YYYY-MM-DD}.log",
            level=settings.LOG_LEVEL,
            format=LOG_FORMAT,
            rotation="00:00",
            retention=f"{settings.LOG_RETENTION_DAYS} days",
            encoding="utf-8",
            backtrace=False,
            diagnose=False,
        )


    root_level = max(logging.INFO, logging.getLevelNamesMapping()[settings.LOG_LEVEL])
    logging.basicConfig(handlers=[InterceptHandler()], level=root_level, force=True)
    logging.getLogger("rag").setLevel(settings.LOG_LEVEL)

    for name in UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [InterceptHandler()]
        uvicorn_logger.propagate = False
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(existing, HealthcheckFilter) for existing in access_logger.filters):
        access_logger.addFilter(HealthcheckFilter())
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
