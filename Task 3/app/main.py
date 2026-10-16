"""HTTP API: document ingest and corpus queries, wired to storage, extraction and providers."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Generator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import settings
from app.core.logger import setup_logging
from app.extraction import EXTRACTORS, SEARCHABLE, ExtractionError, extract
from app.prompt import INSUFFICIENT_INFORMATION, is_insufficient
from app.providers import (
    ANSWER_PROVIDERS,
    EMBEDDING_PROVIDERS,
    RERANKER_PROVIDERS,
    AnswerProvider,
    EmbeddingProvider,
    ProviderError,
    RerankerProvider,
    create_provider,
    is_zero,
)
from app.storage import (
    CommitUnknownError,
    bucket_name,
    create_pool,
    delete_original,
    new_s3_client,
    save_document,
    search_chunks,
    wait_for_storage,
)

setup_logging()
logger = logging.getLogger("rag.api")

MAX_FILE_SIZE = 5 * 1024 * 1024
MULTIPART_ALLOWANCE = 64 * 1024 
MAX_QUERY_CHARS = 4000
CITATION_LIMIT = 5

RETRIEVAL_LIMIT = 40
GENERATION_UNAVAILABLE = (
    "Не удалось сформировать ответ: генерация временно недоступна. "
    "Ниже приведены наиболее релевантные фрагменты документов."
)

DEGRADED_HEADER = "X-RAG-Degraded"


@contextmanager
def timed(timings: dict[str, int], stage: str) -> Generator[None, None, None]:
    """Record a stage's duration in milliseconds, also when it fails, so a slow failure shows up too."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round((time.perf_counter() - start) * 1000)


def format_timings(timings: dict[str, int]) -> str:
    return " ".join(f"{stage}={ms}мс" for stage, ms in timings.items())


class RequestSizeLimit:
    """Stop a body far over the file limit before it is spooled to disk.

    A declared Content-Length is checked before anything is read; a chunked body, which declares
    no length, is counted while it is received. The exact 5 MiB check stays in the endpoint.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > self.max_bytes:
            logger.info("Запрос отклонён до чтения тела: заявлено %s байт, лимит %d", declared.decode(), self.max_bytes)
            await JSONResponse({"detail": "File exceeds 5 MiB"}, status_code=413)(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    logger.info("Запрос без Content-Length отклонён: тело больше %d байт", self.max_bytes)
                    raise HTTPException(status_code=413, detail="File exceeds 5 MiB")
            return message

        await self.app(scope, limited_receive, send)


@asynccontextmanager
async def lifespan(application: FastAPI):
    started = time.perf_counter()
    logger.info(
        "Запуск API: эмбеддинги=%s, реранкер=%s, генерация=%s",
        settings.EMBEDDING_PROVIDER,
        settings.RERANKER_PROVIDER,
        settings.ANSWER_PROVIDER,
    )
    application.state.embedding_provider = create_provider(EMBEDDING_PROVIDERS, "EMBEDDING_PROVIDER")
    application.state.reranker_provider = create_provider(RERANKER_PROVIDERS, "RERANKER_PROVIDER")
    application.state.answer_provider = create_provider(ANSWER_PROVIDERS, "ANSWER_PROVIDER")

    # One client for all requests: boto3 clients are thread-safe, but creating them concurrently is not.
    application.state.s3 = new_s3_client()
    await asyncio.to_thread(wait_for_storage, application.state.s3)
    application.state.pool = await asyncio.to_thread(create_pool)
    logger.info("API готово к работе, старт занял %.1f с", time.perf_counter() - started)
    try:
        yield
    finally:
        logger.info("Остановка API: закрываю пул соединений с БД")
        await asyncio.to_thread(application.state.pool.close)


app = FastAPI(title="RAG API", lifespan=lifespan)
app.add_middleware(RequestSizeLimit, max_bytes=MAX_FILE_SIZE + MULTIPART_ALLOWANCE)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
    logger.error("Необработанная ошибка в %s %s: %s: %s", request.method, request.url.path, type(error).__name__, error)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


class QueryRequest(BaseModel):
    query: str = Field(max_length=MAX_QUERY_CHARS)


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    # Parts that could not be read (a broken page, slide or sheet; the rest was indexed) and fallbacks
    # taken on the way, such as a text file read as Windows-1251.
    warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    text: str
    location: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


def get_s3_client(request: Request):
    return request.app.state.s3


def get_pool(request: Request) -> ConnectionPool:
    return request.app.state.pool


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


def get_reranker_provider(request: Request) -> RerankerProvider:
    return request.app.state.reranker_provider


def get_answer_provider(request: Request) -> AnswerProvider:
    return request.app.state.answer_provider


@app.post("/v1/documents", status_code=201, response_model=DocumentResponse)
def upload_document(
    file: UploadFile = File(...),
    s3=Depends(get_s3_client),
    pool: ConnectionPool = Depends(get_pool),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentResponse:
    started = time.perf_counter()
    timings: dict[str, int] = {}
    filename = Path(file.filename or "").name
    
    suffix = Path(filename).suffix.casefold()
    if suffix not in EXTRACTORS:
        logger.info("Файл %r отклонён: формат %r не поддерживается", filename, suffix)
        raise HTTPException(status_code=415, detail="Unsupported file format")

    content = file.file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        logger.info("Файл %r отклонён: больше %d байт", filename, MAX_FILE_SIZE)
        raise HTTPException(status_code=413, detail="File exceeds 5 MiB")
    
    try:
        with timed(timings, "извлечение"):
            extraction = extract(suffix, content)
    except ExtractionError as error:
        logger.warning("Файл %r не проиндексирован: %s", filename, error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    
    try:
        with timed(timings, "эмбеддинги"):
            vectors = embedding_provider.embed_many([chunk.text for chunk in extraction.chunks])
    except ProviderError as error:
        logger.error(
            "Эмбеддинги для %r (чанков: %d) не получены за %d мс: %s",
            filename, len(extraction.chunks), timings["эмбеддинги"], error, exc_info=error,
        )
        raise HTTPException(status_code=503, detail=str(error)) from error

    document_id = uuid.uuid4()
    object_key = f"{document_id}/{filename}"
    
    try:
        with timed(timings, "s3"):
            s3.put_object(Bucket=bucket_name(), Key=object_key, Body=content)
    except Exception as error:
        logger.error("Не удалось загрузить оригинал %s в S3: %s", object_key, error)
        raise HTTPException(status_code=503, detail="S3 unavailable") from error
    
    try:
        with timed(timings, "бд"):
            save_document(pool, document_id, filename, object_key, extraction.chunks, vectors)
    except CommitUnknownError as error:
        logger.error(
            "Связь с БД оборвалась во время COMMIT документа %s: неизвестно, сохранён ли он; оригинал %s оставлен в S3",
            document_id, object_key, exc_info=error,
        )
        raise HTTPException(status_code=500, detail="Could not save document") from error
    
    except psycopg.OperationalError as error:
        logger.error(
            "БД недоступна при сохранении документа %s за %d мс, удаляю оригинал из S3: %s",
            document_id, timings["бд"], error,
        )
        delete_original(s3, object_key)
        raise HTTPException(status_code=503, detail="Database unavailable") from error
    
    except Exception as error:
        logger.error("Не удалось сохранить документ %s в БД, удаляю оригинал из S3", document_id, exc_info=error)
        delete_original(s3, object_key)
        raise HTTPException(status_code=500, detail="Could not save document") from error
    
    if extraction.warnings:
        logger.warning(
            "Документ %s (%r) проиндексирован с предупреждениями: %s",
            document_id, filename, "; ".join(extraction.warnings),
        )

    logger.info(
        "Документ %s (%r) проиндексирован: размер %d КБ, чанков: %d, всего %d мс | %s",
        document_id,
        filename,
        len(content) // 1024,
        len(extraction.chunks),
        round((time.perf_counter() - started) * 1000),
        format_timings(timings),
    )
    
    return DocumentResponse(
        document_id=str(document_id),
        filename=filename,
        chunks_count=len(extraction.chunks),
        warnings=extraction.warnings,
    )


@app.post("/v1/query", response_model=QueryResponse)
def query_documents(
    body: QueryRequest,
    response: Response,
    pool: ConnectionPool = Depends(get_pool),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    reranker_provider: RerankerProvider = Depends(get_reranker_provider),
    answer_provider: AnswerProvider = Depends(get_answer_provider),
) -> QueryResponse:
    started = time.perf_counter()
    timings: dict[str, int] = {}
    
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")

    
    logger.debug("Запрос: %r", body.query)
    if not SEARCHABLE.search(body.query):
        logger.info("В запросе нет слов для поиска (длина %d), отвечаю «недостаточно информации»", len(body.query))
        return QueryResponse(answer=INSUFFICIENT_INFORMATION, citations=[])
    
    try:
        with timed(timings, "эмбеддинг"):
            query_vector = embedding_provider.embed(body.query)
    except ProviderError as error:
        logger.error("Эмбеддинг запроса не получен: %s", error, exc_info=error)
        raise HTTPException(status_code=503, detail=str(error)) from error
    
    if is_zero(query_vector):
        logger.info("Эмбеддинг запроса нулевой, отвечаю «недостаточно информации»")
        return QueryResponse(answer=INSUFFICIENT_INFORMATION, citations=[])

    try:
        with timed(timings, "поиск"):
            rows = search_chunks(pool, query_vector, RETRIEVAL_LIMIT)
    except psycopg.OperationalError as error:
        logger.error("БД недоступна при поиске за %d мс: %s", timings["поиск"], error)
        raise HTTPException(status_code=503, detail="Database unavailable") from error
    except Exception as error:
        logger.error("Векторный поиск не выполнен", exc_info=error)
        raise HTTPException(status_code=500, detail="Search failed") from error
    
    if not rows:
        logger.info("Поиск ничего не нашёл: в базе нет ни одного чанка")
        return QueryResponse(answer=INSUFFICIENT_INFORMATION, citations=[])

    degraded: list[str] = []
    try:
        with timed(timings, "реранкинг"):
            best = reranker_provider.rerank(body.query, [row["text"] for row in rows], CITATION_LIMIT)
    except ProviderError as error:
        logger.warning("Реранкинг не выполнен, беру порядок векторного поиска: %s", error, exc_info=error)
        best = list(range(min(CITATION_LIMIT, len(rows))))
        degraded.append("rerank")

    citations = [
        Citation(
            document_id=str(row["document_id"]),
            filename=row["filename"],
            chunk_id=str(row["chunk_id"]),
            text=row["text"],
            location=row["source_location"],
        )
        for row in (rows[index] for index in best)
    ]

    try:
        with timed(timings, "генерация"):
            text = answer_provider.answer(body.query, [citation.text for citation in citations])
    except ProviderError as error:
        logger.error(
            "Ответ не сгенерирован за %d мс, отдаю найденные фрагменты: %s",
            timings["генерация"], error, exc_info=error,
        )
        text = GENERATION_UNAVAILABLE
        degraded.append("generation")

    insufficient = "generation" not in degraded and is_insufficient(text)
    logger.info(
        "Запрос обработан за %d мс: кандидатов %d → цитат %d%s%s | %s",
        round((time.perf_counter() - started) * 1000),
        len(rows),
        0 if insufficient else len(citations),
        ", в документах нет ответа" if insufficient else "",
        f", деградация: {', '.join(degraded)}" if degraded else "",
        format_timings(timings),
    )
    
    if degraded:
        response.headers[DEGRADED_HEADER] = ",".join(degraded)
    
    if insufficient:
        return QueryResponse(answer=INSUFFICIENT_INFORMATION, citations=[])
    
    return QueryResponse(answer=text, citations=citations)
