"""HTTP API: document ingest and corpus queries, wired to storage, extraction and providers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Receive, Scope, Send

from app.extraction import EXTRACTORS, ExtractionError, extract
from app.providers import (
    ANSWER_PROVIDERS,
    EMBEDDING_PROVIDERS,
    AnswerProvider,
    EmbeddingProvider,
    ProviderError,
    answer,
    create_provider,
    embed,
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag.api")

MAX_FILE_SIZE = 5 * 1024 * 1024
MULTIPART_ALLOWANCE = 64 * 1024  # room for the multipart envelope around the file
MAX_QUERY_CHARS = 4000
CITATION_LIMIT = 5
INSUFFICIENT_INFORMATION = "Недостаточно информации в загруженных документах."


class RequestSizeLimit:
    """Reject a body declared far over the file limit before it is received and spooled to disk.

    The exact 5 MiB check stays in the endpoint; this only stops grossly oversized uploads early.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            declared = dict(scope["headers"]).get(b"content-length", b"")
            if declared.isdigit() and int(declared) > self.max_bytes:
                await JSONResponse({"detail": "File exceeds 5 MiB"}, status_code=413)(scope, receive, send)
                return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Providers are created once; an unknown name fails the start instead of every request.
    application.state.embedding_provider = create_provider(EMBEDDING_PROVIDERS, "EMBEDDING_PROVIDER")
    application.state.answer_provider = create_provider(ANSWER_PROVIDERS, "ANSWER_PROVIDER")
    # One client for all requests: boto3 clients are thread-safe, but creating them concurrently is not.
    application.state.s3 = new_s3_client()
    await asyncio.to_thread(wait_for_storage, application.state.s3)
    application.state.pool = await asyncio.to_thread(create_pool)
    try:
        yield
    finally:
        await asyncio.to_thread(application.state.pool.close)


app = FastAPI(title="RAG API", lifespan=lifespan)
app.add_middleware(RequestSizeLimit, max_bytes=MAX_FILE_SIZE + MULTIPART_ALLOWANCE)


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, _error: Exception) -> JSONResponse:
    # Keeps the JSON error contract for bugs too; uvicorn still logs the traceback.
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


class QueryRequest(BaseModel):
    query: str = Field(max_length=MAX_QUERY_CHARS)


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    # Parts that could not be read (a broken page, slide or sheet); the rest was indexed.
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


def get_answer_provider(request: Request) -> AnswerProvider:
    return request.app.state.answer_provider


@app.post("/v1/documents", status_code=201, response_model=DocumentResponse)
def upload_document(
    file: UploadFile = File(...),
    s3=Depends(get_s3_client),
    pool: ConnectionPool = Depends(get_pool),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> DocumentResponse:
    # A plain def runs in FastAPI's threadpool: parsing, embedding and S3/DB calls are blocking,
    # and inside async def they would stall every other request on the event loop.
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.casefold()
    if suffix not in EXTRACTORS:
        raise HTTPException(status_code=415, detail="Unsupported file format")
    # One byte past the limit is enough to detect an oversize file without loading all of it.
    content = file.file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 5 MiB")
    try:
        extraction = extract(suffix, content)
    except ExtractionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    try:
        vectors = [embed(embedding_provider, chunk.text) for chunk in extraction.chunks]
    except ProviderError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    document_id = uuid.uuid4()
    object_key = f"{document_id}/{filename}"
    try:
        s3.put_object(Bucket=bucket_name(), Key=object_key, Body=content)
    except Exception as error:
        logger.warning("S3 upload of %s failed: %s", object_key, error)
        raise HTTPException(status_code=503, detail="S3 unavailable") from error
    try:
        save_document(pool, document_id, filename, object_key, extraction.chunks, vectors)
    except CommitUnknownError as error:
        # The rows may have been committed: deleting the original could leave them pointing at nothing.
        logger.error("COMMIT outcome unknown for %s, original kept: %s", document_id, object_key, exc_info=error)
        raise HTTPException(status_code=500, detail="Could not save document") from error
    except Exception as error:
        logger.error("Could not save document %s", document_id, exc_info=error)
        delete_original(s3, object_key)
        raise HTTPException(status_code=500, detail="Could not save document") from error
    if extraction.warnings:
        logger.warning("Document %s indexed partially: %s", document_id, "; ".join(extraction.warnings))
    return DocumentResponse(
        document_id=str(document_id),
        filename=filename,
        chunks_count=len(extraction.chunks),
        warnings=extraction.warnings,
    )


@app.post("/v1/query", response_model=QueryResponse)
def query_documents(
    body: QueryRequest,
    pool: ConnectionPool = Depends(get_pool),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    answer_provider: AnswerProvider = Depends(get_answer_provider),
) -> QueryResponse:
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty")
    try:
        query_vector = embed(embedding_provider, body.query)
    except ProviderError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if is_zero(query_vector):
        # Nothing searchable in the query (only punctuation, say): any ranking would be arbitrary.
        return QueryResponse(answer=INSUFFICIENT_INFORMATION, citations=[])

    try:
        rows = search_chunks(pool, query_vector, CITATION_LIMIT)
    except Exception as error:
        logger.error("Search failed", exc_info=error)
        raise HTTPException(status_code=500, detail="Search failed") from error
    if not rows:
        return QueryResponse(answer=INSUFFICIENT_INFORMATION, citations=[])
    citations = [
        Citation(
            document_id=str(row["document_id"]),
            filename=row["filename"],
            chunk_id=str(row["chunk_id"]),
            text=row["text"],
            location=row["source_location"],
        )
        for row in rows
    ]
    try:
        text = answer(answer_provider, body.query, [citation.text for citation in citations])
    except ProviderError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return QueryResponse(answer=text, citations=citations)
