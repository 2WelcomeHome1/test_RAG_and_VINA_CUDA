"""Storage: PostgreSQL/pgvector (pool, migrations, documents, search) and S3 (client, bucket)."""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

import boto3
import psycopg
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.extraction import Chunk

logger = logging.getLogger("rag.storage")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
# Without explicit limits a hung dependency holds a worker thread for minutes
# (libpq waits for the OS TCP timeout; boto3 defaults to 60 s timeouts with 5 attempts).
DB_CONNECT_TIMEOUT_S = 5
DB_STATEMENT_TIMEOUT_MS = 30_000
DB_POOL_WAIT_S = 5
# The pool retries a failed connect with 1, 2, 4 ... s pauses until this timeout, and meanwhile new
# requests cannot start an attempt. With the default 300 s, the API kept failing for minutes after
# the database came back; a short timeout lets the next request reconnect at once.
DB_RECONNECT_TIMEOUT_S = 5
S3_CONNECT_TIMEOUT_S = 3
S3_READ_TIMEOUT_S = 10
S3_MAX_ATTEMPTS = 2
STARTUP_ATTEMPTS = 40
# Errors worth waiting out while the stack starts; anything else (a broken migration) fails at once.
TRANSIENT_STARTUP_ERRORS = (psycopg.OperationalError, BotoCoreError, ClientError)


class CommitUnknownError(Exception):
    """The connection broke during COMMIT, so the document may or may not have been saved."""


def database_url() -> str:
    return os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")


def bucket_name() -> str:
    return os.environ.get("S3_BUCKET", "rag-documents")


def connect_db(*, autocommit: bool = False, conninfo: str | None = None) -> psycopg.Connection:
    """A standalone connection, for migrations and tests; requests use the pool."""
    connection = psycopg.connect(
        conninfo or database_url(), autocommit=autocommit, row_factory=dict_row, connect_timeout=DB_CONNECT_TIMEOUT_S
    )
    vector_type = connection.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'vector') AS available"
    ).fetchone()["available"]
    if vector_type:
        register_vector(connection)
    return connection


def configure_pooled_connection(connection: psycopg.Connection) -> None:
    register_vector(connection)
    connection.commit()  # the pool requires the connection to be left idle


def create_pool() -> ConnectionPool:
    """Connections reused across requests: no handshake and no type lookups per request.

    `check` replaces connections broken by a database restart before handing them out.
    """
    pool = ConnectionPool(
        database_url(),
        min_size=1,
        max_size=10,
        timeout=DB_POOL_WAIT_S,
        reconnect_timeout=DB_RECONNECT_TIMEOUT_S,
        kwargs={
            "row_factory": dict_row,
            "connect_timeout": DB_CONNECT_TIMEOUT_S,
            "options": f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}",
        },
        configure=configure_pooled_connection,
        check=ConnectionPool.check_connection,
        open=False,
        name="rag",
    )
    pool.open(wait=True, timeout=DB_POOL_WAIT_S)
    return pool


def apply_migrations(conninfo: str | None = None) -> None:
    """Apply pending migrations/*.sql in name order, each recorded once in schema_migrations."""
    with connect_db(conninfo=conninfo) as connection:
        # Serializes concurrent API starts; released when this transaction ends.
        connection.execute("SELECT pg_advisory_xact_lock(17319902)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {row["version"] for row in connection.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name not in applied:
                connection.execute(path.read_text(encoding="utf-8"))
                connection.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))
                logger.info("Applied migration %s", path.name)


def new_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(
            connect_timeout=S3_CONNECT_TIMEOUT_S,
            read_timeout=S3_READ_TIMEOUT_S,
            retries={"max_attempts": S3_MAX_ATTEMPTS, "mode": "standard"},
            s3={"addressing_style": os.environ.get("S3_ADDRESSING_STYLE", "path")},
        ),
    )


def ensure_bucket(s3) -> None:
    try:
        s3.head_bucket(Bucket=bucket_name())
        return
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"404", "NoSuchBucket", "NotFound"}:
            raise
    try:
        s3.create_bucket(Bucket=bucket_name())
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            raise


def wait_for_storage(s3) -> None:
    """Apply migrations and create the bucket, waiting while PostgreSQL or S3 are still starting."""
    for attempt in range(1, STARTUP_ATTEMPTS + 1):
        try:
            apply_migrations()
            ensure_bucket(s3)
            return
        except TRANSIENT_STARTUP_ERRORS as error:
            if attempt == STARTUP_ATTEMPTS:
                raise RuntimeError("Database or S3 did not become ready") from error
            logger.warning("Storage not ready (attempt %d/%d): %s", attempt, STARTUP_ATTEMPTS, error)
            time.sleep(1)


def save_document(
    pool: ConnectionPool,
    document_id: uuid.UUID,
    filename: str,
    object_key: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Vector],
) -> None:
    """Write the document and all its chunks in one transaction: either everything or nothing."""
    with pool.connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents (id, filename, object_key) VALUES (%s, %s, %s)",
                (document_id, filename, object_key),
            )
            # COPY is about twice as fast as executemany; the HNSW index update dominates the rest.
            with cursor.copy(
                "COPY chunks (id, document_id, ordinal, text, source_location, embedding) FROM STDIN (FORMAT BINARY)"
            ) as copy:
                copy.set_types(["uuid", "uuid", "int4", "text", "text", "vector"])
                for ordinal, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                    copy.write_row((uuid.uuid4(), document_id, ordinal, chunk.text, chunk.location, vector))
        try:
            connection.commit()
        except psycopg.OperationalError as error:
            raise CommitUnknownError from error


def search_chunks(pool: ConnectionPool, query_vector: Vector, limit: int) -> list[dict]:
    """The chunks nearest to the query by cosine distance, across the whole corpus."""
    with pool.connection() as connection:
        return connection.execute(
            """SELECT d.id AS document_id, d.filename, c.id AS chunk_id, c.text, c.source_location
               FROM chunks c JOIN documents d ON d.id = c.document_id
               ORDER BY c.embedding <=> %s
               LIMIT %s""",
            (query_vector, limit),
        ).fetchall()


def delete_original(s3, object_key: str) -> None:
    """Best-effort cleanup after a failed save; a failure leaves an orphan, which is logged for reconciliation."""
    try:
        s3.delete_object(Bucket=bucket_name(), Key=object_key)
    except Exception:
        logger.exception("Orphaned original in S3, delete it manually: %s/%s", bucket_name(), object_key)
