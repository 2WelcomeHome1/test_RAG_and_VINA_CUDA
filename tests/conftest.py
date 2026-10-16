from __future__ import annotations

import contextlib
import os

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.core.config import settings
from app.main import app
from app.storage import bucket_name, connect_db, database_url, new_s3_client


def pytest_configure(config):
    """Point the app at a dedicated database and bucket, so tests never wipe the stack's own data.

    The app reads DATABASE_URL and S3_BUCKET on every call, so switching them here is enough.
    """
    # The suite checks exact rankings and answers, which only the deterministic providers give, and
    # must run without GPU models or Ollama even inside the compose stack, which selects real ones.
    # Providers are created when a test client starts, after this hook.
    settings.EMBEDDING_PROVIDER = settings.RERANKER_PROVIDER = settings.ANSWER_PROVIDER = "deterministic"
    main_url = database_url()
    main_database = conninfo_to_dict(main_url)["dbname"]
    test_database = main_database if main_database.endswith("_test") else f"{main_database}_test"
    with psycopg.connect(main_url, autocommit=True) as connection:
        exists = connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_database,)).fetchone()
        if not exists:
            # Another test run may create it at the same moment.
            with contextlib.suppress(psycopg.errors.DuplicateDatabase):
                connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(test_database)))
    os.environ["DATABASE_URL"] = make_conninfo(main_url, dbname=test_database)
    bucket = bucket_name()
    os.environ["S3_BUCKET"] = bucket if bucket.endswith("-test") else f"{bucket}-test"


@pytest.fixture(scope="function")
def client():
    with TestClient(app) as test_client:
        yield test_client


def clear_state() -> None:
    with connect_db() as connection:
        database = connection.execute("SELECT current_database() AS name").fetchone()["name"]
        if not database.endswith("_test") or not bucket_name().endswith("-test"):
            raise RuntimeError(f"Refusing to wipe non-test storage: database {database!r}, bucket {bucket_name()!r}")
        connection.execute("DROP TRIGGER IF EXISTS fail_chunks_for_test ON chunks")
        connection.execute("DROP FUNCTION IF EXISTS fail_chunks_for_test()")
        connection.execute("DELETE FROM documents")
    # Deleted vectors stay in the HNSW index until vacuumed and can hide live rows from a search,
    # which would make one test depend on what the previous ones inserted.
    with connect_db(autocommit=True) as connection:
        connection.execute("VACUUM chunks")
    s3 = new_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name()):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=bucket_name(), Delete={"Objects": objects})


@pytest.fixture(autouse=True)
def isolated_services(client):
    app.dependency_overrides.clear()
    clear_state()
    yield
    app.dependency_overrides.clear()
    clear_state()


@pytest.fixture
def s3():
    return new_s3_client()
