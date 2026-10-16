from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

import app.main as api
from app import extraction
from app.extraction import MAX_CHUNK_CHARS
from app.main import (
    RequestSizeLimit,
    app,
    get_answer_provider,
    get_embedding_provider,
    get_reranker_provider,
    get_s3_client,
)
from app.prompt import INSUFFICIENT_INFORMATION
from app.providers import (
    EMBEDDING_DIMENSIONS,
    AnswerProvider,
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    RerankerProvider,
)
from app.storage import (
    MIGRATIONS_DIR,
    CommitUnknownError,
    apply_migrations,
    bucket_name,
    connect_db,
    database_url,
    new_s3_client,
)
from tests.file_samples import (
    make_document,
    make_docx_with_merged_cells,
    make_docx_with_sections,
    make_docx_zip_bomb,
    make_encrypted_pdf,
    make_markdown_with_sections,
    make_pdf_without_text_layer,
    make_pptx_with_table_and_group,
    make_sparse_xlsx,
    make_xlsx_with_broken_sheet,
)

INSUFFICIENT = {"answer": "Недостаточно информации в загруженных документах.", "citations": []}
UNDECODABLE = "File must be valid UTF-8, UTF-16 or Windows-1251 text"


class BrokenEmbedding(EmbeddingProvider):
    def _encode(self, texts):
        raise RuntimeError("simulated embedding outage")


class BrokenAnswer(AnswerProvider):
    def _generate(self, query, contexts):
        raise RuntimeError("simulated answer outage")


class BrokenReranker(RerankerProvider):
    def _score(self, query, texts):
        raise RuntimeError("simulated reranker outage")


def arbitrary_rows(*_args):
    """What a search could return for a query with nothing to rank by."""
    return [{"document_id": uuid.uuid4(), "filename": "noise.txt", "chunk_id": uuid.uuid4(),
             "text": "unrelated noise", "source_location": "line 1"}]


def upload(client: TestClient, filename: str, content: bytes):
    return client.post(
        "/v1/documents",
        files={"file": (filename, content, "application/octet-stream")},
    )


def cited(client: TestClient, marker: str) -> dict:
    response = client.post("/v1/query", json={"query": marker})
    assert response.status_code == 200, response.text
    return next(item for item in response.json()["citations"] if marker in item["text"])


def table_counts() -> tuple[int, int]:
    with connect_db() as connection:
        documents = connection.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
        chunks = connection.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    return documents, chunks


def object_count() -> int:
    return new_s3_client().list_objects_v2(Bucket=bucket_name()).get("KeyCount", 0)


# --- Ingest: formats and source locations -------------------------------------------------------


@pytest.mark.parametrize("extension", ["txt", "md", "docx", "pdf", "xlsx", "pptx"], ids=["txt", "markdown", "docx", "pdf", "xlsx", "pptx"])
def test_supported_format_ingest_and_search(client, s3, extension):
    marker = f"formatproof{extension} cobalt-unique-731"
    filename = f"evidence.{extension}"
    original = make_document(extension, f"The verified value is {marker}.")

    response = upload(client, filename, original)

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["filename"] == filename
    assert payload["chunks_count"] > 0
    with connect_db() as connection:
        document = connection.execute(
            "SELECT object_key FROM documents WHERE id = %s", (payload["document_id"],)
        ).fetchone()
        rows = connection.execute(
            "SELECT id, text, source_location, vector_dims(embedding) AS dimensions "
            "FROM chunks WHERE document_id = %s ORDER BY ordinal",
            (payload["document_id"],),
        ).fetchall()
    assert document is not None
    assert s3.get_object(Bucket=bucket_name(), Key=document["object_key"])["Body"].read() == original
    assert len(rows) == payload["chunks_count"]
    assert table_counts() == (1, payload["chunks_count"])
    assert all(row["dimensions"] == EMBEDDING_DIMENSIONS for row in rows)

    query = client.post("/v1/query", json={"query": marker})
    assert query.status_code == 200, query.text
    citation = next(item for item in query.json()["citations"] if item["document_id"] == payload["document_id"])
    matching = next(row for row in rows if row["id"] == uuid.UUID(citation["chunk_id"]))
    assert citation["text"] == matching["text"]
    assert marker in citation["text"]
    assert citation["location"] == matching["source_location"]
    if extension == "md":
        assert citation["location"] == 'section "Test document", lines 1-3'
    if extension == "docx":
        assert citation["location"] == 'section "Test document", paragraphs 1-2'
    if extension == "pdf":
        assert citation["location"].startswith("page ")
    if extension == "xlsx":
        assert citation["location"].startswith("sheet ")
    if extension == "pptx":
        assert citation["location"].startswith("slide ")


def test_markdown_location_names_the_section(client):
    preamble_marker = "preamble-basalt-903"
    section_marker = "mdsection-topaz-615"
    response = upload(client, "guide.md", make_markdown_with_sections(preamble_marker, section_marker))

    assert response.status_code == 201, response.text
    assert cited(client, preamble_marker)["location"] == "line 1"
    # The fenced "# ..." shell comment on line 6 must not start a new section inside "Pricing".
    assert cited(client, section_marker)["location"] == 'section "Pricing", lines 4-8'


def test_docx_location_names_the_section(client):
    paragraph_marker = "docxparagraph-jade-371"
    table_marker = "docxtable-amber-842"
    appendix_marker = "docxappendix-coral-129"
    response = upload(
        client, "policy.docx", make_docx_with_sections(paragraph_marker, table_marker, appendix_marker)
    )

    assert response.status_code == 201, response.text
    assert cited(client, paragraph_marker)["location"] == 'section "Limits", paragraphs 2-3'
    assert cited(client, table_marker)["location"] == 'section "Limits", table 1, row 1'
    assert cited(client, appendix_marker)["location"] == 'section "Appendix", paragraphs 4-5'


def test_docx_merged_cells_are_not_duplicated(client):
    marker = "mergedcell-beryl-208"
    response = upload(client, "merged.docx", make_docx_with_merged_cells(marker))

    assert response.status_code == 201, response.text
    assert cited(client, marker)["text"] == f"{marker} | other"


def test_xlsx_rows_with_empty_cells_are_ingested(client):
    marker = "sparsecell-garnet-552"
    response = upload(client, "sparse.xlsx", make_sparse_xlsx(marker))

    assert response.status_code == 201, response.text
    citation = cited(client, marker)
    assert f"B3: {marker}" in citation["text"].splitlines()
    assert citation["location"] == "sheet Sparse, cells A1:C3"


def test_pptx_table_and_grouped_shape_text_is_ingested(client):
    table_marker = "tablecell-quartz-417"
    group_marker = "groupedshape-onyx-268"
    response = upload(client, "deck.pptx", make_pptx_with_table_and_group(table_marker, group_marker))

    assert response.status_code == 201, response.text
    assert cited(client, table_marker)["location"] == "slide 1"
    assert cited(client, group_marker)["location"] == "slide 2"


def test_text_encoding_details_do_not_shift_locations(client):
    # Windows editors start UTF-8 files with a BOM; a form feed is not a line break.
    bom_marker = "bomheading-agate-774"
    response = upload(client, "notes.md", f"﻿# Title\n{bom_marker}".encode())
    assert response.status_code == 201, response.text
    citation = cited(client, bom_marker)
    assert citation["location"] == 'section "Title", lines 1-2'
    assert citation["text"].startswith("# Title")

    feed_marker = "formfeed-jasper-651"
    response = upload(client, "report.txt", f"page one\x0cstill line one\n{feed_marker}".encode())
    assert response.status_code == 201, response.text
    assert cited(client, feed_marker)["location"] == "lines 1-2"


def test_short_lines_are_packed_into_few_chunks(client):
    # One chunk per line cost ~15 KB of RAM per line: a 0.4 MB file of short lines took 3 GB.
    marker = "packedline-opal-quill"
    lines = [f"row {number}" for number in range(1, 10_001)]
    lines[5_000] = f"row 5001 {marker}"
    content = "\n".join(lines).encode()

    response = upload(client, "lines.txt", content)

    assert response.status_code == 201, response.text
    assert response.json()["chunks_count"] <= len(content) // (MAX_CHUNK_CHARS // 2) + 1
    citation = cited(client, marker)
    first, last = map(int, citation["location"].removeprefix("lines ").split("-"))
    assert first <= 5_001 <= last
    assert citation["text"].splitlines()[5_001 - first] == f"row 5001 {marker}"


def test_repeated_chunks_are_stored_once(client):
    # Thousands of identical vectors trap HNSW searches: other documents stop being found at all.
    target = upload(client, "target.txt", b"the reliable sapphire fact")
    response = upload(client, "repetitive.txt", b"a" * 600_000)  # 500 identical 1200-character chunks

    assert response.status_code == 201, response.text
    assert response.json()["chunks_count"] == 1
    citations = client.post("/v1/query", json={"query": "reliable sapphire fact"}).json()["citations"]
    assert any(citation["document_id"] == target.json()["document_id"] for citation in citations)


def test_partially_damaged_xlsx_keeps_readable_sheets(client):
    marker = "survivingsheet-coral-482"
    response = upload(client, "partial.xlsx", make_xlsx_with_broken_sheet(marker))

    assert response.status_code == 201, response.text
    assert response.json()["warnings"] == ["sheet 2 could not be read and was skipped"]
    assert cited(client, marker)["location"].startswith("sheet Intact, ")


def test_pdf_encryption(client):
    # Permission-only encryption (empty user password) opens in any viewer, so it must be indexed.
    marker = "encryptedpdf-onyx-930"
    response = upload(client, "open.pdf", make_encrypted_pdf(f"The value is {marker}.", user_password=""))
    assert response.status_code == 201, response.text
    assert cited(client, marker)["location"] == "page 1"

    response = upload(client, "locked.pdf", make_encrypted_pdf("unreachable text", user_password="secret"))
    assert response.status_code == 422
    assert response.json()["detail"] == "File is password-protected"


# --- Ingest: errors and resource limits ----------------------------------------------------------


def test_unsupported_extension_returns_415_and_writes_nothing(client):
    response = upload(client, "data.csv", b"not supported")
    assert response.status_code == 415
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_empty_document_returns_422(client):
    response = upload(client, "empty.txt", b" \n\t")
    assert response.status_code == 422
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_damaged_files_return_422_and_write_nothing(client):
    for extension in ["docx", "pdf", "xlsx", "pptx"]:
        intact = make_document(extension, "this text is never reached")
        response = upload(client, f"damaged.{extension}", intact[: len(intact) // 2])
        assert response.status_code == 422, (extension, response.text)
        assert response.json()["detail"] == "File is damaged or unreadable", extension
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_pdf_without_text_layer_returns_422(client):
    response = upload(client, "scan.pdf", make_pdf_without_text_layer())
    assert response.status_code == 422
    assert response.json()["detail"] == "File contains no extractable text"
    assert table_counts() == (0, 0)
    assert object_count() == 0


@pytest.mark.parametrize(
    "content",
    [b"text \x98\x98 more", bytes(range(32)) + "текст".encode("cp1251")],
    # 0x98 is invalid in UTF-8 and undefined in Windows-1251; control bytes mean binary data, not text.
    ids=["undefined-byte", "binary"],
)
def test_undecodable_text_returns_422(client, content):
    response = upload(client, "invalid.txt", content)
    assert response.status_code == 422
    assert response.json()["detail"] == UNDECODABLE
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_legacy_text_encodings_are_decoded(client):
    # Older Russian files are often Windows-1251; Notepad saves "Unicode" as UTF-16 with a BOM.
    legacy_marker = "кодировка-рубин-417"
    response = upload(client, "legacy.txt", f"Значение: {legacy_marker}.".encode("cp1251"))
    assert response.status_code == 201, response.text
    assert response.json()["warnings"] == ["file is not UTF-8 and was read as Windows-1251"]
    assert cited(client, legacy_marker)["text"] == f"Значение: {legacy_marker}."

    unicode_marker = "блокнот-сапфир-268"
    response = upload(client, "notepad.txt", f"Значение: {unicode_marker}.".encode("utf-16"))
    assert response.status_code == 201, response.text
    assert response.json()["warnings"] == []
    assert cited(client, unicode_marker)["text"] == f"Значение: {unicode_marker}."


def test_oversize_file_returns_413(client):
    # A byte invalid in every supported encoding makes an incorrectly accepted oversize file fail
    # quickly with 422 instead of being indexed.
    response = upload(client, "oversize.txt", b"\x98" * (5 * 1024 * 1024 + 1))
    assert response.status_code == 413
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_grossly_oversized_body_is_rejected_before_it_is_read():
    # A body declared far over the limit is refused without receiving and spooling it.
    async def endpoint(_scope, _receive, _send):
        raise AssertionError("the request must not reach the application")

    async def receive():
        raise AssertionError("the body must not be read")

    sent = []

    async def send(message):
        sent.append(message)

    middleware = RequestSizeLimit(endpoint, max_bytes=1000)
    scope = {"type": "http", "headers": [(b"content-length", b"1001")]}
    asyncio.run(middleware(scope, receive, send))
    assert sent[0]["status"] == 413


def test_chunked_body_over_the_limit_is_rejected(client):
    # A chunked body declares no length, so it is counted while received; without that it would be
    # spooled to disk in full. A JSON query is used because only the middleware can answer it with 413:
    # the endpoint itself would reject an overlong query with 422.
    def body():
        yield b'{"query": "'
        for _ in range((api.MAX_FILE_SIZE + api.MULTIPART_ALLOWANCE) // 65_536 + 1):
            yield b"a" * 65_536
        yield b'"}'

    response = client.post("/v1/query", content=body(), headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json() == {"detail": "File exceeds 5 MiB"}


def test_ooxml_zip_bomb_is_rejected_before_parsing(client):
    # About 140 KB on the wire, 36 MB of XML inside: parsed, it would take roughly 0.7 GB of RAM.
    response = upload(client, "bomb.docx", make_docx_zip_bomb(36 * 1024 * 1024))
    assert response.status_code == 422
    assert response.json()["detail"] == "Document is too large to process"
    assert table_counts() == (0, 0)


def test_document_over_the_chunk_limit_is_rejected(client, monkeypatch):
    monkeypatch.setattr(extraction, "MAX_CHUNKS", 2)
    # Three distinct 700-character lines: no two fit into one chunk, and none repeats another.
    lines = [f"line {number} " + "word " * 140 for number in range(3)]
    response = upload(client, "long.txt", "\n".join(lines).encode())
    assert response.status_code == 422
    assert response.json()["detail"] == "Document is too large to index"
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_s3_failure_returns_503_without_database_rows(client):
    class BrokenS3:
        def put_object(self, **_kwargs):
            raise RuntimeError("simulated S3 outage")

    app.dependency_overrides[get_s3_client] = lambda: BrokenS3()
    response = upload(client, "file.txt", b"a small valid file")
    assert response.status_code == 503
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_embedding_failure_returns_503_without_s3_or_database_writes(client):
    app.dependency_overrides[get_embedding_provider] = lambda: BrokenEmbedding()
    response = upload(client, "file.txt", b"a small valid file")
    assert response.status_code == 503
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_invalid_embedding_vector_returns_503_and_writes_nothing(client):
    class WrongDimension(EmbeddingProvider):
        def _encode(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    app.dependency_overrides[get_embedding_provider] = lambda: WrongDimension()
    response = upload(client, "file.txt", b"a small valid file")
    assert response.status_code == 503
    assert response.json()["detail"] == "Embedding provider returned an invalid vector"
    assert table_counts() == (0, 0)
    assert object_count() == 0


def fail_chunk_inserts() -> None:
    """Make every chunk INSERT fail after the document row is written; conftest drops the trigger."""
    with connect_db() as connection:
        connection.execute(
            """CREATE FUNCTION fail_chunks_for_test() RETURNS trigger LANGUAGE plpgsql AS $$
               BEGIN RAISE EXCEPTION 'injected chunk failure'; END; $$"""
        )
        connection.execute(
            "CREATE TRIGGER fail_chunks_for_test BEFORE INSERT ON chunks "
            "FOR EACH ROW EXECUTE FUNCTION fail_chunks_for_test()"
        )


def test_database_chunk_error_rolls_back_and_deletes_s3_object(client):
    fail_chunk_inserts()
    response = upload(client, "rollback.txt", b"force one chunk and trigger rollback")
    assert response.status_code == 500
    assert response.json()["detail"] == "Could not save document"
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_s3_cleanup_failure_after_database_error_still_returns_500(client):
    class S3WithBrokenDelete:
        def __init__(self):
            self.real = new_s3_client()

        def put_object(self, **kwargs):
            return self.real.put_object(**kwargs)

        def delete_object(self, **_kwargs):
            raise RuntimeError("simulated S3 outage during cleanup")

    fail_chunk_inserts()
    app.dependency_overrides[get_s3_client] = lambda: S3WithBrokenDelete()
    response = upload(client, "orphan.txt", b"database fails, then S3 cleanup fails too")
    assert response.status_code == 500
    assert response.json()["detail"] == "Could not save document"
    assert table_counts() == (0, 0)
    # Cleanup is best-effort: the original stays in S3 as an orphan without database rows.
    assert object_count() == 1


def test_commit_outcome_unknown_keeps_the_original(client, monkeypatch):
    # If the connection breaks during COMMIT the rows may exist; deleting the original would
    # leave the corpus citing a document whose source is gone.
    def broken_commit(*_args):
        raise CommitUnknownError

    monkeypatch.setattr(api, "save_document", broken_commit)
    response = upload(client, "uncertain.txt", b"commit outcome is unknown")
    assert response.status_code == 500
    assert response.json()["detail"] == "Could not save document"
    assert object_count() == 1


def test_database_outage_on_save_returns_503_and_deletes_the_original(client, monkeypatch):
    # Before COMMIT nothing is saved, so the original must not stay behind; 503 tells the client to retry.
    def database_down(*_args):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(api, "save_document", database_down)
    response = upload(client, "outage.txt", b"the database is down right now")
    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    assert table_counts() == (0, 0)
    assert object_count() == 0


def test_document_write_gets_a_longer_statement_timeout(client, monkeypatch):
    # COPY of a large document is one statement; the 30 s limit for other queries could cut it off.
    seen = []
    real_copy = psycopg.Cursor.copy

    def spy_copy(cursor, *args, **kwargs):
        seen.append(cursor.connection.execute("SHOW statement_timeout").fetchone()["statement_timeout"])
        return real_copy(cursor, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, "copy", spy_copy)
    response = upload(client, "timeout.txt", b"statement timeout check")
    assert response.status_code == 201, response.text
    assert seen == ["5min"]
    with app.state.pool.connection() as connection:
        # The longer limit is local to the write transaction: the pooled connection is back to 30 s.
        assert connection.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "30s"


# --- Query -----------------------------------------------------------------------------------------


def test_blank_query_returns_422(client):
    response = client.post("/v1/query", json={"query": "  \t\n"})
    assert response.status_code == 422


def test_overlong_query_returns_422(client):
    response = client.post("/v1/query", json={"query": "a" * 4001})
    assert response.status_code == 422


class ModelLikeEmbedding(EmbeddingProvider):
    """Like a real model, it gives every text a non-zero vector, punctuation included."""

    def _encode(self, texts):
        return [[1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1) for _ in texts]


@pytest.mark.parametrize(
    ("query", "provider"),
    [("?!... --- ***", ModelLikeEmbedding), ("___", DeterministicEmbeddingProvider)],
    # No word characters at all, whatever the provider; or words the provider maps to a zero vector.
    ids=["punctuation-any-provider", "unrepresentable-tokens"],
)
def test_query_without_searchable_words_returns_no_citations(client, monkeypatch, query, provider):
    # What an HNSW scan returns for such a query depends on the plan: nothing on a small table,
    # arbitrary rows on a large one. The search must not run at all.
    searches = []
    monkeypatch.setattr(api, "search_chunks", lambda *args: searches.append(args) or arbitrary_rows())
    app.dependency_overrides[get_embedding_provider] = lambda: provider()
    response = client.post("/v1/query", json={"query": query})
    assert response.status_code == 200
    assert response.json() == INSUFFICIENT
    assert searches == []


def test_query_embedding_failure_returns_503(client):
    app.dependency_overrides[get_embedding_provider] = lambda: BrokenEmbedding()
    response = client.post("/v1/query", json={"query": "a valid question"})
    assert response.status_code == 503


def test_empty_corpus_returns_fixed_answer(client):
    response = client.post("/v1/query", json={"query": "anything"})
    assert response.status_code == 200
    assert response.json() == INSUFFICIENT


def test_searches_all_documents_not_only_the_latest(client):
    first = upload(client, "first.txt", b"unique firstonly fact is cobalt seven hundred")
    second = upload(client, "second.txt", b"the latest document describes an unrelated amber topic")
    assert first.status_code == second.status_code == 201

    response = client.post("/v1/query", json={"query": "firstonly cobalt seven hundred"})

    assert response.status_code == 200
    assert any(item["document_id"] == first.json()["document_id"] for item in response.json()["citations"])


def test_citations_are_exact_saved_chunks_and_answer_is_grounded(client):
    text = "The documented recovery key is silver-willow-984 and must remain private."
    uploaded = upload(client, "citation.md", text.encode())
    response = client.post("/v1/query", json={"query": "recovery key silver-willow-984"})
    assert uploaded.status_code == 201
    assert response.status_code == 200
    result = response.json()
    assert result["citations"]
    assert len(result["citations"]) <= 5
    citation = result["citations"][0]
    assert citation["document_id"] == uploaded.json()["document_id"]
    assert text in citation["text"]
    assert result["answer"] == citation["text"]
    with connect_db() as connection:
        chunk = connection.execute(
            "SELECT text, source_location FROM chunks WHERE id = %s",
            (citation["chunk_id"],),
        ).fetchone()
    assert chunk is not None
    assert citation["text"] == chunk["text"]
    assert citation["location"] == chunk["source_location"]


def test_query_returns_at_most_five_citations(client):
    for index in range(6):
        response = upload(client, f"match-{index}.txt", f"sharedfact-emerald {index}".encode())
        assert response.status_code == 201
    response = client.post("/v1/query", json={"query": "sharedfact-emerald"})
    assert response.status_code == 200
    assert len(response.json()["citations"]) == 5


def test_answer_provider_failure_returns_citations_without_summary(client):
    # The search already worked: losing the sources along with the summary would waste it.
    uploaded = upload(client, "answer-error.txt", b"answer-error target token")
    assert uploaded.status_code == 201

    app.dependency_overrides[get_answer_provider] = lambda: BrokenAnswer()
    response = client.post("/v1/query", json={"query": "target token"})
    assert response.status_code == 200
    assert response.headers[api.DEGRADED_HEADER] == "generation"
    result = response.json()
    assert result["answer"] == api.GENERATION_UNAVAILABLE
    assert [citation["document_id"] for citation in result["citations"]] == [uploaded.json()["document_id"]]


def test_reranker_failure_keeps_the_vector_order(client):
    first = upload(client, "rerank-a.txt", b"rerankfallback opal alpha")
    second = upload(client, "rerank-b.txt", b"rerankfallback opal beta")
    assert first.status_code == second.status_code == 201
    expected = client.post("/v1/query", json={"query": "rerankfallback opal"}).json()

    app.dependency_overrides[get_reranker_provider] = lambda: BrokenReranker()
    response = client.post("/v1/query", json={"query": "rerankfallback opal"})
    assert response.status_code == 200
    assert response.headers[api.DEGRADED_HEADER] == "rerank"
    # The deterministic reranker keeps the vector order, so a working one must give the same result.
    assert response.json() == expected


def test_quoted_no_answer_phrase_drops_the_citations(client):
    # The prompt shows the phrase in «», and models often repeat the quotes or drop the period.
    class QuotingAnswer(AnswerProvider):
        def _generate(self, query, contexts):
            return f"«{INSUFFICIENT_INFORMATION.rstrip('.')}»"

    assert upload(client, "unrelated.txt", b"quotedanswer garnet fact").status_code == 201
    app.dependency_overrides[get_answer_provider] = lambda: QuotingAnswer()
    response = client.post("/v1/query", json={"query": "quotedanswer garnet"})
    assert response.status_code == 200
    assert response.json() == INSUFFICIENT


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (psycopg.OperationalError("connection refused"), 503, "Database unavailable"),
        (psycopg.errors.UndefinedTable('relation "chunks" does not exist'), 500, "Search failed"),
    ],
    # The database being down is worth a retry; a broken query or schema is not.
    ids=["database-down", "broken-query"],
)
def test_search_failure_status(client, monkeypatch, error, status, detail):
    def failing_search(*_args):
        raise error

    monkeypatch.setattr(api, "search_chunks", failing_search)
    response = client.post("/v1/query", json={"query": "any question"})
    assert response.status_code == status
    assert response.json() == {"detail": detail}


# --- Concurrency and schema --------------------------------------------------------------------------


def test_parallel_ingests_create_independent_documents(client):
    payloads = [
        ("parallel-a.txt", b"parallel unique alpha information"),
        ("parallel-b.txt", b"parallel unique beta information"),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda item: upload(client, *item), payloads))
    assert [response.status_code for response in responses] == [201, 201]
    ids = [response.json()["document_id"] for response in responses]
    assert len(set(ids)) == 2
    with connect_db() as connection:
        rows = connection.execute(
            "SELECT document_id, count(*) AS n, count(DISTINCT ordinal) AS ordinals, "
            "string_agg(text, ' ') AS text FROM chunks GROUP BY document_id"
        ).fetchall()
    assert {str(row["document_id"]): row["text"] for row in rows} == {
        ids[0]: "parallel unique alpha information",
        ids[1]: "parallel unique beta information",
    }
    assert all(row["n"] == row["ordinals"] and row["n"] > 0 for row in rows)


def test_ingests_are_processed_concurrently(client):
    # Each one-line upload embeds exactly once. The barrier releases only when both uploads are
    # inside embedding at the same time; if the API handles them one by one, it times out.
    barrier = threading.Barrier(2, timeout=5)

    class RendezvousEmbedding(DeterministicEmbeddingProvider):
        def _encode(self, texts):
            # An upload embeds all its chunks in one batch, so this runs once per upload.
            barrier.wait()
            return super()._encode(texts)

    app.dependency_overrides[get_embedding_provider] = lambda: RendezvousEmbedding()
    payloads = [("together-a.txt", b"concurrent upload alpha"), ("together-b.txt", b"concurrent upload beta")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda item: upload(client, *item), payloads))
    assert [response.status_code for response in responses] == [201, 201], [r.text for r in responses]
    assert table_counts() == (2, 2)


def test_migrations_create_pgvector_schema_on_empty_database(client):
    # The shared database is already migrated, so the migrations are verified on a fresh one.
    database = f"rag_migration_test_{uuid.uuid4().hex}"
    with connect_db(autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        conninfo = make_conninfo(database_url(), dbname=database)
        apply_migrations(conninfo)
        apply_migrations(conninfo)  # a second run must be a no-op
        with connect_db(conninfo=conninfo) as connection:
            versions = [row["version"] for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version")]
            extension = connection.execute("SELECT count(*) AS n FROM pg_extension WHERE extname = 'vector'").fetchone()["n"]
            tables = {
                row["tablename"]
                for row in connection.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            }
            column = connection.execute(
                """SELECT format_type(a.atttypid, a.atttypmod) AS data_type
                   FROM pg_attribute a JOIN pg_class t ON t.oid = a.attrelid
                   WHERE t.relname = 'chunks' AND a.attname = 'embedding'"""
            ).fetchone()["data_type"]
            index = connection.execute(
                "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'"
            ).fetchone()
    finally:
        with connect_db(autocommit=True) as connection:
            connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(database)))

    assert versions == sorted(path.name for path in MIGRATIONS_DIR.glob("*.sql"))
    assert extension == 1
    assert {"documents", "chunks", "schema_migrations"} <= tables
    assert EMBEDDING_DIMENSIONS == 1024
    assert column == f"vector({EMBEDDING_DIMENSIONS})"
    assert index is not None, "HNSW vector index was not created"
    assert "USING hnsw" in index["indexdef"]
    assert "vector_cosine_ops" in index["indexdef"]
