"""Run one targeted, source-level mutant against every acceptance-test behavior."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COPIED_DIRECTORIES = ("app", "migrations", "tests")
# A mutant that makes a test hang is detected, not survived; the limit only stops the wait.
TEST_TIMEOUT_S = 600
API = "app/main.py"
EXTRACTION = "app/extraction.py"
PROMPT = "app/prompt.py"
LOGGER = "app/core/logger.py"
EMBEDDING = "app/providers/embedding.py"
ANSWER = "app/providers/answer.py"
STORAGE = "app/storage.py"
# 002 sets the column type and recreates the index, so mutating 001 would change nothing in the end.
MIGRATION = "migrations/002_embedding_1024.sql"
TESTS = "tests/test_acceptance.py::"
COMPONENTS = "tests/test_components.py::"


@dataclass(frozen=True)
class Mutation:
    name: str
    test: str
    path: str
    before: str
    after: str


MUTATIONS = [
    # Formats and source locations
    Mutation("drop-original-object", TESTS + "test_supported_format_ingest_and_search[txt]", API,
             "s3.put_object(Bucket=bucket_name(), Key=object_key, Body=content)", "pass"),
    Mutation("reject-txt", TESTS + "test_supported_format_ingest_and_search[txt]", EXTRACTION,
             '    ".txt": text_blocks,\n', ""),
    Mutation("reject-markdown", TESTS + "test_supported_format_ingest_and_search[markdown]", EXTRACTION,
             '    ".md": markdown_blocks,\n', ""),
    Mutation("reject-docx", TESTS + "test_supported_format_ingest_and_search[docx]", EXTRACTION,
             '    ".docx": docx_blocks,\n', ""),
    Mutation("reject-text-pdf", TESTS + "test_supported_format_ingest_and_search[pdf]", EXTRACTION,
             '    ".pdf": pdf_blocks,\n', ""),
    Mutation("reject-xlsx", TESTS + "test_supported_format_ingest_and_search[xlsx]", EXTRACTION,
             '    ".xlsx": xlsx_blocks,\n', ""),
    Mutation("reject-pptx", TESTS + "test_supported_format_ingest_and_search[pptx]", EXTRACTION,
             '    ".pptx": pptx_blocks,\n', ""),
    Mutation("markdown-ignore-headings", TESTS + "test_markdown_location_names_the_section", EXTRACTION,
             "elif not in_code and (heading := MARKDOWN_HEADING.match(line)):", "elif False:"),
    Mutation("markdown-heading-inside-code-fence", TESTS + "test_markdown_location_names_the_section", EXTRACTION,
             "in_code = not in_code", "in_code = False"),
    Mutation("docx-ignore-heading-styles", TESTS + "test_docx_location_names_the_section", EXTRACTION,
             'if style.startswith("Heading") and item.text.strip():',
             'if style.startswith("Mutant") and item.text.strip():'),
    Mutation("docx-table-without-section", TESTS + "test_docx_location_names_the_section", EXTRACTION,
             'scope = join_scope(section, f"table {tables}")', 'scope = f"table {tables}"'),
    Mutation("docx-merged-cells-repeated", TESTS + "test_docx_merged_cells_are_not_duplicated", EXTRACTION,
             "if id(cell._tc) not in seen:", "if True:"),
    Mutation("xlsx-location-from-padded-row", TESTS + "test_xlsx_rows_with_empty_cells_are_ingested", EXTRACTION,
             "Cells(filled[0].row, filled[0].column, filled[-1].row, filled[-1].column)",
             "Cells(row[0].row, row[0].column, row[-1].row, row[-1].column)"),
    Mutation("pptx-skip-table-text", TESTS + "test_pptx_table_and_grouped_shape_text_is_ingested", EXTRACTION,
             'elif getattr(shape, "has_table", False):', "elif False:"),
    Mutation("pptx-skip-grouped-shape-text", TESTS + "test_pptx_table_and_grouped_shape_text_is_ingested", EXTRACTION,
             "texts.extend(pptx_shape_texts(shape.shapes))", "texts.extend([])"),
    Mutation("utf8-bom-kept", TESTS + "test_text_encoding_details_do_not_shift_locations", EXTRACTION,
             'return content.decode("utf-8-sig")', 'return content.decode("utf-8")'),
    Mutation("form-feed-splits-lines", TESTS + "test_text_encoding_details_do_not_shift_locations", EXTRACTION,
             "    for number, line in enumerate(iter_lines(decode_text(content, warnings)), start=1):\n"
             '        yield Block(line, Units("line", number, number))\n',
             "    for number, line in enumerate(decode_text(content, warnings).splitlines(), start=1):\n"
             '        yield Block(line, Units("line", number, number))\n'),
    Mutation("legacy-encoding-rejected", TESTS + "test_legacy_text_encodings_are_decoded", EXTRACTION,
             "text = content.decode(LEGACY_ENCODING)", 'text = content.decode("utf-8")'),
    Mutation("legacy-encoding-not-reported", TESTS + "test_legacy_text_encodings_are_decoded", EXTRACTION,
             'warnings.append("file is not UTF-8 and was read as Windows-1251")', "pass"),
    Mutation("utf16-bom-ignored", TESTS + "test_legacy_text_encodings_are_decoded", EXTRACTION,
             "if content.startswith(UTF16_BOMS):", "if False:"),
    Mutation("chunk-per-line", TESTS + "test_short_lines_are_packed_into_few_chunks", EXTRACTION,
             "if not block.packable or len(text) > MAX_CHUNK_CHARS:", "if True:"),
    Mutation("repeated-chunks-stored", TESTS + "test_repeated_chunks_are_stored_once", EXTRACTION,
             "if text in self._seen:", "if False:"),
    Mutation("broken-sheet-fails-whole-file", TESTS + "test_partially_damaged_xlsx_keeps_readable_sheets", EXTRACTION,
             '            warnings.append(f"{unit} {number} could not be read and was skipped")', "            raise"),
    Mutation("password-pdf-reported-as-damaged", TESTS + "test_pdf_encryption", EXTRACTION,
             'if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:', "if False:"),
    # Ingest errors and resource limits
    Mutation("unsupported-format-wrong-status", TESTS + "test_unsupported_extension_returns_415_and_writes_nothing", API,
             'raise HTTPException(status_code=415, detail="Unsupported file format")',
             'raise HTTPException(status_code=422, detail="Unsupported file format")'),
    Mutation("accept-empty-document", TESTS + "test_empty_document_returns_422", EXTRACTION,
             "    if not builder.chunks:", "    if False:"),
    Mutation("damaged-file-wrong-detail", TESTS + "test_damaged_files_return_422_and_write_nothing", EXTRACTION,
             'raise ExtractionError("File is damaged or unreadable") from error',
             'raise ExtractionError("File contains no extractable text") from error'),
    Mutation("pdf-placeholder-for-missing-text-layer", TESTS + "test_pdf_without_text_layer_returns_422", EXTRACTION,
             'page.extract_text() or ""', 'page.extract_text() or "[scanned page]"'),
    Mutation("undecodable-text-silently-replaced", TESTS + "test_undecodable_text_returns_422[undefined-byte]",
             EXTRACTION, "text = content.decode(LEGACY_ENCODING)",
             'text = content.decode(LEGACY_ENCODING, errors="replace")'),
    Mutation("binary-text-accepted", TESTS + "test_undecodable_text_returns_422[binary]", EXTRACTION,
             "if CONTROL_CHARACTERS.search(text):", "if False:"),
    Mutation("oversize-limit-off-by-one", TESTS + "test_oversize_file_returns_413", API,
             "MAX_FILE_SIZE = 5 * 1024 * 1024", "MAX_FILE_SIZE = 5 * 1024 * 1024 + 1"),
    Mutation("oversized-body-read-anyway", TESTS + "test_grossly_oversized_body_is_rejected_before_it_is_read", API,
             "if declared.isdigit() and int(declared) > self.max_bytes:", "if False:"),
    Mutation("chunked-body-not-counted", TESTS + "test_chunked_body_over_the_limit_is_rejected", API,
             "if received > self.max_bytes:", "if False:"),
    Mutation("zip-bomb-parsed", TESTS + "test_ooxml_zip_bomb_is_rejected_before_parsing", EXTRACTION,
             "if unpacked > limit:", "if False:"),
    Mutation("chunk-limit-ignored", TESTS + "test_document_over_the_chunk_limit_is_rejected", EXTRACTION,
             "if len(self.chunks) >= MAX_CHUNKS:", "if False:"),
    Mutation("s3-error-wrong-status", TESTS + "test_s3_failure_returns_503_without_database_rows", API,
             'raise HTTPException(status_code=503, detail="S3 unavailable")',
             'raise HTTPException(status_code=500, detail="S3 unavailable")'),
    Mutation("embedding-error-wrong-status", TESTS + "test_embedding_failure_returns_503_without_s3_or_database_writes", API,
             '            filename, len(extraction.chunks), timings["эмбеддинги"], error, exc_info=error,\n'
             "        )\n"
             "        raise HTTPException(status_code=503, detail=str(error)) from error",
             '            filename, len(extraction.chunks), timings["эмбеддинги"], error, exc_info=error,\n'
             "        )\n"
             "        raise HTTPException(status_code=500, detail=str(error)) from error"),
    Mutation("invalid-vector-accepted", TESTS + "test_invalid_embedding_vector_returns_503_and_writes_nothing", EMBEDDING,
             "if matrix.shape != (len(texts), EMBEDDING_DIMENSIONS) or not np.isfinite(matrix).all():", "if False:"),
    Mutation("commit-before-chunks", TESTS + "test_database_chunk_error_rolls_back_and_deletes_s3_object", STORAGE,
             "                (document_id, filename, object_key),\n            )\n",
             "                (document_id, filename, object_key),\n            )\n            connection.commit()\n"),
    Mutation("s3-cleanup-error-escapes", TESTS + "test_s3_cleanup_failure_after_database_error_still_returns_500", STORAGE,
             "    try:\n"
             "        s3.delete_object(Bucket=bucket_name(), Key=object_key)\n"
             '        logger.info("Оригинал %s удалён из S3 после неудачной записи в БД", object_key)\n'
             "    except Exception:\n"
             '        logger.exception("Оригинал остался в S3 без записи в БД, удалите его вручную: %s/%s", bucket_name(), object_key)',
             "    s3.delete_object(Bucket=bucket_name(), Key=object_key)"),
    Mutation("unknown-commit-deletes-original", TESTS + "test_commit_outcome_unknown_keeps_the_original", API,
             "            document_id, object_key, exc_info=error,\n"
             "        )\n"
             '        raise HTTPException(status_code=500, detail="Could not save document") from error\n',
             "            document_id, object_key, exc_info=error,\n"
             "        )\n"
             "        delete_original(s3, object_key)\n"
             '        raise HTTPException(status_code=500, detail="Could not save document") from error\n'),
    Mutation("database-outage-on-save-wrong-status",
             TESTS + "test_database_outage_on_save_returns_503_and_deletes_the_original", API,
             "        delete_original(s3, object_key)\n"
             '        raise HTTPException(status_code=503, detail="Database unavailable") from error',
             "        delete_original(s3, object_key)\n"
             '        raise HTTPException(status_code=500, detail="Database unavailable") from error'),
    Mutation("database-outage-keeps-original",
             TESTS + "test_database_outage_on_save_returns_503_and_deletes_the_original", API,
             "        delete_original(s3, object_key)\n"
             '        raise HTTPException(status_code=503, detail="Database unavailable") from error',
             '        raise HTTPException(status_code=503, detail="Database unavailable") from error'),
    Mutation("save-timeout-not-raised", TESTS + "test_document_write_gets_a_longer_statement_timeout", STORAGE,
             "            cursor.execute(\"SELECT set_config('statement_timeout', %s, true)\", "
             "(str(SAVE_STATEMENT_TIMEOUT_MS),))\n", ""),
    Mutation("save-timeout-leaks-to-pool", TESTS + "test_document_write_gets_a_longer_statement_timeout", STORAGE,
             "set_config('statement_timeout', %s, true)", "set_config('statement_timeout', %s, false)"),
    # Query
    Mutation("accept-blank-query", TESTS + "test_blank_query_returns_422", API,
             "if not body.query.strip():", "if False:"),
    Mutation("accept-overlong-query", TESTS + "test_overlong_query_returns_422", API,
             "query: str = Field(max_length=MAX_QUERY_CHARS)", "query: str"),
    Mutation("rank-punctuation-only-query",
             TESTS + "test_query_without_searchable_words_returns_no_citations[punctuation-any-provider]", API,
             "if not SEARCHABLE.search(body.query):", "if False:"),
    Mutation("rank-zero-vector-query",
             TESTS + "test_query_without_searchable_words_returns_no_citations[unrepresentable-tokens]", API,
             "if is_zero(query_vector):", "if False:"),
    Mutation("query-embedding-error-wrong-status", TESTS + "test_query_embedding_failure_returns_503", API,
             '        logger.error("Эмбеддинг запроса не получен: %s", error, exc_info=error)\n'
             "        raise HTTPException(status_code=503, detail=str(error)) from error",
             '        logger.error("Эмбеддинг запроса не получен: %s", error, exc_info=error)\n'
             "        raise HTTPException(status_code=500, detail=str(error)) from error"),
    Mutation("wrong-empty-corpus-answer", TESTS + "test_empty_corpus_returns_fixed_answer", PROMPT,
             'INSUFFICIENT_INFORMATION = "Недостаточно информации в загруженных документах."',
             'INSUFFICIENT_INFORMATION = "Попробуйте ещё раз."'),
    Mutation("search-only-latest-document", TESTS + "test_searches_all_documents_not_only_the_latest", STORAGE,
             "FROM chunks c JOIN documents d ON d.id = c.document_id\n               ORDER BY",
             "FROM chunks c JOIN documents d ON d.id = c.document_id\n"
             "               WHERE d.id = (SELECT id FROM documents ORDER BY created_at DESC LIMIT 1)\n"
             "               ORDER BY"),
    Mutation("alter-citation-text", TESTS + "test_citations_are_exact_saved_chunks_and_answer_is_grounded", API,
             'text=row["text"],', 'text=row["text"] + " mutated",'),
    Mutation("answer-outside-retrieved-context", TESTS + "test_citations_are_exact_saved_chunks_and_answer_is_grounded",
             ANSWER, 'return "\\n".join(contexts)', 'return "\\n".join(contexts) + " unsupported claim"'),
    Mutation("return-six-citations", TESTS + "test_query_returns_at_most_five_citations", API,
             "CITATION_LIMIT = 5", "CITATION_LIMIT = 6"),
    Mutation("generation-outage-fails-query",
             TESTS + "test_answer_provider_failure_returns_citations_without_summary", API,
             "        text = GENERATION_UNAVAILABLE\n"
             '        degraded.append("generation")',
             "        raise HTTPException(status_code=503, detail=str(error)) from error"),
    Mutation("degraded-header-missing",
             TESTS + "test_answer_provider_failure_returns_citations_without_summary", API,
             '        response.headers[DEGRADED_HEADER] = ",".join(degraded)', "        pass"),
    Mutation("reranker-outage-fails-query", TESTS + "test_reranker_failure_keeps_the_vector_order", API,
             "        best = list(range(min(CITATION_LIMIT, len(rows))))\n"
             '        degraded.append("rerank")',
             "        raise HTTPException(status_code=503, detail=str(error)) from error"),
    Mutation("quoted-no-answer-keeps-citations", TESTS + "test_quoted_no_answer_phrase_drops_the_citations", API,
             'insufficient = "generation" not in degraded and is_insufficient(text)',
             'insufficient = "generation" not in degraded and text == INSUFFICIENT_INFORMATION'),
    Mutation("search-outage-wrong-status", TESTS + "test_search_failure_status[database-down]", API,
             '        raise HTTPException(status_code=503, detail="Database unavailable") from error\n'
             "    except Exception as error:\n"
             '        logger.error("Векторный поиск не выполнен", exc_info=error)',
             '        raise HTTPException(status_code=500, detail="Database unavailable") from error\n'
             "    except Exception as error:\n"
             '        logger.error("Векторный поиск не выполнен", exc_info=error)'),
    Mutation("search-error-wrong-status", TESTS + "test_search_failure_status[broken-query]", API,
             'raise HTTPException(status_code=500, detail="Search failed") from error',
             'raise HTTPException(status_code=503, detail="Search failed") from error'),
    # Concurrency and schema
    Mutation("reuse-document-id", TESTS + "test_parallel_ingests_create_independent_documents", API,
             "document_id = uuid.uuid4()", "document_id = uuid.UUID(int=1)"),
    Mutation("blocking-upload-on-event-loop", TESTS + "test_ingests_are_processed_concurrently", API,
             "def upload_document(", "async def upload_document("),
    Mutation("migration-wrong-vector-dimension", TESTS + "test_migrations_create_pgvector_schema_on_empty_database", MIGRATION,
             "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024);",
             "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(128);"),
    Mutation("migration-without-vector-index", TESTS + "test_migrations_create_pgvector_schema_on_empty_database", MIGRATION,
             "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);", ""),
    # Components
    Mutation("log-error-breaks-request", COMPONENTS + "test_log_call_with_bad_arguments_does_not_fail_the_caller", LOGGER,
             "        except Exception:\n            self.handleError(record)",
             "        except Exception:\n            raise"),
    Mutation("warm-up-without-options", COMPONENTS + "test_ollama_warm_up_uses_the_query_options", ANSWER,
             "model=settings.LLM_MODEL, messages=[], options=self._options, keep_alive=settings.LLM_KEEP_ALIVE",
             "model=settings.LLM_MODEL, messages=[], keep_alive=settings.LLM_KEEP_ALIVE"),
    Mutation("generation-slots-unbounded", COMPONENTS + "test_ollama_generation_over_the_cap_fails_fast", ANSWER,
             "if not self._slots.acquire(timeout=settings.LLM_QUEUE_TIMEOUT):", "if not self._slots.acquire():"),
    Mutation("generation-slot-leaks", COMPONENTS + "test_ollama_generation_over_the_cap_fails_fast", ANSWER,
             "        finally:\n            self._slots.release()", "        finally:\n            pass"),
]


def run_pytest(cwd: Path, tests: list[str]) -> subprocess.CompletedProcess | None:
    """Run the tests; None means they did not finish in time."""
    try:
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", *tests],
            cwd=cwd,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=TEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None


def main() -> int:
    # A kill only means something if the targeted test passes on the unmutated code.
    targets = sorted({mutation.test for mutation in MUTATIONS})
    baseline = run_pytest(ROOT, targets)
    if baseline is None or baseline.returncode != 0:
        print("BASELINE FAILED: the targeted tests must pass before mutating")
        print(baseline.stdout[-6000:] if baseline else "timed out")
        return 1
    print(f"Baseline: {len(targets)} targeted tests pass.", flush=True)

    for mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix=f"rag-mutant-{mutation.name}-") as temp:
            project = Path(temp)
            for directory in COPIED_DIRECTORIES:
                shutil.copytree(ROOT / directory, project / directory)
            mutant_source = project / mutation.path
            source_text = mutant_source.read_text(encoding="utf-8")
            count = source_text.count(mutation.before)
            if count != 1:
                print(f"ERROR {mutation.name}: expected one mutation site in {mutation.path}, found {count}")
                return 1
            mutant_source.write_text(source_text.replace(mutation.before, mutation.after, 1), encoding="utf-8")

            started = time.perf_counter()
            completed = run_pytest(project, [mutation.test])
            elapsed = f"{time.perf_counter() - started:.1f}s"
            if completed is None:
                print(f"KILLED {mutation.name} by {mutation.test} (test timed out after {elapsed})", flush=True)
                continue
            if completed.returncode == 0 or f"FAILED {mutation.test}" not in completed.stdout:
                print(f"SURVIVED OR INVALID KILL: {mutation.name} / {mutation.test}")
                print(completed.stdout[-6000:])
                return 1
            print(f"KILLED {mutation.name} by {mutation.test} ({elapsed})", flush=True)

    print(f"Mutation check passed: {len(MUTATIONS)}/{len(MUTATIONS)} targeted mutants killed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
