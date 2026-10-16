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
PROVIDERS = "app/providers.py"
STORAGE = "app/storage.py"
MIGRATION = "migrations/001_init.sql"
TESTS = "tests/test_acceptance.py::"


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
             "    for number, line in enumerate(iter_lines(decode_utf8(content)), start=1):\n"
             '        yield Block(line, Units("line", number, number))\n',
             "    for number, line in enumerate(decode_utf8(content).splitlines(), start=1):\n"
             '        yield Block(line, Units("line", number, number))\n'),
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
    Mutation("invalid-utf8-silently-replaced", TESTS + "test_invalid_utf8_returns_422", EXTRACTION,
             'return content.decode("utf-8-sig")', 'return content.decode("utf-8-sig", errors="replace")'),
    Mutation("oversize-limit-off-by-one", TESTS + "test_oversize_file_returns_413", API,
             "MAX_FILE_SIZE = 5 * 1024 * 1024", "MAX_FILE_SIZE = 5 * 1024 * 1024 + 1"),
    Mutation("oversized-body-read-anyway", TESTS + "test_grossly_oversized_body_is_rejected_before_it_is_read", API,
             "if declared.isdigit() and int(declared) > self.max_bytes:", "if False:"),
    Mutation("zip-bomb-parsed", TESTS + "test_ooxml_zip_bomb_is_rejected_before_parsing", EXTRACTION,
             "if unpacked > limit:", "if False:"),
    Mutation("chunk-limit-ignored", TESTS + "test_document_over_the_chunk_limit_is_rejected", EXTRACTION,
             "if len(self.chunks) >= MAX_CHUNKS:", "if False:"),
    Mutation("s3-error-wrong-status", TESTS + "test_s3_failure_returns_503_without_database_rows", API,
             'raise HTTPException(status_code=503, detail="S3 unavailable")',
             'raise HTTPException(status_code=500, detail="S3 unavailable")'),
    Mutation("embedding-error-wrong-status", TESTS + "test_embedding_failure_returns_503_without_s3_or_database_writes", API,
             "        vectors = [embed(embedding_provider, chunk.text) for chunk in extraction.chunks]\n"
             "    except ProviderError as error:\n"
             "        raise HTTPException(status_code=503, detail=str(error)) from error",
             "        vectors = [embed(embedding_provider, chunk.text) for chunk in extraction.chunks]\n"
             "    except ProviderError as error:\n"
             "        raise HTTPException(status_code=500, detail=str(error)) from error"),
    Mutation("invalid-vector-accepted", TESTS + "test_invalid_embedding_vector_returns_503_and_writes_nothing", PROVIDERS,
             "if values.shape != (EMBEDDING_DIMENSIONS,) or not np.isfinite(values).all():", "if False:"),
    Mutation("commit-before-chunks", TESTS + "test_database_chunk_error_rolls_back_and_deletes_s3_object", STORAGE,
             "                (document_id, filename, object_key),\n            )\n",
             "                (document_id, filename, object_key),\n            )\n            connection.commit()\n"),
    Mutation("s3-cleanup-error-escapes", TESTS + "test_s3_cleanup_failure_after_database_error_still_returns_500", STORAGE,
             "    try:\n"
             "        s3.delete_object(Bucket=bucket_name(), Key=object_key)\n"
             "    except Exception:\n"
             '        logger.exception("Orphaned original in S3, delete it manually: %s/%s", bucket_name(), object_key)',
             "    s3.delete_object(Bucket=bucket_name(), Key=object_key)"),
    Mutation("unknown-commit-deletes-original", TESTS + "test_commit_outcome_unknown_keeps_the_original", API,
             '        raise HTTPException(status_code=500, detail="Could not save document") from error\n'
             "    except Exception as error:",
             "        delete_original(s3, object_key)\n"
             '        raise HTTPException(status_code=500, detail="Could not save document") from error\n'
             "    except Exception as error:"),
    # Query
    Mutation("accept-blank-query", TESTS + "test_blank_query_returns_422", API,
             "if not body.query.strip():", "if False:"),
    Mutation("accept-overlong-query", TESTS + "test_overlong_query_returns_422", API,
             "query: str = Field(max_length=MAX_QUERY_CHARS)", "query: str"),
    Mutation("rank-punctuation-only-query", TESTS + "test_punctuation_only_query_returns_no_citations", API,
             "if is_zero(query_vector):", "if False:"),
    Mutation("query-embedding-error-wrong-status", TESTS + "test_query_embedding_failure_returns_503", API,
             "        query_vector = embed(embedding_provider, body.query)\n"
             "    except ProviderError as error:\n"
             "        raise HTTPException(status_code=503, detail=str(error)) from error",
             "        query_vector = embed(embedding_provider, body.query)\n"
             "    except ProviderError as error:\n"
             "        raise HTTPException(status_code=500, detail=str(error)) from error"),
    Mutation("wrong-empty-corpus-answer", TESTS + "test_empty_corpus_returns_fixed_answer", API,
             'INSUFFICIENT_INFORMATION = "Недостаточно информации в загруженных документах."',
             'INSUFFICIENT_INFORMATION = "Попробуйте ещё раз."'),
    Mutation("search-only-latest-document", TESTS + "test_searches_all_documents_not_only_the_latest", STORAGE,
             "FROM chunks c JOIN documents d ON d.id = c.document_id\n               ORDER BY",
             "FROM chunks c JOIN documents d ON d.id = c.document_id\n"
             "               WHERE d.id = (SELECT id FROM documents ORDER BY created_at DESC LIMIT 1)\n"
             "               ORDER BY"),
    Mutation("alter-citation-text", TESTS + "test_citations_are_exact_saved_chunks_and_answer_is_grounded", API,
             'text=row["text"],', 'text=row["text"] + " mutated",'),
    Mutation("answer-outside-retrieved-context", TESTS + "test_citations_are_exact_saved_chunks_and_answer_is_grounded", PROVIDERS,
             'return "\\n".join(contexts)', 'return "\\n".join(contexts) + " unsupported claim"'),
    Mutation("return-six-citations", TESTS + "test_query_returns_at_most_five_citations", API,
             "CITATION_LIMIT = 5", "CITATION_LIMIT = 6"),
    Mutation("answer-error-wrong-status", TESTS + "test_answer_provider_failure_returns_503", API,
             "        text = answer(answer_provider, body.query, [citation.text for citation in citations])\n"
             "    except ProviderError as error:\n"
             "        raise HTTPException(status_code=503, detail=str(error)) from error",
             "        text = answer(answer_provider, body.query, [citation.text for citation in citations])\n"
             "    except ProviderError as error:\n"
             "        raise HTTPException(status_code=500, detail=str(error)) from error"),
    # Concurrency and schema
    Mutation("reuse-document-id", TESTS + "test_parallel_ingests_create_independent_documents", API,
             "document_id = uuid.uuid4()", "document_id = uuid.UUID(int=1)"),
    Mutation("blocking-upload-on-event-loop", TESTS + "test_ingests_are_processed_concurrently", API,
             "def upload_document(", "async def upload_document("),
    Mutation("migration-wrong-vector-dimension", TESTS + "test_migrations_create_pgvector_schema_on_empty_database", MIGRATION,
             "embedding vector(384) NOT NULL", "embedding vector(128) NOT NULL"),
    Mutation("migration-without-vector-index", TESTS + "test_migrations_create_pgvector_schema_on_empty_database", MIGRATION,
             "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);", ""),
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
