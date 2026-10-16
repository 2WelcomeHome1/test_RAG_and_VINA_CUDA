-- Baseline schema. IF NOT EXISTS lets this adopt databases created by the former startup DDL.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY,
    filename text NOT NULL,
    object_key text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY,
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal integer NOT NULL,
    text text NOT NULL,
    source_location text NOT NULL,
    -- Must match EMBEDDING_DIMENSIONS in app/providers.py.
    embedding vector(384) NOT NULL,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_document_ordinal_idx ON chunks (document_id, ordinal);
