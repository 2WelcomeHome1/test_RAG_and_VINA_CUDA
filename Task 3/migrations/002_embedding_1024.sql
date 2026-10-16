-- Embeddings switch to deepvk/USER-bge-m3 (1024-d). Old 384-d vectors cannot be converted,
-- so the corpus is cleared and must be uploaded again.
DROP INDEX IF EXISTS chunks_embedding_hnsw;
DELETE FROM documents;  -- chunks are removed by ON DELETE CASCADE
ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
