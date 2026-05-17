-- Migration 0001 — initial schema.
--
-- Enables the pgvector extension and creates experiment_runs, the
-- structured log of every evaluation run. Each row captures the
-- configuration that was scored and the metrics it produced; the
-- paper's results section is generated from this table.
--
-- The cards table and its embedding column live in a later migration
-- (Phase 2) — this migration only sets up the pieces that need to exist
-- before Phase 1's logging infrastructure can write its first row.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE experiment_runs (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    eval_set_version TEXT        NOT NULL,
    config           JSONB       NOT NULL,
    metrics          JSONB       NOT NULL,
    per_query        JSONB,
    notes            TEXT
);

CREATE INDEX experiment_runs_created_at_idx ON experiment_runs (created_at DESC);
CREATE INDEX experiment_runs_eval_set_idx   ON experiment_runs (eval_set_version);

COMMENT ON TABLE experiment_runs IS
    'One row per evaluation invocation. config + metrics are JSONB so the schema does not need to change as the pipeline gains new knobs.';
COMMENT ON COLUMN experiment_runs.config IS
    'Full configuration that produced this row: model name, prompt version, preprocessing version, filter flags, K, etc.';
COMMENT ON COLUMN experiment_runs.metrics IS
    'Aggregate metrics: recall@1, recall@5, recall@10, MRR, latency_p50, latency_p95.';
COMMENT ON COLUMN experiment_runs.per_query IS
    'Array of per-query results: {query, expected_ids, retrieved_ids, hit_rank, latency_ms}.';
