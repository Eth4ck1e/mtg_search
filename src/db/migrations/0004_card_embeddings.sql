-- Migration 0004 — versioned embeddings table.
--
-- Why: cards.embedding holds ONE vector per face, so every checkpoint swap
-- (base ↔ tuned) re-embeds the corpus and any process that pinned the other
-- version silently finds zero rows (the dashboard did exactly that on
-- 2026-09-22). With the M6 ablation grid producing several checkpoints,
-- vectors from every version must coexist. Searchers pin a version; nothing
-- is overwritten.
--
-- Design: one row per (face, embedding_version). The existing vectors in
-- cards.embedding are copied in so nothing is lost; the cards.embedding
-- columns are left in place (nullable, unused) rather than dropped in the
-- same migration — a later cleanup migration drops them once every reader
-- has moved. No vector index yet (same reasoning as 0002: exact scan over
-- ~32k × few versions is fast enough; add HNSW per-version if it stops being).

CREATE TABLE card_embeddings (
    oracle_id           UUID         NOT NULL,
    face_index          SMALLINT     NOT NULL,
    embedding_version   TEXT         NOT NULL,
    embedding           vector(768)  NOT NULL,
    embedding_text_hash TEXT         NOT NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (oracle_id, face_index, embedding_version),
    FOREIGN KEY (oracle_id, face_index) REFERENCES cards (oracle_id, face_index) ON DELETE CASCADE
);

CREATE INDEX card_embeddings_version_idx ON card_embeddings (embedding_version);

INSERT INTO card_embeddings (oracle_id, face_index, embedding_version, embedding, embedding_text_hash)
SELECT oracle_id, face_index, embedding_version, embedding, embedding_text_hash
FROM cards
WHERE embedding IS NOT NULL;

COMMENT ON TABLE card_embeddings IS
    'One vector per (face, embedding_version). Versions coexist; searchers pin one. Replaces cards.embedding (retained, unused).';
COMMENT ON COLUMN card_embeddings.embedding_version IS
    'settings.embedding_version at encode time: "<model path or HF id>|preproc=<v>".';
