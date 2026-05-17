-- Migration 0002 — cards table.
--
-- Stores one row per card face. Single-faced cards have face_index = 0.
-- Multi-faced cards (transform, modal_dfc, split, adventure, flip,
-- meld, prepare) have face_index = 0, 1, ...; dedupe by oracle_id at
-- display time.
--
-- Column layering:
--   * Per-face columns vary between faces of the same card:
--       name, mana_cost, colors, type_line, oracle_text,
--       power, toughness, loyalty.
--   * Card-level columns are duplicated across face rows of the same
--     oracle_id (Scryfall does not expose them per-face):
--       cmc, color_identity, keywords, layout, released_at,
--       legalities, raw.
--   * Audit: created_at, updated_at. The ingest script's UPSERT sets
--     updated_at = NOW() explicitly — no trigger.
--   * Embedding: filled by Phase 3's embed.py. All three of embedding,
--     embedding_version, embedding_text_hash must be NULL together or
--     NOT NULL together — enforced by CHECK. This prevents the
--     "silent inconsistency" failure mode flagged in CLAUDE.md §3.
--
-- Intentional omissions:
--   * No separate card_faces JSONB column. Asymmetric to store only on
--     face_index = 0; wasteful to duplicate. The raw JSONB escape hatch
--     already contains card_faces; queries that need cross-face data
--     read it from there.
--   * No lang column. The oracle_cards bulk is English-only; defer
--     until a multilingual evaluation actually needs it.
--   * No vector index. Exact search over ~30k 768-dim vectors is fast
--     enough at this scale; add HNSW/IVFFlat in Phase 5 once
--     measurements justify it.

CREATE TABLE cards (
    -- ----- Key -----
    oracle_id            UUID         NOT NULL,
    face_index           SMALLINT     NOT NULL CHECK (face_index >= 0),

    -- ----- Per-face columns -----
    name                 TEXT         NOT NULL,
    mana_cost            TEXT,
    colors               TEXT[]       NOT NULL DEFAULT '{}',
    type_line            TEXT         NOT NULL,
    oracle_text          TEXT         NOT NULL DEFAULT '',
    power                TEXT,
    toughness            TEXT,
    loyalty              TEXT,

    -- ----- Card-level columns (duplicated across face rows) -----
    cmc                  NUMERIC(4,2) NOT NULL CHECK (cmc >= 0),
    color_identity       TEXT[]       NOT NULL DEFAULT '{}',
    keywords             TEXT[]       NOT NULL DEFAULT '{}',
    layout               TEXT         NOT NULL,
    released_at          DATE,
    legalities           JSONB        NOT NULL,
    raw                  JSONB        NOT NULL,

    -- ----- Audit -----
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- ----- Embedding (filled by Phase 3) -----
    embedding            vector(768),
    embedding_version    TEXT,
    embedding_text_hash  TEXT,

    PRIMARY KEY (oracle_id, face_index),

    CONSTRAINT embedding_triple_paired CHECK (
        (embedding IS NULL AND embedding_version IS NULL AND embedding_text_hash IS NULL)
        OR
        (embedding IS NOT NULL AND embedding_version IS NOT NULL AND embedding_text_hash IS NOT NULL)
    )
);

-- Indexes — filter columns get index support so SQL pre-filter stays cheap.
CREATE INDEX cards_colors_idx         ON cards USING GIN (colors);
CREATE INDEX cards_color_identity_idx ON cards USING GIN (color_identity);
CREATE INDEX cards_keywords_idx       ON cards USING GIN (keywords);
CREATE INDEX cards_cmc_idx            ON cards (cmc);
CREATE INDEX cards_layout_idx         ON cards (layout);
CREATE INDEX cards_released_at_idx    ON cards (released_at);

-- ----- Comments (per-column documentation in the catalog) -----

COMMENT ON TABLE  cards IS
    'One row per card face. Single-faced cards: face_index=0. Multi-face: 0..N. Dedupe by oracle_id at display.';

COMMENT ON COLUMN cards.oracle_id IS
    'Scryfall oracle_id — stable identifier for the canonical card (same across reprints).';
COMMENT ON COLUMN cards.face_index IS
    'Zero-based face index. Always 0 for single-faced cards.';
COMMENT ON COLUMN cards.mana_cost IS
    'Per-face mana cost like ''{1}{U}{U}''. NULL for back faces of transform DFCs.';
COMMENT ON COLUMN cards.colors IS
    'Per-face active colors. May differ from color_identity (which is card-level).';
COMMENT ON COLUMN cards.cmc IS
    'Card-level converted mana value. NUMERIC because Un-set half-mana costs exist.';
COMMENT ON COLUMN cards.keywords IS
    'Scryfall-parsed canonical keyword list (card-level). Case-sensitive (e.g. ''First strike'').';
COMMENT ON COLUMN cards.legalities IS
    'Card-level Scryfall legalities JSONB. Filtered by format at query time, not at ingest.';
COMMENT ON COLUMN cards.raw IS
    'Full original Scryfall record. Escape hatch — read from here for fields not promoted to columns.';
COMMENT ON COLUMN cards.embedding IS
    'Per-face 768-dim vector from the embedding model. NULL until Phase 3''s embed.py fills it.';
COMMENT ON COLUMN cards.embedding_version IS
    'Composite version: ''<model>|preproc=<v>''. Identifies what produced this vector.';
COMMENT ON COLUMN cards.embedding_text_hash IS
    'SHA-256 of the exact text fed to the encoder. Detects silent preprocessing drift.';
