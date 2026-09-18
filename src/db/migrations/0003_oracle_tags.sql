-- Migration 0003 — Scryfall Tagger oracle tags.
--
-- Source: the official ``oracle_tags`` bulk-data file
-- (https://scryfall.com/docs/api/tags), published daily by Scryfall from
-- the community Tagger project. One Tag object per line, each carrying
-- its hierarchy (parent_ids / child_ids), aliases, an optional
-- description, and a list of taggings keyed by oracle_id.
--
-- Purpose: distant supervision for M6 embedder fine-tuning. A functional
-- tag ("sweeper", "cantrip", "ramp") names a set of cards that share a
-- gameplay function regardless of how their Oracle text phrases it —
-- exactly the jargon → rules-text mapping the base encoder lacks
-- (see docs/journal/2026-09-18-*.md). Tags are NOT embedded and NOT
-- used by the SQL pre-filter; they are training-signal only.
--
-- Design:
--   * Two tables, not a column on ``cards``. Tags are many-to-many with
--     cards, arrive from a different bulk file on a different cadence,
--     and cover oracle_ids outside our filtered corpus (~36k tagged
--     vs ~30k ingested). A TEXT[] on ``cards`` would force the tag
--     refresh to rewrite the cards table and would lose the hierarchy.
--   * ``oracle_tags.id`` is Scryfall's stable UUID. Slugs and labels are
--     documented as mutable ("Do not treat tag slugs or labels as
--     permanent identifiers") so every join goes through ``id``.
--   * No FK from ``card_tags.oracle_id`` to ``cards``: the cards PK is
--     composite (oracle_id, face_index) and taggings legitimately
--     reference cards we filtered out. Join at query time; the ingest
--     script logs the coverage ratio.
--   * ``parent_ids`` stored as UUID[] on the tag row (multi-parent DAG,
--     max observed depth 6, no cycles as of 2026-09-18). The
--     ``oracle_tag_closure`` view materialises the transitive
--     ancestor→descendant relation with a recursive CTE so that
--     "all cards under removal" resolves the same way Scryfall's
--     ``otag:removal`` search does (parent tag ∪ all descendants —
--     verified against the live API 2026-09-18).
--   * The ingest is a full replace inside one transaction (tags are
--     derived data with no local edits), so idempotency is trivial.
--
-- Weight semantics (Scryfall): very_strong / strong / median / weak.
-- As of 2026-09-18 the file is 99.7% ``median``; the column is kept for
-- forward-compatibility, not because it currently carries signal.

CREATE TABLE oracle_tags (
    id               UUID         PRIMARY KEY,
    slug             TEXT         NOT NULL,
    label            TEXT         NOT NULL,
    description      TEXT,
    aliases          TEXT[]       NOT NULL DEFAULT '{}',
    parent_ids       UUID[]       NOT NULL DEFAULT '{}',
    tagging_count    INTEGER      NOT NULL DEFAULT 0 CHECK (tagging_count >= 0),
    bulk_updated_at  TIMESTAMPTZ  NOT NULL,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX oracle_tags_slug_idx ON oracle_tags (slug);
CREATE INDEX oracle_tags_parent_ids_idx ON oracle_tags USING GIN (parent_ids);

CREATE TABLE card_tags (
    tag_id           UUID         NOT NULL REFERENCES oracle_tags (id) ON DELETE CASCADE,
    oracle_id        UUID         NOT NULL,
    weight           TEXT         NOT NULL DEFAULT 'median',
    annotation       TEXT,
    PRIMARY KEY (tag_id, oracle_id)
);

CREATE INDEX card_tags_oracle_id_idx ON card_tags (oracle_id);

-- Transitive closure: (ancestor_id, descendant_id, depth). depth = 0 is
-- the reflexive row so a simple join on ancestor_id yields the tag's own
-- taggings plus everything beneath it.
CREATE VIEW oracle_tag_closure AS
WITH RECURSIVE walk AS (
    SELECT id AS ancestor_id, id AS descendant_id, 0 AS depth
    FROM oracle_tags
    UNION
    SELECT w.ancestor_id, t.id, w.depth + 1
    FROM walk w
    JOIN oracle_tags t ON w.descendant_id = ANY (t.parent_ids)
)
SELECT ancestor_id, descendant_id, depth FROM walk;

COMMENT ON TABLE  oracle_tags IS
    'Scryfall Tagger oracle (functional) tags from the official oracle_tags bulk file. Training signal for M6; not embedded, not filtered on.';
COMMENT ON COLUMN oracle_tags.id IS
    'Scryfall stable tag UUID. The only safe join key — slug and label are documented as mutable.';
COMMENT ON COLUMN oracle_tags.parent_ids IS
    'Direct parent tag ids (multi-parent DAG). Use oracle_tag_closure for transitive queries.';
COMMENT ON COLUMN oracle_tags.tagging_count IS
    'Number of DIRECT taggings in the bulk file. Descendant taggings are not included.';
COMMENT ON COLUMN oracle_tags.bulk_updated_at IS
    'Scryfall updated_at of the bulk file this row was loaded from — cite this date in the paper.';
COMMENT ON TABLE  card_tags IS
    'Direct (tag, card) taggings. oracle_id may reference cards outside the ingested corpus; join to cards at query time.';
COMMENT ON VIEW   oracle_tag_closure IS
    'Reflexive-transitive ancestor→descendant pairs. JOIN card_tags ON descendant_id to reproduce Scryfall otag: semantics (parent ∪ descendants).';
