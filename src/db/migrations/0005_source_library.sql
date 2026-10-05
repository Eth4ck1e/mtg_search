-- Migration 0005 — source library (the paper's own retrieval index).
--
-- Purpose: citations and quotations in the paper must come from the source
-- PDFs, never from a model's memory. These tables hold the extracted text of
-- every PDF in docs/sources/ so that (a) passages can be found by meaning or
-- by exact wording, and (b) any quotation can be VERIFIED verbatim against
-- the page it claims to come from (scripts/source_search.py --verify).
--
-- source_pages  — full extracted text per page; the ground truth for
--                 verification. One row per (file, page).
-- source_chunks — paragraph-sized passages with an embedding and a full-text
--                 index, for retrieval. page is where the passage STARTS.
--
-- source_key is the BibTeX key in docs/thesis/references.bib, so a retrieved
-- passage maps directly to a citable entry. Full replace on re-index (derived
-- data). Embedded with the stock Nomic model: this is general academic prose,
-- not card text, so the MTG-tuned checkpoint is the wrong encoder.

CREATE TABLE source_pages (
    file        TEXT    NOT NULL,
    page        INTEGER NOT NULL CHECK (page >= 1),
    source_key  TEXT    NOT NULL,
    text        TEXT    NOT NULL,
    PRIMARY KEY (file, page)
);

CREATE TABLE source_chunks (
    id                BIGSERIAL PRIMARY KEY,
    file              TEXT        NOT NULL,
    page              INTEGER     NOT NULL,
    chunk_index       INTEGER     NOT NULL,
    source_key        TEXT        NOT NULL,
    text              TEXT        NOT NULL,
    embedding         vector(768) NOT NULL,
    embedding_version TEXT        NOT NULL,
    text_tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (file, page, chunk_index)
);

CREATE INDEX source_chunks_key_idx ON source_chunks (source_key);
CREATE INDEX source_chunks_tsv_idx ON source_chunks USING GIN (text_tsv);

COMMENT ON TABLE source_pages  IS 'Extracted text of each source PDF page. Ground truth for quote verification.';
COMMENT ON TABLE source_chunks IS 'Retrieval passages over the source PDFs (semantic + full-text). source_key = BibTeX key.';
