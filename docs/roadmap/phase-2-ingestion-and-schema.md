# Phase 2 — Ingestion & Schema

**Weeks:** 3–4
**Status:** Not started
**Depends on:** Phase 1 deliverables (Postgres + pgvector running, `logging_utils.py`, raw Scryfall file on disk, corpus survey done)

## Goal

Turn the raw Scryfall bulk file into a queryable Postgres schema with one row per card face, a tested idempotent ingest script, and an auto-extracted keyword reminder-text dictionary ready for use by Phase 3's embedding pipeline. By end of phase the database has the full corpus loaded (minus filtered junk layouts), the multi-faced card model is settled, and the preprocessing function that produces "text to be embedded" exists and is unit-tested.

## Deliverables

- [ ] Schema migration in `src/db/migrations/0001_initial_schema.sql`
- [ ] `scripts/ingest.py` — Scryfall JSON → Postgres, idempotent
- [ ] `scripts/build_keyword_dict.py` — corpus scan for reminder text, outputs `data/keywords/reminder_text.json` + manual override file
- [ ] `src/preprocess_text.py` — produces the canonical "embedding text" for a card row, with a version string
- [ ] Tests in `tests/test_ingest.py` and `tests/test_preprocess.py` covering parser edge cases
- [ ] Journal entry on the multi-faced card decision (one row vs. concatenation vs. front-face-only)
- [ ] Journal entry on the keyword extraction approach and any keywords that fell back to manual overrides

## Sub-tasks

### Schema design
1. [ ] Decide on the exact column set. Real columns (must support efficient SQL pre-filter): `oracle_id`, `face_index`, `name`, `mana_cost`, `cmc`, `colors` (text[]), `color_identity` (text[]), `type_line`, `oracle_text`, `keywords` (text[]), `power`, `toughness`, `loyalty`, `layout`, `lang`, `released_at`.
2. [ ] JSONB columns: `legalities`, `card_faces` (only meaningful for `layout in ('transform', 'modal_dfc', 'split', 'adventure', 'flip', 'meld')` — see decision in sub-task 6), `raw` (full original Scryfall record, escape hatch).
3. [ ] Embedding columns: `embedding vector(768)`, `embedding_version text`, `embedding_text_hash text` (hash of the exact text fed to the encoder — lets us detect when preprocessing changes silently).
4. [ ] Indexes: btree on `oracle_id`, `(oracle_id, face_index)`; GIN on `colors`, `color_identity`, `keywords`; btree on `cmc`, `released_at`; pgvector HNSW or IVFFlat on `embedding` (decide based on Phase 5 measurements — start with no ANN index, exact search is fine at 30k).
5. [ ] Migration runner already exists from Phase 1; just add the new migration file.

### Multi-faced card decision
6. [ ] Read Scryfall's docs on `layout` values and `card_faces`. Confirm which layouts actually have multiple printed faces vs. which just have weird `card_faces` for other reasons.
7. [ ] **Decision:** one row per face with composite key `(oracle_id, face_index)`. Single-face cards have `face_index = 0`. Multi-face cards have `face_index = 0, 1, ...`. Dedupe on display by `oracle_id`.
8. [ ] Document the alternatives considered (concatenate faces with separator; front-face-only) in a journal entry, with reasoning.
9. [ ] Edge cases to cover in tests: standard transform (`Delver of Secrets`), modal DFC (`Valki, God of Lies`), split card (`Fire // Ice`), adventure (`Bonecrusher Giant`), flip card (`Akki Lavarunner`), meld card (`Bruna, the Fading Light`).

### Ingest script
10. [ ] `scripts/ingest.py` — streams the bulk JSON (do NOT load 500MB into memory), iterates records, applies the junk-layout filter (rules from Phase 1's corpus survey), explodes multi-face records into face rows, builds the row dict, batches inserts via `psycopg.copy()` or `INSERT ... ON CONFLICT`.
11. [ ] Idempotency via `INSERT ... ON CONFLICT (oracle_id, face_index) DO UPDATE SET ... WHERE excluded.<everything> IS DISTINCT FROM <existing>` so updates only fire when a card's data actually changed.
12. [ ] Pipeline logging via `PipelineRun` context manager: counts (read, parsed, filtered by each rule with rule label, written-as-new, written-as-update, errors), duration, input file checksum, schema version.
13. [ ] CLI: `python scripts/ingest.py --bulk data/raw/oracle-cards-YYYYMMDD.json [--dry-run]`. Dry-run mode parses but doesn't write.
14. [ ] Tests in `tests/test_ingest.py`: parser edge cases (the seven multi-face layouts above), CMC parsing for cards with X costs and Phyrexian mana, color extraction for hybrid/devoid cards, idempotency (run twice, second run produces zero update rows).

### Keyword reminder-text extraction
15. [ ] `scripts/build_keyword_dict.py` — scans `oracle_text` across all rows for patterns like `<KeywordName> (<parenthetical text>)`. Extracts (keyword, reminder_text) pairs, dedupes, prefers the most-recent printing's wording when there are variants.
16. [ ] Output: `data/keywords/reminder_text.json` — `{ "Flying": "This creature can't be blocked except by creatures with flying or reach.", ... }`. Versioned via a top-level `version` field and `generated_at`.
17. [ ] Output: `data/keywords/manual_overrides.json` — for keywords never printed with reminder text. Initially empty; add entries as Phase 3's evaluation surfaces gaps.
18. [ ] Log which keywords were extracted (count, first occurrence card), which had no reminder text in any printing (fall back to manual), and which had conflicting reminder texts across printings (decision logic chose newest).
19. [ ] Test: extract from a small synthetic dataset with known keywords and verify the dict comes out right.

### Preprocessing function
20. [ ] `src/preprocess_text.py:build_embedding_text(card_row, keyword_dict, version=...) -> str` returns the canonical text that will be fed to the encoder. Logic: start with `oracle_text`; for each keyword in `card_row.keywords` that doesn't already have an inline parenthetical, append the reminder text from `keyword_dict`.
21. [ ] Versioning: the `version` parameter is a string that gets stored in `embedding_text_hash`'s metadata. Bumping it (or changing the function's logic) invalidates downstream embeddings — Phase 3's `embed.py` picks up the change and re-embeds affected rows.
22. [ ] Unit tests: cards with no keywords (passthrough), cards with one keyword and no inline reminder (augmented), cards with inline reminder text already present (not duplicated), cards with a keyword that has no dict entry and no inline reminder (logged warning, passthrough).

## Decisions log

| Date | Decision | Reasoning | Alternatives considered |
|---|---|---|---|
| | | | |

## Open questions

- Should we promote `set_code` / `released_at` to real columns? Useful for the paper's discussion of terminology drift over time. Defer until needed.
- Where does the connection string live in dev vs. CI? Probably `DATABASE_URL` env var, with a `.env` file gitignored and a `.env.example` checked in.
- For the `keywords` column: Scryfall's `keywords` array uses canonical capitalization (`"Flying"`, `"First strike"`). Our reminder-text dict needs to match exactly. Document the case-sensitivity contract.
- `card_faces` JSONB: do we want to denormalize anything into the face row, or is the face row's own columns enough? Lean toward "face row's own columns enough" since we're splitting per face anyway; `card_faces` JSONB lives only on the `face_index = 0` row as a complete record.

## Notes for final report

### Methodology — corpus & schema
- Schema diagram (real columns vs. JSONB vs. embedding columns). Defensible: every real column is one that supports SQL pre-filter; everything else is JSONB.
- Filter rules table with counts excluded per rule (from Phase 1 survey) + the final row count after filtering.
- Multi-faced card statistics + the one-row-per-face decision and why.

### Methodology — embedding text construction
- The reminder-text augmentation algorithm, with a worked example: pick a card with a non-evergreen keyword, show before/after embedding text.
- The keyword-extraction approach as a self-maintaining alternative to hand-built dictionaries. Cite the POC's failure mode here.
- Versioning the preprocessing function with a hash, and the `embedding_version` column, are part of the reproducibility story.

### Discussion
- Note any keywords that required manual override and why. These are interesting cases (the paper can use them as examples of MTG's evolution).
- Any unexpected gotchas in multi-faced card handling.

## Journal entries

- (none yet)
