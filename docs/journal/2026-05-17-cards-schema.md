# 2026-05-17 — Cards schema and the per-face row decision

## Context

First substantive Phase 2 work: turning the per-record design that Phase 1's corpus survey ([entry](2026-05-17-corpus-survey.md)) implied into an actual Postgres schema. The single largest schema question — *how do we represent multi-faced cards?* — gets settled here, before `scripts/ingest.py` makes the choice concrete in code. The corpus survey counted 944 real multi-faced cards across seven layouts (transform 401, adventure 157, split 137, modal_dfc 96, prepare 47, flip 26, meld 21), so the decision has to handle them all.

## What happened

Inspected one record from each major layout against the bulk file to confirm field shapes:

- `oracle_id` is a UUID string (`'00037840-…'`) — use the PostgreSQL `UUID` type.
- `cmc` is a `Decimal` (Un-set half-mana costs exist) — `NUMERIC(4,2)`.
- `power`/`toughness`/`loyalty` are strings (`"*"`, `"X"`, `"*+1"` are valid values) — `TEXT`, not numeric.
- **Multi-faced cards expose fields differently than I expected**: per-face `cmc` is always missing — top-level `cmc` is the authoritative value. Per-face `mana_cost` carries the individual cost; top-level `mana_cost` is either `None` (transform) or the joined `"X // Y"` string (split). `keywords` and `color_identity` live only at the top level, never per-face.

That field-layering directly drove the column split in [`0002_cards.sql`](../../src/db/migrations/0002_cards.sql):

| Tier | Columns | Why |
|---|---|---|
| Per-face | `name`, `mana_cost`, `colors`, `type_line`, `oracle_text`, `power`, `toughness`, `loyalty` | Scryfall exposes these per face; they're the fields a semantic-search query about *one face* needs to match. |
| Card-level (duplicated across face rows) | `cmc`, `color_identity`, `keywords`, `layout`, `released_at`, `legalities`, `raw` | Scryfall does not expose these per face. Duplicating them across face rows costs ~944 extra rows of duplication out of ~30k — negligible. |
| Audit | `created_at`, `updated_at` | `updated_at` is set by ingest's UPSERT, not a trigger. |
| Embedding | `embedding vector(768)`, `embedding_version`, `embedding_text_hash` | Filled by Phase 3. The triple is paired by a CHECK — see below. |

Applied via `scripts/migrate.py`, verified the resulting schema with `\d cards` and a functional test of the embedding CHECK constraint (a partial UPDATE that sets only `embedding_version` raises `check_violation`).

## Decision

**One row per card face, primary key `(oracle_id, face_index)`.** Single-faced cards: `face_index = 0`. Multi-faced cards: `face_index = 0, 1, ...`. Dedupe by `oracle_id` at display time, not in the schema.

The schema imposes one load-bearing invariant via CHECK constraint: **`(embedding, embedding_version, embedding_text_hash)` are paired** — all three NULL or all three NOT NULL. The DB refuses any write that would put them out of sync.

## Reasoning

Per-face rows fall out of two independent requirements both pointing the same direction:

1. **Semantic search quality.** The two faces of a transform card are mechanically distinct objects. *Delver of Secrets* (a 1-mana blue sorcery-trigger creature) and *Insectile Aberration* (a 3/2 flyer) live in completely different semantic neighborhoods. Embedding their concatenated oracle text produces a centroid that matches *neither* face well — the worst kind of failure because it looks like search "works" until you measure recall. Per-face embeddings let each face be findable in its own neighborhood.
2. **SQL pre-filter precision.** Mana cost, power/toughness, type line, and `colors` are all per-face. The query "creatures with flying under 3 mana" needs to match on the active face's properties. Per-face rows make this `WHERE cmc < 3 AND 'Flying' = ANY(keywords)`. Combined-row schemes would force JSONB unpacking on every filter — fighting the SQL tower's design.

The cost — display-time deduplication — is an application-layer problem solved with `DISTINCT ON (oracle_id)` or a thin wrapper at the result-rendering layer. Cheap compared to the recall cost of the alternatives.

The embedding-triple CHECK constraint is the single most important schema-level invariant. Without it, `embed.py` could update `embedding_version` while leaving a stale `embedding` in place — and CLAUDE.md §3 explicitly calls out silent embedding-version drift as the failure mode the `embedding_version` column exists to prevent. Pushing the invariant into the schema means embed.py *cannot* drift, regardless of bugs in its update logic.

## Alternatives considered

- **Concatenate faces with `" // "` and embed once.** Rejected: produces an averaged vector that matches neither face. Concrete failure mode: searching "flying creature" against a Delver row would return Delver based on Insectile Aberration's text alone — but a search for "free counterspell" would also match Delver because of *Delver*'s front face. Cross-contamination across faces is a hard recall regression to debug after the fact.
- **Front-face only.** Rejected: werewolves and meld cards put their interesting abilities on the back. *Bruna, the Fading Light* alone is fine; *Brisela, Voice of Nightmares* (the meld result) is the actual reason anyone plays Bruna. Discarding back-face text is data loss in the format that matters most for the paper's "find me a card that does X" benchmark.
- **Separate `card_faces JSONB` column on every row.** Considered as a way to surface cross-face data alongside per-face rows. Rejected: redundant with the `raw JSONB` escape hatch, which already contains `card_faces`. Anyone needing all faces of a card reads `raw -> 'card_faces'` from any face row.
- **Asymmetric `card_faces JSONB` on `face_index = 0` only.** Rejected: every query that needs cross-face data has to remember `WHERE face_index = 0`, which is a subtle correctness footgun.
- **`lang` column on cards.** Defer. The `oracle_cards` bulk is English-only; adding a column that will always be `'en'` is noise. Adds in a future migration if Phase 6's evaluation experiments with non-English variants.
- **Vector ANN index now (HNSW / IVFFlat).** Defer. 30k × 768-float exact search is sub-100ms with pgvector. Building an HNSW index now would lock in parameters before Phase 5 measurements tell us what tuning matters. Easy to add later; expensive to swap.

## Notes for final report

- **Methodology — schema:** schema diagram (per-face / card-level / embedding tiers) goes into the methodology section. The argument that *every real column is one that supports SQL pre-filter, everything else is JSONB* is the defensible design principle.
- **Methodology — multi-face handling:** the per-face decision, with the three rejected alternatives and concrete failure modes for each, is a strong methodology subsection. Worked example: pick a transform card from the corpus and show what its two rows look like.
- **Methodology — reproducibility:** the embedding-triple CHECK is the kind of structural invariant that's worth a sentence. *"We enforce at the schema level that the embedding vector, the version string identifying what produced it, and the hash of the text that was fed to the encoder are populated as a unit — the database refuses partial updates."* Cite [`src/db/migrations/0002_cards.sql:62`](../../src/db/migrations/0002_cards.sql).
- **Discussion — what we did NOT do:** the deferred decisions (no `lang`, no ANN index, no `card_faces` column) belong in the discussion as evidence that schema scope was disciplined.

## Open follow-ups

- [ ] `scripts/ingest.py` (Phase 2 sub-task 10) — needs to know about the per-face decision when exploding multi-face records. The seven layouts that produce multi-face rows: `transform`, `modal_dfc`, `split`, `adventure`, `flip`, `meld`, `prepare`. Adventure cards are a quirk — the "adventure" half is sometimes treated as a separate spell, sometimes as a mode. Investigate before deciding adventure's face count.
- [ ] Confirm `oracle_id` is present on every record kept after the Phase 1 filters. Sub-task carried over from the corpus-survey journal entry. Cheap to add a counter to `scripts/survey_corpus.py`.
- [ ] First ingest run will produce the actual filtered row count. Compare against the corpus-survey estimate (~28,000–29,000 oracle records, ~29,000–30,000 face rows) — large divergence indicates a Scryfall data shift or a filter bug.
- [ ] Phase 5: revisit the vector-index decision once a baseline recall@K is measured. If exact search at 30k stays under target latency, no ANN index ever needs to land.

## Related

- [Phase 2 roadmap](../roadmap/phase-2-ingestion-and-schema.md) — sub-tasks 1–9 covered by this entry
- [Corpus survey](2026-05-17-corpus-survey.md) — the filter rules and multi-face counts feeding this design
- [`src/db/migrations/0002_cards.sql`](../../src/db/migrations/0002_cards.sql) — the schema itself
- [`CLAUDE.md`](../../CLAUDE.md) §3 — the storage philosophy this implements
