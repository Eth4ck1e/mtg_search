# Milestone checkpoint framework

This project is structured as **seven milestones (M0–M7)** rather than calendar weeks. Each milestone produces artifacts (code + journal entries + experiment_runs rows + measured numbers). Between each pair of milestones is a **checkpoint** the curator (Mitchell) must clear before the next milestone's work begins.

The point of the checkpoint is not gating for its own sake — it's to ensure the curator can defend the work at the end of the project. Artifact velocity from LLM assistance is high enough that without explicit checkpoints we would finish in a few weeks without the curator having internalised anything. The checkpoint is the mechanism that makes calendar-pace match comprehension-pace.

## Checkpoint structure

Every Mn → Mn+1 transition has four sections:

1. **Artifact review** — journal entries, code, data changes, and experiment_runs rows produced during the milestone. The curator reads or re-reads everything substantive.
2. **Source vetting** — external sources cited during the milestone are read at least at the abstract level. For each source: confirm the application of the source's principle is correct, decide whether to keep citing it in the final paper, note any caveats.
3. **Required subtasks** — concrete deliverables the curator produces that gate progression. Usually a prediction or reflection journal entry. These are committed before next-milestone work begins.
4. **Interactive review** — a Socratic conversation between the curator and the LLM. The curator walks through the 5 W's (Who / What / When / Where / Why) of the milestone. The LLM pushes on weak spots, asks for defenses of specific decisions, surfaces gaps. **The interactive review is process, not record — it does not go in the journal.** Both parties must agree the curator can defend the work before proceeding.

## Milestone map and status

| ID | Milestone | Status |
|---|---|---|
| M0 | Project skeleton, POC archived, CLAUDE.md installed | ✓ Complete |
| M1 | DB + logging + corpus characterized | ✓ Complete |
| M2 | Corpus ingested + preprocessing pipeline ready | ✓ Complete |
| M3 | First baseline measured (experiment_runs id=13) | ✓ Complete |
| M4 | HyDE + SQL pre-filter | Pending — gated by M3→M4 checkpoint |
| M5 | Systematic evaluation + report generation | Pending |
| M6 | Evidence-driven optimisations | Pending |
| M7 | Final paper + presentation | Pending |

## Checkpoint: M3 → M4 (backfills M0 / M1 / M2)

**Status:** Active.

**Special note:** The checkpoint framework was introduced *at* M3 — milestones M0, M1, and M2 produced artifacts that were never formally reviewed under this rubric. This checkpoint therefore serves as a **one-time backfill**: a comprehensive review of everything from project kickoff through the first measured baseline. Subsequent checkpoints (M4 → M5 onward) cover only the milestone just completed.

### 1. Artifact review

Group by milestone. Read in roughly this order within each group; the order across milestones can be top-down architecture-first or chronological — curator's choice.

**M0 — Project kickoff and architecture:**

- `CLAUDE.md` — the architecture spec and working conventions. **The single most important document in the repo.** Re-read it; every architectural decision downstream is grounded here.
- `docs/journal/2026-05-17-redesign-kickoff.md` — what was done in the repo cleanup and why the POC was archived
- `docs/journal/2026-05-17-poc-retrospective.md` — the failure modes of the prior approach (DistilBERT-without-rewriting, FAISS-standalone) and why each was discarded. M3's baseline empirically reproduces these failures — confirm the alignment.
- `docs/archive/2025-11-03-original-proposal.md` — the original proposal as a reference point. Note the gap between proposed (FAISS, fine-tuning-first, single-tower) and final design (pgvector, evaluation-first, three-tower).
- `docs/roadmap/phase-0-overview.md` and the per-phase roadmap files — confirm understanding of the phase-by-phase deliverable structure (which has now been replaced by this milestone framework, but the phase docs remain the source of truth for sub-task lists).

**M1 — Foundation, logging, corpus characterised:**

- `docs/roadmap/phase-1-foundation-and-logging.md` — read the deliverable list and confirm each was met
- `docs/journal/2026-05-17-corpus-survey.md` — the 37,442-record corpus characterisation and the four filter rules it produced. Quote-ready material for the paper's Data subsection.
- `docker-compose.yml` — local Postgres + pgvector setup. Understand the env-var injection pattern (post-security fix).
- `src/db/migrations/0001_initial.sql` — vector extension + experiment_runs schema. Note the JSONB design choice for config/metrics/per_query.
- `src/db/experiment_log.py` — the writer that lands one row per evaluation in experiment_runs
- `src/logging_utils.py` — `PipelineRun` context manager (JSONL audit trail in `logs/<script>/<date>.jsonl`)
- `scripts/migrate.py`, `scripts/download_scryfall.py`, `scripts/survey_corpus.py` — the M1 operational tooling
- `src/data_processing/scryfall_classify.py` — the pure-predicate classifier used to apply the filter rules. Note the design choice to keep classification logic separate from the filter-rule call site.

**M2 — Schema, ingest, preprocessing pipeline:**

- `docs/roadmap/phase-2-ingestion-and-schema.md` — deliverable list
- `docs/journal/2026-05-17-cards-schema.md` — the per-face row decision (one row per face, PK `(oracle_id, face_index)`) and the alternatives rejected (concatenate-with-separator, front-face-only, separate `card_faces` JSONB column). The schema embeds the load-bearing `embedding_triple_paired` CHECK constraint.
- `docs/journal/2026-05-17-keyword-augmentation.md` — the three failure classes (cross-talk false positives, variable-instance wording, em-dash ability words) and the 100-entry manual overrides curation. The "net-positive repeatable" framing for card-draw engines came from the curator and is documented here.
- `src/db/migrations/0002_cards.sql` — the cards table schema. Confirm the per-face vs card-level column split and the embedding-triple invariant.
- `scripts/ingest.py` and `src/data_processing/ingest_transform.py` — the streaming UPSERT with `IS DISTINCT FROM` idempotency check.
- `scripts/build_keyword_dict.py` and `src/data_processing/keyword_extract.py` — corpus scan for reminder text. The Skyhunter Patrol joint-keyword case is the load-bearing test.
- `data/keywords/reminder_text.json` (auto-extracted, 205 entries) and `data/keywords/manual_overrides.json` (hand-curated, 100 entries). Skim both; note Mind Stone, Commander's Sphere, Kenrith's Transformation, Spelunking are in `rejected` per the engine criterion.
- `src/preprocess_text.py` — `build_embedding_text` is the function that decides what gets fed to the encoder. The inline-check (don't duplicate existing parenthetical reminders) is the load-bearing design choice.

**M3 — First baseline measured:**

- `docs/journal/2026-05-18-eval-set-construction.md` — methodology (tri-state, three testing purposes)
- `docs/journal/2026-05-18-baseline-results.md` — measured numbers + three-class failure analysis
- `data/eval/queries_v1_draft.yaml` — at minimum the schema and a few representative queries (q_014 flicker for the methodology-defining tri-state case; q_001 fliers for the sampling-bias example; q_020 red-creatures for the pure-structural case)
- `data/eval/review_batch_*.html` — skim all four; the visual review of the eval set
- `scripts/embed.py` — corpus embedding pipeline. The embedding-triple invariant is enforced on every UPDATE.
- `scripts/eval_lookup.py` and `scripts/render_review.py` — the curation tooling
- `scripts/evaluate.py` and `src/eval/metrics.py` — the harness and the metric math
- `configs/baseline.yaml` — the Phase 3 baseline configuration
- `experiment_runs.id = 13` — the row this milestone's measurement came from. `SELECT per_query FROM experiment_runs WHERE id=13` and read the top-10 trace for q_014 (flicker), q_011 (tutor), q_001 (flying). The Class B asymmetry pattern is visible in this raw data.

### 2. Source vetting

**Academic citations (paper-grade):**

- `data/eval/methodology_references.md` — three IR-evaluation papers backing the tri-state methodology:
  - **Järvelin & Kekäläinen 2002** — abstract (minimum) + Section 4 if time. Confirms graded relevance is established IR practice. Where do we cite it? Methodology — Evaluation Design.
  - **Voorhees 2000** — abstract. Confirms single-curator judgements yield stable comparative rankings. Where do we cite it? Defends our one-curator eval set.
  - **Sormunen 2002** — abstract + Section 4 if time. Empirical evidence that ~50% of TREC's "relevant" pool is marginal. Where do we cite it? Motivates the `borderline` bucket directly.
- **Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels"** — the HyDE paper (referenced in `CLAUDE.md` §2 but not yet vetted). Read abstract minimum; this is M4's primary citation.

**Internal sources (vet for accuracy, alignment, defensibility):**

- `docs/journal/2026-05-17-poc-retrospective.md` — re-read. The baseline reproduces the failures predicted here; confirm the alignment is real, not hand-waved.
- `CLAUDE.md` §3 (storage philosophy), §5 (keywords/reminder text), §6 (evaluation before optimization), §11 (anti-suggestions) — each shaped a downstream decision. Confirm each section's claims hold up against the artifacts we built.

**Community / non-academic sources (used during curation, vet for limitations):**

- EDHREC theme pages (Ramp, Reanimator) — cited as cross-validation source in the keyword-augmentation entry. Caveat: EDHREC conflates cards that *are* an archetype with cards commonly *played in* it. Confirm we understand the limitation.
- Star City Games / Sheldon Menery board wipe taxonomy, Draftsim card-list articles, MTG Wiki — referenced by the research agent that justified the q_008 board wipe scope expansion. Worth knowing exist; not paper citations.

For each source: does our application stand up? Is the citation defensible to a hostile committee? Should we keep, drop, or supplement?

### 3. Required subtasks (blocking M4 start)

- **Pre-M4 predictions journal entry** — `docs/journal/<date>-pre-m4-predictions.md`. For each of the 22 zero-scoring queries from the baseline, predict the M4 outcome:
  - `HyDE-fixable` — HyDE rewriting alone will lift this query
  - `SQL-fixable` — SQL pre-filter alone will resolve it
  - `needs-both` — only the combined pipeline will work
  - `probably-still-zero` — structural limitation neither tower addresses
  - Date the entry. Predictions become falsifiable claims when M4 measures against them.
- **Add HyDE primary source** to `data/eval/methodology_references.md`: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels" (already referenced in `CLAUDE.md` §2). At minimum the citation block; ideally the abstract summary.

### 4. Interactive review

The curator drives a walk-through with the LLM. Topics span M0–M3 because this is the backfill checkpoint. Specific defenses the LLM will push on, grouped by milestone:

**M0 architecture:**

- Why pgvector instead of FAISS-standalone? (`CLAUDE.md` §3, anti-suggestions §11.)
- Why three-tower instead of single-tower retrieval? What does each tower contribute that the others cannot?
- Why evaluate-first rather than fine-tune-first? (`CLAUDE.md` §6.)
- Why a PHP frontend assumption shaped the storage decision toward a single Postgres instance.

**M1 foundation:**

- Why are pipeline-run logs (JSONL) separate from evaluation logs (experiment_runs)? When does each get written?
- The corpus survey discovered that token-length p99 is 101 (max 359, threshold 512). What did this confirm? What would have changed if max had been 800?
- Why did the four filter rules end up where they did? What's the reasoning for *not* filtering empty `oracle_text`?

**M2 schema and preprocessing:**

- Why per-face row schema instead of concatenation? Name a specific card and the failure mode concatenation would produce.
- The `(embedding, embedding_version, embedding_text_hash)` paired-NULL CHECK constraint — what failure mode does it prevent? What would a bug look like without it?
- The corpus-driven keyword extraction had three empirically-observed failure classes (cross-talk, variable-instance, em-dash). Name an example of each. Why is the manual overrides file the right response (rather than a stricter regex)?
- What's the reminder-text augmentation strategy? What can it not reach? (Card-side vs query-side asymmetry — this is the direct precursor to M4's HyDE work.)
- Why was Sensei's Divining Top reclassified to borderline (not relevant) for "card draw engines"? Apply the framing to a different card you've never considered.

**M3 evaluation:**

- Why tri-state relevance instead of binary? Cite Sormunen 2002's empirical finding from memory.
- What's "query-document asymmetry" and how does the baseline confirm it? Name two concrete failure examples from the per_query trace (e.g., q_011 "tutor" returns Strixhaven Lessons-mechanic cards).
- The four queries that scored (q_007, q_017, q_018, q_024) — what do they have in common structurally? What does this predict about jargon vs natural-language queries in general?
- Why no ANN index on the embedding column? When would we add one?
- What does the SQL tower add that the embedding tower can't? Cite a specific Class C query and explain why it scored zero by design.
- The Sampling Bias caveat on broad queries — explain it. What's the v2 mitigation strategy and why is it deferred?

The review ends when both parties agree the curator can defend the work. If gaps surface, return to source material rather than papering over.

---

## Forward sketches (firm up as milestone artifacts emerge)

### Checkpoint: M4 → M5

- **Artifact review**: M4 journal entries (HyDE design + SQL pre-filter design + combined results). ≥4 new `experiment_runs` rows (baseline / HyDE-only / SQL-only / combined). Specific per-query traces — what did HyDE rewrite for q_014 "flicker effects"? Did the SQL extractor correctly route q_020 red-creatures?
- **Source vetting**: Gao et al. 2022 HyDE paper in depth (Sections 1–4 minimum). Anthropic API docs for the HyDE model in use (likely Claude Haiku). Any prompt-engineering references consulted during HyDE design.
- **Required subtasks**: **M4 retrospective journal entry** comparing pre-M4 predictions to actual results. For each prediction class (HyDE-fixable / SQL-fixable / needs-both / still-zero), how many were right? Where wrong, what does that say about our model of the system? **Pre-M5 hypotheses entry** — what per-category performance changes do we expect under each of the M5 ablations?
- **Interactive review**: defend the HyDE prompt design (every word in the prompt earns its place). Defend the SQL extractor's heuristics. Explain why HyDE works conceptually using Gao et al.'s framing.

### Checkpoint: M5 → M6

- **Artifact review**: `scripts/generate_report.py` and its outputs. All accumulated `experiment_runs` rows aggregated into paper-ready figures and tables.
- **Source vetting**: any additional IR or NLP papers cited for systematic-eval methodology. External baseline comparators (Scryfall keyword search? TF-IDF? Lexical BM25?) — do we need them and are they defensible?
- **Required subtasks**: **Targeted-optimisation plan journal entry** — based on M5's per-query data, identify the weakest categories or query types. Pre-register optimisation strategies (e.g., "try cross-encoder reranking for hard-jargon queries, expect +X% recall@10") so M6 is hypothesis-testing rather than flailing.
- **Interactive review**: defend the choice of optimisations and why other candidates were rejected.

### Checkpoint: M6 → M7

- **Artifact review**: M6 optimisation journal entries. Comparative `experiment_runs` rows. Updated paper outline.
- **Source vetting**: final pass on every citation in the project. Anything missing in the related-work or discussion sections?
- **Required subtasks**: **Paper outline journal entry**. Decide main body vs appendix. Identify any remaining experiments needed to close the argument.
- **Interactive review**: defend the overall research narrative. Walk the LLM through the paper's argument structure in your own words.

### M7 completion

- **Final deliverables**: paper PDF, presentation slides, clean and documented public repo.
- **Interactive review**: mock-presentation defense. LLM plays hostile committee. Repeat until comfortable.

---

## How to use this document

- The curator opens this file at the start of every milestone transition.
- Items are worked through in order: 1 → 2 → 3 → 4.
- A milestone transition does not happen until section 4 (interactive review) ends with mutual agreement.
- This document lives outside the journal. It is **process**, not **record**. Journal entries continue to capture the substantive what/why/decisions; this document captures the workflow rules.
- If a milestone produces unexpected artifacts (a new tool, a new data file), update the relevant section of this doc to add the artifact-review item before closing the checkpoint.
