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

## Checkpoint: M3 → M4

**Status:** Active.

### 1. Artifact review

Read or re-read, in this order:

- `docs/journal/2026-05-18-eval-set-construction.md` (methodology)
- `docs/journal/2026-05-18-baseline-results.md` (results + failure analysis)
- `data/eval/queries_v1_draft.yaml` — at minimum the schema and a few representative queries (q_014 flicker for the methodology-defining tri-state case; q_001 fliers for the sampling-bias example; q_020 red-creatures for the pure-structural case)
- `data/eval/review_batch_*.html` — skim all four; the visual review of the eval set
- `scripts/evaluate.py` and `src/eval/metrics.py` — the harness and the metric math
- `experiment_runs.id = 13` — the row this checkpoint's measurement came from. In particular: `SELECT per_query FROM experiment_runs WHERE id=13` and read the top-10 trace for q_014 (flicker), q_011 (tutor), q_001 (flying). The Class B asymmetry pattern is visible in this raw data.

### 2. Source vetting

- `data/eval/methodology_references.md` — the three papers backing the tri-state methodology:
  - **Järvelin & Kekäläinen 2002** — abstract (minimum) + Section 4 if time. Confirms graded relevance is established IR practice.
  - **Voorhees 2000** — abstract. Confirms single-curator judgements yield stable comparative rankings.
  - **Sormunen 2002** — abstract + Section 4 if time. Empirical evidence that ~50% of TREC's "relevant" pool is marginal.
- `docs/journal/2026-05-17-poc-retrospective.md` — re-read. The baseline reproduces the failure modes this entry predicted; confirm the alignment.
- For each source: does our application stand up? Is the citation defensible to a hostile committee? Should we keep, drop, or supplement?

### 3. Required subtasks (blocking M4 start)

- **Pre-M4 predictions journal entry** — `docs/journal/<date>-pre-m4-predictions.md`. For each of the 22 zero-scoring queries from the baseline, predict the M4 outcome:
  - `HyDE-fixable` — HyDE rewriting alone will lift this query
  - `SQL-fixable` — SQL pre-filter alone will resolve it
  - `needs-both` — only the combined pipeline will work
  - `probably-still-zero` — structural limitation neither tower addresses
  - Date the entry. Predictions become falsifiable claims when M4 measures against them.
- **Add HyDE primary source** to `data/eval/methodology_references.md`: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels" (already referenced in `CLAUDE.md` §2). At minimum the citation block; ideally the abstract summary.

### 4. Interactive review

The curator drives a walk-through with the LLM. Specific defenses the LLM will push on:

- Why per-face row schema instead of concatenation? (Schema entry's reasoning.)
- Why tri-state relevance instead of binary? Cite the supporting paper from memory.
- What's the reminder-text augmentation strategy? What can it not reach? (Card-side vs query-side asymmetry.)
- What's "query-document asymmetry" and how does the baseline confirm it? Name two concrete failure examples from the per_query trace.
- Why pgvector instead of FAISS-standalone? (CLAUDE.md §3 + the anti-suggestions list.)
- Why no ANN index on the embedding column? When would we add one?
- What does the SQL tower add that the embedding tower can't? Cite a specific Class C query and explain why it scored zero.

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
