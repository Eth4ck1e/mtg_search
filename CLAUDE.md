# MTG Semantic Search — Project Guide

CSCI 5953 independent study, CSUSB. Author: Mitchell Trafford.
Repo: https://github.com/Eth4ck1e/mtg_search
Conference target: end of Fall 2026 or Spring 2027.

This document governs Claude Code's behavior in this project. It overrides defaults. It is the source of truth for architecture, working style, and conventions. The week-by-week schedule lives in `docs/roadmap/`; daily decisions and analysis live in `docs/journal/`.

---

## 1. Mission

Build a natural-language semantic search system for Magic: The Gathering cards (~30k unique) that solves the **query-document asymmetry problem** identified by the original POC: short informal user queries ("a blue counterspell that costs 2") embed too far from dense Oracle text vectors to produce useful matches.

The contribution of this project is a **three-tower retrieval architecture** that addresses that asymmetry, plus a measurement methodology that lets us defend the design with numbers.

## 2. Architecture

Three concerns, kept separate:

1. **Structured filter (SQL pre-filter).** Color identity, mana value, type line, P/T, legality — anything categorical or numeric. Exact-match facts. Postgres handles them.
2. **Query rewriting (HyDE).** A generative model transforms the user's NL query into a hypothetical card. The hypothetical's *ability text portion* is what gets embedded. The structured attributes it extracts become SQL predicates for tower 1. Reference: Gao et al. 2022, "Precise Zero-Shot Dense Retrieval without Relevance Labels."
3. **Semantic search.** `sentence-transformers/multi-qa-distilbert-cos-v1` over Oracle ability text. The hypothetical card text from tower 2 is embedded with the same model. ANN search runs **inside** the candidate set already filtered by tower 1.

**Order of operations:** SQL filter first, vector search within the filtered set. Not post-filter on top-K. Pre-filter is critical or recall collapses on constrained queries.

## 3. Storage

- **Single store:** Postgres + pgvector. ~30k cards is well inside pgvector's comfort zone, and the eventual PHP frontend talks to one database. No separate vector DB at this scale.
- **Schema philosophy:** real columns for well-defined attributes (`name`, `cmc`, `colors`, `type_line`, `oracle_text`, `power`, `toughness`, `keywords`, `layout`); JSONB for genuinely variable nested data (`legalities`, `card_faces`); `raw JSONB` as escape hatch for any Scryfall field not promoted to a column. Do not dump raw JSON into a single column.
- **`embedding vector(768)` column** with a paired **`embedding_version`** column. The version string identifies the model + text-representation combination that produced each vector. Without this, silent inconsistency creeps in the first time the embedding pipeline changes.
- **Multi-faced cards** (transform, modal DFC, split, adventure): one row per face, composite key `(oracle_id, face_index)`. Dedupe by `oracle_id` at display time.

## 4. Ingestion

Two stages, two scripts:

- **Parse stage** (`scripts/ingest.py`): Scryfall `oracle-cards` bulk JSON → Postgres rows. No ML, no GPU. **Idempotent** via `INSERT ... ON CONFLICT (oracle_id, face_index) DO UPDATE`. Filters out tokens, emblems, art series, digital-only, silver-bordered, memorabilia at this stage.
- **Embed stage** (`scripts/embed.py`): reads rows where `embedding IS NULL OR embedding_version != $current`, runs the model, writes back. Slow, expensive, frequently re-run.

## 5. Embedding Text & Keywords

- Embed Oracle text only — not type line, not mana cost, not color. Those are SQL territory.
- Before embedding, augment Oracle text with **auto-extracted reminder text** for any keyword the card has but doesn't already explain inline.
- The reminder-text dictionary is built once by scanning the whole Scryfall corpus for parenthetical patterns. Wizards inconsistently prints reminder text — that's the lever: somewhere across all printings of every keyword, the canonical Wizards-written definition exists. The dictionary harvests it. New keywords in future sets get picked up automatically on re-ingestion.
- A small manual override file handles the few keywords that have never received reminder text in any printing.
- **Do not** hand-build a keyword definition dictionary. **Do not** "train DistilBERT to understand keywords" via custom definitions. Reminder-text augmentation puts the canonical text into what gets embedded — that's what actually moves vectors.

## 6. Evaluation Before Optimization

Before fine-tuning, before HyDE prompt tuning, before any other optimization: build a hand-curated evaluation set of **30–50 (NL query, list-of-relevant-oracle-ids) pairs**. (The list is plural — most queries have multiple relevant cards.) This is the harness; every change after it is measured against it.

Fine-tuning with synthetic query/card pairs (`MultipleNegativesRankingLoss` on top of `multi-qa-distilbert-cos-v1`) is on the roadmap but **deferred** until baseline + reminder-text augmentation + HyDE numbers are measured. Do not preemptively suggest fine-tuning. It is the last optimization, not the foundation.

## 7. Logging — First-Class Concern

The final paper and the conference presentation are built from logs. If we don't log structurally from day one, week 14 becomes archaeology. Three categories:

| Category | Where it lives | Format |
|---|---|---|
| Pipeline runs (ingest, embed, parse, keyword-extract) | `logs/<script_name>/<YYYY-MM-DD>.jsonl` | One JSON object per line: timestamp, script version, input count, output count, count skipped per rule, duration, any version strings (model, prompt, preprocessing). |
| Evaluation runs | `experiment_runs` table in Postgres | One row per (configuration, eval-set version): config JSON, recall@K, MRR, per-query results JSON, timestamp. |
| Human decisions and analysis | `docs/journal/<YYYY-MM-DD>-<topic>.md` | Markdown. Source material for the paper. |

**Rule:** every script that mutates state writes a pipeline-run log entry. Every evaluation invocation writes an `experiment_runs` row. Every non-trivial design decision gets a journal entry. Journals are committed to the repo — they are part of the project record.

The shape consistency across all logs is what makes week 14's `scripts/generate_report.py` a one-script job instead of a week of cleanup.

## 8. Working Timeline

The project is structured as **seven milestones (M0–M7)** rather than calendar weeks. The original 14-week schedule (in the roadmap files below) assumed a student writing every line by hand; LLM-assisted artifact production runs roughly 6× faster, so the calendar pace and the comprehension pace would diverge without explicit checkpoints. Each milestone-transition is gated by a **checkpoint** the curator must clear before the next milestone begins — see [`docs/process/milestone-checkpoints.md`](docs/process/milestone-checkpoints.md) for the framework and the current open checkpoint specification.

**Milestone status (as of M3 complete, awaiting M3 → M4 review):**

| ID | Milestone | Roadmap mapping | Status |
|---|---|---|---|
| M0 | Project skeleton, POC archived, CLAUDE.md installed | Phase 0 + pre-Phase-1 cleanup | ✓ Complete |
| M1 | DB + logging + corpus characterized | [Phase 1](docs/roadmap/phase-1-foundation-and-logging.md) | ✓ Complete |
| M2 | Corpus ingested + preprocessing pipeline | [Phase 2](docs/roadmap/phase-2-ingestion-and-schema.md) | ✓ Complete |
| M3 | First baseline measured (`experiment_runs.id=13`) | [Phase 3](docs/roadmap/phase-3-baseline-and-eval.md) | ✓ Complete |
| M4 | HyDE + SQL pre-filter | [Phase 4](docs/roadmap/phase-4-hyde-and-prefilter.md) | Gated by M3 → M4 checkpoint |
| M5 | Systematic evaluation + report generation | [Phase 5](docs/roadmap/phase-5-systematic-eval.md) | Pending |
| M6 | Evidence-driven optimisations | [Phase 6](docs/roadmap/phase-6-optimization.md) | Pending |
| M7 | Final paper + presentation | [Phase 7](docs/roadmap/phase-7-finalization.md) | Pending |

The roadmap files remain the source of truth for per-phase sub-task lists and "Notes for final report" sections. They no longer drive the schedule. **Treat deliverables and logging discipline as the contract; week numbers in the roadmap are historical context only.**

## 9. Repo Layout

```
mtg_search/
├── CLAUDE.md                            # This file — architecture + working conventions
├── pyproject.toml                       # Pinned deps; Python >= 3.11
├── docker-compose.yml                   # pgvector/pgvector:pg16 on localhost:5432
├── .env / .env.example                  # Postgres credentials (.env gitignored)
├── archive/poc_v1/                      # POC snapshot, preserved
├── configs/
│   └── baseline.yaml                    # Phase 3 baseline retrieval config
├── data/
│   ├── raw/                             # Scryfall bulk JSON (gitignored, large)
│   ├── processed/                       # Corpus survey JSON (gitignored, regenerable)
│   ├── eval/                            # Hand-curated eval set + tooling outputs
│   │   ├── queries_v1_draft.yaml        # 26 queries with tri-state relevance
│   │   ├── methodology_references.md    # The 3 IR-eval papers backing tri-state
│   │   └── review_batch_*.html          # Visual review (gitignored, regenerable)
│   └── keywords/                        # Reminder-text dict + manual overrides
├── docs/
│   ├── archive/                         # Original proposal + similar historical
│   ├── journal/                         # Dated decision/analysis entries
│   ├── process/                         # Workflow rulebooks (milestone-checkpoints.md)
│   └── roadmap/                         # Phase files (M0–M7 mapping in §8 above)
├── scripts/                             # Entry-point scripts
│   ├── migrate.py                       # SQL migration runner
│   ├── download_scryfall.py             # Streamed bulk JSON fetch
│   ├── survey_corpus.py                 # Corpus characterisation
│   ├── ingest.py                        # Bulk → cards table UPSERT
│   ├── build_keyword_dict.py            # Reminder-text extraction
│   ├── embed.py                         # Corpus embedding pipeline
│   ├── eval_lookup.py                   # Scryfall candidate finder
│   ├── render_review.py                 # Eval-set HTML reviewer
│   └── evaluate.py                      # Run a config, write experiment_runs row
├── src/
│   ├── config.py                        # Pydantic Settings (single source of truth)
│   ├── logging_utils.py                 # PipelineRun JSONL context manager
│   ├── preprocess_text.py               # build_embedding_text + load_keyword_dict
│   ├── data_processing/                 # scryfall_classify, ingest_transform, keyword_extract
│   ├── db/                              # experiment_log writer + SQL migrations
│   ├── eval/                            # Pure-Python metric calculation
│   └── utils/                           # select_device etc.
├── tests/                               # ~80 tests, mix of unit + integration
└── logs/                                # JSONL pipeline-run logs (gitignored)
```

## 10. Working Style

**Collaboration is mentor / pair-programmer, not hand-holder.** Mitchell has prior dev experience and is leveling up toward senior. Skip the patronizing teaching tone.

- **Push back when a worse choice is about to be made.** Silence on a bad approach is a disservice. Argue with reasoning, not deference.
- **Lead with reasoning on design calls.** "Here's why, here's the trade-off, here's the code" — not just the code.
- **Follow Python best practices:** type hints where they earn their keep, docstrings on public functions, sensible module structure, tests where they catch real bugs (no test theater).
- **Work within the existing repo structure.** Don't restructure without a reason.
- **Cite industry practice** for style/structure questions.

## 11. Anti-Suggestions

Do not propose:

- A separate vector DB (Qdrant, Weaviate, Pinecone, FAISS-standalone) at this scale. pgvector is the answer.
- Post-filter on top-K vector results when pre-filter is what's needed.
- Embedding mana cost, color, CMC, or type line into the text representation. Those are SQL fields.
- Hand-maintained keyword definition dictionaries.
- Fine-tuning before a baseline is measured.
- Dumping raw Scryfall JSON into a single JSONB column. Parse properly; `raw` is escape hatch only.
- Restructuring the repo layout without a reason.

## 12. Reports & The Final Paper

The conference presentation is end of Fall 2026 or Spring 2027. The final paper draws from:

- `docs/journal/` → methodology section, design narrative
- `experiment_runs` table → results section, comparison tables, ablation tables
- `docs/roadmap/*.md` "Notes for final report" sections → structured arguments per phase
- `scripts/generate_report.py` → automated tables/figures from logged data

**`scripts/generate_report.py` is a deliverable, not an afterthought.** It must exist by Phase 5 (week 9–10) and run cleanly by Phase 7.

## 13. Quick Reference

```bash
# Activate environment
source venv/bin/activate            # macOS/Linux
.\venv\Scripts\Activate.ps1         # Windows / PowerShell

# Pipeline
python scripts/ingest.py            # Scryfall JSON -> Postgres
python scripts/build_keyword_dict.py
python scripts/embed.py             # missing/stale embeddings -> updated
python scripts/evaluate.py --config configs/baseline.yaml

# Reporting
python scripts/generate_report.py --since 2026-05-01 --out docs/reports/
```
