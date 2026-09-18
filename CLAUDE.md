# MTG Semantic Search — Project Guide

CSCI 5953 independent study, CSUSB. Author: Mitchell Trafford.
Repo: https://github.com/Eth4ck1e/mtg_search
Conference target: end of Fall 2026 or Spring 2027.

This document governs Claude Code's behavior in this project. It overrides defaults. It is the source of truth for architecture, working style, and conventions. The week-by-week schedule lives in `docs/roadmap/`; daily decisions and analysis live in `docs/journal/`.

---

## 1. Mission

Consumer catalogs — trading card games, e-commerce, media libraries, technical documentation — increasingly need natural-language search that works for average users, not just experts fluent in the catalog's structured query syntax. Traditional keyword and faceted-filter search excels when the user knows the domain's vocabulary, but excludes the majority who query informally. Naive dense retrieval fails on the same catalogs because short informal queries embed too far from formal domain text. This project uses Magic: The Gathering (~30k unique cards) as a test bed for a research question that generalizes across consumer catalog domains: **can natural-language semantic search match or beat the domain-standard search tool (Scryfall) for non-expert users, and does the pattern extend to similar catalog domains?**

The technical response is a **three-stage retrieval cascade** that addresses the **query–document asymmetry problem** identified by the original POC. Stage 1 is a HyDE-style query rewriter (local instruction-tuned LLM) that transforms the natural-language query into structured attributes plus a hypothetical card ability text. Stage 2 is a SQL pre-filter that narrows the candidate set on the extracted structured attributes. Stage 3 is semantic vector search over Oracle text embeddings inside the pre-filtered candidate set. The methodological response is a measurement discipline that quantifies per-component contribution through the –SQL ablation and comparison against Scryfall's expert-crafted queries. Real-world outcome — accessibility for non-experts — anchors the paper's argument; there is no internal-baseline diff.

## 2. Architecture

Three sequential stages in a retrieval cascade (Wang et al. 2011 tradition). Execution order matches numbering:

1. **HyDE query rewriter (Stage 1).** A local instruction-tuned LLM (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` — MLX-quantized for Apple Silicon inference via `mlx_lm.server`; final choice pending M4 candidate evaluation) transforms the user's natural-language query into a structured JSON output with two fields: (a) filter attributes for Stage 2, and (b) a hypothetical MTG card ability text for Stage 3. Served over an OpenAI-compatible HTTP endpoint so `query_rewriter.py` is backend-agnostic (swap MLX → vLLM/llama.cpp/TGI without touching Python). Reference: Gao et al. 2022, *"Precise Zero-Shot Dense Retrieval without Relevance Labels."* Preserved at [`docs/sources/2022_gao_hyde.pdf`](docs/sources/2022_gao_hyde.pdf).
2. **SQL pre-filter (Stage 2).** Structured attributes handed off from Stage 1 — color identity, mana value, type line, P/T, legality, and other categorical or numeric facts — narrow the candidate set. Postgres handles this natively; the pre-filter runs before any vector operation.
3. **Semantic vector search (Stage 3).** The Stage 1 hypothetical ability text is embedded with `nomic-ai/nomic-embed-text-v1.5` (137M-param bi-encoder, 768-dim output). ANN search runs **inside** the candidate set narrowed by Stage 2 — pre-filter, not post-filter on top-K. Post-filter on top-K collapses recall on constrained queries.

**Design invariant:** the semantic stage always executes over the SQL-narrowed candidate set. This is the load-bearing design decision separating this cascade from generic dense retrieval.

**Naming note:** the architecture was originally called "three-tower" internally (2026-05 documents); it was renamed to "three-stage retrieval cascade" on 2026-09-11 for terminology accuracy (see `docs/journal/2026-09-11-*.md`). The "tower" metaphor is standard in *parallel* dual-encoder ranking systems (Guo et al. 2016); a sequential filter→rewrite→rank pipeline is a cascade.

## 3. Storage

- **Single store:** Postgres + pgvector. ~30k cards is well inside pgvector's comfort zone; corpus scale, structured attribute filtering, and vector search all coexist in one database. No separate vector DB at this scale.
- **Schema philosophy:** real columns for well-defined attributes (`name`, `cmc`, `colors`, `type_line`, `oracle_text`, `power`, `toughness`, `keywords`, `layout`); JSONB for genuinely variable nested data (`legalities`, `card_faces`); `raw JSONB` as escape hatch for any Scryfall field not promoted to a column. Do not dump raw JSON into a single column.
- **`embedding vector(768)` column** with a paired **`embedding_version`** column. The version string identifies the model + text-representation combination that produced each vector. Without this, silent inconsistency creeps in the first time the embedding pipeline changes.
- **Multi-faced cards** (transform, modal DFC, split, adventure): one row per face, composite key `(oracle_id, face_index)`. Dedupe by `oracle_id` at display time.

## 4. Ingestion

Two stages, two scripts:

- **Parse stage** (`scripts/ingest.py`): Scryfall `oracle-cards` bulk file (gzipped JSON-Lines, `.jsonl.gz`) → Postgres rows. No ML, no GPU. **Idempotent** via `INSERT ... ON CONFLICT (oracle_id, face_index) DO UPDATE`. Corpus filters exclude tokens, emblems, art series, digital-only, silver-bordered, memorabilia, Un-set (`set_type=funny`) novelty cards, and token-only booster products (`set_type=token`). Un-sets and token-set products were added to the filter list on 2026-09-11.
- **Embed stage** (`scripts/embed.py`): reads rows where `embedding IS NULL OR embedding_version != $current`, applies the Nomic Embed document prefix, runs the model, writes back. Slow, expensive, frequently re-run.

The Scryfall bulk-data endpoint (2026-09-11 API contract) returns metadata at `https://api.scryfall.com/bulk-data/oracle-cards` with fields `jsonl_download_uri` and `compressed_size`. The download itself is `.jsonl.gz` (gzipped JSON-Lines; one card per line). Downstream consumers must decompress + line-parse rather than treat the file as a single JSON array.

## 5. Embedding Text & Keywords

- Embed Oracle text only — not type line, not mana cost, not color. Those are SQL territory.
- Before embedding, augment Oracle text with **auto-extracted reminder text** for any keyword the card has but doesn't already explain inline.
- The reminder-text dictionary is built once by scanning the whole Scryfall corpus for parenthetical patterns. Wizards inconsistently prints reminder text — that's the lever: somewhere across all printings of every keyword, the canonical Wizards-written definition exists. The dictionary harvests it. New keywords in future sets get picked up automatically on re-ingestion.
- A small manual override file handles the few keywords that have never received reminder text in any printing.
- **Do not** hand-build a keyword definition dictionary. **Do not** attempt to fine-tune the embedding model to understand keywords via custom definitions. Reminder-text augmentation puts the canonical text into what gets embedded — that's what actually moves vectors.
- **Nomic Embed prefix requirement:** `nomic-ai/nomic-embed-text-v1.5` requires task-specific prefixes on inputs — `search_document: ` for the corpus side, `search_query: ` for the query side. Applied at encode time via helpers in `src/preprocess_text.py`. Omitting the prefixes measurably degrades retrieval quality.

## 6. Evaluation Before Optimization

Before fine-tuning, before HyDE prompt tuning, before any other optimization: a hand-curated evaluation set of ~26 queries with tri-state relevance judgments (relevant / partially relevant / not relevant) is the harness. Every change after it is measured against it. The current set (`data/eval/queries_v1_draft.yaml`) spans six query categories: natural language, jargon, fragmented, hybrid, constrained, mechanical.

Fine-tuning the embedding model on synthetic (query, card) training pairs is on the roadmap but **deferred** — it is only pursued if M4/M5 measurements show HyDE + SQL pre-filter alone is insufficient. Do not preemptively suggest fine-tuning. It is the last optimization, not the foundation.

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

The project is structured as **seven milestones (M0–M7)** rather than calendar weeks. The original 14-week schedule (in the roadmap files below) assumed a student writing every line by hand; LLM-assisted artifact production runs roughly 6× faster, so the calendar pace and the comprehension pace would diverge without explicit checkpoints. Each milestone-transition is gated by a **checkpoint** the curator must clear before the next milestone begins — see [`docs/process/milestone-checkpoints.md`](docs/process/milestone-checkpoints.md) for the framework.

**Milestone status (updated 2026-09-11 after the baseline-abandonment pivot):**

| ID | Milestone | Roadmap mapping | Status |
|---|---|---|---|
| M0 | Project skeleton, POC archived, CLAUDE.md installed | Phase 0 + pre-Phase-1 cleanup | ✓ Complete |
| M1 | DB + logging + corpus characterized | [Phase 1](docs/roadmap/phase-1-foundation-and-logging.md) | ✓ Complete |
| M2 | Corpus ingested + preprocessing pipeline | [Phase 2](docs/roadmap/phase-2-ingestion-and-schema.md) | ✓ Complete (2026-09-11 rebuild with Nomic Embed v1.5) |
| M3 | First baseline measured | [Phase 3](docs/roadmap/phase-3-baseline-and-eval.md) | ⚠️ Superseded — baseline abandoned 2026-09-11; the previously claimed `experiment_runs.id=13` measurements were confirmed fabricated. Paper reframed from diff-vs-baseline to outcome-vs-Scryfall. See `docs/journal/2026-09-11-pivot-baseline-abandonment-and-encoder-switch.md`. |
| M4 | HyDE + SQL pre-filter | [Phase 4](docs/roadmap/phase-4-hyde-and-prefilter.md) | In progress |
| M5 | Systematic evaluation + report generation | [Phase 5](docs/roadmap/phase-5-systematic-eval.md) | Pending |
| M6 | Evidence-driven optimisations (fine-tuning candidate) | [Phase 6](docs/roadmap/phase-6-optimization.md) | Pending; deferrable |
| M7 | Final paper + presentation | [Phase 7](docs/roadmap/phase-7-finalization.md) | Pending |

The roadmap files remain the source of truth for per-phase sub-task lists and "Notes for final report" sections. They no longer drive the schedule. **Treat deliverables and logging discipline as the contract; week numbers in the roadmap are historical context only.** The `docs/process/timeline.md` document is the current source of truth for the Fall 2026 semester schedule.

## 9. Repo Layout

```
mtg_search/
├── CLAUDE.md                            # This file — architecture + working conventions
├── README.md                            # Public overview + quickstart
├── pyproject.toml                       # Pinned deps; Python >= 3.11
├── docker-compose.yml                   # pgvector/pgvector:pg16 on localhost:5432
├── .env / .env.example                  # Postgres credentials (.env gitignored)
├── archive/poc_v1/                      # POC snapshot, preserved
├── _planning-archive/                   # Pre-repo local planning docs (historical)
├── configs/
│   └── baseline.yaml                    # Retrieval-run configuration
├── data/
│   ├── raw/                             # Scryfall bulk .jsonl.gz (gitignored, large)
│   ├── processed/                       # Corpus survey JSON (gitignored, regenerable)
│   ├── eval/                            # Hand-curated eval set + tooling outputs
│   │   ├── queries_v1_draft.yaml        # 26 queries with tri-state relevance
│   │   ├── methodology_references.md    # IR-eval papers backing tri-state
│   │   └── review_batch_*.html          # Visual review (gitignored, regenerable)
│   └── keywords/                        # Reminder-text dict + manual overrides
├── docs/
│   ├── archive/                         # Original proposal + historical planning
│   ├── journal/                         # Dated decision/analysis entries
│   ├── process/                         # Workflow rulebooks + timeline
│   ├── roadmap/                         # Phase files (M0–M7 mapping in §8 above)
│   ├── sources/                         # Academic source PDFs (gitignored) + bibliography
│   └── thesis/                          # Thesis-class deliverables (abstracts, timeline)
├── scripts/                             # Entry-point scripts
│   ├── migrate.py                       # SQL migration runner
│   ├── download_scryfall.py             # Scryfall bulk .jsonl.gz fetch
│   ├── survey_corpus.py                 # Corpus characterisation
│   ├── ingest.py                        # Bulk → cards table UPSERT
│   ├── build_keyword_dict.py            # Reminder-text extraction
│   ├── embed.py                         # Corpus embedding pipeline (Nomic Embed v1.5)
│   ├── eval_lookup.py                   # Scryfall candidate finder
│   ├── render_review.py                 # Eval-set HTML reviewer
│   ├── evaluate.py                      # Run a config, write experiment_runs row
│   └── test_search.py                   # Ad-hoc naive-dense-retrieval CLI
├── src/
│   ├── config.py                        # Pydantic Settings (single source of truth)
│   ├── logging_utils.py                 # PipelineRun JSONL context manager
│   ├── preprocess_text.py               # build_embedding_text + Nomic prefix helpers
│   ├── data_processing/                 # scryfall_classify, ingest_transform, keyword_extract
│   ├── db/                              # experiment_log writer + SQL migrations
│   ├── eval/                            # Pure-Python metric calculation
│   └── utils/                           # device selection, warning suppression
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
- Fine-tuning the embedding model before HyDE + SQL pre-filter measurements are in and the Scryfall comparator is run. Fine-tuning is M6 optional work only if evidence demands it.
- Dumping raw Scryfall JSON into a single JSONB column. Parse properly; `raw` is escape hatch only.
- Restructuring the repo layout without a reason.
- Citing the fabricated `experiment_runs.id=13` numbers or the `docs/journal/2026-05-18-baseline-results.md` figures as if they were real measurements. They are not; the entire M3 baseline was superseded on 2026-09-11.
- The old "three-tower" architecture name. It is a **three-stage retrieval cascade**.

## 12. Reports & The Final Paper

The conference presentation is end of Fall 2026 or Spring 2027. The final paper draws from:

- `docs/journal/` → methodology section, design narrative
- `experiment_runs` table → results section, comparison tables, ablation tables
- `docs/roadmap/*.md` "Notes for final report" sections → structured arguments per phase
- `docs/sources/` → academic source PDFs with an annotated bibliography
- `scripts/generate_report.py` → automated tables/figures from logged data

**`scripts/generate_report.py` is a deliverable, not an afterthought.** It must exist by M5 and run cleanly by M7.

## 13. Quick Reference

```bash
# Activate environment
source .venv/bin/activate            # macOS/Linux (.venv, not venv)

# Note: Python 3.13.0 has a .pth-file processing bug that breaks editable-install
# imports. Prefix scripts with PYTHONPATH="$PWD" as a workaround; upgrading to a
# patched Python 3.13.x removes this need.

# Bring up pgvector (once per session; runs on localhost:5432)
docker compose up -d

# Migrations (idempotent — safe to re-run)
PYTHONPATH="$PWD" python scripts/migrate.py

# Ingestion pipeline
PYTHONPATH="$PWD" python scripts/download_scryfall.py   # oracle-cards .jsonl.gz → data/raw/
PYTHONPATH="$PWD" python scripts/ingest.py              # → cards table UPSERT
PYTHONPATH="$PWD" python scripts/build_keyword_dict.py  # reminder-text dictionary
PYTHONPATH="$PWD" python scripts/embed.py               # missing/stale embeddings updated

# Start MLX HyDE server (Apple Silicon; leave running in a separate shell)
PYTHONPATH="$PWD" python -m mlx_lm server \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --port 8080 --log-level WARNING
# ~57 tok/sec generation, ~4.8GB peak on M3. OpenAI-compatible endpoint at
# http://localhost:8080/v1 — query_rewriter.py hits /v1/chat/completions.

# Ad-hoc query testing (naive dense retrieval only; no HyDE, no SQL pre-filter)
PYTHONPATH="$PWD" python scripts/test_search.py --dedupe "cheap red removal"

# Evaluation (writes to experiment_runs)
PYTHONPATH="$PWD" python scripts/evaluate.py --config configs/baseline.yaml

# Reporting (M5+)
PYTHONPATH="$PWD" python scripts/generate_report.py --since 2026-09-01 --out docs/reports/
```
