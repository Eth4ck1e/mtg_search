# Phase 1 — Foundation & Logging Infrastructure

**Weeks:** 1–2
**Status:** Not started

## Goal

Stand up the environment, the database, and — most importantly — the logging spine **before any pipeline runs.** Logging infrastructure has to exist before the work it's measuring, or early data is lost permanently and the final report is built on guesses.

By end of phase: Postgres + pgvector is running locally, the project skeleton matches the repo layout in `CLAUDE.md` §9, `src/logging_utils.py` is in place, the `experiment_runs` schema exists, the journal directory is wired up with a working template, and the raw Scryfall bulk file is downloaded and characterized.

## Deliverables

- [ ] Python virtualenv + `requirements.txt` (initial set)
- [ ] Local Postgres 16 + `pgvector` extension running (Docker Compose)
- [ ] `src/logging_utils.py` — structured JSON-lines logger; pipeline-run context manager
- [ ] `experiment_runs` table created via migration in `src/db/migrations/`
- [ ] `docs/journal/TEMPLATE.md` and at least one real entry (kickoff)
- [ ] `data/raw/oracle-cards-YYYYMMDD.json` downloaded
- [ ] `docs/journal/<date>-corpus-survey.md` — corpus characterization writeup
- [ ] `.gitignore` excluding `data/raw/`, `logs/`, `venv/`, `__pycache__/`, `*.pyc`, `.env`

## Sub-tasks

### Environment
1. [ ] Create `requirements.txt` with initial pins (psycopg[binary], pgvector, sentence-transformers, torch CPU, pyyaml, tqdm, pydantic)
2. [ ] `python -m venv venv`; activate; install
3. [ ] Confirm Python version is 3.11+ (3.12 preferred). Document in journal if forced lower.

### Database
4. [ ] `docker-compose.yml` for Postgres 16 + pgvector image (`pgvector/pgvector:pg16`)
5. [ ] Create `mtg_search` database; enable `CREATE EXTENSION vector;`
6. [ ] Decide on connection management approach (psycopg connection pool vs. raw connections). Note decision in log.

### Logging spine — non-negotiable for this phase
7. [ ] `src/logging_utils.py` — module with:
   - `PipelineRun` context manager that opens a JSONL log file at `logs/<script_name>/<YYYY-MM-DD>.jsonl`, captures start time, script version (from git short SHA or fallback), and writes a closing entry with duration, input/output counts, and arbitrary metadata.
   - Helper for emitting per-record events without flooding (counters, sampled records).
8. [ ] `experiment_runs` table schema:
   - `id` serial PK
   - `created_at` timestamptz default now()
   - `eval_set_version` text
   - `config` jsonb — full configuration that produced this row (model name, prompt versions, preprocessing version, filter flags, etc.)
   - `metrics` jsonb — recall@1, recall@5, recall@10, MRR, latency_p50, latency_p95
   - `per_query` jsonb — array of {query, expected_ids, retrieved_ids, hit_rank, latency_ms}
   - `notes` text (optional human note)
9. [ ] Migration runner script (`scripts/migrate.py`) — applies SQL files in `src/db/migrations/` idempotently.
10. [ ] Verify by running a dummy "fake evaluation" that writes a synthetic row.

### Journal & roadmap
11. [ ] `docs/journal/TEMPLATE.md` — frontmatter, sections (context, what happened, decision, reasoning, alternatives, next).
12. [ ] First real journal entry: redesign kickoff — what the POC showed, why the new architecture, what changes from the old plan.

### Corpus survey
13. [ ] `scripts/download_scryfall.py` — fetch `oracle-cards` bulk. Logs file size, URL, checksum, fetch timestamp.
14. [ ] One-off analysis (Jupyter notebook or `scripts/survey_corpus.py`):
    - Total record count
    - Layout distribution (`normal`, `transform`, `modal_dfc`, `split`, `adventure`, `token`, `emblem`, `art_series`, `meld`, `flip`, `saga`, ...)
    - Layouts to filter out at ingest (tokens, emblems, art series, digital-only, silver-bordered, memorabilia) → produce concrete filter rule list
    - Multi-faced card count by layout (needed for face-row design in Phase 2)
    - Keyword frequency distribution (informs reminder-text dict work in Phase 2)
    - Oracle text length distribution (chars + tokens via the embedding model's tokenizer) — informs whether truncation is a concern
    - Cards with null/empty oracle_text (probably lands, tokens) — confirm filter logic
15. [ ] Commit results as `docs/journal/<date>-corpus-survey.md` with figures/tables.

## Decisions log

| Date | Decision | Reasoning | Alternatives considered |
|---|---|---|---|
| | | | |

## Open questions

- How to identify "script version" in logs reproducibly when working with uncommitted changes — git SHA + dirty flag? File hash?
- Should `experiment_runs` `per_query` be in the same row as JSONB, or a separate `experiment_query_results` table? JSONB is simpler for v1; flat table is easier to aggregate. Defer; revisit when first comparison report is written in Phase 5.
- Where does the LLM API key for HyDE (Phase 4) live? `.env` + `python-dotenv` is the obvious choice; confirm and add to `.gitignore` before any commits.

## Notes for final report

### Introduction / problem framing
- What the original POC actually did (DistilBERT on both queries and cards, no SQL filter, no query rewriting).
- The query-document asymmetry failure pattern with concrete examples from the POC era.
- Why this matters: it's a general retrieval problem, not MTG-specific — frames the contribution beyond the domain.

### Methodology — corpus
- Corpus survey numbers go straight into the paper's "Data" subsection. Total count, layout breakdown, filter rules with counts excluded by rule.
- Filter rule list with justification per rule (why exclude tokens, why exclude digital-only, etc.).
- Multi-faced card statistics — frames the schema decision in Phase 2.

### Methodology — infrastructure
- The logging-first decision itself is worth a paragraph in methodology. It's not standard practice in student projects and it's defensible: every reported number traces back to a logged row.

## Journal entries

- (none yet — add links as written)
