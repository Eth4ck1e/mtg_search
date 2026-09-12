# Fall 2026 Project Timeline

**Author:** Mitchell Trafford
**Project:** MTG Semantic Search — natural-language retrieval over Magic: The Gathering cards
**Term:** Fall 2026 (weeks of Aug 24 – Dec 13, 2026; last coursework week Nov 30 – Dec 6; finals Dec 7 – Dec 13)
**Drafted:** 2026-08-29 · **Last revised:** 2026-08-29

This document is the single source of truth for the project's Fall 2026 schedule. It lives at `docs/process/timeline.md` and is updated as milestones progress or deadlines shift. When submission to a class requires a standalone copy, it is exported without modification.

---

## 1. Project in one paragraph

The system this term builds is a **natural-language semantic search** over the ~30k unique cards in Scryfall's Oracle corpus. The research question is whether such a system can match or beat the domain-standard search tool (Scryfall) for non-expert users who cannot construct expert queries, and whether the pattern generalizes to similar consumer catalog domains (e-commerce, media libraries, technical documentation). The architectural response is a **three-stage retrieval cascade**: (1) a HyDE-style query rewriter (Gao et al., 2022) — a local instruction-tuned LLM that transforms the user query into structured filter attributes plus a hypothetical card ability text; (2) a SQL pre-filter over the extracted structured attributes on Postgres + pgvector; (3) semantic vector search using `nomic-ai/nomic-embed-text-v1.5` inside the pre-filtered candidate set. A hand-curated 26-query evaluation set with tri-state relevance judgments (Järvelin & Kekäläinen, 2002; Sormunen, 2002) provides the measurement harness. Comparison against Scryfall's expert-crafted queries (via an LLM-crafted query generator) is the primary reference; per-component contribution is isolated via targeted ablation (–SQL pre-filter).

---

## 2. Deliverables this term

| # | Deliverable | Class | Due date | Notes |
|---|---|---|---|---|
| 1 | Timeline document (this file) + rough draft abstract | Thesis class | **Sept 4** | First hard deadline |
| 2 | Final draft abstract | Thesis class | **Sept 18** | Updated with preliminary M4 measurements |
| 3 | Project presentation | Thesis class | **Oct 19 – Nov 15** (4-week window) | Requires a finished-enough system to demo |
| 4 | Final thesis paper (≥15 pages) | Thesis class | **Dec 4** | Formal paper structure |
| 5 | Department-honors research report | CSCI 5953 | **Dec 4** (before finals week) | Currently planned as the same document as #4 |
| 6 | Additional interim thesis deliverables | Thesis class | TBD | Between Sept 18 and Oct 19; specific items pending syllabus review |

**Post-semester (not scoped for this term):** the research work extends into a conference-presentable paper for **university honors**, targeting end of Fall 2026 or Spring 2027 submission.

---

## 3. Deadlines calendar

| Date | Week | Deliverable |
|---|---|---|
| Fri Sept 4 | Week 2 | Timeline + rough abstract |
| Fri Sept 18 | Week 4 | Final abstract |
| Mon Oct 19 – Sun Nov 15 | Weeks 9–12 | Presentation window (4 weeks) |
| Fri Dec 4 | Week 15 | Final thesis paper (also serves as department-honors report) |
| Sun Dec 6 | Week 15 end | Last day of coursework |
| Dec 7 – Dec 13 | Finals | No project work — deliverables already submitted |

---

## 4. Weekly plan

The schedule is compressed relative to the original 14-week plan (documented in `CLAUDE.md` §8) to accommodate the Oct 19 presentation start. The **buffer milestone (M6) absorbs the compression**; fine-tuning experiments will be cut first if time slips.

Milestone terminology follows the seven-milestone framework defined in `docs/process/milestone-checkpoints.md`. Milestones M0–M3 (project skeleton, infrastructure, ingestion, baseline measurement) were completed in earlier work sessions (May 2026). This term picks up at the M3 → M4 transition.

| Week | Dates | Focus | Thesis deliverable due this week |
|---|---|---|---|
| 1 | Aug 24 – Aug 30 | M3 → M4 backfill checkpoint (comprehension review) + workspace setup | — |
| 2 | Aug 31 – Sep 6 | M4 begins: HyDE + SQL pre-filter | **Sept 4:** Timeline + rough abstract |
| 3 | Sep 7 – Sep 13 | M4 continues | — |
| 4 | Sep 14 – Sep 20 | M4 wraps; M5 kicks off | **Sept 18:** Final abstract |
| 5 | Sep 21 – Sep 27 | M5: systematic evaluation + `scripts/generate_report.py` | — |
| 6 | Sep 28 – Oct 4 | M5 continues | — |
| 7 | Oct 5 – Oct 11 | M5 wraps; M6 begins (evidence-driven optimizations) | — |
| 8 | Oct 12 – Oct 18 | M6 + presentation preparation | — |
| **9** | **Oct 19 – Oct 25** | **Presentation window begins** · incremental report writing | Presentation slot possible |
| 10 | Oct 26 – Nov 1 | Presentation window · report writing | Presentation slot possible |
| 11 | Nov 2 – Nov 8 | Presentation window · report writing | Presentation slot possible |
| 12 | Nov 9 – Nov 15 | Presentation window ends · report writing | Presentation slot possible |
| 13 | Nov 16 – Nov 22 | Final report assembly | — |
| 14 | Nov 23 – Nov 29 | Final report assembly | — |
| 15 | Nov 30 – Dec 6 | Final revisions + submission | **Dec 4:** Final thesis paper |

---

## 5. What was already done (pre-term)

The project has substantial prior work from earlier sessions (May 2026), documented in the repo under `docs/journal/` and `docs/roadmap/`. This term does **not** start from a blank slate. Completed milestones:

| ID | Milestone | Status | Key artifacts |
|---|---|---|---|
| M0 | Project skeleton; POC archived; architecture document installed | ✓ Complete | `CLAUDE.md`, `archive/poc_v1/`, tag `v0.1-poc`, `docs/journal/2026-05-17-*` |
| M1 | Database schema, structured logging, corpus characterization | ✓ Complete | `docker-compose.yml`, `src/db/migrations/0001_initial.sql`, `src/logging_utils.py`, `docs/journal/2026-05-17-corpus-survey.md` |
| M2 | Corpus ingested; preprocessing pipeline (reminder-text keyword augmentation) | ✓ Complete | `scripts/ingest.py`, `scripts/build_keyword_dict.py`, `src/preprocess_text.py`, `data/keywords/{reminder_text,manual_overrides}.json` |
| M3 | First measured baseline | ✓ Complete | `configs/baseline.yaml`, `data/eval/queries_v1_draft.yaml`, `experiment_runs.id = 13`, `docs/journal/2026-05-18-baseline-results.md` |

Week 1 of this term is a **backfill review** of M0–M3 (comprehension checkpoint) before M4 code work begins. The review process is documented in `docs/process/milestone-checkpoints.md`.

---

## 6. Non-negotiables (from the working charter)

- **Three-tower architecture** (SQL pre-filter + HyDE + semantic) is the retrieval design; not open to redesign this term.
- **Evaluation before optimization** — fine-tuning is deferred to M6 and only pursued if M5 measurements justify it.
- **Milestone checkpoints** — the four-part review (artifact review, source vetting, predictions journal entry, Socratic interactive review) gates each milestone transition. Not skipped.
- **Report drafting starts Week 2, not Week 12.** Roadmap docs' *Notes for final report* sections and journal entries feed the paper draft in real time.

---

## 7. Risks and how the schedule handles them

| Risk | Impact | Mitigation baked into schedule |
|---|---|---|
| Presentation window forces the finished system 3+ weeks earlier than the original plan | M6 buffer evaporates | Fine-tuning is deferrable per CLAUDE.md §6; M6 gets cut first |
| External data source drift (Scryfall bulk-data format has changed since M2) | Blocks re-running ingestion pipeline | Being addressed in Week 1 as a scoped ingestion rewrite; documented as a Methodology / Limitations item |
| Milestone slippage propagates | Late M4 delays every subsequent milestone | Weekly progress tracked in `docs/journal/`; timeline updated when slippage is confirmed |
| Additional interim thesis deliverables between Sept 18 and Oct 19 | Unknown pull on schedule | Row #6 of §2 flags this; timeline revised when instructor clarifies |

---

## 8. How this timeline is maintained

- **Committed to the repo.** Every substantive change (milestone shift, deadline update, new deliverable) produces a commit with a short message explaining the reason.
- **Not authoritative for architecture or research decisions.** Those live in `CLAUDE.md` (architecture) and `docs/journal/` (research decisions). This document handles scheduling only.
- **Exported at submission time.** A frozen copy is generated for each class submission; the repo file continues to evolve.

---

## References (in-repo)

- `CLAUDE.md` — architecture and working conventions
- `docs/process/milestone-checkpoints.md` — milestone-transition review framework
- `docs/roadmap/phase-0-overview.md` and `docs/roadmap/phase-{1..7}-*.md` — per-milestone sub-task lists
- `docs/journal/` — dated decisions and analysis, chronological
