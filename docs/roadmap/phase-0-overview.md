# Phase 0 — Roadmap Overview

**Purpose:** index over all phases. Every phase has the same shape so progress, decisions, and report-material live in predictable places.

## How to use these files

Each phase file contains:

1. **Goal** — what success looks like for the phase, in one paragraph.
2. **Deliverables** — concrete artifacts. Tickable, named.
3. **Sub-tasks** — ordered work items with `[ ]` / `[x]` status. Add new ones as scope expands. Strike out ones that turn out unnecessary (with a reason).
4. **Decisions log** — table: `date | decision | reasoning | alternatives considered`. Append-only. Every non-trivial design call lands here.
5. **Open questions** — unresolved items that need an answer before later phases can proceed cleanly.
6. **Notes for final report** — structured material for the paper. Organized by paper section (Introduction, Methodology, Results, Discussion). Write as you go; do not reconstruct at week 14.
7. **Journal entries** — links to `docs/journal/` entries that elaborate on this phase.

## Working principles

- The schedule is a guide, not a contract. Phases will slip. The deliverables and the logging discipline matter more than week numbers.
- Don't skip the "Notes for final report" sections. They are why we log everything.
- Decision-log entries cost nothing to write and save weeks of memory archaeology at the end.
- When in doubt about whether something belongs in a sub-task vs. a journal entry: sub-tasks are *what we will do*, journal entries are *what we learned doing it*.

## Phase index

| Phase | Title | Weeks | Status |
|---|---|---|---|
| 1 | [Foundation & Logging Infrastructure](phase-1-foundation-and-logging.md) | 1–2 | Not started |
| 2 | [Ingestion & Schema](phase-2-ingestion-and-schema.md) | 3–4 | Not started |
| 3 | [Baseline Embedding & Evaluation Harness](phase-3-baseline-and-eval.md) | 5–6 | Not started |
| 4 | [HyDE & SQL Pre-Filter Integration](phase-4-hyde-and-prefilter.md) | 7–8 | Not started |
| 5 | [Systematic Evaluation & Reporting](phase-5-systematic-eval.md) | 9–10 | Not started |
| 6 | [Evidence-Driven Optimization](phase-6-optimization.md) | 11–13 | Not started |
| 7 | [Finalization & Paper](phase-7-finalization.md) | 14–15 | Not started |

## Final-paper section map

Where each phase's notes feed into the paper:

| Paper section | Source phases |
|---|---|
| Abstract | All |
| Introduction (problem framing) | 1, 3 (baseline failures) |
| Background / Related Work | 1, 4 (HyDE citation) |
| Methodology — corpus & schema | 1, 2 |
| Methodology — embedding & augmentation | 2 (keyword dict), 3 |
| Methodology — retrieval architecture | 4 |
| Methodology — evaluation design | 3, 5 |
| Experiments | 5, 6 |
| Results | 5, 6, 7 (final numbers) |
| Discussion & limitations | 5, 6 |
| Future work | 6 (deferred fine-tuning, etc.), 7 |
| Conclusion | 7 |
| Appendix | All (eval set, full results table) |

## Cross-phase artifacts

- **`docs/journal/`** — chronological, committed.
- **`experiment_runs` table** — every measured configuration ever run.
- **`scripts/generate_report.py`** — turns the above into shareable reports. Built in Phase 5, polished in Phase 7.
- **`data/eval/queries.yaml`** — hand-curated test set. Versioned (eval-set v1, v2, ...). Every `experiment_runs` row references which version it scored against.
