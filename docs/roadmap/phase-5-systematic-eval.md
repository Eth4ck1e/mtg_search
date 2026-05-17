# Phase 5 — Systematic Evaluation & Reporting

**Weeks:** 9–10
**Status:** Not started
**Depends on:** Phase 4 (three-tower pipeline working end-to-end, several `experiment_runs` rows already)

## Goal

Turn the pile of `experiment_runs` rows into a coherent comparison report, expand the eval set if it's not detecting meaningful differences, run a deliberate configuration sweep across the design axes that emerged from Phases 3–4, and identify the specific weaknesses that Phase 6's optimization will target. By end of phase the report-generation script exists, the design space has been explored systematically, and there's a concrete answer to "what is the biggest remaining gap and what would close it".

## Deliverables

- [ ] `data/eval/queries_v2.yaml` (only if v1 turns out too small to detect meaningful deltas — decision made in this phase)
- [ ] `scripts/generate_report.py` — queries `experiment_runs`, emits a markdown/HTML report with comparison tables, per-query breakdowns, ablation summaries
- [ ] `docs/reports/2026-MM-DD-phase-5.md` — the first real generated report, hand-curated narrative wrapped around auto-generated tables
- [ ] Configuration sweep results: rows in `experiment_runs` covering at least 8–12 distinct configurations
- [ ] Journal entry: per-query failure pattern analysis across the sweep
- [ ] Journal entry: identified weaknesses and Phase 6 hypothesis (what to fix next, and why this and not that)

## Sub-tasks

### Eval set expansion (conditional)
1. [ ] Look at variance in Phase 4 measurements. If two configurations differ by less than ~3 percentage points on recall@10 with the current 30–50 query set, the set isn't large enough to detect the kind of improvements Phase 6 will produce. Decide.
2. [ ] If expanding: target 75–100 queries. Maintain the same coverage breakdown from Phase 3 (difficulty × category × color). Add queries specifically targeting the categories that showed the most variance in Phase 4.
3. [ ] Version as `queries_v2`. Every subsequent `experiment_runs` row references either v1 or v2; the eval set version is part of the `config` JSON.
4. [ ] Don't compare v1-scored rows to v2-scored rows directly — they're different exams. The paper's results table needs to be careful about this.

### Configuration sweep
5. [ ] **Reminder-text augmentation on vs. off.** Re-embed corpus without reminder-text augmentation (new `embedding_version`), re-run the full pipeline. The delta is the empirical value of the augmentation itself. **This is one of the paper's headline results** — design it carefully.
6. [ ] **HyDE prompt variants.** Run the best 2–3 prompt versions from Phase 4 against the (possibly expanded) eval set. If a clear winner, fix it; if not, choose by latency / token cost.
7. [ ] **Pre-filter vs. post-filter ablation.** Run both configurations on the same prompt, same eval set. Pre-filter should win, especially on constrained queries — this confirms the architectural choice.
8. [ ] **Hard vs. soft filters.** If Phase 4 left this open: implement soft filtering (boost matched cards by some factor rather than exclude unmatched), run it, compare.
9. [ ] **Top-K sensitivity.** Run the final config at K=5, 10, 20, 50. Recall@K curves go in the appendix.
10. [ ] **Alternative embedding models, optional.** If the schedule permits: swap `multi-qa-distilbert-cos-v1` for `all-mpnet-base-v2` (larger, often better but slower) and `BAAI/bge-base-en-v1.5` (often state-of-the-art for retrieval). Re-embed, re-run. Models go in the config JSON.
11. [ ] **Cosine vs. dot product, optional.** pgvector supports both. The model is trained for cosine; should be a no-op unless something is off.

### Report generation
12. [ ] `scripts/generate_report.py` — argparse interface: `--since DATE`, `--eval-set-version`, `--metrics recall@5,mrr,...`, `--out PATH`. Default: latest run per unique config, pretty markdown output.
13. [ ] Auto-generated comparison table: rows = configurations, columns = metrics. Include eval-set version per row.
14. [ ] Per-query breakdown table: for two named configurations (e.g., baseline vs. best), show which queries improved, which regressed, which stayed the same.
15. [ ] Ablation table: define a "default" config and a list of "knobs" to vary; the script generates the table showing the effect of each knob.
16. [ ] Latency table: p50, p95, p99 per configuration, with HyDE LLM time broken out separately from retrieval time.
17. [ ] CLI output is markdown; an `--out html` mode using a simple template (Jinja or even f-strings) is a stretch.
18. [ ] Test the report generator on `experiment_runs` rows as they exist today, before running the full sweep. Easier to debug the script when there's less data to wade through.

### Failure analysis
19. [ ] For the best-so-far config, dump the bottom 10 queries by recall@10. Categorize the failures:
    - **Filter too restrictive** — HyDE extracted a constraint that excluded valid answers
    - **Filter too lenient** — system returned cards from outside the user's intended scope
    - **Embedding miss** — the relevant cards' embedding text doesn't match the hypothetical text well enough
    - **Eval set issue** — the labeled "relevant" set is incomplete or wrong (re-label the query and re-run)
    - **Fundamentally hard** — the query is ambiguous or requires meta-knowledge no part of the pipeline has
20. [ ] Count failures by category. Identify the dominant category.
21. [ ] Journal entry: weaknesses identified, hypotheses for Phase 6. Crystallize Phase 6's plan: "Based on this analysis, the highest-leverage optimization is X, because Y." If "X" is fine-tuning, motivate it with the data; if "X" is something else (better prompt, additional preprocessing, query parser, hybrid BM25+semantic), motivate that.

## Decisions log

| Date | Decision | Reasoning | Alternatives considered |
|---|---|---|---|
| | | | |

## Open questions

- Should the report include statistical significance tests on the metric deltas? At eval-set size 50, the noise is non-trivial; a paired test (per-query) is more meaningful than aggregate comparisons. Worth investigating, but not required.
- Per-query results in `experiment_runs.per_query` are JSONB arrays. At 100 queries × 10 results = 1000 entries per row. Fine for v1. If it gets unwieldy, normalize to a separate `experiment_query_results` table — defer that decision until it's a real problem.

## Notes for final report

### Methodology — evaluation expansion (if applicable)
- The decision logic for expanding to v2, with the variance numbers that justified it.
- The coverage breakdown of v2.

### Results — main comparison
- The headline table: baseline vs. each major configuration. This is the result the paper hinges on.
- The ablation table (knob by knob) is the second-headline result.
- Per-category breakdown — the paper benefits from showing where each design choice helps most.

### Results — reminder-text augmentation
- Dedicated subsection. Before/after numbers + a worked example showing a query where augmentation made the difference (which keyword was missing inline, what the reminder text added, what cards moved into the top-K).

### Discussion — failure analysis
- The five failure categories above, with example queries per category, are the structure of the Discussion section. They also feed Future Work.

## Journal entries

- (none yet)
