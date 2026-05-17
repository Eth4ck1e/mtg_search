# Phase 4 — HyDE & SQL Pre-Filter Integration

**Weeks:** 7–8
**Status:** Not started
**Depends on:** Phase 3 (baseline measured, eval harness running, `experiment_runs` populated)

## Goal

Add the two layers that the POC was missing: an LLM-based query-rewriter (HyDE) that turns informal user queries into a hypothetical card representation, and a SQL pre-filter that uses the structured attributes the rewriter extracts. By end of phase the full three-tower retrieval pipeline runs end-to-end, every prompt change is versioned, and `experiment_runs` contains at least one row per major prompt variant + the baseline — so the **measured delta** is visible.

## Deliverables

- [ ] `src/query_rewriter.py` — HyDE-style transformer: NL query → `{structured_filters: {...}, hypothetical_text: "..."}`
- [ ] `prompts/hyde/v1.txt` — versioned prompt (and v2, v3 as iteration produces them)
- [ ] `src/sql_filter.py` — translates structured filter dict → SQL `WHERE` clause + parameterized values
- [ ] `src/search.py` — orchestrates query → HyDE → SQL filter → vector search within filtered set → ranked results
- [ ] Updated `configs/` — new configurations exercising HyDE + filter at various combinations
- [ ] Multiple `experiment_runs` rows: baseline, HyDE-only (no filter), filter-only (deterministic constraints, no HyDE), full pipeline, HyDE-prompt v2/v3
- [ ] Journal entries: HyDE prompt design, filter logic decisions, comparison of HyDE-only vs. filter-only vs. combined

## Sub-tasks

### Query rewriter
1. [ ] Decide which LLM. Default: Claude (Anthropic API) — Mitchell has experience and the SDK supports prompt caching well. Alternatives: GPT-4o-mini, local Llama. Document the decision.
2. [ ] Define the structured output schema: `{filters: {colors?, color_identity?, cmc_lte?, cmc_gte?, cmc?, type_includes?: [], type_excludes?: [], legality?}, hypothetical_text: str, confidence: float}`. The `filters` keys map 1:1 to columns we have real indexes on. Anything the rewriter "wants" to filter on that isn't in this schema is dropped or attached to `hypothetical_text`.
3. [ ] First prompt (`prompts/hyde/v1.txt`): few-shot examples covering all coverage categories from the eval set. Output format is strict JSON. Include explicit "if no constraint, omit the key" instruction.
4. [ ] `src/query_rewriter.py:rewrite(query: str, prompt_version: str) -> RewriteResult` with prompt version threaded through. Logs every call (input query, prompt version, raw model output, parsed result, model name, model revision, tokens used, latency).
5. [ ] Cache rewrites by `(query, prompt_version, model)` — same input shouldn't pay the LLM cost twice during evaluation.
6. [ ] Schema validation on the output — if the model returns invalid JSON or unknown filter keys, log and fall back to "no filter, raw query as hypothetical text". Don't crash the eval run; capture the failure rate as a separate metric.

### SQL filter
7. [ ] `src/sql_filter.py:build_where(filters: dict) -> tuple[str, list]` produces a parameterized SQL fragment + values. Defensive: never interpolate filter values into the SQL string; always parameterize.
8. [ ] Support: exact color identity match, color subset match, CMC range, type-line `LIKE` for inclusion, type-line negative match for exclusion, legality lookup against `legalities` JSONB.
9. [ ] **Hard constraints vs. soft**: at start treat all filters as hard. If a filter is wrong (model hallucinated "white" for a blue counterspell query), recall craters. Plan for an experiment in Phase 5 where filters become soft (boost matched cards rather than exclude unmatched).
10. [ ] Test against the eval set's filter-bearing queries: for "counterspell that costs 2", verify the rewriter produces `{colors: ["U"]}` (or `{color_identity: ["U"]}`, choose one), `{cmc: 2}`, and the SQL produces the expected candidate set.

### Three-tower pipeline
11. [ ] `src/search.py:search(query: str, *, prompt_version, top_k, model) -> SearchResult` is the single entry point. Orchestration: rewrite → build SQL filter → SQL query returning candidate IDs + their embeddings → cosine score the hypothetical-text embedding against candidates → return top-K with metadata.
12. [ ] Pre-filter vs. post-filter: pre-filter (SQL narrows candidates first) is the architecture. Implement it cleanly. If a config flag for post-filter is needed for an ablation, name it `--post-filter` and document it as an explicit ablation, not the default.
13. [ ] Edge case: filter is too restrictive and returns zero candidates. Decision: log it as a "filter-too-tight" event, then either (a) widen the filter automatically (drop the most-constraining clause) or (b) return empty. Default (b); flip to (a) only if Phase 5 measurements suggest it.
14. [ ] Latency budget: pre-filter SQL should be <20ms with our indexes. Vector search on a filtered set of <1000 candidates should be <50ms. Total budget (including HyDE LLM call): <2s p95. The LLM dominates — that's the price of HyDE.

### Configurations and measurements
15. [ ] `configs/hyde_v1_no_filter.yaml` — rewriter on, but discard the structured filters; embed the hypothetical text and search the full corpus. Isolates "did the query-rewriting help on its own".
16. [ ] `configs/filter_only.yaml` — use Phase 2's preprocessing to extract structured constraints from the query deterministically (no LLM), apply the SQL filter, embed the raw query. Isolates "did the pre-filter help on its own". (This is a real ablation worth running — it tells us how much of the win is structural vs. semantic.)
17. [ ] `configs/three_tower_v1.yaml` — full pipeline with HyDE prompt v1.
18. [ ] Run all of the above against `queries_v1` eval set. Each is a separate `experiment_runs` row.
19. [ ] Prompt iteration: when the first per-query failure analysis on `three_tower_v1` surfaces patterns (e.g., "rewriter always overfilters by color"), draft `prompts/hyde/v2.txt`, run as `three_tower_v2.yaml`, log results, write a journal entry comparing v1 vs. v2.
20. [ ] Plan for at least 2–3 prompt iterations before declaring the prompt-tuning phase done.

## Decisions log

| Date | Decision | Reasoning | Alternatives considered |
|---|---|---|---|
| | | | |

## Open questions

- One LLM call per query is fine for an eval set of 50, but it bloats latency and cost. Should the eval cache rewrites in `data/eval/cache/` so re-running evaluation doesn't repay the LLM cost? Lean yes.
- HyDE rewrites can be misleading when the user query is itself well-formed ("Counterspell"). Should we have a confidence gate where low-confidence rewrites fall back to the raw query? Test it as an ablation.
- Color identity vs. colors: which field does HyDE target? Both exist in the schema for a reason (deck-builder color identity uses `color_identity`; effect-level color uses `colors`). The prompt needs to be precise about this. Document.

## Notes for final report

### Methodology — query rewriting
- HyDE paper citation (Gao et al. 2022).
- The structured-output adaptation: not pure HyDE — we extract SQL filters as a side product of the rewrite. This is the original contribution at the architectural level.
- The exact prompt (or final iteration of it) goes in the appendix.

### Methodology — pre-filter vs. post-filter
- The pre-filter-first design is non-trivial to justify; cite that on constrained queries, post-filter on top-K throws away recall when constraints are tight. Reference the POC retrospective.

### Results — ablation
- Three-row table: baseline vs. HyDE-only vs. filter-only vs. full. This is the canonical ablation that justifies the architecture. Worth a dedicated subsection.
- Per-category breakdown of the full pipeline: which query categories benefited most? (Jargon and constrained queries should win big; literal queries should be roughly tied.)

### Discussion
- HyDE prompt iteration story. The journal entries on v1 → v2 → v3 are the source material. Lead with the failure pattern in v1 that prompted v2.

## Journal entries

- (none yet)
