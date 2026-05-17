# Phase 3 — Baseline Embedding & Evaluation Harness

**Weeks:** 5–6
**Status:** Not started
**Depends on:** Phase 2 (corpus loaded, preprocessing function exists, keyword dict generated)

## Goal

Produce the first measured number. Embed the corpus with `sentence-transformers/multi-qa-distilbert-cos-v1` on the preprocessed (oracle-text + augmented reminder-text) representation, build a hand-curated evaluation set, score the baseline pipeline (raw query → embed → cosine search, no SQL filter, no HyDE), and capture the result as the first row in `experiment_runs`. **This phase produces the number every subsequent change is measured against.**

The single most valuable deliverable in this phase is the **evaluation set**. Everything else can be rebuilt mechanically; the eval set is the one piece of judgment-intensive human work.

## Deliverables

- [ ] `scripts/embed.py` — embeds rows where `embedding IS NULL OR embedding_version != $current`
- [ ] `data/eval/queries_v1.yaml` — hand-curated 30–50 `(query, [relevant_oracle_ids])` pairs
- [ ] `data/eval/README.md` — methodology for how the eval set was constructed, what coverage it aims for
- [ ] `scripts/evaluate.py` — runs a configuration over the eval set and writes a row to `experiment_runs`
- [ ] Baseline row in `experiment_runs` with config + metrics for the no-HyDE, no-SQL-filter pipeline
- [ ] Journal entry: baseline metrics with per-query breakdown and a failure-mode analysis
- [ ] Journal entry: eval set construction methodology, what queries were chosen, what was deliberately omitted

## Sub-tasks

### Embedding script
1. [ ] `scripts/embed.py` — selects rows to embed (`WHERE embedding IS NULL OR embedding_version != $current`), batches them, runs through `multi-qa-distilbert-cos-v1`, writes back via `UPDATE`. Batch size starts at 32, configurable.
2. [ ] Use `sentence-transformers` library directly (`SentenceTransformer.encode()`). Do not hand-roll tokenization + `[CLS]` extraction — the library handles mean-pooling and normalization correctly for the model.
3. [ ] Device selection via `src/utils/device.py`: `cuda > mps > cpu`.
4. [ ] Logging: count read, count embedded, count skipped (already at current version), batch durations, total duration, model name, model version (huggingface revision hash), preprocessing version. Sample one record's embedding-text per N batches for spot-checking.
5. [ ] Idempotency: running `embed.py` twice in a row on a clean state produces zero new embeddings the second time.
6. [ ] CLI: `python scripts/embed.py [--limit N] [--batch-size N] [--dry-run]`. Limit lets you smoke-test on 100 cards before committing to the full run.
7. [ ] Embedding-version string convention: `"<model_name>@<hf_revision>|preproc=<preprocess_version>"` so two embeddings with the same string are byte-comparable.

### Evaluation set construction
8. [ ] Define coverage targets. The eval set must include queries spanning:
   - **Query style**: natural-language sentences, fragmented "search-y" queries, jargon ("ramp", "ETB", "flicker"), constraints ("under 3 mana"), mechanical descriptions ("counter target spell").
   - **Card types**: creatures, instants, sorceries, enchantments, artifacts, lands, planeswalkers.
   - **Color spread**: at least one query per color identity (W, U, B, R, G, colorless, multicolor).
   - **Difficulty bands**: easy (literal keyword in text), medium (one inference required), hard (player-jargon that needs reminder text to bridge).
9. [ ] Construct ~30 initial queries. Format in `data/eval/queries_v1.yaml`:
   ```yaml
   version: "v1"
   queries:
     - id: q_001
       query: "cards that flicker creatures"
       relevant: [<oracle_id_1>, <oracle_id_2>, ...]
       difficulty: hard      # easy | medium | hard
       category: jargon      # natural | fragmented | jargon | constrained | mechanical
       notes: "Tests reminder-text augmentation for 'blink' / 'exile and return'"
   ```
10. [ ] For each query, look up ~5–15 expected-relevant `oracle_id`s by consulting Scryfall's own search (`https://scryfall.com/search`) and verifying by hand. Document anything contentious in the per-query `notes` field.
11. [ ] **Do not include queries the new architecture is known to handle trivially.** The eval set's job is to surface differences; queries everyone agrees on are noise.
12. [ ] Validate that every `relevant` oracle_id actually exists in the DB. Add a one-shot check script.
13. [ ] Commit the eval set with a journal entry documenting the construction process, the coverage breakdown, and queries deliberately excluded.

### Evaluation harness
14. [ ] `scripts/evaluate.py` — takes a configuration YAML (which retrieval pipeline to use, which eval set version, etc.), runs every query, computes:
    - **Recall@1, Recall@5, Recall@10** — what fraction of `relevant` IDs appear in top-K.
    - **MRR** (mean reciprocal rank of the first relevant hit). Note: this measures "did we surface *something* relevant", not "did we get them all" — pair with recall.
    - **Latency**: per-query p50 and p95 ms.
    - **Per-query results**: rank of each relevant hit, full top-10 with similarity scores.
15. [ ] Write one row to `experiment_runs` per evaluation invocation. Config column captures everything that could affect results (model name, version, preprocessing version, prompt version (none yet in Phase 3), filter logic, K).
16. [ ] CLI: `python scripts/evaluate.py --config configs/baseline.yaml`. Configs live in a versioned directory.
17. [ ] Test the harness on a single hand-crafted dummy eval set (3 queries, 2 cards) to verify metric calculations are correct before pointing it at the real one.

### Baseline run + analysis
18. [ ] Configuration `configs/baseline.yaml` — embed query as-is with `multi-qa-distilbert-cos-v1`, cosine search across ALL cards (no SQL filter), top-K = 10, no HyDE.
19. [ ] Run `evaluate.py --config configs/baseline.yaml`. Capture the resulting `experiment_runs` row ID.
20. [ ] Per-query failure analysis: sort queries by recall@10, look at the bottom quartile. For each failing query:
    - What did the system return instead?
    - Why? Look at the cosine similarities — are the returned cards in a related embedding region?
    - Is the failure mode consistent with the POC retrospective's predictions, or new?
21. [ ] Journal entry capturing the baseline numbers and per-query analysis. **This entry is the heart of the paper's "baseline" subsection** — write it well.

## Decisions log

| Date | Decision | Reasoning | Alternatives considered |
|---|---|---|---|
| | | | |

## Open questions

- Should the eval set have queries with **zero** relevant cards in the corpus? Tests false-positive behavior but complicates the metric definition. Defer to v2 of the eval set.
- Should we normalize embeddings before storing or at query time? `multi-qa-distilbert-cos-v1`'s `encode()` produces L2-normalized vectors by default — confirm and document.
- What about diacritics / special characters in card names (`Lim-Dûl's Vault`, `Lupinflower Village`)? Probably handled by the tokenizer, but worth one test case in the eval set.

## Notes for final report

### Methodology — evaluation design
- The eval set's coverage breakdown (table: difficulty × category × color, with counts) is paper-ready. Include it.
- Justify the size choice: 30–50 is small enough to hand-curate carefully, large enough to detect 5+ percentage-point differences in recall@K with reasonable confidence. Cite IR literature on minimum eval-set sizes if relevant.
- The (query, list-of-relevant-oracle-ids) format and why it matters — most queries have multiple relevant cards, single-correct-answer would be wrong.

### Methodology — baseline
- The exact baseline configuration. Single sentence: "We embed both query and oracle text (augmented with reminder text) with `multi-qa-distilbert-cos-v1`, do cosine similarity over the full corpus, and take top-K."
- Embedding-version string convention and `embedding_text_hash` for reproducibility.

### Results — baseline
- Headline metric table: recall@1, recall@5, recall@10, MRR, latency p50/p95.
- Per-category and per-difficulty breakdown table. This is the structure that motivates HyDE: jargon and constrained queries should be visibly worse than literal queries.
- Per-query examples in the appendix: 3–5 illustrative failures.

### Discussion — baseline failure modes
- Map observed failures back to POC retrospective predictions. Where they line up, the redesign's premises are validated. Where they diverge, that's interesting and worth a paragraph.

## Journal entries

- (none yet)
