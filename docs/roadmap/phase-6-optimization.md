# Phase 6 — Evidence-Driven Optimization

**Weeks:** 11–13
**Status:** Not started
**Depends on:** Phase 5 (failure analysis complete, dominant weakness identified, Phase 6 hypothesis stated)

## Goal

Address whatever Phase 5's analysis identified as the highest-leverage remaining weakness. **The work in this phase is contingent on Phase 5's findings, not pre-decided.** Fine-tuning a sentence-transformer is the candidate optimization that's been on the roadmap the longest, but it is one option among several, and the choice is driven by evidence.

The phase is structured to support **whichever optimization is chosen**, with explicit sub-task branches per candidate. Pick one (or in priority order, two), execute, measure against Phase 5's best baseline, document.

## Deliverables

- [ ] Journal entry: the Phase 6 plan, written immediately after Phase 5 closes — restates the hypothesis, names the optimization, predicts the expected metric delta and on which query categories
- [ ] Implementation of the chosen optimization (see sub-task branches)
- [ ] At least one new `experiment_runs` row measuring the optimization vs. Phase 5's best
- [ ] Journal entry: actual delta vs. predicted, what worked, what didn't, what surprised
- [ ] (Optionally) a second optimization if the first leaves measurable headroom and time permits

## Sub-tasks — branch on Phase 5's identified weakness

### Branch A — Fine-tuning the encoder
*Pick this if Phase 5 shows "embedding miss" is the dominant failure category and the embedding text genuinely doesn't capture the semantics needed.*

1. [ ] Generate synthetic (query, card) training pairs. Use an LLM (Claude) to produce candidate queries for ~200–500 cards, sampled to cover the failure-category distribution. Quality-control by spot-check: are the generated queries plausible NL queries a player would type?
2. [ ] Split 80/20 train/validation. Reserve the actual eval set entirely from training — never include any oracle_id from the eval set in the training pairs.
3. [ ] Fine-tune `multi-qa-distilbert-cos-v1` with `MultipleNegativesRankingLoss`. Hyperparameters to start: lr=2e-5, batch_size=16, epochs=3, warmup_steps=10% of total.
4. [ ] Log training metrics (loss curves), save checkpoints, save final model with a new `embedding_version` string.
5. [ ] Re-embed the corpus with the fine-tuned model. Re-run the eval pipeline. Compare to Phase 5's best.
6. [ ] If overfitting is suspected (training loss drops but eval recall doesn't move): try fewer epochs, larger batch, or expand training data with more diverse cards.
7. [ ] Document what changed. The journal entry on this work is paper-section-grade.

### Branch B — HyDE prompt redesign
*Pick this if Phase 5 shows "filter too restrictive" or "filter too lenient" is dominant and the failures correlate with how HyDE is interpreting queries.*

1. [ ] Categorize the filter-related failures: when does HyDE over-constrain, when does it under-constrain?
2. [ ] Redesign the prompt to address the specific pattern. Possible changes: more conservative filter extraction (when in doubt, omit the filter), explicit "no filter" example cases, decompose into two passes (first decide what to filter on, then decide on values).
3. [ ] Add prompt-level confidence and use it as a gate — low-confidence rewrites fall back to raw query with no filter.
4. [ ] Run measurements, compare.

### Branch C — Hybrid BM25 + semantic
*Pick this if Phase 5 shows the pipeline systematically misses queries with very specific terminology (e.g., card names mentioned in the query, distinctive single-word abilities) that the embedding generalizes away.*

1. [ ] Add a Postgres full-text-search index on `oracle_text`. Compute BM25 score for each query against the candidate set.
2. [ ] Combine BM25 and cosine via reciprocal rank fusion (RRF) or weighted sum. Tune the weight on a held-out portion of the eval set.
3. [ ] Run measurements, compare.

### Branch D — Better preprocessing
*Pick this if Phase 5 shows the embedding text itself is missing information — e.g., reminder text isn't being included for cards that need it, or non-keyword phrases would benefit from synonym expansion.*

1. [ ] Audit the cards on the failure list. What's in `oracle_text`? What's missing from the embedding text that would have helped?
2. [ ] Iterate on `src/preprocess_text.py`. Each preprocessing change is a new `preprocess_version` and triggers a re-embedding of affected rows.
3. [ ] Run measurements, compare.

### Branch E — Something Phase 5 surfaced that isn't above
1. [ ] Write up the chosen approach as a journal entry first — pitch the case, predict the outcome.
2. [ ] Implement, measure, document.

## Decision template

When entering this phase, write a journal entry titled `YYYY-MM-DD-phase-6-plan.md` with this structure:

```markdown
## Phase 5's dominant failure category
<one paragraph from Phase 5's analysis>

## Chosen optimization (branch X)
<one paragraph: what and why this and not the others>

## Predicted outcome
- Expected metric delta: recall@10 +X percentage points
- Expected effect on each query category: ...
- Expected cost: <time, compute, complexity>

## Falsification criteria
What result would make me conclude this was the wrong choice?
```

This entry exists so the post-experiment journal entry can compare predictions to outcomes directly.

## Decisions log

| Date | Decision | Reasoning | Alternatives considered |
|---|---|---|---|
| | | | |

## Open questions

- If the first chosen optimization fails (delta is negative or zero), do we try a second branch, or accept the result and move to Phase 7? Default: try one more, only if there's clear time. Failed experiments are still paper content.
- For Branch A specifically: do we publish the fine-tuned model (HuggingFace) as a project artifact? Worth considering — it's the kind of deliverable a conference reviewer notes positively.

## Notes for final report

### Methodology — chosen optimization
- Full description of what was done. The branch-specific sub-tasks above already structure this.
- Training data construction (if Branch A), with the synthetic-data quality-control story.

### Results — optimization delta
- Before/after table with statistical context (paired test if eval-set is large enough).
- Per-category breakdown: which queries the optimization moved most.

### Discussion
- Why this optimization worked (or didn't). The falsification-criteria framing above lets you write this honestly.
- Limitations: what would still be wrong even with this optimization. Connects to Future Work.

### Future work
- Whatever Phase 5's failure analysis identified as the *second* most important category, that wasn't addressed in this phase. Frame as concrete next steps.

## Journal entries

- (none yet)
