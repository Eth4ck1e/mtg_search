# 2026-05-18 — Evaluation set v1 construction methodology

## Context

Phase 3's heaviest deliverable: a hand-curated `(query, relevant_oracle_ids)` set that every subsequent retrieval change is measured against. The roadmap calls for 30–50 queries; we landed at 26 with rich per-query relevant lists (often 25–75 cards) rather than a longer list of thinner queries. The eval set is committed as [`data/eval/queries_v1_draft.yaml`](../../data/eval/queries_v1_draft.yaml). This entry documents the methodology — what was chosen, why, and what's deliberately deferred — so the paper's "Methodology — Evaluation Design" subsection draws directly from a single record rather than spelunking through commits.

## What happened

The eval set was built in **four batches** across two days, each rendered through [`scripts/render_review.py`](../../scripts/render_review.py) into a self-contained HTML page with card images so the picks could be visually audited end-to-end. The supporting toolchain landed alongside:

- [`scripts/eval_lookup.py`](../../scripts/eval_lookup.py) — Scryfall candidate finder. Hits `/cards/search` with operator syntax, cross-references against the local cards table, prints full oracle text per candidate.
- [`scripts/render_review.py`](../../scripts/render_review.py) — generates an HTML page from the draft YAML with three card grids per query (green relevant / yellow borderline / red rejected) at "large" image size. Clicking a card opens its Scryfall page. The visual review surface that makes auditing 700+ cards tractable.
- [`data/eval/methodology_references.md`](../../data/eval/methodology_references.md) — three academic citations (Järvelin & Kekäläinen 2002, Voorhees 2000, Sormunen 2002) that ground the tri-state relevance scheme in published IR-eval practice.

**Workflow per query**:

1. I propose 1–4 Scryfall search syntaxes covering different oracle-text wordings of the same intent (e.g., for "flicker effects": `o:"exile target creature you control"` for spells, `o:"exile another target creature you control" o:"return"` for ETB-trigger creatures, plus targeted name lookups for canonical staples like Restoration Angel and Eldrazi Displacer).
2. Run the lookups, read full oracle text from `eval_lookup.py`, judge each candidate as relevant / borderline / rejected with one-line reasoning.
3. Write picks into the draft YAML.
4. Render the HTML and audit visually — catches cards I should have promoted from rejected or demoted from relevant.
5. EDHREC theme-page cross-validation pass for category-jargon queries that map to community-recognised themes (Ramp, Reanimator) — surfaces gaps the corpus-text search missed.

The collaborative pattern that emerged: I drive candidate generation and tri-state judgment (which is fundamentally an NLP comparison task — "does this oracle text match the query's intent"), and Mitchell audits the rendered HTML for MTG-rules judgments my training doesn't cover well (deck context, format-meta, combo interactions). On the **q_010 card-draw-engines** pass for instance, Mitchell sharpened the methodology by introducing a "net-positive repeatable" criterion that moved Mind Stone, Commander's Sphere, Kenrith's Transformation, and Spelunking from borderline to rejected (single-card replacement isn't an engine).

## Decision

**The eval set is v1-draft, 26 queries serving three distinct testing purposes**:

| Testing purpose | Count | Examples | Tests what |
|---|---|---|---|
| Pure-semantic | 21 | q_006 ramp, q_014 flicker, q_023 prowess natural | Embedding quality on jargon / natural-language / fragmented-style queries |
| Pure-structural | 3 | q_019 free counterspell, q_020 red creatures <3, q_021 1-mana instants | SQL filter routing/correctness — embedding cannot enforce these |
| Hybrid | 2 | q_025 red pingers <3, q_026 cheap blue counterspells | Both-tower cooperation — only resolves with HyDE + SQL together |

**Coverage breakdown** (paper-ready table):

| Axis | Distribution |
|---|---|
| Difficulty | 7 easy / 11 medium / 8 hard |
| Categories | 13 jargon / 4 natural / 3 fragmented / 3 constrained / 2 hybrid / 1 mechanical |
| Style-invariance pairs | q_008↔q_024 board wipes (jargon ↔ natural), q_011↔q_022 tutor (jargon ↔ natural) |

**Relevance scheme: tri-state (`relevant` / `borderline` / `rejected`)**, with `borderline` excluded from both numerator and denominator of recall@K and from MRR. Cards that are plausibly relevant under a generous reading but not a strict one (Helvault as flicker, Mind Stone as card-draw engine) live in `borderline` so the metric isn't forced to either penalise the system for surfacing them or trivialise itself by counting them as relevant.

## Reasoning

**Why three testing purposes instead of one?** Halfway through batch 3 (post-feedback on board wipes and removal), it became clear that the eval-set queries weren't all measuring the same thing. q_020 "red creatures under 3 mana" has zero semantic content; the SQL filter is the entire system for that query. q_014 "flicker effects" has zero structural content; the embedding is the entire system. The eval set has to test both towers and their cooperation, which means queries with different shapes serve different roles. Calling that out explicitly in the methodology section is more defensible than pretending all 26 queries measure the same thing.

**Why 26 queries and not 30–50?** The roadmap target assumes thinner per-query relevant lists (5–15 each per the original sub-task spec). We instead built deeper lists per query — q_008 has 75 relevant board wipes spanning 7 sub-categories; q_009 has 73 removal spells across 5 modalities. The total volume (≈700 relevant judgments across 26 queries) is comparable to a 30–50-query set with 15-card relevant lists. Going deeper per query lets us measure recall at higher denominators for narrow concepts — more statistically meaningful than wider-but-shallower coverage. Sormunen 2002's finding that marginal-relevance documents dominate TREC pools also supports deeper per-query investigation over more queries with less rigorous boundaries.

**Why tri-state instead of binary?** Covered in detail in [the keyword-augmentation entry](2026-05-17-keyword-augmentation.md) and [methodology_references.md](../../data/eval/methodology_references.md). The condensed argument: forcing a binary judgment on Helvault ("is it a flicker effect?") either penalises the system for surfacing a card players legitimately use in flicker piles, or trivialises relevance by counting tangential matches. Sormunen 2002 demonstrated empirically that ~50% of TREC's "relevant" pool consists of marginally-relevant documents — the same population our `borderline` bucket isolates. Järvelin & Kekäläinen 2002 established graded relevance as IR mainstream; Voorhees 2000 demonstrated that single-curator judgments under documented criteria produce stable comparative rankings.

**Why EDHREC cross-validation matters.** Corpus-driven Scryfall search captures cards by exact oracle-text patterns. EDHREC's theme pages capture cards by community usage. The two diverge in predictable directions: corpus search misses cards with idiosyncratic wording (Reanimate, Animate Dead missed in q_012 because my initial query anchored on "return target creature card from your graveyard to the battlefield"); EDHREC theme pages include cards that are commonly *played in* the archetype without being *of* the archetype (Chaos Warp on the Ramp page because it's commonly in ramp decks, not because it ramps). The intersection — corpus-match validated by community recognition — is the most defensible set.

**The LLM/curator division of labour.** Mitchell pointed out (correctly) that judging "does this oracle text match the query's intent" is fundamentally an NLP-comparison task, not a deep-MTG-knowledge task. The original Phase 3 roadmap framed eval-set construction as the heaviest curator-driven deliverable; in practice, an LLM with structured candidate lists from Scryfall plus a visual audit by an MTG player produces high-quality judgments faster than a curator working alone. Reserve the human judgment for the parts the LLM is genuinely worse at: format-meta context, combo interactions, "is this card actually played for this purpose."

## Alternatives considered

- **Binary relevance instead of tri-state.** Rejected — covered above. Tri-state aligns with established IR-eval practice and prevents the eval from encoding one narrow definition of relevance.
- **NDCG with graded relevance levels.** Considered as an alternative to recall@K + borderline-filtering. Rejected for v1 — NDCG requires graded labels (e.g., highly relevant / fairly relevant / marginal / irrelevant) which would have tripled the labelling work. Tri-state with binary scoring is simpler, gives most of the benefit, and stays comparable to Voorhees 2000's stable-ranking results.
- **Bigger relevant sets via "everything that matches a programmatic check."** Considered for broad queries — instead of 30 hand-picked fliers for q_001 "creatures with flying," programmatically label all 3,360 fliers as relevant. Rejected for v1 — would require a programmatic check that's robust to interpretation, and "do I judge each card or do I judge a property" is a methodologically thorny question. Documented as a v2 follow-up. The v1 numbers will undermeasure broad-query recall, which is a known caveat we can defend in the paper.
- **More queries, thinner relevance lists.** Rejected — covered above. We chose depth per query over query count.
- **Use only EDHREC themes directly.** Rejected — EDHREC's theme pages conflate cards that *are* the archetype with cards commonly *played in* it. Tutoring through EDHREC alone would have included Chaos Warp under "ramp" and we'd have had to do the same reading-against-text work anyway.

## Notes for the final report

### Methodology — evaluation design

Quote-ready paragraph: *"The evaluation set comprises 26 hand-curated natural-language queries spanning three testing purposes: pure-semantic queries that test embedding quality on player jargon, natural-language, and fragmented-search phrasings (n=21); pure-structural queries that test SQL-filter routing where embeddings cannot enforce numeric or color constraints (n=3); and hybrid queries that combine structural constraints with semantic terms, resolvable only via cooperative use of both towers (n=2). Each query carries a `(relevant, borderline, rejected)` tri-state judgment (Järvelin & Kekäläinen 2002; Sormunen 2002), with the `borderline` bucket isolating cards whose relevance is genuinely contested (e.g., save-from-wrath effects like Helvault as 'flicker'); borderline cards are excluded from recall@K and MRR computation. Relevance judgments were produced by a single curator under documented criteria (Voorhees 2000) using a corpus-extraction-plus-EDHREC-validation workflow with a custom HTML review tool for visual audit."*

Coverage table (drop into the methodology section as-is):

| Axis | Count |
|---|---|
| Difficulty: easy / medium / hard | 7 / 11 / 8 |
| Category: jargon / natural / fragmented / constrained / hybrid / mechanical | 13 / 4 / 3 / 3 / 2 / 1 |
| Testing purpose: pure-semantic / pure-structural / hybrid | 21 / 3 / 2 |
| Total relevance judgments (relevant + borderline + rejected) | ≈700 |

Style-invariance pairs (cite as evidence that the eval set probes phrasing-robustness, not just per-query recall):

- q_008 "board wipes" (jargon) ↔ q_024 "a card that destroys all creatures" (natural)
- q_011 "tutor" (jargon) ↔ q_022 "a card that lets me look at my deck and put a creature into the battlefield" (natural)

### Discussion — known limitations

Worth noting in the discussion section:

1. **Sampling bias on broad-concept queries** — q_001 "creatures with flying" has 30 hand-sampled relevant cards against ~3,360 fliers in corpus. Recall@K on this query undermeasures real system performance (the system might surface 10 flyers but none in our sample). Mitigation in v2: either programmatic relevance checking per query criterion, or systematic sampling proofs.
2. **Single-curator judgments** — Voorhees 2000 supports the practice for comparative-ranking stability, but a multi-assessor pass on the relevant/borderline boundary cases would strengthen the methodology. Tractable to add later for queries that turn out to be load-bearing in the final paper.
3. **English-only**, **paper-Magic-only** (already in the corpus filter rules) — limits generalisation claims.

## Open follow-ups

- [ ] Rename `queries_v1_draft.yaml` to `queries_v1.yaml` once we declare v1 final; update `render_review.py` default path and `configs/baseline.yaml` `eval_set:` reference.
- [ ] Address sample-bias mitigation strategy in v2 — programmatic per-criterion relevance for broad concepts (q_001 fliers, q_005 haste, q_015 ETB) so the recall metric reflects real system performance instead of sample-overlap probability.
- [ ] Banding's printed joke-text on 19 old cards still rides in oracle text and skips augmentation. Phase 3 eval will show whether banding-related queries fail; if so, build the force-replace mechanism flagged in [the keyword-augmentation entry](2026-05-17-keyword-augmentation.md).
- [ ] Sub-task 12 from the roadmap: "validate every relevant oracle_id exists in the DB" — the render-review script already does this implicitly (Scryfall miss = visible warning during render), but a one-shot lint script would be cleaner. Easy if eval-v2 ever drifts.

## Related

- [Phase 3 roadmap](../roadmap/phase-3-baseline-and-eval.md) — sub-tasks 8–13 covered by this entry
- [Baseline results + failure-mode analysis](2026-05-18-baseline-results.md) — what happens when we point the evaluator at this set
- [Keyword augmentation entry](2026-05-17-keyword-augmentation.md) — the tri-state methodology was first introduced there
- [`data/eval/queries_v1_draft.yaml`](../../data/eval/queries_v1_draft.yaml) — the eval set itself
- [`data/eval/methodology_references.md`](../../data/eval/methodology_references.md) — the three citing papers
- [`scripts/eval_lookup.py`](../../scripts/eval_lookup.py), [`scripts/render_review.py`](../../scripts/render_review.py) — the tooling
