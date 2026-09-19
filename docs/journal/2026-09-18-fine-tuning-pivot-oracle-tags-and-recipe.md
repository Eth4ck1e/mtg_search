# 2026-09-18 — Fine-tuning pivot: oracle tags landed, source review, training recipe

**Status:** decision + data + literature + control measurements. Tag pipeline built and run; `src/search.py` landed and the four base-embedder grid rows are logged (`experiment_runs` 11–14, §5a); training pairs built (§7 item 3); trainer drafted and smoke-tested, full run pending (§7 item 4).
**Follows:** `2026-09-18-hyde-prompt-v1-design-notes.md` (the 10-query test series and the 8B-vs-27B comparison that motivated this).
**Supersedes:** the "scrape Scryfall tags slowly under rate limits" plan discussed during the test series. No scrape is needed.

---

## 1. Why we are here

The 10-query HyDE test series exposed three failure families in the 8B rewriter: jargon knowledge gap, over-narrowing on keywords, and compositional attention drift. The 27B comparison showed the jargon gap is a domain-knowledge ceiling, not an architecture problem — the bigger model knew what "cantrip" and "sac outlet" meant, the smaller one guessed. Prompt engineering cannot close a knowledge gap; it can only route around it one example at a time.

The pivot: teach the domain to the **embedder** first, using Scryfall's community oracle tags as distant supervision. If the embedder maps "cantrip" and "draw a card when this resolves" to the same region, the HyDE stage no longer has to translate jargon into rules text — it only has to extract filters and pass the concept through. That shrinks the prompt, shrinks the few-shot set, and shrinks the model we need. LoRA on the HyDE model stays on the table as a second step if the rewriter's *structural* failures (filter over-narrowing, attention drift) persist after the embedder is fixed.

Order of operations agreed: embedder first, direct fine-tune. HyDE LoRA second, only if needed.

## 2. Scryfall oracle tags — what changed and what we built

### 2.1 No scrape required

Scryfall now publishes oracle tags as an **official daily bulk-data file** (`type: oracle_tags`, documented at https://scryfall.com/docs/api/tags, listed in `GET /bulk-data`). This replaces the plan to page through `otag:` searches at 1 req/s (~5,200 requests, ~90 minutes) with a single 6 MB download from `data.scryfall.io`, which Scryfall's rate-limit page explicitly exempts from limits. The only rate-limited request we make is the one metadata call to `api.scryfall.com`, sent with the required `User-Agent` and `Accept` headers already in `src/config.py`.

Policy check (all from https://scryfall.com/docs/api and `/docs/api/rate-limits`, fetched today):
- "performing research" is a named permitted purpose under the Fan Content Policy.
- "If you need to rapidly look up card names ... you must use the bulk data files." We do.
- Tagger data has no separate license; it falls under the general Scryfall data terms. The paper attributes Oracle text to Wizards of the Coast and tags to "Scryfall Tagger (community-maintained)", cites the bulk-file date, and does not redistribute the tag file.
- Tag slugs and labels are documented as mutable. We join on the stable `id` UUID everywhere.

### 2.2 File shape (2026-09-18 bulk, `updated_at` 2026-09-18T21:00:35Z)

| Measure | Value |
|---|---|
| Oracle tags | 4,551 |
| Taggings (tag, oracle_id) | 236,170 |
| Distinct tagged oracle_ids | 36,238 |
| Hierarchy | multi-parent DAG, 921 roots, max depth 6, no dangling refs |
| Tags with a description | 1,210 |
| Tags with aliases | 729 |
| Weight field | 99.7% `median` — effectively unused |

The hierarchy matters. `removal` has zero direct taggings and 54 descendants; Scryfall's `otag:removal` search returns the union (6,737 cards, verified against the live API today). Our closure view reproduces that number exactly.

### 2.3 Coverage against our corpus

| Measure | Value |
|---|---|
| Taggings whose oracle_id is in `cards` | 205,262 / 236,170 (86.9%) |
| Corpus cards with ≥1 direct tag | 30,928 / 31,124 (**99.4%**) |
| Corpus cards with ≥3 direct tags | 28,949 |
| Tags with ≥5 corpus cards | 3,022 |
| Tags with ≥20 corpus cards | 1,033 |
| Tags with ≥100 corpus cards | 369 |
| Tags with a description AND ≥5 corpus cards | 1,015 |
| (tag, card) pairs over tags with ≥5 cards | 202,903 |

The 13% of taggings outside the corpus are tokens, extras, and Un-set cards we filter at ingest. Nothing to recover there.

### 2.4 What was built

- `scripts/download_scryfall.py --dataset oracle-tags` — the existing downloader generalised with a dataset flag instead of a second script. Same `.partial` staging, SHA-256, run log.
- `src/db/migrations/0003_oracle_tags.sql` — `oracle_tags` (id, slug, label, description, aliases, parent_ids, tagging_count, bulk_updated_at), `card_tags` (tag_id, oracle_id, weight, annotation), and the `oracle_tag_closure` recursive view. Two tables rather than a column on `cards`: many-to-many, different refresh cadence, and 13% of taggings reference cards we don't hold. No FK to `cards` for that reason; coverage is measured and logged instead.
- `scripts/ingest_tags.py` — full replace in one transaction (tags are derived data with no local edits), logs coverage to `logs/ingest_tags/`.

Tags are **not** embedded and **not** used by the SQL pre-filter. They are training signal only. CLAUDE.md §5's "embed Oracle text only" holds.

## 3. Source review — what the literature supports

Two passes: the seven PDFs already in `docs/sources/` that touch embedder training, then a search for 2025–2026 work matching our setup (bounded corpus, no relevance labels, tag-derived supervision, small encoder, Apple Silicon compute). Fourteen new PDFs fetched; `docs/sources/README.md` has the annotated entries.

One correction: the file we had labelled Nigam et al. 2020 (Amazon) is actually **Choi et al. 2020** (Home Depot, arXiv:2008.08180). Renamed on disk; README fixed. Nigam is only cited inside it.

### 3.1 Consensus from the existing sources

- **Loss:** InfoNCE with in-batch negatives, everywhere (DPR, Nomic, GTE/BGE/E5, Flipkart 2026). Nomic's own supervised stage: cosine similarity, 7 hard negatives, batch 256, AdamW lr 2e-5, wd 0.01, grad clip 1.0, 400-step warmup, **one epoch** — and an explicit note that "training for multiple epochs hurts performance" (Nussbaum et al. §4.2.5).
- **Weak positives work.** DPR's distant-supervision positives cost ~1 point vs gold (App. A). SBERT's Wikipedia-section triplets beat the prior SOTA with no human labels (§4.4). Choi trains entirely on click-thresholded logs. Tag → card pairs are the same species.
- **One hard negative is the cheapest large win** (DPR Table 3: +10 points top-5), with diminishing returns past 7 (Nomic).
- **Sample efficiency:** DPR beats BM25 at 1,000 pairs. We have ~200k.
- **Full fine-tuning at this scale.** Every 100–300M model in these sources is fully fine-tuned. LoRA appears only on 4–7B backbones.
- **Prefixes stay on during training** (Nomic §4.2.4). `search_query: ` on the anchor, `search_document: ` on the card.
- **Fine-tuning is distribution-specific.** Nomic's BEIR fine-tune hurt on LoCo; BEIR Finding 1 says in-domain gains don't predict out-of-domain. Nobody in this set measures forgetting — they just report it when it bites.

### 3.2 What the 2025–2026 papers add

Ranked by fit. Full entries in the sources README.

1. **CustomIR (Paull 2025)** — closest recipe: known corpus, LLM-synthesised queries, BM25-mined negatives, LLM verification of negatives. Measured **20–32% of mined negatives were actually relevant** before verification. That number is the argument for tag-aware negative filtering.
2. **Tamber et al. 2025** — plain InfoNCE fine-tuning *degraded* strong 110–137M base models (BGE-base on SciFact 74.1→72.1). Strongest evidence that naive MNRL on a good base can go backwards. Also: 56k synthetic queries matched 56k human queries.
3. **Murtaza et al. 2026 (EMNLP Industry)** — near-identical setup: 0.6B retriever, ~15k synthetic pairs, ~34k-item catalog. Aggressive fine-tuning dropped out-of-distribution recall 0.85→0.65; an embedding-anchor regulariser (L2 to the frozen encoder's outputs) restored it with ~14% in-distribution gain intact.
4. **BioHiCL (Lan et al. 2026, ACL)** — turns hierarchical community tags (MeSH) into contrastive supervision. Positives by label overlap, negatives by *zero* label overlap. LoRA ≈ full on bge-base. This is our situation with a different taxonomy.
5. **ChEmbed (2025)** — the only paper fine-tuning the Nomic lineage. Nomic's own lr 2e-5. Notable: **in-batch-only beat mined hard negatives** on a jargon-heavy chemistry corpus. So the –hard-negative ablation is mandatory, not optional.
6. **CoHyDE (2026)** — directly answers "does HyDE still help after fine-tuning": yes for vague/informal queries, and the fine-tuned encoder wins on well-formed ones. Both stages stay; the 2×2 (HyDE on/off × fine-tuned on/off) is the ablation table the paper needs.
7. **Gwon et al. 2025 (CIKM)** — 3B–8B open LLMs are as good as proprietary generators for synthetic queries. Licenses our local Llama 3.1 8B for query generation.
8. **NV-Retriever (2024/25)** — positive-aware mining (drop negatives scoring within a margin of the positive). This is exactly `mine_hard_negatives(relative_margin=0.05)` in Sentence Transformers.
9. **Wang et al. 2026 (KDD)** — multi-positive objectives. Sampling one positive per step is a defensible baseline when a tag has hundreds of positives.
10. **Pande et al. 2025** — cautionary: every fine-tuning variant of all-MiniLM-L6 fell below the untouched baseline. The frozen base is a permanent ablation row.

### 3.3 Sentence Transformers state (Sept 2026)

v6.1.0 current. Pin ≥5.7 — that release fixed silent gradient corruption in the cached losses. `SentenceTransformerTrainer` with `prompts={"anchor": "search_query: ", "positive": "search_document: "}` handles the Nomic prefix contract. `CachedMultipleNegativesRankingLoss` gives a large effective batch on a memory-bound Mac. `mine_hard_negatives` implements NV-Retriever's margin rule. `InformationRetrievalEvaluator` + `NanoBEIREvaluator` cover in-domain and forgetting probes.

## 4. The hypothesis, and what it fixes about the pipeline

*(Restated by Mitchell 2026-09-18 after the first draft of this entry lost it to context compaction. This section governs §4a.)*

**Hypothesis.** Fine-tuning the embedder on tag-derived pairs teaches it MTG jargon, mechanics, and keyword semantics. Once the embedder knows what "cantrip" means, the HyDE stage no longer has to translate jargon into hypothetical rules text. Its rewrite job gets lighter and its prompt gets simpler: fewer rules, fewer few-shot examples, and a focus on **filter extraction**. Where the user's phrasing matches an existing tag context, HyDE should be guided to **normalise toward tag vocabulary** ("board wipe" → "sweeper", "sac outlet" → "sacrifice outlet") rather than expanding the query to cover the large variance of rules text those tags represent. The tag vocabulary becomes a shared language between Stage 1 and Stage 3.

**What Stage 3 embeds after the pivot.** Not a hypothetical card. The query-side text becomes, in priority order: (1) tag-vocabulary concepts HyDE mapped the query to; (2) the user's own phrasing, passed through, when no tag fits; (3) hypothetical rules text only as a fallback. This is why the training anchors in §4a are tag labels and natural-language queries, not pseudo-cards.

**What this changes in the prompt.** The HyDE output schema gains a concepts field (tag-vocabulary strings), `hypothetical_card` becomes optional, and the few-shot set is rebuilt around filter extraction plus concept normalisation. The simplification is measurable — count rules and examples in `hyde_v1.yaml` vs the post-pivot prompt — and is itself a paper result.

**The trap to stay out of.** Once HyDE can emit tag labels, the shortcut is to filter `card_tags` directly and skip the embedder. That is Scryfall's `otag:` search rebuilt locally: it only works for concepts the community has tagged and cards the community has tagged. Routing the concept through the tuned embedder is what generalises to untagged cards and to phrasings no tag covers. This distinction is a paper argument and is why CLAUDE.md keeps tags as training signal only, never a filter column. A direct-tag-filter row is a legitimate *ablation* (it bounds what the embedder adds over a lookup), not a system configuration.

**The accessibility framing (Mitchell, 2026-09-18).** The paper's primary claim is not "beats Scryfall on results." Scryfall's `otag:` search plus its filter syntax already lets a power user get these results. The claim is that the cascade gets a novice **the same results from plain language** — Stage 1 absorbs the query-syntax expertise, Stage 2 applies it, and the user never learns `otag:`, `c<=`, `mv<=`, or `t:`. Parity with an expert-crafted Scryfall query is therefore a *win*, not a tie, because the bar to reach it dropped. Two measurements follow:
- *Result parity* — objective. For each eval query, the result-set overlap between the cascade and the LLM-crafted expert Scryfall query (the M5 comparator already planned). Recall against the expert set is the number; the tri-state judgments still catch cases where the expert query itself was too narrow.
- *Ease of use* — a user study is out of scope this term, so a proxy: the syntactic complexity of the Scryfall query a user would have needed (operator count, distinct operator types, presence of `otag:`) versus the plain-language query they typed. Report per query alongside parity. If the results match and the complexity gap is large, the accessibility claim stands on measurement rather than assertion.

**Experimental grid.** Stage 1 mode {hypothetical text (v1 prompt), tag-normalised concepts (v2 prompt), raw pass-through} × embedder {base, tuned}, plus –SQL. Every cell is a retrieval run over the 26-query eval set logged to `experiment_runs`. The base-embedder rows are the control (see §5).

## 4a. Recommended recipe

Each line cites what backs it. Decisions Mitchell has confirmed are marked **[confirmed]**.

**Training pairs — anchors chosen to match what the post-pivot Stage 3 will actually see (§4). [confirmed 2026-09-18]**
1. *Tag-anchored (primary).* Anchor = tag label, and separately each alias, so the interlingua strings themselves are trained ("sweeper", "board wipe" via alias where Scryfall records one). Positive = one tagged card per step (Wang 2026 Rand1LH). Walk the closure so `removal` anchors draw from all 54 descendants.
2. *Tag descriptions.* Anchor = the tag's description sentence (1,015 tags have one and ≥5 corpus cards). Short natural language, one step removed from the label — the bridge between jargon and pass-through phrasing.
3. *Synthetic user queries.* 3–4 per card from the local Llama 3.1 8B (Gwon 2025), forced across our six eval categories (Tamber: type mix; Yuksel & Kamps: type distribution must match the test queries). Round-trip filter: keep a query only if the frozen base ranks its card in the top 20. Covers the pass-through case where no tag fits.
4. *Doc-like anchors: none in v1.* Hypothetical text is fallback-only after the pivot. The NanoBEIR forgetting probe (§4a eval (c)) is the check that dropping them didn't cost rules-text → rules-text matching; add a small slice only if it does.

**Negatives.** In-batch by default, with **tag-aware batching**: no two anchors in a batch share a tag (structural false-negative guard, BioHiCL's zero-overlap rule; Nomic's one-source-per-batch is the same instinct). Hard negatives as a separate ablation: `mine_hard_negatives(relative_margin=0.05, num_negatives=3)` then drop any negative sharing a tag with the anchor. ChEmbed says mined negatives may hurt here; CustomIR and DPR say they help. Measure.

**Loss.** `CachedMultipleNegativesRankingLoss`, scale 20 (τ=0.05, cosine — stay in Nomic's lineage; do not switch to DPR's dot product). `CachedGISTEmbedLoss` with the frozen base as guide is the ablation for guide-based false-negative masking.

**Full fine-tune vs LoRA. [confirmed 2026-09-18: full fine-tune primary]** lr 2e-5 (Nomic's own supervised setting, ChEmbed), 1 epoch, 5% warmup, wd 0.01, AdamW. Every sub-300M paper does it this way, Nomic included, and it matches the "direct on embedder" decision. Secondary, only if the forgetting probe drops: LoRA r=16 + embedding-anchor regulariser (Murtaza 2026), or weight-interpolate the tuned and base checkpoints (Khattab 2026). Nomic's custom `nomic-bert` modeling code may complicate PEFT target-module naming; full fine-tune sidesteps that.

**Epochs.** 1, with early stopping on a held-out synthetic-query split. Nomic and Gill 2025 both stop at one; Tamber's 30 epochs used 4k batches and distillation, not applicable. At ~200k pairs one epoch is ~800 steps at batch 256 — enough.

**Held-out tags.** Reserve ~15% of tags (stratified by size) entirely from training. Retrieval on those tags' cards, queried by their label, is the **jargon-generalisation probe**: did the model learn "jargon means a function" or just memorise 3,000 vocabulary items? Report both.

**Evaluation protocol per run → `experiment_runs`:**
- (a) 26-query eval set, Recall@K / MRR, tri-state, per category.
- (b) Held-out-tag probe.
- (c) `NanoBEIREvaluator` on 3–4 small BEIR sets as the forgetting probe. ChEmbed and Murtaza only caught regression because they measured it.
- (d) The grid from §4: Stage 1 mode {hypothetical, tag-normalised, pass-through} × embedder {base, tuned}, plus –SQL and the direct-tag-filter ablation. Frozen base rows are permanent.
- (e) The same 10 HyDE test queries on the **8B** rewriter, compared on **retrieved cards** (not rewriter JSON — the rewriter doesn't change when the embedder does) before and after tuning, with the 27B rewriter results as the reference ceiling for the Stage 1 half.

**Disclosure.** Training on tag "cantrip" and evaluating on eval query "cantrips that dig" is in-distribution by design — that is the point of the intervention — and the paper says so. The held-out-tag probe is the honest complement.

## 5. Ordering — one concern, one fix

Fine-tuning before the cascade has run on the eval set leaves no before-number. The 10-query series inspected rewriter *output* — JSON filters and hypothetical text, judged by eye. It never retrieved a card: nothing was embedded, no SQL filter ran, no top-K came back, nothing was scored against the eval set. Fine-tuning changes the **embedder**, and the rewriter's JSON is identical before and after, so the effect is only visible one stage later, in which cards come back. There is also no retrieval number for the current system at all — `scripts/evaluate.py` still runs raw-query embedding-only, and the only logged result is the fabricated May row.

Fix: draft `src/search.py` (Stage 2+3 orchestration), run the eval set once with the base embedder, log the row. That row is the control every fine-tuning claim rests on, and the same module runs every cell of the §4 grid. It is a day of work and nothing in §4a waits on it — the training-pair builder can be written in parallel.

## 5a. Stage 2+3 orchestration landed (`src/search.py`) — first dry-run numbers

Built the same day, after Mitchell confirmed the ordering. `src/search.py` compiles rewriter filters to a parameterised WHERE clause, embeds the query-side text with the Nomic query prefix, and runs pgvector cosine search inside the filtered set, deduped by `oracle_id`. `scripts/evaluate.py` now runs every config through it and logs Stage 1 output, the WHERE clause, candidate count, and per-stage timings per query. Four configs cover the base-embedder cells of the §4 grid; the old `baseline.yaml` and `scripts/test_search.py` are gone.

**Filter semantics encoded (each was a test-series decision):**
- Colour ops: `contains_any` (default) admits colourless cards; `exactly` is the only op that excludes them. Identity mirrors the colour op.
- `types` are ORed ("instants or sorceries"); `subtypes` are ANDed ("elf warriors"). The first smoke test ANDed types and matched zero cards — regression test added.
- `keywords` filter is **off by default** (selective strictness, the Prowess/Trample over-narrowing). `FilterPolicy(keywords=True)` turns it on for the ablation.
- Power/toughness compare only when the text column parses as an integer (`*`, `X` are skipped).
- Every value is a bound parameter; only whitelisted operators reach SQL text. Out-of-contract values raise `FilterError`, which the searcher logs as a warning and drops the filters rather than returning zero silently.
- The embedding column is pinned to `settings.embedding_version`, so a tuned checkpoint (new `EMBEDDING_MODEL`, re-embed) is searched in isolation from the base vectors.

**Logged results, 26-query eval set (v1-draft), base Nomic v1.5, 8B rewriter, v1 prompt, keywords filter off.** Mitchell approved logging with the current policy; precision@10 was added to `src/eval/metrics.py` first because recall@10 is capped by relevant-set size (q_018 has 34 relevant cards, so a perfect top-10 scores 0.29).

| `experiment_runs.id` | Config | P@10 | R@10 | MRR | p50 latency |
|---|---|---|---|---|---|
| 14 | `raw_dense` (no rewriter, no filter) | 0.031 | 0.017 | 0.067 | 83 ms |
| 13 | `cascade_passthrough` (HyDE filters, raw query embedded) | 0.050 | 0.028 | 0.130 | 985 ms |
| 12 | `cascade_hyde_v1_nosql` (–SQL ablation) | 0.085 | 0.036 | 0.269 | 967 ms |
| 11 | `cascade_hyde_v1` — **control** | 0.112 | 0.050 | 0.294 | 978 ms |

Each stage adds, monotonically. On the base embedder the hypothetical text contributes more than the filter (–0.062 vs –0.027 P@10 when removed) — expected before fine-tuning, since the untuned encoder depends on Stage 1 to translate jargon. Per-category table and reading are in the paper draft §5.3.

**Note on ids:** the `experiment_runs` table was rebuilt on 2026-09-11, so `id=13` now refers to a real row (`cascade_passthrough`), not the fabricated May baseline. The paper cites rows by config name + date, never by bare id.

**What the per-query trace shows:**
- Six queries got no `hypothetical_card` and fell back to embedding the raw query (q_001 flying, q_005 haste, q_016 mana dorks, q_020 red creatures <3, q_021 1-mana instants, q_023 spell-triggered growth). Two of those (flying, haste) are *user-explicit* keywords that the off-by-default keywords policy then ignored → R@10 = 0. This is the selective-strictness case made concrete: a keyword the user typed should be strict. Cheapest rule to test: apply the keywords filter when the rewriter returned filters but no hypothetical text (it judged the query purely structural). **[decide]**
- Two zero-candidate queries, both Stage 1 jargon failures, not search bugs: "mana dorks" → invented subtype `Dork` + P/T = 1/1; "red pingers" → `types: [Instant, Sorcery]` plus P/T filters (pingers are creatures). Exactly the jargon-gap family the pivot targets.
- Stage 1 is ~1 s of the ~1.04 s p50; Stage 2+3 together are under 100 ms at this corpus size.

## 6. CLAUDE.md revisions

§5 (do not fine-tune the embedder on keyword definitions), §6 (fine-tuning deferred to M6), and §11 (anti-suggestion) all encode the pre-pivot position. Revised today to: reminder-text augmentation stays the corpus-side lever; tag-derived contrastive fine-tuning is the query-side lever, motivated by the 2026-09-18 test evidence and gated on the base-embedder control run in §5 above. Hand-written definition dictionaries remain banned.

## 7. Next actions

1. ~~Confirm anchors and full-vs-LoRA~~ — both confirmed 2026-09-18 (§4, §4a). **[Mitchell]** ordering in §5 still open.
2. ~~`src/search.py` + base-embedder eval run (control row)~~ — done, rows 11–14 (§5a).
3. ~~`scripts/build_training_pairs.py`~~ — done for tag label/alias/description anchors: 207,062 train pairs over 2,729 tags, 482 held-out tags (33,862 probe pairs), `data/training/pairs_v1.jsonl` + `manifest_v1.json`. Synthetic-query anchors are a separate script (needs the LLM server; hours). Note: the random stratified hold-out withheld `burn` — so q_004 ("burn spell that deals 3 damage") becomes a genuine generalisation test rather than in-distribution; keep the seed and disclose.
4. ~~`scripts/finetune_embedder.py`~~ — drafted and smoke-tested on MPS (5 steps, batch 64, 2k pairs, ~32 s): full fine-tune, `CachedMultipleNegativesRankingLoss` (scale 20, mini-batch 32), prompts on, lr 2e-5, 5 % warmup, wd 0.01, one epoch, grad clip 1.0, `max_seq_length` 256. Tag-disjoint batch sampler (no shared anchor tag; no positive carrying another pair's anchor tag; constraint-starved tail packed loosely and counted). Probes: in-training dev on train tags; post-training held-out-tag probe over the full corpus, base vs tuned; optional NanoBEIR (`--nanobeir`). One `experiment_runs` row per run. **Measured throughput ~10 pairs/s on the M3 → the 207k-pair epoch is ~6 h.** Not yet run in full. Requires ST ≥ 5.7 (have 6.0), `datasets`, `accelerate` (added to pyproject).
5. Re-embed corpus with the tuned checkpoint under a new `embedding_version`; run (d) and (e).
6. `prompts/hyde_v2.yaml` — concepts field, optional `hypothetical_card`, few-shot rebuilt around filters + tag normalisation. Count rules/examples vs v1 for the paper.
