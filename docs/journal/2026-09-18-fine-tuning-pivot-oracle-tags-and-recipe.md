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

## 5b. First fine-tuning run — `nomic-mtg-v1` (2026-09-18 evening)

Run as specified in §4a on the Mac (M3, MPS): 207,062 pairs, batch 256 (cached mini-batch 32), lr 2e-5, one epoch = 809 steps, tag-disjoint batching, prompts on. `experiment_runs.id = 19`. Checkpoint at `models/nomic-mtg-v1/` (547 MB safetensors, gitignored).

**Throughput correction.** The 5-step smoke run predicted ~10 pairs/s (≈6 h); that was startup overhead. Steady state was **~49 pairs/s → 71 min of training**, ~1 h 45 min wall including the pre-training probe, the full-corpus held-out probe (base + tuned), and NanoBEIR. Two to three runs a day on the Mac is realistic; the 5060 Ti workstation is a convenience, not a requirement.

**Loss:** 4.2 (≈ ln 256, chance) → 2.64 at step 100 → 1.80 at step 400 → 1.46 at step 805. Still falling at the end of the epoch, but the dev probe flattened after step ~600 (ndcg@10 0.523 → 0.536 over the last 200 steps), so one epoch is close to the knee. A second epoch is an ablation, not a default (Nussbaum: multiple epochs hurt).

**Probe results (base → tuned):**

| Probe | ndcg@10 | mrr@10 | map@10 | recall@10 | recall@100 |
|---|---|---|---|---|---|
| Dev (100 *training* tags, 6.2k-card corpus) | 0.195 → **0.536** | 0.249 → **0.600** | 0.150 → **0.467** | 0.097 → 0.229 | — |
| **Held-out (482 tags never seen, full 31k corpus)** | 0.114 → **0.242** | 0.193 → **0.325** | 0.079 → **0.187** | 0.046 → 0.114 | 0.164 → **0.327** |

- The dev probe is in-distribution and expected to jump; it says the loss is doing what it should.
- **The held-out probe is the result.** On tags the model never saw, queried by label alone against the whole corpus, every metric roughly doubled. The model did not memorise 2,729 vocabulary items; it learned that a short functional phrase maps to a region of rules text. This is the generalisation evidence §4 said the paper needs, and it is what separates the tuned embedder from a tag lookup.
- Absolute held-out numbers are modest because many held-out tags are narrow (`bounceland`, `blood-artist-ability`) and the corpus has 31k distractors; recall@100 doubling is the more honest read of the ranking shift.

**NanoBEIR (forgetting probe).** Base measured separately after the run (`models/nomic-mtg-v1/nanobeir_base.json`).

| NanoBEIR ndcg@10 | base | tuned | delta |
|---|---|---|---|
| SciFact | 0.731 | 0.736 | +0.005 |
| FiQA2018 | 0.485 | 0.441 | −0.044 |
| NFCorpus | 0.326 | 0.309 | −0.017 |
| mean | 0.514 | 0.495 | −0.019 |

Mild, uneven forgetting: two points mean, four on FiQA (financial QA, the most "query-like" of the three). Within what Murtaza et al. and ChEmbed report for unregularised fine-tunes and far from the 0.85→0.65 collapse Murtaza saw with aggressive settings. Acceptable for a domain-specialised deployment; the embedding-anchor regulariser (or weight interpolation with the base) is the ablation to run if a later recipe change pushes this past ~5 points. **Every future run reports this table**, base column fixed.

### 5c. Tuned embedder through the cascade — grid rows 20–23 (2026-09-18, late)

Corpus re-embedded with `models/nomic-mtg-v1` (`embedding_version = models/nomic-mtg-v1|preproc=v1`), same four configs, same 8B rewriter and v1 prompt, keywords filter off.

| Config | Stage 1 text embedded | P@10 base → tuned | MRR base → tuned | rows |
|---|---|---|---|---|
| `raw_dense` | user query, no filters | 0.031 → **0.069** | 0.067 → **0.102** | 14 → 20 |
| `cascade_passthrough` | user query + HyDE filters | 0.050 → **0.081** | 0.130 → **0.206** | 13 → 21 |
| `cascade_hyde_v1` (control) | hypothetical rules text + filters | **0.112** → 0.089 | **0.294** → 0.270 | 11 → 22 |
| `cascade_hyde_v1_nosql` | hypothetical rules text, no filters | 0.085 → 0.077 | 0.269 → 0.196 | 12 → 23 |

**Reading.** The tuned embedder helps exactly where it was trained to and hurts exactly where §4a item 4 warned it might:

- **Modes that embed the user's own words improved.** Raw dense more than doubled P@10; pass-through went 0.050 → 0.081 P@10 and 0.130 → 0.206 MRR. The encoder now maps short player phrasing closer to rules text without any rewriter help.
- **Modes that embed hypothetical rules text got worse.** The control dropped 0.112 → 0.089. Training only on short anchors (labels, aliases, descriptions) pulled the query-side representation toward short functional phrases; a paragraph of Wizards-style rules text used *as a query* is now further from the document side than it was in the base model. This is the doc-like-anchor trade-off, measured.
- **Best single cell on the eval set is still base + hypothetical text** (0.112 / 0.294). Pass-through + tuned is second on MRR (0.206) and closes most of the gap on the jargon category (P@10 0.115 vs 0.131 for hyde/tuned; MRR 0.158 vs 0.263).
- **Per category (P@10):** natural 0.000 → 0.125 raw and 0.000 → 0.075 pass-through; jargon 0.054 → 0.077 raw, 0.062 → 0.115 pass-through, 0.146 → 0.131 hyde; constrained unchanged where the filter is present (it's Stage 2's category); fragmented still 0.000 in every cell — three queries no configuration touches.
- **Vocabulary gap, concretely.** `sweeper` was trained with aliases `wipe`, `boardwipe`, `mass removal`, `wrath of god` — and "board wipes" (pass-through, tuned) still scored 0, while the same query through hypothetical text scored 0.20. "flicker effects" is 0 in every cell although `flicker-creature` (137 cards) and `flicker-slow` (110) were both trained. Two follow-ups: (i) check how much of each eval query's relevant set is even inside the closure pool of the tag we'd expect HyDE to pick — if the judgments and the tags disagree, no embedder fixes it; (ii) the `boardwipe`-vs-"board wipes" miss says single-alias exposure is not enough; the synthetic-query anchors (§4a source 3) are what teach user phrasing.

**What this means for the hypothesis.** The pass-through row is *not yet* the hypothesis test, because the hypothesis was: HyDE normalises the query toward **tag vocabulary**, then the tuned embedder does the rest. Pass-through embeds raw user phrasing ("board wipes"), which is exactly what the tag-anchored training never showed the model. The cell that tests the hypothesis is **v2 prompt (concepts in tag vocabulary) × tuned embedder**, and it is not built yet. Two cheaper probes are available immediately and should run first tomorrow:
1. **Tag-label oracle.** Hand-map each eval query to the tag(s) a perfect v2 rewriter would emit (e.g. "board wipes" → `sweeper`), embed the label, search with the tuned model. This is the upper bound for v2 + tuned before any prompt work. If it beats 0.112 / 0.294, build v2. If it doesn't, the training data needs source 3 before v2 is worth writing.
2. **Mixed anchors run** (`pairs_v2`): add a doc-like slice (hypothetical-style anchors: the card's own rules text with names stripped, or one-sentence LLM paraphrases) at ~20–30 % so the hypothetical-text mode stops regressing. Then base + hyde vs tuned-v2 + hyde is a fair comparison, and the v2-prompt cell inherits an embedder that handles both input shapes.

**Latency.** Unchanged, as expected: p50 ~1.0–1.2 s with the rewriter, 88 ms without. The embedder swap costs nothing at query time.

### 5d. Tag-label oracle — the v2 ceiling, measured (2026-09-19, rows 26–29)

**What the oracle is.** The v2 prompt was going to make HyDE emit the query's concept in tag vocabulary ("board wipes" → `sweeper`) and let the tuned embedder do the rest. Before writing that prompt, we can measure its *ceiling* by playing the perfect rewriter by hand: for each eval query, a human picks the tag a flawless v2 would emit, the harness embeds that tag's **label** instead of the user's words (filters still come from the v1 rewriter), and we score against the eval judgments. No prompt can beat its own oracle, so if the oracle loses to the current control, v2-as-label-emitter is not worth building. Mapping: `data/eval/tag_label_oracle_v1.yaml` (21 of 26 queries mapped; keyword/structural/ETB queries have no functional tag and fall through to the raw query). Configs: `configs/oracle_tag_label{,_nosql}.yaml`; `Searcher.search(embed_text=...)` is the override hook.

**Eval-judgment coverage first.** Before trusting the oracle, checked what fraction of each query's *relevant* set lies inside the chosen tag's closure pool: 93–100 % for every jargon query (sweeper 75/76, removal 71/72, ramp 26/28, flicker 30/32, tutor 37/37, recursion 32/33, mana-dork 23/23). So the tags *can* reach the judgments; the question is whether the embedder finds them from the label.

**Result.**

| Row | Embedder | Filters | P@10 | MRR |
|---|---|---|---|---|
| 28 | base | HyDE v1 | 0.065 | 0.185 |
| 29 | base | none | 0.035 | 0.100 |
| 26 | **tuned** | HyDE v1 | **0.073** | **0.153** |
| 27 | tuned | none | 0.081 | 0.122 |
| 11 | base | HyDE v1 + hypothetical text (control) | 0.112 | 0.294 |
| 21 | tuned | HyDE v1 + raw user query (pass-through) | 0.081 | 0.206 |

**The oracle loses to the control and to plain pass-through.** Embedding the tag label — even with the embedder trained on exactly those labels — scores 0.073 / 0.153 against the eval judgments. The pass-through row that embeds the user's own words scores higher (0.206 MRR). The base+hypothetical control is nearly double.

**Why: a tag is a category, the judgments are its archetypes.** Per-query oracle P@10 (tuned, no filters) against the tag's pool size:

| Tag | pool | P@10 | | Tag | pool | P@10 |
|---|---|---|---|---|---|---|
| counterspell-free | 13 | 0.20 | | tutor | 1,120 | 0.10 |
| extra-turn | 53 | 0.60 | | removal-artifact | 1,105 | 0.00 |
| fetchland | 53 | 0.40 | | cast-trigger-you | 1,175 | 0.00 |
| wheel | 137 | 0.20 | | draw-engine | 1,497 | 0.00 |
| flicker | 179 | 0.10 | | recursion | 2,090 | 0.00 |
| mana-dork | 414 | 0.20 | | ramp | 2,136 | 0.00 |
| counterspell | 513 | 0.10 | | draw | 4,043 | 0.00 |
| sweeper | 871 | 0.10 | | removal | 5,968 | 0.00 |

Narrow tags work; broad tags fail, monotonically. "Ramp" has 2,136 tagged cards and the eval set judges 28 of them relevant — the canonical ramp spells. The label "ramp" embeds to the *centre of the pool*, so the top 10 are ten arbitrary ramp-tagged cards (a land-fetching creature, a mana-doubling enchantment, a treasure maker), almost none of which are the 28 archetypes a player means. The hypothetical text "Search your library for a basic land card and put it onto the battlefield" is more specific than the category and lands on the archetypes. This is exactly the held-out probe's blind spot: there, relevant = the whole pool, so category-level retrieval scores well; against human judgments, category-level retrieval is too coarse.

**Consequences for the hypothesis.** The strong form — HyDE emits a tag label, the embedder does the rest — is not supported. A label carries the *category* but drops the *specificity* that both the hypothetical text and the user's own phrasing carry. Three things survive intact:
1. The tuned embedder's real win is on **user phrasing** (pass-through MRR 0.130 → 0.206, raw 0.067 → 0.102). The right query-side input is the user's words, or a HyDE output that keeps their specificity — not a category name.
2. The hypothetical-text regression (control 0.112 → 0.089) is a **training-data** problem, fixable with doc-like anchors (`pairs_v2`), not evidence against HyDE.
3. The prompt-simplification hypothesis becomes: **HyDE gets shorter because the embedder no longer needs a full paragraph of rules text** — a one-clause hypothetical ("search for a basic land, put it onto the battlefield") or the user's phrasing plus a concept word should suffice. That is testable: shorten `hypothetical_card` to one sentence in v2 and measure tokens and accuracy.

**Revised next steps (replaces §7 items 6+):**
- `pairs_v2`: keep label/alias/description anchors, add **doc-like anchors** (one-sentence rules-text paraphrase per card, LLM-written, or the card's first sentence) at ~30 %, and **synthetic user queries** (source 3) once the LLM batch job exists. Retrain; expect the control row to recover and pass-through to hold.
- `hyde_v2`: same two-field contract, but `hypothetical_card` capped at one sentence and the rules/examples trimmed; concepts field **dropped** on this evidence. Measure output tokens and Stage 1 latency alongside P@10/MRR.
- Direct-tag-lookup ablation is now more interesting, not less: it will fail on broad tags for the same reason, and that failure is the paper's argument for embedding over lookup.

DB state after this section: `cards.embedding` holds the **base** vectors again (re-embedded for rows 28–29).

**Schema note.** `cards.embedding` holds one vector per row, so re-embedding with the tuned model *replaces* the base vectors; switching back means re-embedding (~minutes). Fine for now — base rows are logged — but if the ablation grid grows past two or three checkpoints, move embeddings to a `card_embeddings (oracle_id, face_index, embedding_version)` table so checkpoints coexist and the searcher just changes its version pin. Candidate migration 0004.

### 5e. CORRECTION to §5d — the oracle's "misses" are unjudged category members (2026-09-19)

§5d concluded the tag-label oracle fails on broad tags. Checking that claim against tag membership instead of the eval judgments reverses the reading. For each mapped query, how many of the top 10 are members of the chosen tag's closure pool (no filters):

| Row | judged-relevant @10 | **in tag pool @10** |
|---|---|---|
| Oracle, **tuned** (27) | 0.10 | **0.95** |
| Oracle, base (29) | 0.04 | 0.35 |
| Control, base + hypothetical text (11) | 0.12 | 0.56 |

With the tuned embedder the label retrieves genuine category members almost every time: `ramp` 10/10 (base 0/10), `recursion` 10/10 (base 0/10), `flicker` 10/10 (base 0/10), `tutor` 10/10 (base 1/10), `sweeper` 10/10 (base 2/10). `burn` was a **held-out** tag and still scores 10/10. The only weak cell is `counterspell-free` (3/10), a 13-card pool.

**So the mechanism is not "too few samples per tag" and not "label lands on junk".** The model learned the categories. The eval set judges ~30 archetypes per query out of pools of 1,000-6,000 legitimate members, and scores every unjudged member as a miss. This is the annotation-hole effect BEIR documents (Thakur et al. 2021, §6: ANCE went from below BM25 to above it once holes were judged). Our judgments were built from lexical Scryfall lookups, so they favour cards whose text matches the obvious phrasing, which is also what hypothetical text retrieves. The eval set is biased toward the control.

**What is and is not established.**
- Established: tuned + label retrieves tag members at 0.95 vs 0.35 base. Large, real, and the held-out `burn` case shows it is not pure memorisation.
- Caveat: 20 of 21 mapped tags were in training, and "is tagged X" is the training objective, so this is mostly in-distribution. Tag membership is community-curated, not the same as "what a player wants".
- Still true from §5d: a label returns *arbitrary* members, not the *best-known* ones. A player typing "ramp" probably wants Cultivate before an obscure land-fetching creature. That is a ranking-within-category question (popularity/EDHREC rank as a tiebreak), not an embedding failure.
- Not established: that label-emitting v2 beats the control for users. Neither metric settles it; P@10-judged favours the control by construction, pool-membership favours the oracle by construction.

**Revised plan.**
1. **Fix the measuring stick first.** Judge the holes: pool the top-10 from every logged row per query, have Mitchell (or an LLM judge with Mitchell spot-checks, task #25) label the unjudged cards, freeze as eval v2. Until then, report both metrics side by side and say why.
2. The concepts field is **back on the table** for v2; dropping it in §5d was premature.
3. `pairs_v2` with doc-like anchors still stands (the hypothetical-text regression is real on either metric: control in-pool 0.56 is on the *base* embedder).
4. Add a within-category ranking signal (Scryfall `edhrec_rank` is already in `raw`) as a cheap ablation.

### 5f. Reframing the target: complete set retrieval, not top-10 relevance (Mitchell, 2026-09-19)

**Mitchell's position.** Relevance ranking was never the thing being solved. Scryfall's tag + attribute search returns a *set* — every card that matches, good or bad — and orders it by a separate sort (EDHREC popularity being the practical relevance proxy, since the most-played cards are the ones people are looking for). The goal here is the same contract reached from plain language: **every card that could be a match, in descending order of match, paged m × n**. Ordering by popularity is then a sort option available to both systems and out of scope as a research claim. The baseline is Scryfall's result set, not a hand-picked list of archetypes.

**Consequence for measurement.** P@10 against ~30 judged archetypes is the wrong instrument for that goal. The right ones are set-retrieval measures against the full target set: **R-precision** (precision at depth = size of the target set; 1.0 means the first |pool| results are exactly the pool), **P@100**, and **depth to 90 % of the pool** (how far a user must page to have seen nearly everything). `scripts/probes/set_retrieval_probe.py`, no filters, 21 mapped queries, target = the tag's closure pool:

| Query-side text / embedder | P@100 | R-precision | R@500 | median depth to 90 % of pool |
|---|---|---|---|---|
| tag label / **tuned** | **0.82** | **0.73** | 0.55 | **1.8 × pool** |
| raw user query / **tuned** | 0.74 | 0.60 | 0.47 | 3.4 × |
| hypothetical text / tuned | 0.57 | 0.46 | 0.36 | 3.8 × |
| hypothetical text / base (the v1 control) | 0.47 | 0.27 | 0.25 | 13.2 × |
| raw user query / base | 0.31 | 0.21 | 0.18 | 13.5 × |
| tag label / base | 0.29 | 0.19 | 0.17 | 16.4 × |

Under the goal as Mitchell states it, the ordering of systems **inverts** relative to §5c/§5d: the tuned embedder with a tag label is best, the tuned embedder with the *user's own words* is second, and the v1 control is fourth. Per tag, label/tuned R-precision: tutor 0.96, extra-turn 0.94, burn 0.92 (**held-out tag**), counterspell 0.92, fetchland 0.91, draw 0.89, recursion 0.86, removal 0.84 (5,968 cards), ramp 0.70, sweeper 0.65; weak spots mana-dork 0.42, cast-trigger-you 0.30, counterspell-free 0.23 (13 cards). To see 90 % of all tutors a user pages 1,030 results with label/tuned versus 12,156 with the control.

**This rehabilitates the original hypothesis.** Raw user phrasing on the tuned embedder (0.60) already beats hypothetical text on either embedder, and tag vocabulary (0.73) beats both. HyDE's rewrite job genuinely can shrink to filters plus concept normalisation. The concepts field is in for v2.

**Caveats to carry into the paper.**
1. 20 of 21 targets are tags seen in training; the target *is* the training signal. `burn` (held out, 0.92) and the 482-tag held-out probe (§5b) are the generalisation evidence. The Scryfall comparator with expert queries that are *not* bare `otag:` lookups is the independent check.
2. **A ranking is not a set.** Scryfall's result ends; ours is all 31k cards in order. "Every card that could match" needs a stopping rule — a similarity threshold, a score-gap heuristic, or simply paging with the score shown. Depth-to-90 % at 1.8 × pool says a naive cutoff would either truncate the set or pad it ~45 % with non-members. Choosing and evaluating that cutoff is now a real design item.
3. Tag membership is community-curated and incomplete; some "non-members" in the top ranks are untagged true matches (the generalisation benefit), which set metrics against the pool under-credit. The dashboard judgments measure that.

**Mitchell's corrections (2026-09-22).**
1. Only queries with no SQL component rank the full 31k corpus. In the control run 15 of 26 eval queries carried filters, with candidate sets of 3 to 17,428 cards; on constrained queries Stage 2 already leaves a set close to the target size. So the unbounded-ranking concern applies to the 11 pure-jargon queries, and Table 5 (run with no filters) is pessimistic for the filtered majority. The set-retrieval probe should run **with filters on**, measuring R-precision inside the Stage 2 candidate set — that is the cascade's number, not the embedder's.
2. **No stopping rule.** Restricting the returned set has no computational value: pgvector scores every candidate in the WHERE set regardless of LIMIT, so a threshold saves nothing. Results are the full ranked candidate set; the interface soft-caps with "load more" purely to bound payload and rendering; the match score is shown so users decide how far into the tail to dig. Depth-to-90 % remains a *ranking-quality* measure, not a cutoff. Caveat 2 above is withdrawn.

**Revised plan.**
- Primary metrics become R-precision / P@100 / depth-to-90 % against (a) tag pools and (b) expert Scryfall query result sets; P@10-on-archetypes stays as a secondary "famous cards first" measure, reported with the EDHREC-sort caveat.
- `evaluate.py` gets a deep-retrieval mode (k = target-set size) and these metrics, so grid rows are logged under the new instrument.
- Build `hyde_v2` with the concepts field; measure prompt size, output tokens, Stage 1 latency, and the set metrics with filters on.
- Add the EDHREC sort as a display option in the dashboard (data already in `raw`).
- `pairs_v2` doc-like anchors drop in priority: hypothetical text is no longer the mode we are optimising for.

### 5g. Set metrics logged inside the cascade — rows 38–39 (base) and 42–46 (tuned), 2026-09-22

`scripts/evaluate.py` now ranks the **entire Stage 2 candidate set** (`Searcher.rank_prepared`, no LIMIT — pgvector scores every candidate regardless) and scores it against the query's tag pool. Two R-precisions are reported: against the full pool (charges Stage 2's exclusions to the cascade) and against the *reachable* pool (Stage 3's ranking quality given what Stage 2 admitted). `reachable frac` is Stage 2 on its own.

| Row | Config | Embedder | R-prec | R-prec (reachable) | P@100 | reachable | depth90 / target |
|---|---|---|---|---|---|---|---|
| 38 | hyde v1 + SQL (control) | base | 0.215 | — | 0.444 | 0.68 | 8.4× |
| 39 | passthrough + SQL | base | 0.185 | — | 0.343 | 0.68 | 8.4× |
| 42 | hyde v1 + SQL | tuned | 0.326 | 0.517 | 0.572 | 0.68 | 3.0× |
| 44 | hyde v1, no SQL | tuned | 0.459 | 0.459 | 0.566 | 1.00 | 3.8× |
| 45 | raw query, no SQL | tuned | 0.598 | 0.598 | 0.736 | 1.00 | 3.4× |
| 43 | **passthrough + SQL** | tuned | 0.456 | **0.641** | 0.713 | 0.68 | **1.2×** |
| 46 | tag-label oracle + SQL | tuned | 0.536 | **0.728** | 0.765 | 0.68 | **1.1×** |

(Rows 38–39 predate the reachable column; re-run when the base vectors are back.)

**Three readings.**
1. **Stage 2 over-narrows on jargon queries, and it costs a third of the set.** `reachable = 0.68` on every filtered row. Split by cause: user-explicit narrowing is correct and the tag pool is simply the wrong target ("instants that draw cards" → draw ∩ instants; "cheap blue counterspells"); model-inferred narrowing is the loss — "ramp spells" gets `types: [Instant, Sorcery]` and drops 1,784 ramp permanents; "burn spell that deals 3 damage" gets `cmc = 1` (hallucinated); "mana dorks" and "pingers" filter to zero. This is the selective-strictness problem (prompt design notes §2) measured at the set level.
2. **Given what Stage 2 admits, Stage 3 on the tuned embedder ranks well.** R-prec(reachable) 0.64 for the user's own words, 0.73 for tag labels; depth-to-90 % drops to 1.1–1.2× target. Compared with raw/no-SQL (0.60, 3.4×), the filter *helps* ranking inside the admitted set even as it hurts coverage — the two effects the ablation table has to show separately.
3. **Hypothetical text loses on the tuned embedder either way** (0.52 reachable with SQL, 0.46 without) — consistent with §5c/§5f: the v1 prompt's paragraph-length rewrite is the wrong query-side shape once the encoder knows the vocabulary.

**Fix order.** (a) Fix the target: for filtered queries, intersect the tag pool with the *user-explicit* constraint so the metric stops charging correct narrowing — add an optional `target_filter` per query to `tag_label_oracle_v1.yaml`. (b) Fix Stage 2: the v2 prompt must stop inferring `types`/`cmc`/`power` from jargon (rule: filters come only from words the user typed), and the keywords-off policy should extend to inferred types. Both are prompt work, which is the next task.

DB state: `cards.embedding` holds **tuned** vectors (left in place so the dashboard reviews the tuned model).

### 5h. `hyde_v2` — concept normalisation + explicit-only filters (2026-09-22, rows 51–53)

**Contract.** Three fields: `filters` (only from words the user typed; "spell" is not a type; slang nouns are never keywords), `concepts` (1–3 phrases in deckbuilder vocabulary, community term preferred: "board wipe" → "sweeper"), `hypothetical_card` (one sentence, fallback only). `prompts/hyde_v2.yaml`; `Stage1Mode.CONCEPTS` embeds the concept phrases joined by ", ", falling back to the hypothetical, then the raw query. Two robustness fixes landed with it: the rewriter now sends stop sequences and decodes only the first JSON object (the 8B model kept generating extra "Query:/Output:" pairs), and `Searcher` drops any keyword not present in the corpus's canonical list (a hallucinated "Burn"/"Pinger" keyword otherwise zeroes the result set). Two v2 draft examples were verbatim eval queries and were replaced — none of the 8 examples appear in the eval set.

**Prompt size** (`scripts/probes/prompt_size.py`, Llama 3.1 tokenizer): v1 = 11 rules, 7 examples, 2,079 request tokens; v2 = 8 rules, 8 examples, **1,414 request tokens (−32 %)**. Rules shrank; examples did not — the 8B model needed one example per failure shape (slang-as-keyword, implied-type-not-a-filter) that a rule alone did not fix.

**Results, tuned embedder, set metrics inside the Stage 2 set (n = 21):**

| Row | Config | R-prec | R-prec (reachable) | reachable | depth90 | P@10 | MRR | Stage 1 ms | out tokens |
|---|---|---|---|---|---|---|---|---|---|
| 42 | v1 hyde + SQL | 0.326 | 0.517 | 0.68 | 3.0× | 0.088 | 0.270 | 1,186 | — |
| 43 | v1 passthrough + SQL | 0.456 | 0.641 | 0.68 | 1.2× | 0.081 | 0.206 | 1,064 | — |
| **51** | **v2 concepts + SQL** | **0.558** | **0.693** | **0.81** | 1.3× | 0.096 | 0.234 | 1,384 | 62 |
| 52 | v2 hypothetical + SQL | 0.533 | 0.625 | 0.81 | 1.2× | 0.085 | 0.153 | 1,389 | 62 |
| 53 | v2 concepts, no SQL | 0.631 | 0.631 | 1.00 | 2.1× | 0.081 | 0.184 | 1,383 | 62 |
| 46 | tag-label oracle + SQL (ceiling) | 0.536 | 0.728 | 0.68 | 1.1× | 0.073 | 0.153 | — | — |

**Readings.**
1. **v2 concepts is the best real cascade cell**: R-prec 0.558 vs 0.456 (v1 passthrough) vs 0.326 (v1 hyde, same embedder) vs 0.215 (v1 hyde, base — the control). It even beats the hand-mapped oracle on full-pool R-prec (0.536) because it over-narrows less.
2. **Explicit-only filters recovered coverage**: reachable 0.68 → 0.81. "ramp spells", "mana dorks", "board wipes", "removal", "burn spell" now emit no type/cost filter. Remaining over-narrowing: "destroy target artifact" → `types: [Instant]` (user typed no type); "red pingers" → `Creature` + `Flying` (canonical, so the guard can't drop it). Both are the 8B knowledge ceiling.
3. **Concepts beat the one-sentence hypothetical on the same filters** (row 51 vs 52: 0.693 vs 0.625 reachable) — and v2's hypothetical was null on 24/26 queries anyway; the model treats it as the fallback it was told to be.
4. **Archetype precision is roughly flat** (P@10 0.096 / MRR 0.234 vs control 0.112 / 0.294): the famous-cards-first measure did not improve, which is consistent with §5f — that is the popularity sort's job.
5. **Latency did not fall, and the reason is instructive.** Stage 1 is ~1.4 s for v2 vs ~1.1–1.2 s for v1 despite a 32 % shorter request. `mlx_lm.server` caches the shared prompt prefix across calls, so prompt length barely matters; **output tokens dominate** (62 tokens ≈ 1.1 s at ~57 tok/s), and v2's three-field JSON with explicit nulls is not shorter than v1's. Cheapest next win: tell the model to omit null fields → expect ~30 output tokens. The "simpler prompt" claim therefore has to be stated as request tokens and rule count, not latency, until output is trimmed.
6. Remaining knowledge-gap misses: "a card that lets me look at my deck and put a creature into the battlefield" → concepts `[card draw, token maker]` (should be tutor); "ETB triggers" → nothing usable. These are the cases the 27B comparison already predicted.

**Next.** Omit-nulls output rule (latency); re-run v2 on the **base** embedder to complete the 2×2 (v2 × base is the missing cell for the paper's "does fine-tuning make the simpler prompt viable" argument); then the Scryfall expert-query comparator.

### 5i. Omit-nulls, the 2×2, and versioned embeddings (2026-09-22, rows 56–61)

**Migration 0004 — `card_embeddings`.** The dashboard was found pinned to the base version while `cards.embedding` held tuned vectors: every search returned nothing. Root cause is the one-vector-per-face design. New table keyed by `(oracle_id, face_index, embedding_version)`; `embed.py` upserts per version and only encodes faces missing *that* version; `Searcher(embedding_model=...)` pins any version; the dashboard loads every embedder with vectors and offers a selector, plus an EDHREC sort and "load more" paging. Base and tuned now coexist (31,972 rows each). No more re-embedding to switch.

**Omit-nulls output rule** (task 1). Stage 1 latency **1,384 → 772 ms**, output tokens **62 → 28**. The prompt change shifted some filter outputs (model variance under a changed prompt, not randomness — temperature is 0): R-prec 0.558 → 0.510, reachable 0.81 → 0.72 (rows 51 → 61). One regression exposed a contract hazard — `colors: []` meant "colourless" and zeroed "free counterspell" — fixed: `[]` is no constraint unless `colors_op = exactly`. Latency win kept; the quality delta is two or three queries flipping on a 21-query set and is reported as-is.

**The 2×2** (task 2), set metrics inside the Stage 2 set:

| Prompt \ Embedder | base | tuned |
|---|---|---|
| v1 (hypothetical paragraph) + SQL | 0.215 (row 38) | 0.326 (42) |
| v2 (concepts, explicit filters) + SQL | **0.163** (57) | **0.510** (61) |
| v2, no SQL | 0.198 (58) | 0.631 (53) |

**On the base embedder the simpler prompt is *worse* than v1** (0.163 vs 0.215): a bare concept phrase like "ramp" means nothing to an encoder that never learned the vocabulary, whereas a paragraph of rules text at least overlaps lexically. On the tuned embedder the same prompt is the best cell. That is the paper's central mechanism, now measured: **fine-tuning is what makes the lighter rewriter viable; neither half works alone.** Stage 1 cost with v2 is 0.77 s and 28 output tokens on the same 8B model.

Committed rows: 56 (v2 tuned, omit-nulls, pre-fix), 57–58 (v2 base), 61 (v2 tuned, fixed).

### 5j. Scryfall comparator — result parity and query complexity (2026-09-22, rows 68–79)

**What was built.** `data/eval/scryfall_expert_queries_v1.yaml`: for each of the 26 eval queries, the Scryfall search an expert would write, in two variants — `expert` (community `otag:` allowed) and `no_tag` (Oracle text + attributes only, the pre-Tagger baseline). Drafted by Claude (a different model family from the Llama rewriter; `scripts/craft_scryfall_queries.py` will regenerate via the API once a key is set); **Mitchell reviews every query**. `scripts/scryfall_comparator.py fetch` pulled all 52 result sets through `/cards/search` at 1 req/s (Scryfall's hard limit is 2/s) with the required headers — 52 queries, ~300 pages, cached in `data/eval/scryfall_results_v1.json` (committed; oracle_ids only, ~230 KB). `compare --row N` scores a logged cascade row's stored ranking (`ranking_top`, now saved for every query when `set_metrics` is on) against each Scryfall set restricted to our corpus.

**Result parity — cascade ranking vs the expert's result set (n = 26):**

| Cascade row | vs `expert` (otag) R-prec | Jaccard@\|S\| | depth90 | vs `no_tag` R-prec | Jaccard |
|---|---|---|---|---|---|
| v1 prompt, base embedder (control, row 73) | 0.343 | 0.273 | 1.32× | 0.351 | 0.280 |
| v2 prompt, tuned embedder (row 72) | 0.523 | 0.446 | 0.98× | 0.391 | 0.310 |
| **v2 + tuned + keywords filter on (row 78)** | **0.565** | **0.492** | **0.95×** | — | — |

Per query (v2 tuned, expert sets): instants that cost 1 mana 1.00, haste creatures 1.00 (with keywords on), tutor 0.96, cheap blue counterspells 0.95, extra turns 0.94, counterspells 0.92, fetch lands 0.91, instants that draw cards 0.86, red creatures under 3 mana 0.85, recursion 0.74, wheels 0.74, flicker 0.73, ramp 0.68, board wipes 0.61. **Depth-to-90 % below 1.0×** means a user paging the cascade's results sees 90 % of the expert's set before having scrolled past as many cards as the set contains.

Failures are the known ones: "destroy target artifact" 0.09 (inferred Instant filter), "a card that destroys all creatures" 0.08 (the concept "sweeper" retrieves the whole 870-card sweeper pool, but the expert's `o:"destroy all creatures"` set is 84 cards — the cascade is *broader* than the expert here, which is not obviously wrong), "burn spell that deals 3 damage" 0.03, the tutor-a-creature query 0.00, and the prowess-style query 0.00 (Scryfall's own set for that one has 6 % recall against the judgments — the expert query is bad too).

**The comparator is itself imperfect, and that matters.** Scryfall's expert `otag:` sets have **11 % precision and 90 % recall** against our eval judgments; the `no_tag` sets 17 % / 72 %. Tags are categories; judgments are archetypes (§5e) — the same annotation-hole effect from the other side. Parity with the expert set, not precision against the 30 judged archetypes, is the paper's number.

**Query complexity — what the user would have had to type (mean over 26):**

| | plain language (ours) | Scryfall `expert` | Scryfall `no_tag` |
|---|---|---|---|
| length | 3.8 words | 20 chars | 59 chars |
| operators | 0 | 1.5 | 3.3 |
| needs `otag:` | — | 100 % of jargon queries | 0 % |
| boolean / grouping / negation | 0 | rare | common (`(o:… or o:…) -t:land`) |

The expert path is short *only because* `otag:` exists — every jargon query needs it, and a user has to know the tag's exact slug (`sweeper`, not `board wipe`; `mana-dork`; `counterspell-free`). Without tags, the same intent takes three operators, quoted Oracle phrases, and boolean grouping, and still lands at 72 % recall. The cascade takes the 3.8-word query and reaches 0.57 R-precision against the tagged expert's set with no syntax at all. That is the accessibility claim (§4, paper §1.1) with a number on it.

**Caveats.** The expert queries are drafted, not collected from real experts — Mitchell's review is the check; a small expert-user sample is future work. 20 of 21 tag-mapped targets were seen in training (the `no_tag` column is the independent check: 0.39 vs 0.35 for the control). Rows 74–79 in `experiment_runs` (kind `scryfall_comparator`).

### 5k. Dashboard review — two observations and a null result (Mitchell, 2026-09-22)

**Observation 1 (weakness).** "destroy all creatures" → v2 concepts `sweeper, board wipe` → the whole 870-card sweeper category (land destruction, mass bounce included), which by review is far less useful than Scryfall's `o:"destroy all creatures"`. Checked: `boardwipe`, `board-wipe`, `board wipe`, `wipe`, `mass-removal` are all aliases of `sweeper` on Scryfall (identical 942-card result set); there is no narrower community tag, and the specificity lives only in Oracle text. The rewriter replaced a specific phrase with its category.

Tried the principled fix — embed the user's words first, then the concepts (`Stage1Mode.QUERY_PLUS_CONCEPTS`, row 84/85). On that query: P@10 0 → 0.20, first result Day of Judgment. Overall: **R-prec 0.486 vs 0.510, Scryfall parity 0.563 vs 0.565** — a wash; gains on the specific queries, losses on bare-jargon ones ("mana dorks" + "mana dork" is noise). Kept as an ablation row, not the default. Mitchell's read, recorded verbatim: *"it's likely that our simple attempts to fix now may make our overall results worse and this just drives home the point that while the idea is showing merit there is still a lot of work that would need to be done to make it foolproof as a search tool."* That sentence belongs in the Discussion.

**Observation 2 (strength).** *"I did a search 'is a planeswalker' and it did exactly what you would expect right away and filtered by type planeswalker, and the same search on Scryfall would return nothing. You would have to use the advanced interface or know the shortcuts to type `type:planeswalker` into the search bar to do the same thing. Relatively simple task either way but one has no prior-knowledge requirement compared to the other."* — the accessibility claim in one example, for the paper's Discussion.

**Also noted:** the dashboard's "match %" is raw cosine, and the tuned model's scores sit at 30–40 % even on excellent results (base: 60–75 %; nonsense query on tuned: 22 %). Contrastive fine-tuning pulls the distribution down and spreads it; the value is meaningful relative to the query's own distribution, not on a fixed scale. To do: show relative match (top = 100 %) with raw cosine in a tooltip.

### 5l. Future fine-tuning directions (Mitchell, 2026-09-22)

Mitchell's two proposals, verbatim, then the mechanics.

1. *"Maybe there is additional fine tuning that can be done to provide overlap training? Right now if a query closely matches 1 tag it embeds to that tag. What if a query matches multiple tags and a multiple-tag search resulted in a more specific subset of cards? If that was possible then maybe we would get better results."*
2. *"On a different vein we want to have an LLM do a large task of pre-processing our tag pool to make things more specific. It is a good starting point for training data but possible that tags are too generic and we would need subtags to split categories like sweepers into more specific subsets. The LLM could evaluate all the tag groups for possible set splitting and build a new training dataset."*

**(1) Intersection anchors — cheap, data already in hand.** `card_tags` gives every co-occurrence. Build anchors from tag *pairs* with a meaningful intersection (say ≥ 10 cards, and the intersection ≤ 50 % of the smaller pool so it is actually narrower): anchor text = the two labels joined ("sweeper, creature removal"; "flicker, enters trigger"; "counterspell, cantrip"), positives = the intersection, and — the important part — the tag-disjoint batch sampler must treat cards in *either* parent pool as unsafe negatives. This teaches the encoder that "A, B" sits inside A ∩ B rather than at the midpoint of A and B, which is what a plain average of two label embeddings gives today. Literature: BioHiCL's label-overlap positives (Lan et al. 2026) are the single-label form of this; multi-positive objectives (Wang et al. 2026) handle the many-positives case. Expected effect: exactly the "destroy all creatures" class of query — the v2 rewriter can emit two concepts and the embedder lands in the intersection. Risk: pair explosion (3,000 tags → millions of pairs); sample by co-occurrence count and cap per pair.

**(2) LLM-refined subtags — the bigger lever, and where the training signal's ceiling is.** Scryfall's `sweeper` covers land destruction, mass bounce, and creature wraths alike; no community subtag separates them. Procedure: for each tag with a large pool (≥ 100 cards), cluster the pool *using the tuned embedder* (k-means on the card vectors, k chosen by silhouette), then have an LLM read each cluster's Oracle texts and propose a label and a one-line description ("sweeper → destroy-all-creatures / mass-bounce / land-destruction / -X/-X"). Clusters the LLM cannot name coherently are left merged. Output: `oracle_tags_refined` with `parent_id` pointing at the Scryfall tag, and a `pairs_v2` built from the refined leaves plus the parents. Cheaper than having the LLM read 236k taggings from scratch, and the LLM only names structure the embedder already found. Literature: UAE/AnglE and CustomIR both use LLMs as annotators for supervision the corpus lacks; this is the taxonomy-refinement form. Risk: LLM-invented subtags are not community-validated — evaluate on the held-out judgments and the Scryfall parity rows, never on the refined tags themselves (that would be the model grading its own homework twice over). Cost: ~370 tags × a few clusters × one LLM call ≈ an afternoon on the API.

Both feed the same `pairs_v2` → retrain → grid → comparator loop that already exists. Order: (1) first (no LLM cost, one training run), then (2).

## 6. CLAUDE.md revisions

§5 (do not fine-tune the embedder on keyword definitions), §6 (fine-tuning deferred to M6), and §11 (anti-suggestion) all encode the pre-pivot position. Revised today to: reminder-text augmentation stays the corpus-side lever; tag-derived contrastive fine-tuning is the query-side lever, motivated by the 2026-09-18 test evidence and gated on the base-embedder control run in §5 above. Hand-written definition dictionaries remain banned.

## 7. Next actions

1. ~~Confirm anchors and full-vs-LoRA~~ — both confirmed 2026-09-18 (§4, §4a). **[Mitchell]** ordering in §5 still open.
2. ~~`src/search.py` + base-embedder eval run (control row)~~ — done, rows 11–14 (§5a).
3. ~~`scripts/build_training_pairs.py`~~ — done for tag label/alias/description anchors: 207,062 train pairs over 2,729 tags, 482 held-out tags (33,862 probe pairs), `data/training/pairs_v1.jsonl` + `manifest_v1.json`. Synthetic-query anchors are a separate script (needs the LLM server; hours). Note: the random stratified hold-out withheld `burn` — so q_004 ("burn spell that deals 3 damage") becomes a genuine generalisation test rather than in-distribution; keep the seed and disclose.
4. ~~`scripts/finetune_embedder.py`~~ — drafted and smoke-tested on MPS (5 steps, batch 64, 2k pairs, ~32 s): full fine-tune, `CachedMultipleNegativesRankingLoss` (scale 20, mini-batch 32), prompts on, lr 2e-5, 5 % warmup, wd 0.01, one epoch, grad clip 1.0, `max_seq_length` 256. Tag-disjoint batch sampler (no shared anchor tag; no positive carrying another pair's anchor tag; constraint-starved tail packed loosely and counted). Probes: in-training dev on train tags; post-training held-out-tag probe over the full corpus, base vs tuned; optional NanoBEIR (`--nanobeir`). One `experiment_runs` row per run. **Measured throughput ~10 pairs/s on the M3 → the 207k-pair epoch is ~6 h.** Not yet run in full. Requires ST ≥ 5.7 (have 6.0), `datasets`, `accelerate` (added to pyproject).
5. Re-embed corpus with the tuned checkpoint under a new `embedding_version`; run (d) and (e).
6. `prompts/hyde_v2.yaml` — concepts field, optional `hypothetical_card`, few-shot rebuilt around filters + tag normalisation. Count rules/examples vs v1 for the paper.
