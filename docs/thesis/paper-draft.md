---
title: "Natural-Language Semantic Search for Trading Card Catalogs: A Three-Stage Retrieval Cascade with Query Rewriting, Structured Pre-Filtering, and Dense Retrieval"
author: "Mitchell Trafford"
date: "Fall 2026 · CSCI 5953 · CSUSB"
---

# Paper Draft — Working Document

**Editing notes for Mitchell:** sections marked *[YOUR PROSE]* are pulled verbatim from your notes with minor grammar polish. Sections marked *[DRAFTED — REVIEW]* are drafted from our conversation and the outline; rewrite in your voice as needed. Sections marked *[TODO]* have writing prompts but need real results (M4/M5 measurements) or your input to populate.

---

## Abstract

*[YOUR PROSE — Sept 4 draft, refreshed 2026-09-18 for post-pivot scope: two comparators (dropped naive-baseline reference), Scryfall comparator uses LLM-crafted expert queries, dropped fabricated-baseline framing]*

Keyword or faceted filtering is still commonly used across the internet for consumer-facing product catalogs. Magic: The Gathering (MTG), the world's most popular and established trading card game, is a particular case where keyword search and faceted filtering dominate retrieval. For many product catalogs, faceted filtering and sorting work well. However, the compositional rules text in products like trading cards introduces a layer of complexity that simple filtering and sorting cannot address. This research addresses these limitations in trading card product catalogs of a small-to-medium corpus (~30,000 documents) through a three-stage retrieval cascade. The Hypothetical Document Embeddings (HyDE) query rewriter (Stage 1) uses a generative large language model to split each natural-language query into (a) structured attributes handed to Stage 2 and (b) a hypothetical card ability text handed to Stage 3. The SQL pre-filter (Stage 2) narrows the candidate set using structured attributes such as color, mana value, types, and legality. Finally, Stage 3 embeds the generated hypothetical card ability text using Nomic Embed v1.5 for vector search over card ability text embeddings inside the narrowed candidate set from Stage 2. This architecture is evaluated on a benchmark of 26 hand-curated queries with graded relevance labels, measuring how many correct results appear in the top 10 and the rank of the first correct match (recall@10 and MRR). Performance is compared against two points of reference: a version of the cascade with the SQL pre-filter removed, isolating that stage's contribution; and the Scryfall search tool that MTG players use today, evaluated with LLM-crafted expert-level queries to represent the strongest form of the competing structured-search paradigm. Full cascade and ablation measurements are in progress; the primary claim concerns whether the cascade matches or exceeds Scryfall for non-expert users on the same eval queries. While MTG serves as the test bed, the retrieval principles used in this research generalize to any consumer catalog domain where structured-query interfaces currently reward expert users and exclude those who query in natural language.

---

## 1. Introduction

### 1.1 Problem statement

*[DRAFTED — REVIEW in your voice]*

Consumer catalogs — trading card games, e-commerce product listings, media libraries, technical documentation — increasingly need natural-language search that works for average users, not just experts fluent in the catalog's structured query syntax. Traditional keyword and faceted-filter search excels for users who already know the domain's vocabulary and how the search tool decomposes queries, but this same strength excludes the majority of users who query informally, in their own words rather than the tool's grammar.

Magic: The Gathering (MTG), the world's most popular and established trading card game, is a representative case. Its dominant search interface — Scryfall — is powerful, syntax-rich, and used effectively by expert deck-builders. An experienced Scryfall user can construct a query like `t:creature c:wu cmc=2 o:"enters the battlefield"` to find blue-or-white two-mana creatures with enters-the-battlefield abilities. The average player who searches for *"creatures that are blue or white with etb effects that cost 2"* has no path to those results without learning Scryfall's grammar first.

Attempts to bridge this gap with off-the-shelf dense retrieval fail for a specific reason: **the query-document asymmetry problem**. Short informal player queries embed too far in vector space from the formal, prose-heavy card text they should match. A query like *"cheap red removal"* does not lexically overlap with Lightning Bolt's actual text *"Lightning Bolt deals 3 damage to any target,"* and general-purpose sentence encoders have no domain knowledge to bridge that gap. Naive dense retrieval on this corpus fails on the majority of evaluation queries.

*[YOUR PROSE — 2026-09-18, the accessibility framing]*

Scryfall already offers an easy method for much of this: its community-maintained oracle tags (`otag:`) let a power user retrieve cards by function. Both approaches may end up performing the same on a results measure. However, the benefit of this system over Scryfall then becomes ease of use. The pre-filter stage removes the complexity of Scryfall power-user querying, and with plain language the user gets objectively the same results a structured-query power user can get, lowering the bar for novice users searching for cards. So even if it performs no better by a results measure, it is still a win from a user perspective. Parity with an expert-crafted Scryfall query is the accessibility claim proven, not a tie.

### 1.2 Contributions

*[DRAFTED — REVIEW]*

This work makes four contributions:

1. **Architectural.** A three-stage retrieval cascade combining HyDE-style query rewriting, structured SQL pre-filtering, and dense vector search inside the pre-filtered candidate set — designed to address query-document asymmetry without domain-specific fine-tuning as a starting point.
2. **Methodological.** An evaluation methodology anchored on real-world outcome comparison against the domain-standard tool (Scryfall) with LLM-crafted expert queries, rather than diff against an internal baseline. Per-component contribution is quantified through a targeted ablation of the SQL pre-filter stage.
3. **Empirical.** Documentation of specific failure modes observed with an off-the-shelf small (8B parameter) instruction-tuned LLM as the HyDE model, including jargon knowledge gaps, over-narrowing on keyword filters, and compositional attention drift. Comparison against a larger (27B parameter) model shows that most of these failures resolve with more parameters, informing the case for domain fine-tuning of the embedder on Scryfall oracle tags as distant supervision, and a measurement of whether that fine-tune simplifies the query-rewriting stage.
4. **Design.** A framework for interactive user-refinable filters and a taxonomy of jargon-query shapes (structural, ability-text, compound) that maps to distinct HyDE output shapes.

### 1.3 Paper organization

Section 2 reviews related work in dense retrieval, query rewriting, and filtered vector search. Section 3 describes the three-stage cascade architecture and the evaluation methodology. Section 4 presents the experimental configurations. Section 5 reports results, including qualitative analysis of the failure modes observed. Section 6 discusses the design decisions surfaced during development and the path toward fine-tuning as an intervention for the observed limitations. Section 7 concludes and outlines future work.

---

## 2. Related Work

### 2.1 Dense retrieval foundations

*[DRAFTED — REVIEW, cite from docs/sources]*

Sentence-level dense retrieval as a paradigm was established by Reimers and Gurevych (2019) with Sentence-BERT, which introduced a siamese bi-encoder architecture for producing fixed-length semantic embeddings suitable for cosine-similarity retrieval. Dense Passage Retrieval (Karpukhin et al., 2020) extended the paradigm to open-domain question answering with dual-encoder training. Modern encoders in this lineage — Nomic Embed v1.5 (Nussbaum et al., 2024), used in this work, together with the broader family of models evaluated on the MTEB benchmark (Muennighoff et al., 2022) — build on the same bi-encoder + contrastive training foundation while addressing scale, context length, and domain generality.

The evaluation methodology used here builds on Manning, Raghavan, and Schütze (2008), Chapter 8, which establishes recall@K and MRR as the standard metrics for retrieval effectiveness. BEIR (Thakur et al., 2021) provides the standard heterogeneous benchmark methodology for zero-shot dense retrieval evaluation.

### 2.2 Query rewriting and expansion

*[DRAFTED — REVIEW]*

Hypothetical Document Embeddings (HyDE), introduced by Gao et al. (2022), is the primary technique adapted in this work. HyDE addresses the query-document asymmetry by using an instruction-tuned generative language model (Ouyang et al., 2022) to transform a natural-language query into a hypothetical target document. The hypothetical document, rather than the raw query, is embedded and used for retrieval. Query2doc (Wang et al., 2023) is a concurrent technique that concatenates the hypothetical document with the query rather than replacing it. Best-practices studies on LLM query expansion (Wang, 2024) and the recent adaptive-HyDE literature (2025) address extensions and refinements.

A critique worth engaging with is the "Rethinking LLM-based Query Expansion" work (2025), which argues that observed HyDE gains may partially reflect LLM knowledge leakage from pre-training rather than pure query rewriting. In the MTG-specific setting studied here, this concern is somewhat mitigated: MTG's Oracle text is a bounded, formal, well-known corpus, and the empirical failure modes observed with smaller models suggest that domain knowledge is often *absent*, not leaked.

### 2.3 Filtered vector search

*[DRAFTED — REVIEW]*

The SQL pre-filter design decision is grounded in the filtered ANN literature. An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System (2026) is directly applicable — this work uses pgvector as its retrieval backend. Recent experimental studies of attribute filtering in ANN search (2025) and benchmarks on transformer-based embedding vectors (2025) inform the pre-filter-versus-post-filter design decision made here: pre-filtering preserves recall on structurally-constrained queries, while post-filtering top-K vector results collapses recall in those cases.

The cascade architecture itself follows the multi-stage retrieval tradition established by Wang et al. (2011) — filter-then-rank pipelines rather than parallel dual-encoder ranking (Guo et al., 2016).

### 2.4 Semantic product search in consumer catalogs

*[DRAFTED — REVIEW]*

The generalization argument in Section 1.1 draws directly on Amazon's semantic product search work (Nigam et al., 2020), which addresses the same query-document asymmetry problem in an e-commerce catalog. Their approach — combining structured product attributes with semantic retrieval — is architecturally similar to the cascade proposed here, providing a published precedent for the generalization claim.

### 2.5 Domain adaptation of dense retrievers

*[DRAFTED — REVIEW, cite from docs/sources "Embedder fine-tuning" section]*

In-domain fine-tuning of a bi-encoder follows a well-established recipe: contrastive training with in-batch negatives (Karpukhin et al., 2020; Nussbaum et al., 2024), one epoch at a small learning rate, with task prefixes preserved. Positives need not be human-labelled — distant supervision from document structure (Reimers and Gurevych, 2019, §4.4), click logs (Choi et al., 2020), or community taxonomies (Lan et al., 2026) costs roughly one point against gold labels (Karpukhin et al., 2020, Appendix A). When no labels exist, LLM-synthesised queries per document (the Promptagator / InPars lineage; Gwon et al., 2025) are competitive with human queries at equal count (Tamber et al., 2025).

Two cautions from recent work shape the approach here. First, naive contrastive fine-tuning of a strong small encoder can score *below* the untouched base (Tamber et al., 2025; Pande et al., 2025), and out-of-domain ability can drop sharply without a regulariser or a forgetting probe (Murtaza et al., 2026). Second, mined hard negatives help in general (Moreira et al., 2024) but hurt on at least one jargon-heavy corpus (ChEmbed, 2025), because near-duplicate documents make many mined "negatives" actually relevant — CustomIR (Paull, 2025) measured 20–32% false negatives before verification. Finally, CoHyDE (Senthil et al., 2026) shows that query rewriting and an in-domain encoder are complementary: the rewriter helps on vague queries, the encoder on well-formed ones. This work's fine-tuning stage follows that evidence: full fine-tune, one epoch, tag-aware batching against false negatives, hard negatives as an ablation, and a general-retrieval probe run alongside every checkpoint.

### 2.6 IR evaluation methodology

*[DRAFTED — REVIEW]*

The evaluation methodology's use of tri-state graded relevance judgments is grounded in Järvelin and Kekäläinen (2002), which established graded relevance as standard IR practice, and Sormunen (2002), which showed empirically that a large fraction of TREC-"relevant" documents are marginally relevant — motivating the middle-bucket separation. Voorhees (2000) provides the methodological defense for single-curator evaluation sets producing stable comparative rankings.

---

## 3. Methodology

### 3.1 Task and dataset

*[DRAFTED — REVIEW]*

The task is: given a user's natural-language query, return the top-K most semantically relevant MTG cards from a corpus of ~30,000 unique cards.

**Corpus.** The Scryfall `oracle-cards` bulk dataset (2026-09-11 snapshot) contains 38,740 raw entries. Filtering rules exclude non-card layouts (tokens, emblems, art series, vanguard, planar, scheme), digital-only printings (Arena/MTGO exclusive), silver-bordered cards, memorabilia set-types, novelty Un-set (`set_type=funny`) products, and token-only booster products (`set_type=token`). The resulting corpus contains 31,972 face rows across 31,124 unique `oracle_id`s (80.3% retention rate). Multi-face cards (transform, modal-DFC, split, adventure) are stored as separate rows keyed on `(oracle_id, face_index)`; results deduplicate by `oracle_id` at display time.

**Oracle tags (training signal).** *[DRAFTED — REVIEW]* Scryfall publishes the community-maintained Tagger oracle tags as an official daily bulk file (2026-09-18 snapshot: 4,551 functional tags, 236,170 tag-to-card assignments, a multi-parent hierarchy of depth ≤6). Tags name what a card *does* in player vocabulary — `sweeper`, `cantrip`, `ramp`, `sacrifice-outlet-creature` — independent of how its rules text phrases it. Joined on `oracle_id`, 99.4% of corpus cards carry at least one tag and 205,262 assignments fall inside the corpus; 3,022 tags cover five or more corpus cards. Tags are used only as distant supervision for embedder fine-tuning (Section 6.3). They are not embedded and are not used by the SQL pre-filter, so that the system's results never depend on a card having been tagged. Scryfall's data policy names research as a permitted use; the exact bulk-file date is cited for reproducibility.

### 3.2 Three-stage retrieval cascade

*[DRAFTED — REVIEW]*

The cascade consists of three sequential stages, executed in numerical order. The design invariant is that the semantic search stage always executes within the SQL-pre-filtered candidate set — a pre-filter design rather than a post-filter on top-K.

#### 3.2.1 Stage 1 — HyDE query rewriter

*[YOUR PROSE, lightly edited for flow]*

`hyde_v1.yaml` functions as the instruction prompt that tells the HyDE model exactly how to parse a query into SQL pre-filter attributes and a hypothetical card ability text. The main challenge with this method is engineering the prompt to cover the wide range of jargon, keywords, and varying output shapes that any given prompt may produce. For example, the jargon few-shot example teaches the model how to handle "cantrips," but cantrips directly reference structural attributes (cheap instants and sorceries that draw a card) rather than an ability-text pattern. In contrast, "ramp" is jargon that references an ability-text pattern (adding mana or accelerating mana production) with no clear structural filter constraints. This nuance makes it difficult to use simple few-shot examples to teach the model how to handle the variety of outputs a single stage of the cascade must produce.

The base model handles general cases fairly well but struggles on more complex mechanics or jargon-specific prompts.

**HyDE model.** The default HyDE model is `meta-llama/Llama-3.1-8B-Instruct`, served locally on Apple Silicon via `mlx_lm.server` (MLX-quantized 4-bit variant, ~4.8GB peak memory, ~57 tokens/sec generation on an M3 MacBook Pro). The server exposes an OpenAI-compatible endpoint, making the backend swappable for non-Apple hardware. The choice of a small local model preserves the paper's cost-story argument: HyDE inference must be affordable enough to run per-query on modest hardware.

**Output contract.** HyDE produces a JSON object with two fields:

- `filters` — a structured object with optional fields for colors, color identity, types, keywords, converted mana value, power, toughness, and format legality. Populated when the query specifies structural constraints; null otherwise.
- `hypothetical_card` — canonical Wizards-authored MTG rules text describing what a card matching the query would do. Populated when the query has an ability-text component; null when the query is purely structural.

**Planned v2 contract.** *[DRAFTED — REVIEW; lands with the fine-tuned embedder, Section 6.3]* After the embedder is fine-tuned on oracle tags, the rewriter's job narrows to filter extraction plus concept normalisation: a `concepts` field carries the query's functional intent in tag vocabulary where a tag fits ("board wipe" → "sweeper"), the user's own phrasing passes through where none does, and `hypothetical_card` becomes a fallback rather than the default. The v1 and v2 prompts are compared by rule count, example count, and output tokens as a direct measure of the simplification hypothesis.

#### 3.2.2 Stage 2 — SQL pre-filter

*[DRAFTED — REVIEW]*

The structured filter attributes produced by Stage 1 are compiled into a PostgreSQL WHERE clause and used to narrow the candidate set. The schema exposes real columns for the well-defined attributes (colors, color_identity, cmc, type_line, keywords, layout, released_at) with GIN indexes on array columns and B-tree indexes on numeric columns, making common filter combinations inexpensive. Filter compilation is straightforward: array containment for colors and keywords, numeric comparison for cmc, LIKE pattern-matching on type_line for subtypes, JSONB path-extract for legalities.

**Pre-filter, not post-filter.** The semantic search stage runs within the SQL-narrowed candidate set. Post-filtering top-K vector results — the more common approach — collapses recall on structurally-constrained queries because relevant cards may fall outside the initial top-K if the encoder alone cannot bridge the query-document gap.

#### 3.2.3 Stage 3 — Semantic vector search

*[DRAFTED — REVIEW]*

The hypothetical card text produced by Stage 1 is embedded using `nomic-ai/nomic-embed-text-v1.5`, a 137M-parameter bi-encoder in the sentence-transformer lineage with 768-dimensional output. Nomic Embed requires task-specific prefixes on inputs — `search_document:` for corpus-side documents and `search_query:` for query-side text — applied at encode time via helper functions in `src/preprocess_text.py`. Omitting the prefixes measurably degrades retrieval quality.

Cosine similarity between the query embedding and the pre-filtered candidate set's card embeddings is computed via pgvector's `<=>` operator, and the top-K results are returned. Because the semantic search operates on the SQL-narrowed set, this stage is inexpensive in practice — typically dozens to hundreds of candidates rather than the full ~32K corpus.

### 3.3 Corpus text preparation

*[DRAFTED — REVIEW]*

Only the Oracle text of each card is embedded — not the type line, mana cost, or color, which live in SQL. Before embedding, the Oracle text is augmented with canonical Wizards-authored reminder text for any keyword the card has but does not already explain inline. The reminder-text dictionary is built once by scanning the full Scryfall corpus for parenthetical patterns: since Wizards inconsistently prints reminder text across printings, at least one printing of every keyword contains the canonical definition, and this can be harvested automatically. A small manual override file handles the handful of keywords Wizards has never printed reminders for.

### 3.4 Evaluation setup

*[DRAFTED — REVIEW]*

The evaluation set is a hand-curated collection of 26 queries with tri-state relevance judgments (relevant / partially relevant / not relevant). Queries span six categories established during eval-set construction: natural language, jargon, fragmented, hybrid, constrained, and mechanical. Metrics are recall@10 and MRR, following standard IR conventions.

Configurations form a grid over the Stage 1 mode and the embedder, plus two ablations and the external comparator. Every cell is a retrieval run over the same 26 queries, logged as one `experiment_runs` row.

| | Base Nomic Embed v1.5 | Tag-fine-tuned embedder |
|---|---|---|
| Stage 1: hypothetical card text (v1 prompt) | control — the pre-fine-tuning number | |
| Stage 1: tag-normalised concepts (v2 prompt) | | main result |
| Stage 1: raw query pass-through (no rewrite) | | |

- **–SQL ablation** — the main configuration with the pre-filter removed; isolates Stage 2's contribution.
- **Direct-tag lookup ablation** — HyDE's concept mapped straight to `card_tags` membership, no embedder. Bounds what the tuned embedder adds over a tag lookup; a system that only matched this row would be Scryfall's `otag:` search rebuilt locally.
- **Scryfall comparator** — LLM-crafted expert-level Scryfall queries evaluated against the same eval set: what Scryfall can do when operated by an expert-adjacent LLM constructing its queries. Two measures are reported per query: *result parity* (overlap between the cascade's top-K and the expert query's result set, alongside the tri-state judgments) and *query complexity* (operator count and operator types in the expert Scryfall query versus the plain-language query the user typed) as a proxy for ease of use in the absence of a user study.

The base-embedder rows are run and logged before any fine-tuned checkpoint is evaluated; the frozen base remains a permanent row in every results table.

---

## 4. Experiments

### 4.1 Configurations

*[DRAFTED — REVIEW, awaiting M5 formal measurements]*

Experimental configurations are logged as rows in an `experiment_runs` Postgres table, one row per (configuration, eval-set version) pair, capturing config JSON, aggregate metrics, per-query results, and timing. All models used for the experiments reported here are publicly available on HuggingFace; all code, configs, and the evaluation set are publicly versioned in the project repository.

### 4.2 Test-driven prompt development

*[YOUR PROSE + drafted synthesis, REVIEW]*

Before running the formal eval set, we conducted a 10-query test series designed to probe distinct behaviors of the HyDE stage. Queries were selected across the six eval-set categories, and none of the test queries appear in the eval set — a data-leakage check documented in the design-notes journal (2026-09-18). The purpose was diagnostic: to characterize the failure modes of the base HyDE model and the specific classes of queries where prompt engineering alone would be insufficient.

---

## 5. Results

### 5.1 Test series — base 8B model

*[YOUR PROSE for flicker example, integrated with drafted synthesis for the broader pattern]*

The initial method without fine-tuning produced correct structural filter output on most queries but exhibited three distinct failure families on the semantic content (the `hypothetical_card` field).

**Jargon knowledge gap.** The most notable failure was on the query "flicker effects." The prompt examples teaching the model how to handle user queries properly identified "flicker" as jargon rather than as a canonical Scryfall keyword and produced appropriate filters and JSON structure. However, the model hallucinated or misjudged what "flicker" actually is. Flicker is an effect in the game that exiles a card and returns it to play, most often applied to creatures to re-trigger their enters-the-battlefield abilities. Instead, the model could only be partially correct: it generated hypothetical card text suggesting that flicker bounced a creature to its owner's hand and prevented it from untapping — a bounce effect with a stun rider (a "freeze" effect the model hallucinated). This is not the flicker mechanic at all; a downstream semantic search on that hypothetical text would retrieve bounce cards and tap-locks rather than flicker cards.

Two additional queries — "mana rock" and "wheels" — showed the same failure signature. "Mana rock" produced a mana-fixing spell description rather than a canonical tap-for-mana artifact. "Wheels" produced a "draw X cards equal to life total" hypothetical rather than the Wheel of Fortune-family text "each player discards their hand and draws seven cards." Three independent MTG-specific jargon queries hit the same failure mode: the small model reaches for a semantically-adjacent but categorically-different mechanic when the query's true target is outside its trained lexicon.

**Over-narrowing on keyword filters.** A second failure family surfaced on the query "creatures that get bigger every time I cast a spell." The model correctly identified that Prowess — a canonical Scryfall keyword — matches the query pattern (*"Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn"*), and it populated the `keywords: ["Prowess"]` filter accordingly. This is not technically wrong. But Prowess is only one specific mechanic within the broader class of "creatures that grow from spellcasting" — the class also includes creatures with permanent +1/+1 counter accumulation (Chasm Skulker family) and various one-off card designs. Filtering on `keywords: ["Prowess"]` narrows the candidate set to the ~50 Prowess creatures and eliminates the rest before the semantic search can consider them.

This surfaces a deeper architectural concern about filter semantics. The current cascade uses strict pre-filter semantics: every filter field is a hard WHERE clause. When those filters are user-provided, this is what the user wants. But when the filters are model-inferred from an ambiguous natural-language query, false-positive keyword extraction has a much larger blast radius than false-positive extraction on colors or CMC. A partial fix is to move to *selective strictness*: user-explicit attributes stay strict, model-inferred attributes become soft (rank-boost rather than hard filter).

**Compositional attention drift.** A third failure family appeared on "green creatures that create tokens when they enter the battlefield." The filter extraction was fully correct — colors, types, and the color_identity mirroring were all populated as expected. But the hypothetical card text drifted from the query's explicit "when they enter the battlefield" trigger to a "when this creature dies" trigger. The model retained the shape of the ability (create tokens) but swapped the trigger direction. This is not a knowledge-gap failure; the model demonstrably understands ETB triggers elsewhere in the prompt. It appears to be an attention-composition failure over the long input (system prompt plus seven few-shot examples plus the user query).

**Summary.** Of the 10 test queries, the base 8B model produced clean structural filter output on all 10, but only 4 semantic hypotheticals were fully correct, 3 had partial or over-narrowing issues, and 3 were categorically wrong. All 3 fully-wrong cases were MTG-specific jargon queries.

### 5.2 Model size comparison — 8B vs 27B

*[DRAFTED — REVIEW, from our comparison run]*

To characterize whether the observed failures reflect model-size limits or are structural to the approach, the same 10 test queries were run against a larger non-reasoning model of the same architectural class: Gemma 3 27B Instruct (MLX 4-bit, ~15GB memory footprint on the same M3 hardware).

On the three jargon knowledge-gap failures, Gemma 3 27B produced the correct canonical text:

| Query | Llama 3.1 8B hypothetical | Gemma 3 27B hypothetical |
|---|---|---|
| flicker effects | "Return target creature to owner's hand..." (bounce) | "Exile target creature. Return it to the battlefield under its owner's control." |
| mana rock | "Tap two colorless mana sources: Add two mana..." (mana fixing) | "Tap: Add one colorless mana." |
| wheels | "Draw X cards, where X is your life total." | "Each player discards their hand and draws seven cards." |

Gemma 3 27B also resolved the compositional attention drift (correctly producing an ETB trigger on the token-creation query) and produced canonical rules text — *"Whenever you cast a spell, put a +1/+1 counter on this creature."* — for the creatures-that-get-bigger query, instead of over-narrowing to only Prowess.

The finding is that a ~3× parameter jump within the same non-reasoning architecture class closes the jargon knowledge gap. The failures observed at 8B are not architectural — they reflect the domain-knowledge ceiling of a smaller model. Larger models know more MTG.

### 5.3 Full cascade evaluation — base embedder (pre-fine-tuning)

*[DRAFTED — REVIEW. Rows logged to `experiment_runs` 2026-09-18 (ids 11–14, eval set v1-draft). These are the base-embedder cells of the Section 3.4 grid: the control every fine-tuning claim is measured against. Tuned-embedder rows and the Scryfall comparator are pending.]*

All four base-embedder configurations were run over the 26-query evaluation set with the untuned Nomic Embed v1.5 encoder, the Llama 3.1 8B rewriter, and the v1 prompt. The keywords filter was off (selective strictness, Section 6.2). Precision@10 is reported alongside recall@10 because recall is capped by relevant-set size on this evaluation set — a query with 34 relevant cards can score at most 0.29 at K=10 with a perfect top ten — which makes recall unreadable as a headline number.

**Table 1 — Aggregate results, base embedder (n = 26 queries, macro-averaged).**

| Configuration | Stage 1 | Stage 2 | P@10 | R@10 | MRR | p50 latency |
|---|---|---|---|---|---|---|
| Raw dense (floor) | none | none | 0.031 | 0.017 | 0.067 | 83 ms |
| Pass-through | filters only | SQL | 0.050 | 0.028 | 0.130 | 985 ms |
| –SQL ablation | filters + hypothetical | none | 0.085 | 0.036 | 0.269 | 967 ms |
| **Full cascade (control)** | filters + hypothetical | SQL | **0.112** | **0.050** | **0.294** | 978 ms |

Each stage adds, and the ordering is monotone. Removing the SQL pre-filter from the full cascade costs 0.027 P@10 and 0.025 MRR; removing the hypothetical text (pass-through) costs 0.062 P@10 and 0.164 MRR. On the base embedder the rewriter's hypothetical text is doing more work than the filter — which is the expected picture before fine-tuning, since the untuned encoder cannot bridge jargon on its own and depends on Stage 1 to translate it. Stage 1 accounts for roughly 0.9 s of the ~1 s median latency; Stages 2 and 3 together run under 100 ms at this corpus size.

**Table 2 — Precision@10 by query category, base embedder.**

| Category (n) | Raw dense | Pass-through | –SQL | Full cascade |
|---|---|---|---|---|
| natural (4) | 0.000 | 0.000 | 0.075 | 0.100 |
| jargon (13) | 0.054 | 0.062 | 0.123 | 0.146 |
| fragmented (3) | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid (2) | 0.000 | 0.000 | 0.100 | 0.050 |
| constrained (3) | 0.000 | 0.133 | 0.000 | 0.133 |
| mechanical (1) | 0.100 | 0.100 | 0.100 | 0.100 |

Three patterns in the per-category breakdown carry into the fine-tuning design:

- **Constrained queries depend entirely on Stage 2.** The three purely structural queries ("instants that cost 1 mana", "red creatures under 3 mana", "free counterspell") score zero in both no-filter configurations and 0.133 in both filtered ones, regardless of what text is embedded. This is the pre-filter's contribution in isolation.
- **Jargon is where the hypothetical text earns its place — and where it still fails.** The 13 jargon queries move from 0.054 (raw) to 0.146 (full cascade), but 8 of the 13 still score zero, including *flicker effects*, *ramp spells*, *tutor*, *wheels*, *ETB triggers*, and *mana dorks*. Two jargon queries hit zero candidates from rewriter hallucinations: "mana dorks" produced an invented subtype `Dork` with power/toughness 1/1, and "red pingers" placed power/toughness filters on instants and sorceries. These are the jargon-gap family from Section 5.1 reproduced at the retrieval level.
- **Fragmented queries fail everywhere.** All three ("card draw engines", "graveyard recursion", "a card that lets me look at my deck and put a creature…") score zero in every configuration. The rewriter does not recover the intent, and the raw encoder does not either.

Six queries returned filters but no hypothetical text, so the full cascade embedded the raw query for them. Two of those — "creatures with flying" and "haste creatures" — name a canonical keyword explicitly; with the keywords filter off they scored zero, which is the user-explicit case selective strictness is meant to keep strict (Section 6.2). A rule that applies the keywords filter only when the rewriter judged the query purely structural (filters present, no hypothetical text) is the cheapest candidate and is queued as an ablation row.

*[TODO — tuned-embedder rows of the grid after M6 training; Scryfall comparator with result-parity and query-complexity measures; direct-tag-lookup ablation.]*

---

## 6. Discussion

### 6.1 Key challenges observed during development

*[YOUR PROSE, structured as bullet points converted to prose]*

Four challenges dominated the M4 development phase:

**1. The base HyDE model struggles with domain-specific mechanics, jargon, and keywords.** As documented in Section 5, the small (8B) instruction-tuned model cannot reliably translate MTG-specific player language into the canonical Wizards-authored rules text required for effective semantic retrieval. This failure is not correctable through prompt engineering alone; it reflects the model's underlying training-data coverage of MTG-specific vocabulary.

**2. HyDE prompt engineering compounds in complexity and adds latency.** As the prompt is refined to handle more edge cases — different jargon shapes, canonical-keyword hallucination guards, color-filter semantics, over-narrowing prevention — the system prompt grows longer, few-shot examples multiply, and per-query inference costs increase. Rules and examples that address one failure mode can conflict with rules addressing another. This is a general limit of prompt engineering as an intervention for domain-specific behavior.

**3. HyDE tends to over-rely on attribute filtering when a canonical filter closely matches part of a query.** The Prowess case in Section 5 is one example; a similar pattern was observed with Trample and other keyword mechanics. Both Prowess and Trample are legitimate canonical keywords, but many cards without those specific keywords exhibit similar ability profiles. Filtering on keywords should be reserved for queries where the keyword is explicitly mentioned; when the model infers a keyword from a broader natural-language description, the filter is too restrictive.

**4. The default embedding model cannot bridge domain-specific jargon-to-canonical-text asymmetries on its own.** Even if HyDE produced perfect canonical rules text every time, the encoder must still embed that text close to the actual card text in vector space. A general-purpose encoder trained on general web text has no MTG-specific knowledge to know that "flicker" and "exile-then-return-to-battlefield" are semantically equivalent. The reminder-text corpus augmentation partially addresses the corpus side but does not resolve the query-side asymmetry.

### 6.2 The selective strictness design decision

*[DRAFTED — REVIEW]*

The over-narrowing failure family in Section 5.1 surfaced a specific architectural question: should the SQL pre-filter apply the same strict semantics to model-inferred attributes as to user-explicit attributes?

Three positions are defensible: (i) strict pre-filter on all fields (the current baseline), where every filter is a hard WHERE clause; (ii) loose pre-filter where filters act as rank-boosts rather than hard constraints, allowing non-matching cards to remain in the candidate set at lower rank; and (iii) selective strictness, where user-explicit attributes (a color the user typed) remain strict but model-inferred attributes (a keyword the model guessed) are treated as soft.

Selective strictness offers the best tradeoff for this cascade. It preserves the recall benefit of pre-filtering on genuinely-constrained queries while limiting the blast radius of the model's inference errors. It also requires distinguishing user-explicit from model-inferred in the pipeline, which is a small structural addition to the HyDE output contract.

### 6.3 The path toward fine-tuning

*[YOUR PROSE, lightly edited — 2026-09-18. Facts corrected: Scryfall now ships tags as an official bulk file, no scraping.]*

The more tests we ran, the more fine-tuning looked like the best option. Given the observed limits of prompt engineering and off-the-shelf models, we begin a second version of the three-stage retrieval system focused on fine-tuning. Scryfall is a great resource for this: through its community Tagger project it assigns functional tags to cards (`sweeper`, `cantrip`, `ramp`, `sacrifice-outlet-creature`, `flicker-creature`), and these tags provide a canonical mapping from player-language jargon to card-level anchors. The original plan was to slowly collect these tags through the search API within Scryfall's rate limits. That turned out to be unnecessary — Scryfall publishes the full tag set, including the tag hierarchy, as an official daily bulk file — so the tag-to-card data was loaded into the project database in one step (Section 3.1).

**Jargon is a broad category and will have different outputs for different words.** Ramp is jargon, not a keyword; it is a direct reference to ability text, so it would have a hypothetical rules text rather than a filter. Cantrip is a direct reference to a cheap instant or sorcery that draws a card, so it is mostly structural. To fix them all through the prompt would require a rule and an example per jargon shape. To fix them all at once requires specialized training on MTG keywords and jargon using the Scryfall tags.

**The hypothesis.** Fine-tuning the embedding model on the jargon, mechanics, and keywords using tags is what should simplify the HyDE pipeline. HyDE should not need to work as hard to rewrite and can instead focus primarily on filters. If the embedding model is trained to understand the jargon, the rewrite portion of HyDE is less demanding, so the prompt instructions and examples become simpler and focus on filter extraction. HyDE may even be guided to move toward tag vocabulary wherever possible — if the query matches an existing tag context, emit that concept — instead of expanding the query to match the large variance of rules text those tags represent. HyDE does not disappear; its workflow changes, and the rewriting and prompt-engineering examples get far simpler as a result.

Two fine-tuning methods are available, and the question was which to use for which stage:

**Fine-tune the embedder (direct).** Full contrastive fine-tuning of Nomic Embed v1.5 on (anchor, card) pairs derived from the tags — the anchor is the tag label, an alias, its description, or a synthetic player query — using in-batch negatives with tag-aware batching so that two cards sharing a tag never serve as each other's negatives. This teaches the encoder to embed "sweeper" close to sweeper cards and "flicker" close to flicker cards directly, without an intermediate rewriting stage. At 137M parameters the literature fine-tunes the whole model (Section 2.5); adapter methods are the fallback if the general-retrieval probe drops.

**Fine-tune HyDE (LoRA).** Low-Rank Adaptation (Hu et al., 2021) adds small trainable matrices beside the frozen 8B weights, so the model learns MTG-specific query-to-JSON behaviour without touching the base model or requiring a larger one. This is the second step, taken only if the rewriter's *structural* failures — filter over-narrowing, compositional attention drift — persist after the embedder is tuned.

The order of operations is: tag ingestion first (done); the base-embedder control run through the full cascade second, so every fine-tuning claim has a before-number; embedder fine-tuning third; the v2 prompt fourth; HyDE LoRA only if measurements demand it.

### 6.4 What the tuned embedder adds over a tag lookup

*[DRAFTED — REVIEW; pairs with the accessibility prose in Section 1.1]*

Once HyDE can name a tag, the obvious shortcut is to look the tag up in `card_tags` and return those cards. That path is Scryfall's `otag:` search rebuilt locally, and it fails in two ways: it only knows concepts the community has tagged, and it only finds cards the community has tagged. Routing the concept through the tuned embedder is what generalises to untagged cards and to phrasings no tag covers. The direct-tag lookup is kept as an ablation row precisely so this can be measured: the held-out-tag probe (tags withheld entirely from training, queried by their label after training) shows whether the model learned that jargon names a *function* or merely memorised a vocabulary list. The accessibility claim (Section 1.1) and the generalisation claim reinforce each other — parity with expert Scryfall on tagged concepts, plus coverage beyond them.

### 6.5 Interactive filter refinement (future work)

*[DRAFTED — REVIEW]*

An observation from testing suggests a natural production extension: HyDE's structured filter output could be exposed to users as editable UI controls after the initial results display. In such a flow, the user submits a natural-language query, HyDE produces filters and a hypothetical card, the initial results display alongside editable filter controls (color pickers, mana-value sliders, type checkboxes), the user adjusts filters as needed, and the pipeline re-runs against the adjusted filters while reusing the original hypothetical card. HyDE inference — the expensive stage — runs once; filter adjustment is a cheap Postgres query.

This is out of scope for the current work (no production frontend planned this term), but the cascade architecture supports it naturally: a `HyDEResult` object's `filters` field can be modified in place before being passed to the search orchestrator without any pipeline change. The pattern reinforces the accessibility framing — non-experts get LLM-driven filter proposals, experts can override and refine.

### 6.6 Limitations

*[DRAFTED — REVIEW]*

Several limitations of the current work are worth explicit acknowledgement:

- **Fine-tuning is in-distribution by design.** Training on the tag `cantrip` and evaluating on the query *"cantrips that dig"* is the intended intervention, not leakage, but it is disclosed as such. The held-out-tag probe (Section 6.4) is the honest complement.
- **No user study.** The ease-of-use claim rests on a query-complexity proxy (Section 3.4), not on measured user behaviour.
- **No naive dense retrieval baseline is reported.** The paper is anchored on outcome comparison against Scryfall rather than diff against an internal reference; the naive-baseline number would not carry methodological weight in this framing. Qualitative failure-mode observations from naive dense retrieval are documented via ad-hoc CLI test tools.
- **No dedicated –HyDE ablation.** Evidence for HyDE's contribution comes from (i) the Scryfall comparator, which represents SQL-heavy retrieval without HyDE, and (ii) the published HyDE literature on standard benchmarks. A dedicated within-corpus –HyDE ablation is future work.
- **Single-curator evaluation set.** Voorhees (2000) provides the methodological defense for single-curator retrieval evaluation, but the constraint remains a real one.
- **English-only corpus.** The Scryfall `oracle_cards` bulk is English-only; multilingual evaluation is future work.
- **Small corpus (~30K cards).** The pre-filter cost/benefit story may differ meaningfully at web scale.
- **Encoder generation held constant.** Modern SOTA encoders (E5-Mistral, BGE, Qwen3-Embedding) are not compared; the encoder was held constant to isolate the cascade's contribution. Multi-encoder comparison is future work.

---

## 7. Conclusion and Future Work

### 7.1 Conclusion

*[DRAFTED — REVIEW, brief and restated from Introduction contributions]*

This work presents a three-stage retrieval cascade for natural-language semantic search over a bounded consumer catalog, using MTG as the test bed. The cascade combines HyDE-style query rewriting, structured SQL pre-filtering, and dense vector search inside the pre-filtered candidate set. Development surfaced concrete failure modes of the base approach — jargon knowledge gaps, over-narrowing on keyword filters, compositional attention drift — most of which are resolved by moving to a larger base model but persist as evidence that domain-specific fine-tuning is the natural next intervention. The methodological framing (outcome comparison against a domain-standard tool rather than diff against an internal baseline) and the accessibility contribution (bringing expert-level search results to non-expert users) are the paper's core arguments.

### 7.2 Future work

*[DRAFTED — REVIEW]*

Embedder fine-tuning on oracle tags is now in scope (Section 6.3); HyDE LoRA fine-tuning remains future work unless measurements after the embedder fine-tune justify it. Beyond that:

- **Multi-encoder comparison.** Evaluating the cascade against SOTA embeddings (E5-Mistral, BGE, Qwen3-Embedding) at the encoder position.
- **Interactive filter refinement.** Implementing the user-editable filter pattern described in Section 6.5 and measuring its effect on user satisfaction.
- **User study.** Replacing the query-complexity proxy with measured novice task success and time-to-result against Scryfall.
- **Co-training rewriter and encoder.** Following CoHyDE (Senthil et al., 2026), iteratively training the HyDE model on the tuned encoder's retrieval scores and vice versa.
- **Cross-domain generalization test.** Applying the cascade to a non-MTG consumer catalog (e-commerce, media library, technical documentation) to test the generalization claim empirically.

---

## References

*[TODO — compile full bibliography from docs/sources/README.md. BibTeX blocks for the three IR-eval papers are already in data/eval/methodology_references.md.]*

Primary references cited in this draft:

1. Manning, C.D., Raghavan, P., Schütze, H. (2008). *Introduction to Information Retrieval*, Ch. 8.
2. Reimers, N., Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP.*
3. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP.*
4. Nigam, P., et al. (2020). Semantic Product Search for Matching Structured Product Catalogs in E-Commerce.
5. Thakur, N., et al. (2021). BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.
6. Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
7. Wang, L., et al. (2011). A Cascade Ranking Model for Efficient Ranked Retrieval. *SIGIR.*
8. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback (InstructGPT).
9. Gao, L., et al. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE).
10. Wang, L., et al. (2023). Query2doc: Query Expansion with Large Language Models.
11. Nussbaum, Z., et al. (2024). Nomic Embed: Training a Reproducible Long Context Text Embedder.
12. Muennighoff, N., et al. (2022). MTEB: Massive Text Embedding Benchmark.
13. Järvelin, K., Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR Techniques. *ACM TOIS.*
14. Voorhees, E.M. (2000). Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness.
15. Sormunen, E. (2002). Liberal Relevance Criteria of TREC.
16. [2026] An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System.
17. [2025] Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study.
18. [2025] Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support.

---

## Appendices

### Appendix A — HyDE prompt (v1)

*[Include the final version of `prompts/hyde_v1.yaml` verbatim in the appendix. Full system prompt with schema, rules, few-shot examples.]*

### Appendix B — Evaluation set

*[Include `data/eval/queries_v1_draft.yaml` — the 26 queries with tri-state relevance judgments.]*

### Appendix C — Per-query results

*[TODO — populated after M5 measurements land. Full recall@10 and MRR per (query, configuration) pair.]*

### Appendix D — Failure-mode test series

*[The 10 diagnostic test queries used to characterize the base HyDE model's failure modes (Section 5.1), with full HyDE output for each and side-by-side comparison against the larger model (Section 5.2).]*
