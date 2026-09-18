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

### 1.2 Contributions

*[DRAFTED — REVIEW]*

This work makes four contributions:

1. **Architectural.** A three-stage retrieval cascade combining HyDE-style query rewriting, structured SQL pre-filtering, and dense vector search inside the pre-filtered candidate set — designed to address query-document asymmetry without domain-specific fine-tuning as a starting point.
2. **Methodological.** An evaluation methodology anchored on real-world outcome comparison against the domain-standard tool (Scryfall) with LLM-crafted expert queries, rather than diff against an internal baseline. Per-component contribution is quantified through a targeted ablation of the SQL pre-filter stage.
3. **Empirical.** Documentation of specific failure modes observed with an off-the-shelf small (8B parameter) instruction-tuned LLM as the HyDE model, including jargon knowledge gaps, over-narrowing on keyword filters, and compositional attention drift. Comparison against a larger (27B parameter) model shows that most of these failures resolve with more parameters, informing the case for domain fine-tuning.
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

### 2.5 IR evaluation methodology

*[DRAFTED — REVIEW]*

The evaluation methodology's use of tri-state graded relevance judgments is grounded in Järvelin and Kekäläinen (2002), which established graded relevance as standard IR practice, and Sormunen (2002), which showed empirically that a large fraction of TREC-"relevant" documents are marginally relevant — motivating the middle-bucket separation. Voorhees (2000) provides the methodological defense for single-curator evaluation sets producing stable comparative rankings.

---

## 3. Methodology

### 3.1 Task and dataset

*[DRAFTED — REVIEW]*

The task is: given a user's natural-language query, return the top-K most semantically relevant MTG cards from a corpus of ~30,000 unique cards.

**Corpus.** The Scryfall `oracle-cards` bulk dataset (2026-09-11 snapshot) contains 38,740 raw entries. Filtering rules exclude non-card layouts (tokens, emblems, art series, vanguard, planar, scheme), digital-only printings (Arena/MTGO exclusive), silver-bordered cards, memorabilia set-types, novelty Un-set (`set_type=funny`) products, and token-only booster products (`set_type=token`). The resulting corpus contains 31,972 face rows across 31,124 unique `oracle_id`s (80.3% retention rate). Multi-face cards (transform, modal-DFC, split, adventure) are stored as separate rows keyed on `(oracle_id, face_index)`; results deduplicate by `oracle_id` at display time.

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

Three configurations are compared:

1. **Full cascade** (HyDE + SQL pre-filter + Nomic Embed) — the main result.
2. **–SQL ablation** (HyDE + Nomic Embed, no pre-filter) — isolates the SQL pre-filter's contribution.
3. **Scryfall comparator** — LLM-crafted expert-level Scryfall queries evaluated against the same eval set. This serves as a "best-case SQL-heavy retrieval" reference: what Scryfall can do when operated by an expert-adjacent LLM constructing its queries.

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

### 5.3 Full cascade evaluation

*[TODO — populate after M5 measurements land. Table shape: config × recall@10 × MRR × per-category breakdown. Comparison against Scryfall LLM-crafted expert queries. Ablation delta for –SQL pre-filter.]*

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

*[YOUR PROSE + drafted synthesis, REVIEW]*

Given the observed limits of prompt engineering and off-the-shelf models, we begin planning a second version of the three-stage retrieval system focused on fine-tuning. The approach starts by scraping Scryfall for tag-to-card data — Scryfall assigns tags to cards through its Cardtags project (tags like `ramp`, `removal`, `tutor`, `card-draw`, `combo-piece`), and these tags provide a canonical mapping from player-language jargon to card-level anchors. Scryfall's API supports tag-based search directly, and their rate-limit guidelines are respectful (50-100ms between requests), making sanctioned ingestion straightforward.

The tag-to-card data is integrated into the project database and used as training signal. Two distinct fine-tuning interventions become available:

**Fine-tune the embedder.** Contrastive training on (query, positive_card) pairs derived from Scryfall tags, using `MultipleNegativesRankingLoss` on top of Nomic Embed v1.5. This teaches the encoder to embed "ramp" close to ramp cards, "flicker" close to flicker cards, and so on — directly, without an intermediate rewriting stage.

**Fine-tune HyDE.** Supervised training or LoRA (Low-Rank Adaptation, per Hu et al. 2021) on (query, correct JSON output) pairs. This teaches the HyDE model MTG-specific translations without requiring a large increase in model size.

The interventions are complementary but not redundant. A fine-tuned embedder shortens the pipeline for pure-jargon queries: if the encoder already understands "flicker" ↔ "exile then return" natively, HyDE no longer needs to canonicalize that mapping and can be reduced to its filter-extraction role. HyDE remains valuable for compositional queries (where multiple constraints must be extracted and combined) and for natural-language queries whose intent needs canonicalization before the encoder sees it.

The order of operations is: Scryfall tag ingestion first (prerequisite for both interventions); embedder fine-tuning second (smaller model, cheaper, higher leverage per unit effort, well-established SBERT recipe); HyDE fine-tuning only if measurements after the encoder fine-tune show HyDE quality remaining as the bottleneck.

### 6.4 Interactive filter refinement (future work)

*[DRAFTED — REVIEW]*

An observation from testing suggests a natural production extension: HyDE's structured filter output could be exposed to users as editable UI controls after the initial results display. In such a flow, the user submits a natural-language query, HyDE produces filters and a hypothetical card, the initial results display alongside editable filter controls (color pickers, mana-value sliders, type checkboxes), the user adjusts filters as needed, and the pipeline re-runs against the adjusted filters while reusing the original hypothetical card. HyDE inference — the expensive stage — runs once; filter adjustment is a cheap Postgres query.

This is out of scope for the current work (no production frontend planned this term), but the cascade architecture supports it naturally: a `HyDEResult` object's `filters` field can be modified in place before being passed to the search orchestrator without any pipeline change. The pattern reinforces the accessibility framing — non-experts get LLM-driven filter proposals, experts can override and refine.

### 6.5 Limitations

*[DRAFTED — REVIEW]*

Several limitations of the current work are worth explicit acknowledgement:

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

The primary future-work direction is fine-tuning, as discussed in Section 6.3. Scryfall tag ingestion is the prerequisite; embedder contrastive fine-tuning is the next step; HyDE LoRA fine-tuning follows if measurements justify it. Beyond fine-tuning:

- **Multi-encoder comparison.** Evaluating the cascade against SOTA embeddings (E5-Mistral, BGE, Qwen3-Embedding) at the encoder position.
- **Interactive filter refinement.** Implementing the user-editable filter pattern described in Section 6.4 and measuring its effect on user satisfaction.
- **NL-to-SQL alternative architecture.** A parallel investigation of whether direct natural-language-to-SQL generation, leveraging Scryfall tags as a queryable filter attribute, offers an alternative or complementary path to the semantic search stage.
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
