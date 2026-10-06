---
title: "Natural-Language Semantic Search for Trading Card Catalogs: A Three-Stage Retrieval Cascade with Query Rewriting, Structured Pre-Filtering, and Dense Retrieval"
author: "Mitchell Trafford"
date: "Fall 2026 · CSCI 5953 · CSUSB"
---

# Paper Draft — Working Document

**Editing notes for Mitchell:** sections marked *[YOUR PROSE]* are pulled verbatim from your notes with minor grammar polish. Sections marked *[DRAFTED — REVIEW]* are drafted from our conversation and the outline; rewrite in your voice as needed. Sections marked *[TODO]* have writing prompts but need real results (M4/M5 measurements) or your input to populate.

---

## Abstract

> **STALE — to be rewritten by Mitchell (flagged 2026-10-03; not edited).** The abstract below predates the fine-tuning work and the measurements. Facts that have changed, all from `docs/reports/2026-10-03/`:
> - Stage 1 no longer has to produce a paragraph of hypothetical card text. The current rewriter (prompt v2) produces filters plus one to three concept phrases; the hypothetical text is a fallback.
> - The embedder was fine-tuned on 207,062 (tag, card) pairs from Scryfall's community oracle tags (71 minutes on a laptop). Retrieval on 482 held-out tags roughly doubled.
> - Measurements are complete, not "in progress". Parity with expert Scryfall result sets: 0.34 (first version) → 0.57 (current). Nine of 26 queries reach 0.85 or better.
> - The reported measures changed from recall@10 / MRR to set-retrieval measures (R-precision against the full result set; parity with the expert's Scryfall result set), because the goal is every matching card in ranked order, not ten archetypes.
> - Comparators: the first version (v1 rewriter, base embedder), the same cascade with the SQL pre-filter removed, and expert Scryfall queries with and without community tags.
> - The claim is parity for non-experts, not beating Scryfall: 3.7 words of plain language versus 20–52 characters of query syntax.

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

> **STALE — to be rewritten by Mitchell (flagged 2026-10-03; not edited).** What the list below does not yet reflect:
> - Contribution 1 says the design avoids domain fine-tuning "as a starting point"; fine-tuning the embedder is now a central part of the result.
> - A finding not listed: the two changes depend on each other. The simpler rewriter is worse than the original on the base embedder (0.169 vs 0.215 R-precision) and best on the tuned one (0.510).
> - A methodological finding not listed: hand-curated top-10 judgments under-credit correct results (annotation holes); 95% of the tuned model's top results were members of the intended tag though only 10% were judged.
> - The Scryfall comparison now has numbers (parity 0.57; query-complexity table).
> - Contribution 4 (interactive filter refinement) exists as a review dashboard, not a user-facing feature.

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

Hypothetical Document Embeddings (HyDE), introduced by Gao et al. (2022), is the primary technique adapted in this work. HyDE addresses the query-document asymmetry by using an instruction-tuned generative language model (Ouyang et al., 2022) to transform a natural-language query into a hypothetical target document. The hypothetical document, rather than the raw query, is embedded and used for retrieval. Query2doc (Wang et al., 2023) is a concurrent technique that concatenates the hypothetical document with the query rather than replacing it. Best-practices studies on LLM query expansion (Zhang et al., 2024) and the recent adaptive-HyDE literature (Lei et al., 2025) address extensions and refinements.

A critique worth engaging with is the "Rethinking LLM-based Query Expansion" work (Yoon et al., 2025), which argues that observed HyDE gains may partially reflect LLM knowledge leakage from pre-training rather than pure query rewriting. In the MTG-specific setting studied here, this concern is somewhat mitigated: MTG's Oracle text is a bounded, formal, well-known corpus, and the empirical failure modes observed with smaller models suggest that domain knowledge is often *absent*, not leaked.

### 2.3 Filtered vector search

*[DRAFTED — REVIEW]*

The SQL pre-filter design decision is grounded in the filtered ANN literature. An in-depth study of filter-agnostic vector search on a PostgreSQL database system (Lu et al., 2026) is directly applicable — this work uses pgvector as its retrieval backend. Recent experimental studies of attribute filtering in ANN search (Li et al., 2025) and benchmarks on transformer-based embedding vectors (Iff et al., 2025) inform the pre-filter-versus-post-filter design decision made here: pre-filtering preserves recall on structurally-constrained queries, while post-filtering top-K vector results collapses recall in those cases.

The cascade architecture itself follows the multi-stage retrieval tradition established by Wang et al. (2011) — filter-then-rank pipelines rather than parallel dual-encoder ranking (Guo et al., 2016).

### 2.4 Semantic product search in consumer catalogs

*[DRAFTED — REVIEW]*

The generalization argument in Section 1.1 draws directly on Home Depot's semantic product search work (Choi et al., 2020), which addresses the same query-document asymmetry problem in an e-commerce catalog. Their approach — combining structured product attributes with semantic retrieval — is architecturally similar to the cascade proposed here, providing a published precedent for the generalization claim.

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

Our system is given a user query in plain language and returns the full set of cards that could match that query, ranked by how well they match, from a corpus of 31,124 cards. A result is measured against the results an expert could get from Scryfall's own search syntax on a similar query, and success is determined by comparing the two. The two resulting sets are compared with the Scryfall set as the standard to match, comparing how well our system can achieve similar results to an expertly crafted Scryfall search given a plain-language input.

Scryfall provides a full database dump for limited use including research, which includes 38,740 entries as of our snapshot of 2026-09-11. The data was ingested into a local database and scrubbed of all cards that were not relevant for our purposes, including tokens, emblems, art cards, schemes, and planes (3,739). Additionally, digital-only cards (2,058), Un-set novelty cards (998), silver-bordered cards (458), memorabilia (350), and token-only products (13) were removed, and the remaining legal cards resulted in 31,124 cards. Multi-face cards (transform, modal double-faced, split, adventure) each had their own respective rows per face, for 31,972 rows. Finally, we embedded each face separately to retain the abilities of the individual pieces, and the results are merged to one card during search.

Scryfall also provides a separate bulk file for tags, which are a community-maintained resource including parent and child tags. These tags provide useful labels that identify what a card does or what category the card's abilities fall into (e.g. "sweeper", "ramp", "cantrip"). 99.4% of cards in our set carry at least one tag, and 3,022 tags cover five or more cards in the set. Tags are used as the training signal for the embedder fine-tuning and are not embedded themselves or used as a filter directly.

### 3.2 Three-stage retrieval cascade

The system was designed to function in three cascading stages: the HyDE rewriter (Stage 1), the SQL pre-filter (Stage 2), and the ranking embedder (Stage 3). Original work focused only on Stage 3 embedding and data conditioning and inspired the additional research which led to the addition of Stages 1 and 2.

#### 3.2.1 Stage 1 — Query rewriter

The first stage uses Llama 3.1 8B Instruct, 4-bit, served locally through `mlx_lm.server` on an OpenAI-compatible endpoint. The model achieved an average of 57 tokens/s and used 4.8 GB of unified memory on an M3 Max MacBook Pro. The user's input query is served to this stage directly, and it outputs a JSON object which is passed to Stage 2.

The final version of our prompt (v2) has three fields to instruct the model how to handle the rewriting and how to format the JSON object output: `filters` (only from attributes the user types directly), `concepts` (1–3 short phrases in deckbuilder vocabulary, preferring community terms like those given by oracle tags), and `hypothetical_card` (one sentence, used only when no concept fits).

Our control prompt (v1) had two fields, `filters` and a full hypothetical card rules text, after Gao et al. (2022) (HyDE). v1 and v2 had 11 rules, 7 examples, 2,079 request tokens, 42 output tokens and 8 rules, 8 examples, 1,414 request tokens, 28 output tokens respectively. Temperature for both is 0. The client keeps only the first JSON object the model outputs.

#### 3.2.2 Stage 2 — SQL pre-filter

The second stage is where the identified attributes from Stage 1 become parameterised SQL `WHERE` clauses over Postgres columns. These attributes include colors, color identity, types, subtypes, mana value, power, toughness, format legality, and keywords. The rules during this stage are as follows: a color filter always includes colorless cards unless the query instructs exactly certain colors, and an empty color list is not constrained. Multiple types are ORed and subtypes are ANDed. Keyword filters are maintained only if the keyword exists in the corpus. Under v1, keyword filters were switched off entirely because the model inferred keywords from paraphrases in the query and the hard filter then removed every card that achieved the same effect without that keyword. In both v1 and v2 this stage provides the pre-filtering to achieve better matching within the desired subset of cards during the embedding stage.

#### 3.2.3 Stage 3 — Semantic ranking

Concept phrases from v2 or hypothetical text from v1 are embedded with Nomic Embed v1.5 (137 million parameters, 768 dimensions). There were two checkpoints for this stage: the stock model (control) and the fine-tuned model (`nomic-mtg-v1`). Even though some queries are functionally fully handled during Stage 1 and Stage 2, Stage 3 always runs. Therefore, even on a pure filter query, candidates are still ranked by the embedding stage. Every candidate that survived the filter stage is then scored by cosine similarity exactly, with no approximate index. The entire ranked candidate list is the result and is paginated in the UI.

Storage for the system is a single Postgres database with the pgvector extension that contains both the card attributes and the vectors (one row per face per model version) to handle both filtering and ranking with a single query.

### 3.3 Corpus text preparation

The corpus text was prepared using keyword augmentation to unify the document space and remove possible ambiguity between cards that have only a keyword and cards that have both keywords and reminder text for those keywords. Example: a card that says only "Flash" gets "Flash (You may cast this spell any time you could cast an instant.)" added. Of our 31,972 entries, 9,929 faces had text added to them, and our dictionary covers 79% of keyword occurrences in the corpus. Our dictionary covers 240 definitions that are used for augmentation. 143 definitions were parsed automatically from reminder text in the existing database and 97 were hand-written (58 replacing narrow or incorrect harvested definitions and 39 new). Deliberately skipped keywords were Enchant, Food, Gift, and Machina. The excluded keywords had no correct generic wording to substitute or had multiple conflicting references that made augmentation unfeasible. Augmentation was applied early in the research process and was used in fine-tuning pairs. However, augmentation was never tested on its own, so its effect is unmeasured and remains an oversight in the research.

Finally, Nomic Embed requires a task prefix on every input, so each card text is prefixed with `search_document: ` before embedding and each query-side text with `search_query: ` at search time; without these prefixes retrieval quality drops.

### 3.4 Evaluation setup

Twenty-six plain-language queries were hand-written to cover six kinds of input: jargon (13), natural language (4), fragmented (3), constrained (3), hybrid (2), mechanical (1). Jargon queries dominate the queries used for evaluation by design, to address the root question the project aimed to answer. Other categories are too small to support any claims on their own. The full list is in Appendix B.

Each query has two reference sets:

1. **Tag pool.** 21 of the 26 queries map to the one Scryfall oracle tag that names the query's concept; the reference set is every corpus card carrying that tag or one of its child tags. The five remaining queries have no functional tag (keyword and structural queries) and are scored only by the second set.
2. **Expert Scryfall query.** For all 26 queries, a Scryfall query is created to achieve the result an expert would get for the same request, in two versions: one with community tags allowed (`otag:`) and one without. The resulting set, restricted to the corpus, is the reference. The Scryfall queries were generated by Claude (a different model family from the rewriter) and are reviewed by me.

Results were measured using the full ranked candidate list (not top-k) against the reference set. R-precision shows the share of the system's first N results that are in the reference set; 1.0 means the first N results are exactly the reference set. Reachable is the share of the reference set that survives the Stage 2 filter, to isolate what the filter costs. R-precision (reachable) isolates ranking quality by counting only the cards the filter admitted. Depth to 90% is how far down the list a user reads to have seen 90% of the reference set, as a multiple of N. Measures against the expert set use the same R-precision, plus Jaccard overlap at depth N. Finally, we measure query complexity for each query by the length and number of operators of the expert's Scryfall query against the plain-language query, and whether `otag:` was needed. This serves as our accessibility proxy measurement, since no real human users were used.

Every evaluation run writes one row to an `experiment_runs` table with configuration, prompt version and hash, embedder version, and per-query results. All tables and figures for the results section are created from those rows by `generate_report.py`.

Interpreting the results has some nuances worth mentioning. Each query is about 1/21 (0.048) of a tag-pool average and 1/26 (0.038) of an expert-set average, so differences near those values are a single query. Tag membership is also the training signal, and the no-tag expert queries and the held-out tags are the checks that don't share it. The expert's query just represents one possible expert's choice. A syntax-free query is ambiguous, so our expert set is a standard to match, not a guarantee of sameness.

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

### 5.4 Embedder fine-tuning on oracle tags

*[DRAFTED — REVIEW. First run `nomic-mtg-v1`, 2026-09-18, `experiment_runs` id 19. Recipe per Section 6.3; run details in the 2026-09-18 journal §5b.]*

The base encoder was fine-tuned for one epoch on 207,062 (anchor, card) pairs derived from 2,729 oracle tags, with 482 tags (15%, stratified by pool size) withheld entirely. Anchors were tag labels, aliases, and descriptions; positives were reminder-augmented Oracle text; the loss was in-batch-negative InfoNCE with tag-disjoint batching so that no card sharing a tag with an anchor could be sampled as its negative. Training took 71 minutes on an M3 laptop GPU. The training loss fell from chance (ln 256 ≈ 4.2) to 1.46; the in-training probe flattened after roughly 600 of 809 steps, so one epoch is near the knee.

**Table 3 — Retrieval probes before and after fine-tuning (cosine, query = tag label).**

| Probe | ndcg@10 | mrr@10 | map@10 | recall@100 |
|---|---|---|---|---|
| Training tags, 6.2k-card sub-corpus (in-distribution) — base | 0.195 | 0.249 | 0.150 | — |
| Training tags — tuned | 0.536 | 0.600 | 0.467 | — |
| **Held-out tags, full 31k corpus — base** | 0.114 | 0.193 | 0.079 | 0.164 |
| **Held-out tags — tuned** | **0.242** | **0.325** | **0.187** | **0.327** |

The in-distribution probe confirms the objective is doing what it should. The held-out probe is the result that matters for the argument in Section 6.4: on 482 functional tags the model never saw during training, queried by their label alone against the entire corpus, every metric roughly doubled. The encoder did not memorise a vocabulary list; it learned that a short functional phrase in player language maps to a region of rules text, and that mapping transferred to phrases it had not been shown. Absolute held-out scores are modest because many held-out tags are narrow (`bounceland`, `blood-artist-ability`) and compete with 31,000 distractors; the doubling of recall@100 is the cleaner read of the ranking shift.

**Forgetting probe.** NanoBEIR ndcg@10 on three general-domain subsets, base → tuned: SciFact 0.731 → 0.736, FiQA 0.485 → 0.441, NFCorpus 0.326 → 0.309; mean 0.514 → 0.495. The fine-tune costs two points of general retrieval on average and four on the most query-like subset — mild, uneven forgetting consistent with the unregularised runs reported by Murtaza et al. (2026) and ChEmbed (2025), and nowhere near the collapse Murtaza et al. observed with aggressive settings. For a domain-specialised deployment this is an acceptable trade; the embedding-anchor regulariser or base-weight interpolation (Section 2.5) is the planned ablation should a later recipe push the loss past a few points.

**Table 4 — The Section 3.4 grid with the tuned embedder (rows 20–23 vs 11–14; n = 26; keywords filter off).**

| Configuration | Query-side text | P@10 base → tuned | MRR base → tuned |
|---|---|---|---|
| Raw dense | user query, no filters | 0.031 → 0.069 | 0.067 → 0.102 |
| Pass-through | user query + HyDE filters | 0.050 → 0.081 | 0.130 → 0.206 |
| Full cascade (v1 prompt) | hypothetical rules text + filters | **0.112** → 0.089 | **0.294** → 0.270 |
| –SQL ablation | hypothetical rules text, no filters | 0.085 → 0.077 | 0.269 → 0.196 |

The tuned embedder improved every configuration that embeds the user's own words and degraded every configuration that embeds hypothetical rules text. Both effects follow from the training data: anchors were short functional phrases, so the query-side representation moved toward short phrases and away from paragraph-length rules text used as a query. On the jargon category, pass-through with the tuned embedder (P@10 0.115) approaches the v1 cascade with the tuned embedder (0.131) without generating any hypothetical text, and beats it on latency by the full cost of Stage 1 for that text. The best single cell on this evaluation set, however, remains the v1 cascade on the *base* embedder (0.112 / 0.294).

Two observations qualify the pass-through result as a test of the Section 6.3 hypothesis. First, pass-through embeds raw user phrasing, whereas the hypothesis has the rewriter normalise that phrasing toward tag vocabulary before the embedder sees it; the cell that tests the hypothesis is the v2 prompt combined with the tuned embedder, which is future work at the time of writing. Second, the training anchors expose each concept through a handful of strings — `sweeper` was trained with the aliases *wipe*, *boardwipe*, *mass removal*, and *wrath of god* — and the eval query *"board wipes"* still scored zero through pass-through while scoring 0.20 through hypothetical text. Single-string exposure per synonym is not enough to cover user phrasing; the synthetic-query anchor source (Section 6.3) is designed to close that gap. Fragmented queries remain at zero in every cell of the grid, tuned or not.

**The tag-label oracle.** *[DRAFTED — REVIEW; rows 26–29, journal §5d]* Before building a rewriter that emits concepts in tag vocabulary, its ceiling was measured directly: for each evaluation query a human selected the tag a flawless rewriter would emit, the harness embedded that tag's label in place of the user's words, and results were scored against the same judgments. The chosen tags cover 93–100% of each jargon query's relevant set, so the judgments are reachable through them. The oracle nonetheless scored **0.073 P@10 / 0.153 MRR** with the tuned embedder and filters (0.065 / 0.185 with the base), below both the pass-through row and the v1 control. Per-query precision falls monotonically with the tag's pool size: `extra-turn` (53 cards) 0.60, `fetchland` (53) 0.40, `wheel` (137) 0.20, `sweeper` (871) 0.10, and every tag above roughly 1,000 cards — `removal`, `ramp`, `recursion`, `draw` — 0.00.

The explanation is that a tag names a *category* while relevance judgments name its *archetypes*. "Ramp" covers 2,136 tagged cards; the evaluation set judges 28 of them relevant, the canonical ramp spells a player means by the word. The label embeds to the centre of the pool and retrieves ten arbitrary members; the hypothetical text "search your library for a basic land card and put it onto the battlefield" is more specific than the category and lands on the archetypes. This is the held-out probe's blind spot made visible: there, the whole pool counts as relevant, so category-level retrieval scores well; against human judgments it is too coarse. On judged precision alone the strong form of the Section 6.3 hypothesis — rewriter emits a tag, embedder does the rest — would look unsupported; the correction below shows that reading is an artefact of the judgments. What survives is the weaker and more useful form: the tuned embedder handles user phrasing directly (pass-through MRR 0.130 → 0.206), and the rewriter's job shrinks to filters plus a *short* hypothetical clause rather than a paragraph, which is measurable in output tokens. The hypothetical-text regression in Table 4 is a training-data effect, addressable by adding rules-text-shaped anchors, not evidence against the rewriter.

**Correction: the oracle's misses are annotation holes.** *[DRAFTED — REVIEW; journal §5e]* Scoring the same oracle runs against tag membership rather than the relevance judgments reverses the reading above. With the tuned embedder, 95% of the oracle's top-10 results are members of the intended tag's pool, against 35% for the base embedder and 56% for the v1 control; `ramp`, `recursion`, `flicker`, and `tutor` each go from 0-1 of 10 to 10 of 10, and `burn`, a tag withheld from training, also scores 10 of 10. The encoder learned the categories. The evaluation set judges roughly thirty archetypes per query out of pools of one to six thousand legitimate members and counts every unjudged member as a miss, the annotation-hole effect Thakur et al. (2021, §6) document for dense retrievers. Because the judgments were assembled from lexical Scryfall lookups, they favour cards that the hypothetical-text mode also favours. Neither metric alone settles the comparison: judged precision favours the control by construction and pool membership favours the oracle by construction (20 of 21 mapped tags were seen in training). The defensible claims are that the tuned embedder retrieves category members far more reliably than the base, that a bare label returns arbitrary rather than best-known members (a within-category ranking problem), and that the evaluation set needs its holes judged before the v2 comparison can be decided.

### 5.5 Set retrieval: the measure that matches the goal

*[YOUR PROSE, lightly edited — 2026-09-19]*

Relevance ranking is not actually the problem this system is trying to solve, and it is solvable separately. Scryfall gives the ability to sort results by EDHREC rank, which is a relevance marker based on popularity: cards that show up high in EDHREC sorting are the cards used most commonly and will naturally be the cards people are looking for. Scryfall filtering by tag with other basic attribute filtering does not give relevance-ranked results; it gives the set of results, good or bad. The metric that matters is therefore measured against that baseline. To restate the stance: we are not trying to produce a system that outperforms Scryfall's own capabilities. A stack that both produces the matching results and surfaces the most relevant automatically would be great, but it is beyond this scope. The goal is a system that provides every single card that could be a match, in descending order. Results are limited to a top *n* for evaluation, but in real use it would be all results split into *m* pages of *n* cards. Sorting by relevance then becomes no different from Scryfall's, assuming popularity sorting is a function available to both systems.

*[DRAFTED — REVIEW; journal §5f, `scripts/probes/set_retrieval_probe.py`]*

Under that goal the appropriate measures are set-retrieval measures against the full target set rather than precision over ten hand-picked archetypes: R-precision (precision at a depth equal to the size of the target set), P@100, and the depth a user must page to in order to have seen 90% of the set. Table 5 reports them for the 21 evaluation queries with a functional tag, using the tag's full pool as the target and no filters.

**Table 5 — Set retrieval against tag pools (mean over 21 queries; depth as a multiple of pool size, median).**

| Query-side text | Embedder | P@100 | R-precision | Depth to 90% |
|---|---|---|---|---|
| Tag label | tuned | **0.82** | **0.73** | **1.8×** |
| User's raw query | tuned | 0.74 | 0.60 | 3.4× |
| Hypothetical text (v1 prompt) | tuned | 0.57 | 0.46 | 3.8× |
| Hypothetical text (v1 prompt) | base | 0.47 | 0.27 | 13.2× |
| User's raw query | base | 0.31 | 0.21 | 13.5× |
| Tag label | base | 0.29 | 0.19 | 16.4× |

The ordering of systems inverts relative to Table 4. The tuned embedder given the user's own words (0.60) outperforms hypothetical rules text on either embedder, and given tag vocabulary (0.73) it outperforms both; to see 90% of all tutor effects a user pages about 1,000 results instead of 12,000. `burn`, a tag withheld from training, reaches 0.92. This supports the Section 6.3 hypothesis in its original form: once the encoder understands the domain's functional vocabulary, the rewriter's job reduces to filter extraction and concept normalisation. Three caveats apply. Twenty of the 21 targets are tags seen in training, so the held-out results (Table 3, and `burn` here) carry the generalisation claim, and the expert-query Scryfall comparator remains the independent check. A dense ranking has no natural end, but this needs no stopping rule: only queries with no structural component rank the full corpus (11 of the 26 evaluation queries; the other 15 are pre-filtered to between 3 and 17,428 candidates), the vector search scores every candidate regardless of how many are returned, and the interface pages the full ranked set with the match score visible so users decide how far into the tail to look. Depth-to-90% is therefore a ranking-quality measure, and Table 5, run without filters, is pessimistic for the filtered majority. Finally, tag pools are community-curated and incomplete, so untagged true matches in the top ranks are counted as errors here; the human review in the results dashboard measures that directly.

**Table 6 — Set retrieval inside the cascade (Stage 2 filters applied; tuned embedder unless noted; n = 21).**

| Configuration | R-prec (full pool) | R-prec (reachable) | P@100 | Reachable | Depth to 90% |
|---|---|---|---|---|---|
| v1 cascade, base embedder (control) | 0.215 | — | 0.444 | 0.68 | 8.4× |
| v1 cascade, tuned | 0.326 | 0.517 | 0.572 | 0.68 | 3.0× |
| Raw query, no filters, tuned | 0.598 | 0.598 | 0.736 | 1.00 | 3.4× |
| Pass-through (filters + raw query), tuned | 0.456 | **0.641** | 0.713 | 0.68 | **1.2×** |
| Tag-label oracle + filters, tuned | 0.536 | **0.728** | 0.765 | 0.68 | **1.1×** |

*[DRAFTED — REVIEW; journal §5g]* Measured inside the cascade, the pre-filter shows both of its faces at once. On every filtered configuration only 68% of each tag pool survives Stage 2 ("reachable"). Part of that is correct narrowing the user asked for — "instants that draw cards" should exclude non-instants, and the tag pool is then the wrong target — but part is inferred narrowing the user did not ask for: "ramp spells" receives a types filter of instant/sorcery and loses 1,784 ramp permanents; "burn spell that deals 3 damage" receives a mana-value filter of exactly 1. Within what Stage 2 admits, however, the tuned embedder ranks well: R-precision over the reachable set reaches 0.64 for the user's own words and 0.73 for tag vocabulary, and a user reaches 90% of the admitted set within 1.1–1.2× its size, against 3.4× with no filter at all. The pre-filter improves ranking inside the set while reducing coverage of it; the two effects are reported separately because a single number would hide the trade. Both fixes belong to the rewriter: filters should be derived only from constraints the user typed (selective strictness, Section 6.2), and the evaluation target for filtered queries should be the tag pool intersected with the user-explicit constraint.

### 5.6 The v2 rewriter: concept normalisation with explicit-only filters

*[DRAFTED — REVIEW; journal §5h, rows 51–53]*

The second prompt version implements the Section 6.3 hypothesis directly. The rewriter emits (a) filters derived only from attributes the user literally typed, (b) one to three concept phrases in deckbuilder vocabulary with the community's standard term preferred ("board wipe" → "sweeper"), and (c) a one-sentence hypothetical only when no concept phrase captures the intent. Stage 3 embeds the concept phrases. Measured with the Llama 3.1 tokenizer, the v2 request is 1,414 tokens against 2,079 for v1 (−32%), with 8 rules against 11; the example count did not fall (8 vs 7) because the 8B model needed one demonstration per failure shape that a rule alone did not correct.

**Table 7 — v1 vs v2 on the tuned embedder, filters applied (n = 21 tag-mapped queries).**

| Rewriter | Query-side text | R-prec (full) | R-prec (reachable) | Reachable | P@10 | MRR | Stage 1 latency |
|---|---|---|---|---|---|---|---|
| v1 | hypothetical paragraph | 0.326 | 0.517 | 0.68 | 0.088 | 0.270 | 1.19 s |
| v1 | user's raw query | 0.456 | 0.641 | 0.68 | 0.081 | 0.206 | 1.06 s |
| **v2** | **concept phrases** | **0.558** | **0.693** | **0.81** | 0.096 | 0.234 | 1.38 s |
| v2 | one-sentence hypothetical | 0.533 | 0.625 | 0.81 | 0.085 | 0.153 | 1.39 s |
| — | tag-label oracle (ceiling) | 0.536 | 0.728 | 0.68 | 0.073 | 0.153 | — |

Explicit-only filtering raised coverage from 68% to 81% of each target set — "ramp spells", "mana dorks", "board wipes", and "burn spell" no longer receive an inferred type or cost — and concept phrases outrank both the paragraph-length and the one-sentence hypothetical on the same filters. The v2 cascade is the strongest real configuration measured, and on full-pool R-precision it exceeds the hand-mapped oracle because it over-narrows less. Archetype precision is flat, as expected once ranking-by-popularity is treated as a separate sort. Two residual failures are knowledge-gap cases the 27B comparison predicted: "destroy target artifact" still receives an inferred instant filter, and "red pingers" receives an invented Flying keyword. Latency did not fall with the shorter prompt: the inference server caches the shared prompt prefix, so per-query cost is dominated by output tokens (about 62), which v2's three-field JSON did not reduce; trimming null fields from the output is the obvious next step, and the simplification claim is stated here in request tokens and rule count rather than wall-clock time.

**Table 8 — Rewriter × embedder (R-precision against tag pools, filters applied, n = 21).**

| | Base embedder | Tuned embedder |
|---|---|---|
| v1 rewriter (hypothetical paragraph) | 0.215 | 0.326 |
| v2 rewriter (concept phrases, explicit-only filters) | 0.163 | **0.510** |

*[DRAFTED — REVIEW; journal §5i]* The two interventions are not independent. On the base encoder the lighter v2 rewriter is *worse* than v1: a bare concept phrase such as "ramp" carries no signal for an encoder that never learned the domain's vocabulary, whereas a paragraph of hypothetical rules text at least overlaps lexically with card text. On the tuned encoder the same rewriter is the strongest configuration measured. Fine-tuning the encoder is what makes the simpler, cheaper rewriter viable; neither half delivers the result alone. With a final output rule that omits unset fields, the v2 rewriter produces about 28 output tokens per query and runs in 0.77 s on the same 8B model, against 1.19 s for v1 — the simplification claim now holds in wall-clock terms as well as in request size, at a cost of a few queries whose filters shifted under the changed prompt (R-precision 0.558 → 0.510 on the tuned encoder across that change).

### 5.7 Comparison against expert Scryfall queries

*[DRAFTED — REVIEW; journal §5j; numbers from the generated report `docs/reports/2026-10-03/` (grid rows 88–101, comparator rows 102–129). Expert queries in `data/eval/scryfall_expert_queries_v1.yaml` are LLM-drafted and reviewed by the author; result sets fetched 2026-09-22. Corrected 2026-10-03: an earlier draft of this section mixed figures from a 21-query run into tables labelled 26 queries.]*

For each evaluation query an expert Scryfall search was written in two forms: with the community oracle tags (`otag:`) an expert would use today, and without them, from Oracle text and attributes alone. Each was run through Scryfall's search API (all pages, one request per second) and the result set restricted to the corpus. The cascade's full ranking was then scored against that set: R-precision (the fraction of the first |S| results that are in S), Jaccard overlap at depth |S|, and the depth needed to see 90% of S.

**Table 9 — Result parity with the expert's Scryfall result set (n = 26).**

| Cascade configuration | vs expert (tags) R-prec | Jaccard | Depth to 90% | vs expert (no tags) R-prec |
|---|---|---|---|---|
| v1 rewriter, base encoder (control) | 0.343 | 0.273 | 1.32× | 0.351 |
| v2 rewriter, tuned encoder | 0.523 | 0.446 | 0.98× | 0.391 |
| v2 rewriter, tuned encoder, keyword filter on | **0.565** | **0.492** | **0.95×** | — |

A depth below 1.0× means the user sees 90% of the expert's set before scrolling past as many cards as the set contains. Per query, the best configuration reaches parity of 0.85 or higher on nine of the 26 queries (instants that cost 1 mana 1.00, haste creatures 1.00, tutor 0.96, cheap blue counterspells 0.95, extra turns 0.94, counterspells 0.92, fetch lands 0.91, instants that draw cards 0.86, red creatures under 3 mana 0.85) and fails on the rewriter's known knowledge-gap cases. One "failure" is instructive: for *a card that destroys all creatures* the expert wrote `o:"destroy all creatures"` (84 cards) while the cascade retrieved the whole sweeper category (870 cards) — broader than the expert, not wrong.

The comparator is itself imperfect. Against the hand-curated judgments, the tagged expert sets have 9% precision and 92% recall; the untagged sets 14% and 77%. This is the annotation-hole effect of Section 5.4 seen from the other side, and it is why parity with the expert set rather than precision against thirty judged archetypes is the number reported.

**Table 10 — What the user had to type (mean over 26 queries; from `docs/reports/2026-10-03/tables/06_query_complexity.csv`).**

| | Plain language (this work) | Scryfall, tags allowed | Scryfall, no tags |
|---|---|---|---|
| Length | 3.7 words | 20 characters | 52 characters |
| Operators | 0 | 1.7 | 3.2 |
| Requires `otag:` | — | 81% of queries | — |
| Boolean grouping or negation | 0 | rare | common |

The expert path is short only because community tags exist, and using them requires knowing the tag's exact slug: `sweeper`, not "board wipe"; `counterspell-free`; `mana-dork`. Without tags the same intent takes three operators, quoted Oracle phrases, and boolean grouping, and still reaches only 77% recall of the judged cards. The cascade takes the 3.7-word query with no syntax and reaches 0.57 R-precision against the tagged expert's result set. That is the accessibility claim of Section 1.1 with a measurement attached: not that the system beats Scryfall, but that it reaches most of what an expert reaches without the user learning the grammar.

*[TODO — eval v2: pool top-10 across all logged rows, judge the holes, freeze; pairs_v2 with doc-like anchors and synthetic queries, retrain, re-run grid; hyde_v2 with a one-sentence hypothetical and trimmed rules (measure tokens/latency); Scryfall comparator; direct-tag-lookup ablation (expected to fail on broad tags for the same reason — the argument for embedding over lookup).]*

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

### 6.6 What review of the results shows

*[YOUR PROSE — 2026-09-22, from the dashboard review; lightly edited]*

Reviewing results visually surfaced both sides of the system in two searches. For *"destroy all creatures"* the rewriter mapped the query to the sweeper category, which is partially correct but broader than what was asked — Scryfall has no narrower tag either; `boardwipe` and its variants are aliases of `sweeper` — and by review the results are far less useful than Scryfall's literal `o:"destroy all creatures"`. Attempting to fix this by embedding the user's words alongside the concept repaired that query and lost equivalent ground elsewhere (Section 5.6: 0.486 vs 0.510 R-precision), so it is reported as an ablation rather than adopted. It is likely that simple attempts to fix individual failures now make overall results worse, and this drives home the point that while the idea is showing merit, there is still a lot of work needed to make it foolproof as a search tool.

In other cases the results are very good, especially on human-syntax searching. A search for *"is a planeswalker"* did exactly what one would expect right away and filtered by type; the same search on Scryfall returns nothing — the user would have to use the advanced interface or know to type `type:planeswalker`. A relatively simple task either way, but one has no prior-knowledge requirement compared to the other.

### 6.7 Limitations

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
- **Overlap (intersection) training.** *[YOUR PROSE]* Right now if a query closely matches one tag it embeds to that tag. If a query matches multiple tags, a multiple-tag search resulting in a more specific subset of cards may give better results: anchors built from tag pairs with positives drawn from the intersection would teach the encoder that a two-concept query sits inside the intersection rather than between the two categories. The co-occurrence data already exists in the tag table.
- **LLM-refined subtags.** *[YOUR PROSE]* An LLM could pre-process the tag pool to make it more specific. The community tags are a good starting point for training data, but some are too generic — `sweeper` covers creature wraths, land destruction, and mass bounce alike — and would need subtags to split such categories into more specific subsets. The LLM would evaluate each tag group for possible splitting and build a new training dataset; a cheap form clusters each large pool with the tuned encoder first and has the LLM name the clusters. Refined subtags are not community-validated, so evaluation must stay on the held-out judgments and the Scryfall parity rows.
- **Co-training rewriter and encoder.** Following CoHyDE (Senthil et al., 2026), iteratively training the HyDE model on the tuned encoder's retrieval scores and vice versa.
- **Cross-domain generalization test.** Applying the cascade to a non-MTG consumer catalog (e-commerce, media library, technical documentation) to test the generalization claim empirically.

---

## References

*Generated by `scripts/build_references.py` from `docs/sources/arxiv_metadata.json` and `extra_references.yaml`; lists the works cited in the text above. Do not edit by hand — re-run the script after changing citations. BibTeX for every source held: `docs/thesis/references.bib`.*

Choi, J. I., Kallumadi, S., Mitra, B., Agichtein, E., & Javed, F. (2020). Semantic Product Search for Matching Structured Product Catalogs in E-Commerce. *arXiv preprint*. arXiv:2008.08180.

Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. *arXiv preprint*. arXiv:2212.10496.

Gwon, D., Jedidi, N., & Lin, J. (2025). Study on LLMs for Promptagator-Style Dense Retriever Training. *CIKM 2025*. arXiv:2510.02241.

Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv preprint*. arXiv:2106.09685.

Iff, P., Bruegger, P., Chrapek, M., et al. (2025). Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on Transformer-based Embedding Vectors. *arXiv preprint*. arXiv:2507.21989.

Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems, 20(4), 422–446*. doi:10.1145/582415.582418.

Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP 2020*. arXiv:2004.04906.

Kasmaee, A. S., Khodadad, M., Astaraki, M., et al. (2025). ChEmbed: Enhancing Chemical Literature Search Through Domain-Specific Text Embeddings. *arXiv preprint*. arXiv:2508.01643.

Lan, M., Zheng, L., & Kilicoglu, H. (2026). BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels. *ACL 2026*. arXiv:2604.15591.

Lei, F., Mezouar, M. E., Noei, S., & Zou, Y. (2025). Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support. *arXiv preprint*. arXiv:2507.16754.

Li, M., Yan, X., Lu, B., et al. (2025). Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study. *SIGMOD 2026*. arXiv:2508.16263.

Li, M., Lv, X., Zou, J., et al. (2025). Query Expansion in the Age of Pre-trained and Large Language Models: A Comprehensive Survey. *arXiv preprint*. arXiv:2509.07794.

Lu, D., Caminal, H., Chatzakis, M., et al. (2026). An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System: [Experiments and Analysis]. *SIGMOD 2026*. arXiv:2603.23710.

Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. *Cambridge University Press (Chapter 8: Evaluation in information retrieval)*.

Moreira, G. d. S. P., Osmulski, R., Xu, M., et al. (2024). NV-Retriever: Improving text embedding models with effective hard-negative mining. *arXiv preprint*. arXiv:2407.15831.

Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022). MTEB: Massive Text Embedding Benchmark. *arXiv preprint*. arXiv:2210.07316.

Murtaza, S. S., Nie, Y., Soni, U., Wen, E., & Frydenlund, A. (2026). When Synthetic Data Hurts: On Catastrophic Forgetting in Skill Retrieval for LLM Agents. *EMNLP 2026 Industry Track*. arXiv:2609.10750.

Nussbaum, Z., Morris, J. X., Duderstadt, B., & Mulyar, A. (2024). Nomic Embed: Training a Reproducible Long Context Text Embedder. *Transactions on Machine Learning Research*. arXiv:2402.01613.

Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *arXiv preprint*. arXiv:2203.02155.

Pande, M., Kumar, S., & Damle, A. Y. (2025). When Fine-Tuning Fails: Lessons from MS MARCO Passage Ranking. *arXiv preprint*. arXiv:2506.18535.

Paull, N. (2025). CustomIR: Unsupervised Fine-Tuning of Dense Embeddings for Known Document Corpora. *arXiv preprint*. arXiv:2510.21729.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP 2019*. arXiv:1908.10084.

Senthil, V., Hathidara, A., & Schreiber, S. (2026). CoHyDE: Iterative Co-Training of LLM Rewriter & Dense Encoder for Tool Retrieval. *REALM Workshop at EMNLP 2026*. arXiv:2605.29271.

Sormunen, E. (2002). Liberal relevance criteria of TREC: counting on negligible documents?. *Proceedings of SIGIR 2002, 324–330*. doi:10.1145/564376.564433.

Tamber, M. S., Kazi, S., Sourabh, V., & Lin, J. (2025). Conventional Contrastive Learning Often Falls Short: Improving Dense Retrieval with Cross-Encoder Listwise Distillation and Synthetic Data. *arXiv preprint*. arXiv:2505.19274.

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS 2021 Datasets and Benchmarks Track*. arXiv:2104.08663.

Voorhees, E. M. (2000). Variations in relevance judgments and the measurement of retrieval effectiveness. *Information Processing & Management, 36(5), 697–716*. doi:10.1016/S0306-4573(00)00010-8.

Wang, L., Lin, J., & Metzler, D. (2011). A cascade ranking model for efficient ranked retrieval. *Proceedings of SIGIR 2011, 105–114*. doi:10.1145/2009916.2009934.

Wang, L., Yang, N., & Wei, F. (2023). Query2doc: Query Expansion with Large Language Models. *Proceedings of EMNLP 2023*. arXiv:2303.07678.

Yoon, Y., Jung, J., Yoon, S., & Park, K. (2025). Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion. *Findings of ACL 2025*. arXiv:2504.14175.

Zhang, L., Wu, Y., Yang, Q., & Nie, J. (2024). Exploring the Best Practices of Query Expansion with Large Language Models. *arXiv preprint*. arXiv:2401.06311.

**Cited but unresolved** (no publication identified — resolve or remove before submission):

- Guo et al., 2016 — Section 2.3 (dual-encoder / 'tower' terminology). Inherited from an early planning document. No source PDF is held and the intended paper was never recorded. Either identify it or drop the citation.

---

## Appendices

### Appendix A — Query-rewriter prompt (v2)

*Generated by `scripts/generate_report.py --paper` on 2026-10-03; do not edit by hand.* The original v1 prompt is `prompts/hyde_v1.yaml` in the repository.

**System prompt**

````text
You are a search-query rewriter for a Magic: The Gathering card database.

Given a user's natural-language query, produce a JSON object with three fields:
  1. "filters" — structured attributes the user EXPLICITLY stated. Null
     for anything the user did not literally say.
  2. "concepts" — one to three short phrases naming what the card DOES, in
     the vocabulary a deckbuilder uses: sweeper, removal, ramp, mana dork,
     mana rock, tutor, cantrip, card draw, counterspell, flicker, reanimate,
     recursion, burn, pinger, wheel, extra turn, fetchland, sac outlet,
     token maker, lifegain, discard, mill, evasion, prowess, ... Prefer the
     community's standard term over the user's phrasing when one exists
     ("board wipe" → "sweeper"; "creatures that tap for mana" → "mana dork").
     Empty list if the query is purely structural.
  3. "hypothetical_card" — ONE sentence of rules text, only when no concept
     phrase captures the intent (a novel or very specific mechanic).
     Otherwise omit the key.

## RULES

- Output ONLY valid JSON. No prose, no code fences. OMIT every filter field
  you are not setting — do not write "field": null. If there are no
  filters at all, omit the "filters" key too.
- Filters come only from words the user typed: colour words, numbers with
  "mana"/"cost"/"cmc", card types (creature, instant, sorcery, artifact,
  enchantment, land, planeswalker), creature subtypes, power/toughness,
  format names, and keyword abilities the user named (flying, haste,
  trample, ...). NEVER infer a type, cost, or size from a concept word:
  "ramp", "burn", "removal", "pinger", "mana dork", "mana rock" are concepts,
  not filters, even though they imply a card type. "spell"/"spells" is not a
  type.
- "cheap" means cmc <= 2. "expensive"/"big" means cmc >= 5. Other vague
  size words are not filters.
- Colours: when the user names a colour, set BOTH "colors" and
  "color_identity" to the same letters. Default "colors_op" to
  "contains_any" (admits colourless). Use "exactly" only for "mono-",
  "only", or "exclusively". For "colorless" cards use "colors": [] with
  "colors_op": "exactly". Never emit an empty "colors" list otherwise.
- "keywords" holds only canonical keyword ABILITIES the user typed (Flying,
  First strike, Trample, Haste, Deathtouch, Lifelink, ...). A slang noun for a
  kind of card ("pinger", "edict", "burn", "cantrip") is NEVER a keyword — it
  goes in "concepts".
- "concepts" should cover every functional idea in the query, in order of
  importance. Keep each phrase to one to three words.
- Do not include names of real cards.
- If unsure, prefer fewer filters and more concepts.

## FILTER SCHEMA

```json
{
  "filters": {
    "colors": ["W"|"U"|"B"|"R"|"G", ...] | null,
    "colors_op": "contains_any" | "contains_all" | "subset_of" | "exactly" | null,
    "color_identity": [...] | null,
    "types": ["Creature"|"Instant"|"Sorcery"|"Artifact"|"Enchantment"|"Land"|"Planeswalker"|"Battle", ...] | null,
    "subtypes": ["Elf", ...] | null,
    "cmc": {"op": "<="|"="|">="|"<"|">"|"between", "value": <number>|[<lo>,<hi>]} | null,
    "keywords": ["Flying", ...] | null,
    "power": {"op": ..., "value": <number>} | null,
    "toughness": {"op": ..., "value": <number>} | null,
    "format_legality": {"format": "standard"|"modern"|"pioneer"|"legacy"|"vintage"|"commander"|"pauper", "status": "legal"} | null
  },
  "concepts": [<string>, ...],
  "hypothetical_card": <string> | null
}
```
````

**Few-shot examples (8)**

````text
Query: cheap red removal
Output: {
  "filters": {"colors": ["R"], "colors_op": "contains_any", "color_identity": ["R"], "cmc": {"op": "<=", "value": 2}},
  "concepts": ["removal"]
}

Query: board wipes
Output: {
  "concepts": ["sweeper", "board wipe"]
}

Query: green creatures that tap for mana under 2 mana
Output: {
  "filters": {"colors": ["G"], "colors_op": "contains_any", "color_identity": ["G"], "types": ["Creature"], "cmc": {"op": "<", "value": 2}},
  "concepts": ["mana dork"]
}

Query: mana rocks that cost 2
Output: {
  "filters": {"cmc": {"op": "=", "value": 2}},
  "concepts": ["mana rock"]
}

Query: sorceries that make tokens
Output: {
  "filters": {"types": ["Sorcery"]},
  "concepts": ["token maker"]
}

Query: permanents that let me play lands out of my graveyard
Output: {
  "concepts": ["land recursion"],
  "hypothetical_card": "You may play lands from your graveyard."
}

Query: black edicts under 4 mana
Output: {
  "filters": {"colors": ["B"], "colors_op": "contains_any", "color_identity": ["B"], "cmc": {"op": "<", "value": 4}},
  "concepts": ["edict", "sacrifice removal"]
}

Query: flying creatures with vigilance that cost 3 or less
Output: {
  "filters": {"types": ["Creature"], "keywords": ["Flying", "Vigilance"], "cmc": {"op": "<=", "value": 3}},
  "concepts": []
}
````

### Appendix B — Evaluation set

*Generated by `scripts/generate_report.py --paper` on 2026-10-03; do not edit by hand.* 26 queries (`data/eval/queries_v1_draft.yaml`, version v1-draft); the full relevance judgments (card ids with a one-line reason each) are in that file. *Mapped tag* is the Scryfall oracle tag used as the set-retrieval target; the expert queries are the Scryfall comparator inputs.

| ID | Query | Category | Relevant | Borderline | Mapped tag | Expert Scryfall query | Without tags |
|---|---|---|---|---|---|---|---|
| q_001 | creatures with flying | fragmented | 30 | 0 | — | `t:creature kw:flying` | `t:creature kw:flying` |
| q_002 | destroy target artifact | mechanical | 30 | 0 | removal-artifact | `otag:removal-artifact o:destroy` | `o:"destroy target artifact"` |
| q_003 | instants that draw cards | fragmented | 17 | 8 | draw | `t:instant otag:draw` | `t:instant o:"draw"` |
| q_004 | burn spell that deals 3 damage to any target | natural | 20 | 0 | burn | `otag:burn-any o:"3 damage"` | `o:"deals 3 damage to any target"` |
| q_005 | haste creatures | fragmented | 30 | 0 | — | `t:creature kw:haste` | `t:creature kw:haste` |
| q_006 | ramp spells | jargon | 28 | 5 | ramp | `otag:ramp` | `(o:"search your library for" o:"land card" o:"onto the battlefield") or (o:"add" o:"mana" -t:land -o:"{T}: Add")` |
| q_007 | counterspells | jargon | 30 | 0 | counterspell | `otag:counterspell` | `o:"counter target" (t:instant or t:creature or t:enchantment)` |
| q_008 | board wipes | jargon | 76 | 0 | sweeper | `otag:sweeper` | `o:"destroy all creatures" or o:"exile all creatures" or (o:"all creatures get -" o:"until end of turn") or o:"destroy all nonland permanents"` |
| q_009 | removal | jargon | 72 | 4 | removal | `otag:removal` | `(o:"destroy target" or o:"exile target") (o:creature or o:permanent or o:"nonland permanent")` |
| q_010 | card draw engines | jargon | 20 | 1 | draw-engine | `otag:draw-engine` | `-t:instant -t:sorcery (o:"whenever" or o:"at the beginning") o:"draw a card"` |
| q_011 | tutor | jargon | 37 | 0 | tutor | `otag:tutor` | `o:"search your library for a" -o:"basic land card" -t:land` |
| q_012 | graveyard recursion | jargon | 33 | 1 | recursion | `otag:recursion` | `o:"from your graveyard" (o:"return target" or o:"return up to" or o:"put target")` |
| q_013 | fetch lands | jargon | 16 | 7 | fetchland | `otag:fetchland` | `t:land o:"search your library for" o:"land card" o:"sacrifice"` |
| q_014 | flicker effects | jargon | 32 | 3 | flicker | `otag:flicker` | `o:"exile" o:"return" o:"to the battlefield under" (o:"target creature" or o:"target permanent" or o:"another target")` |
| q_015 | ETB triggers | jargon | 33 | 0 | — | `t:creature o:"when" o:"enters"` | `t:creature o:"when" o:"enters"` |
| q_016 | mana dorks | jargon | 23 | 1 | mana-dork | `otag:mana-dork` | `t:creature o:"{T}: Add"` |
| q_017 | wheels | jargon | 9 | 4 | wheel | `otag:wheel` | `o:"discards" o:"hand" o:"draws seven cards"` |
| q_018 | extra turns | jargon | 17 | 2 | extra-turn | `otag:extra-turn` | `o:"take an extra turn"` |
| q_019 | free counterspell | constrained | 11 | 0 | counterspell-free | `otag:counterspell-free` | `o:"counter target spell" (o:"rather than pay" or o:"without paying" or mv=0)` |
| q_020 | red creatures under 3 mana | constrained | 17 | 0 | — | `t:creature c:r mv<3` | `t:creature c:r mv<3` |
| q_021 | instants that cost 1 mana | constrained | 25 | 0 | — | `t:instant mv=1` | `t:instant mv=1` |
| q_022 | a card that lets me look at my deck and put a creature into the battlefield | natural | 10 | 0 | tutor-creature | `otag:tutor-creature o:"onto the battlefield"` | `o:"search your library for a creature card" o:"onto the battlefield"` |
| q_023 | creatures that get bigger every time I cast a spell | natural | 18 | 0 | cast-trigger-you | `t:creature otag:cast-trigger-you o:"+1/+1"` | `t:creature o:"whenever you cast" o:"+1/+1"` |
| q_024 | a card that destroys all creatures | natural | 12 | 0 | sweeper | `otag:sweeper o:"destroy all creatures"` | `o:"destroy all creatures"` |
| q_025 | red pingers under 3 mana | hybrid | 11 | 3 | pinger | `otag:pinger c:r mv<3` | `c:r mv<3 o:"{T}:" o:"deals 1 damage"` |
| q_026 | cheap blue counterspells | hybrid | 16 | 1 | counterspell | `otag:counterspell c:u mv<=2` | `c:u mv<=2 o:"counter target"` |

### Appendix C — Per-query results

*Generated by `scripts/generate_report.py --paper` on 2026-10-03; do not edit by hand.* Source: `docs/reports/2026-10-03/tables/07_per_query.csv`, `experiment_runs` rows 90 (control) and 101 (headline). P@10 is against the hand-curated judgments, R-prec against the query's tag pool, parity against the expert Scryfall result set (tags allowed).

| ID | Query | Category | Control P@10 | Headline P@10 | Control R-prec | Headline R-prec | Control parity | Headline parity |
|---|---|---|---|---|---|---|---|---|
| q_001 | creatures with flying | fragmented | 0.000 | 0.000 | — | — | 0.598 | 0.664 |
| q_002 | destroy target artifact | mechanical | 0.100 | 0.000 | 0.382 | 0.205 | 0.486 | 0.094 |
| q_003 | instants that draw cards | fragmented | 0.000 | 0.100 | 0.157 | 0.157 | 0.706 | 0.862 |
| q_004 | burn spell that deals 3 damage to any target | natural | 0.100 | 0.100 | 0.067 | 0.034 | 0.021 | 0.028 |
| q_005 | haste creatures | fragmented | 0.000 | 0.100 | — | — | 0.902 | 0.998 |
| q_006 | ramp spells | jargon | 0.000 | 0.000 | 0.088 | 0.704 | 0.084 | 0.683 |
| q_007 | counterspells | jargon | 0.200 | 0.100 | 0.669 | 0.916 | 0.671 | 0.916 |
| q_008 | board wipes | jargon | 0.400 | 0.000 | 0.183 | 0.611 | 0.182 | 0.609 |
| q_009 | removal | jargon | 0.400 | 0.000 | 0.410 | 0.839 | 0.215 | 0.328 |
| q_010 | card draw engines | jargon | 0.000 | 0.100 | 0.228 | 0.527 | 0.228 | 0.526 |
| q_011 | tutor | jargon | 0.000 | 0.100 | 0.106 | 0.960 | 0.107 | 0.958 |
| q_012 | graveyard recursion | jargon | 0.100 | 0.000 | 0.573 | 0.767 | 0.562 | 0.743 |
| q_013 | fetch lands | jargon | 0.300 | 0.400 | 0.604 | 0.906 | 0.604 | 0.906 |
| q_014 | flicker effects | jargon | 0.000 | 0.100 | 0.000 | 0.743 | 0.000 | 0.734 |
| q_015 | ETB triggers | jargon | 0.000 | 0.000 | — | — | 0.166 | 0.164 |
| q_016 | mana dorks | jargon | 0.000 | 0.200 | 0.000 | 0.415 | 0.000 | 0.417 |
| q_017 | wheels | jargon | 0.000 | 0.200 | 0.022 | 0.745 | 0.022 | 0.743 |
| q_018 | extra turns | jargon | 0.500 | 0.600 | 0.208 | 0.943 | 0.208 | 0.943 |
| q_019 | free counterspell | constrained | 0.100 | 0.100 | 0.077 | 0.077 | 0.077 | 0.077 |
| q_020 | red creatures under 3 mana | constrained | 0.000 | 0.000 | — | — | 0.852 | 0.850 |
| q_021 | instants that cost 1 mana | constrained | 0.300 | 0.000 | — | — | 0.999 | 0.999 |
| q_022 | a card that lets me look at my deck and put a creature into the battlefield | natural | 0.000 | 0.000 | 0.015 | 0.000 | 0.023 | 0.000 |
| q_023 | creatures that get bigger every time I cast a spell | natural | 0.000 | 0.000 | 0.249 | 0.000 | 0.105 | 0.000 |
| q_024 | a card that destroys all creatures | natural | 0.300 | 0.000 | 0.183 | 0.651 | 0.298 | 0.083 |
| q_025 | red pingers under 3 mana | hybrid | 0.000 | 0.100 | 0.000 | 0.148 | 0.000 | 0.422 |
| q_026 | cheap blue counterspells | hybrid | 0.100 | 0.200 | 0.298 | 0.366 | 0.806 | 0.950 |

### Appendix D — Failure-mode test series

*[The 10 diagnostic test queries used to characterize the base HyDE model's failure modes (Section 5.1), with full HyDE output for each and side-by-side comparison against the larger model (Section 5.2).]*
