---
title: "Natural-Language Semantic Search for MTG Cards — Paper Outline"
author: "Mitchell Trafford"
date: "Fall 2026"
---

# Paper Outline

**How to use this document:** each section lists (a) its **purpose**, (b) **writing prompts** — what to actually write, in your voice, (c) **sources to cite** — papers from `docs/sources/`, (d) **journal cross-refs** — where the material lives, and (e) **notes for filling** — TODO items when real results are available.

Structure follows the format shared by Gao et al. 2022 (HyDE), Wang et al. 2023 (Query2doc), and Nigam et al. 2020 (Amazon semantic product search) — the three closest published precedents.

Target length: 15+ pages (thesis-class requirement). This outline is scoped to ~18–22 pages when written.

---

## Working title

*Placeholder options — refine after Discussion is drafted:*

- "Bridging the Query–Document Asymmetry: A Three-Stage Retrieval Cascade for Non-Expert Semantic Search on Consumer Catalogs"
- "Accessible Natural-Language Search on Structured Catalogs: HyDE + SQL Pre-Filter + Dense Retrieval on Magic: The Gathering"
- "From Expert Queries to Natural Language: A Retrieval Cascade for Consumer Catalog Search"

---

## Abstract

**Purpose:** 200–300 word single-paragraph summary. Follows the standard five-move structure — Problem, Approach, Method, Preliminary Results, Contribution.

**Status:** Rough draft was submitted 2026-09-04 (`docs/thesis/abstract-rough-draft-kit.md`); final draft due 2026-09-18 (task #15). This section of the outline is a placeholder — the final abstract goes here when written.

**Writing prompts:**

- Adapt the Sept 4 rough draft to the outcome-vs-Scryfall framing (see `docs/journal/2026-09-11-*.md`)
- Drop the recall@10 = 0.019 baseline reference (fabricated; no longer in the paper)
- Update encoder reference from `multi-qa-distilbert-cos-v1` to `nomic-ai/nomic-embed-text-v1.5`
- Simplify comparator list to the current three-config matrix

---

## 1. Introduction

### 1.1 Problem statement

**Purpose:** state the research problem in general enough terms that a non-IR reader can follow, then narrow to the specific technical question.

**Writing prompts:**

- Consumer catalogs (trading cards, e-commerce, media libraries) increasingly need natural-language search
- Traditional keyword and faceted-filter search excels when users know the domain's vocabulary but excludes users who query in natural language
- Naive dense retrieval fails on the same catalogs because short informal queries embed far from formal domain text — the query-document asymmetry problem
- MTG serves as a test bed for the generalization question

**Sources to cite:**

- Manning, Raghavan, Schütze 2008 (Ch 8) — foundational IR reference for the problem framing
- Nigam et al. 2020 (`docs/sources/2020_nigam-etal_semantic-product-search-structured-catalogs.pdf`) — Amazon's motivation for the same problem in e-commerce

**Journal cross-refs:**

- `docs/journal/2026-09-11-pivot-baseline-abandonment-and-encoder-switch.md` §2 (Research reframing) — canonical framing of the problem

### 1.2 Motivation

**Purpose:** why does this matter beyond an academic exercise? Ground the accessibility angle.

**Writing prompts:**

- Non-expert users are the majority of the addressable population for consumer catalog search
- Expert-query fluency (Scryfall's advanced syntax, Amazon's category filters) creates an accessibility gap
- Bridging that gap has real commercial and user-experience value — cite the e-commerce literature for scale of the problem
- MTG's ~30k unique cards is a bounded, well-structured test bed where the tradeoff can be measured cleanly

**Sources to cite:**

- Nigam et al. 2020 — commercial motivation for semantic product search
- 2026 review (`docs/sources/2026_semantic-retrieval-product-search-ecommerce.pdf`) — broader e-commerce context

**Journal cross-refs:**

- `docs/journal/2026-09-11-*.md` §2 — full outcome-vs-Scryfall reframing captured here

**Notes for filling:**

- Quote the specific commercial framing (accessibility → expanded user population) from your Sept 11 discussion, in your own words

### 1.3 Contributions

**Purpose:** enumerate the paper's contributions crisply.

**Writing prompts:**

- **Architectural contribution:** a three-stage retrieval cascade combining HyDE query rewriting, SQL pre-filter, and semantic vector search — designed to bridge query-document asymmetry without domain-specific fine-tuning
- **Methodological contribution:** an evaluation methodology anchored on outcome vs. the domain-standard tool (Scryfall) rather than internal-baseline diff; comparator uses LLM-crafted expert queries to test against the strongest form of the competing paradigm
- **Empirical contribution:** measurement of per-component contribution via targeted ablation; identification of which query categories the cascade wins and loses on
- **Generalization contribution:** framing the results in terms that extend to other consumer catalog domains

### 1.4 Paper organization

**Purpose:** roadmap sentence for the reader. Standard boilerplate.

**Writing prompts:**

- One paragraph: "Section 2 reviews related work in dense retrieval, query rewriting, and filtered vector search. Section 3 describes the three-stage cascade architecture and evaluation methodology. Section 4 presents experimental configurations. Section 5 reports results. Section 6 discusses..."

---

## 2. Related Work

### 2.1 Dense retrieval foundations

**Purpose:** situate the paper in the sentence-embedding + bi-encoder retrieval tradition.

**Writing prompts:**

- Sentence embeddings and bi-encoder retrieval as introduced by SBERT (Reimers & Gurevych 2019)
- Modern dense retrievers: DPR as the reference architecture (Karpukhin et al. 2020)
- Zero-shot dense retrieval and the query-document asymmetry problem (context for HyDE)
- Modern encoders: MTEB benchmark landscape, position of Nomic Embed v1.5

**Sources to cite:**

- Reimers & Gurevych 2019 (SBERT) — `docs/sources/2019_reimers-gurevych_sentence-bert.pdf`
- Karpukhin et al. 2020 (DPR) — `docs/sources/2020_karpukhin_dpr.pdf`
- Thakur et al. 2021 (BEIR) — `docs/sources/2021_thakur_beir-benchmark.pdf`
- Nussbaum et al. 2024 (Nomic Embed) — `docs/sources/2024_nussbaum_nomic-embed.pdf`
- 2024 MTEB survey — `docs/sources/2024_mteb-survey.pdf`

### 2.2 Query rewriting and expansion

**Purpose:** establish the HyDE technique and its lineage.

**Writing prompts:**

- HyDE (Gao et al. 2022) as the core query-rewriting technique
- Query2doc (Wang et al. 2023) as a parallel/concurrent technique — describe the technical difference (Query2doc concatenates; HyDE replaces)
- Best-practices literature on LLM query expansion (Wang 2024)
- The Rethinking-LLM-QE (2025) critique — must engage with this, since it directly critiques HyDE
- Ouyang et al. 2022 (InstructGPT / RLHF) — foundational for the "instruction-tuned LLM zero-shot generalizes" argument HyDE depends on

**Sources to cite:**

- Gao et al. 2022 (HyDE) — primary
- Wang et al. 2023 (Query2doc) — concurrent-work comparison
- Wang 2024 (LLM query expansion best practices)
- 2025 Rethinking LLM Query Expansion — critique to engage with
- 2025 Query Expansion Survey — broader landscape

**Journal cross-refs:**

- `docs/journal/2026-09-11-*.md` §5 (Experimental matrix) — defense against Rethinking-LLM-QE for MTG's bounded formal domain

### 2.3 Filtered vector search (SQL pre-filter)

**Purpose:** ground the pre-filter design decision in the filtered-ANN literature.

**Writing prompts:**

- The pre-filter vs. post-filter dichotomy in filtered ANN search
- Why pre-filter is preferred for constrained queries (recall preservation on low-selectivity filters)
- PostgreSQL-specific filtered vector search literature (directly relevant to pgvector)
- Recent benchmarks on modern transformer embeddings with filtered ANN

**Sources to cite:**

- 2026 Filter-Agnostic Vector Search on Postgres — `docs/sources/2026_filter-agnostic-vector-search-postgres.pdf` (**most directly relevant** — same underlying system)
- 2025 Attribute Filtering in ANN — `docs/sources/2025_attribute-filtering-ann-study.pdf`
- 2025 Filtered ANN Benchmark on Transformer Embeddings — `docs/sources/2025_filtered-ann-transformer-embeddings-benchmark.pdf`
- 2023 Hybrid Multi-stage Retrieval TREC 2022 — `docs/sources/2023_hybrid-multi-stage-trec2022.pdf`

### 2.4 Semantic product search in consumer catalogs

**Purpose:** situate the generalization claim — this problem shape appears in e-commerce and other catalog domains.

**Writing prompts:**

- Nigam et al. 2020 as the closest published analogue (Amazon; structured catalog + semantic retrieval)
- Recent semantic product search literature (2026 review)
- How the cascade design mirrors Amazon-style hybrid approaches
- Why MTG generalizes: structured-attribute-rich catalog with formal-vocabulary documentation that non-experts don't speak

**Sources to cite:**

- Nigam et al. 2020 — `docs/sources/2020_nigam-etal_semantic-product-search-structured-catalogs.pdf`
- 2026 semantic retrieval review — `docs/sources/2026_semantic-retrieval-product-search-ecommerce.pdf`

### 2.5 IR evaluation methodology

**Purpose:** justify the eval-set design and metric choices.

**Writing prompts:**

- Graded relevance (Järvelin & Kekäläinen 2002) — nDCG lineage
- Single-curator eval sets (Voorhees 2000) — defensibility of your solo-curator approach
- Tri-state relevance (Sormunen 2002) — direct analogue for your three-level scheme
- Recall@K and MRR as standard metrics (Manning et al. 2008)
- Modern LLM-based relevance judgment methodology (2025 papers) — for Discussion / Future Work

**Sources to cite:**

- Järvelin & Kekäläinen 2002 (paywalled; cited via `data/eval/methodology_references.md`)
- Voorhees 2000 (paywalled)
- Sormunen 2002 (paywalled)
- Manning et al. 2008 Ch 8 — `docs/sources/2008_manning-raghavan-schutze_ir-book-ch08-evaluation.pdf`
- 2025 LLM-based relevance judgment — `docs/sources/2025_arabzadeh_llm-relevance-judgment-benchmark.pdf`, `docs/sources/2025_true-llm-relevance-judgment-framework.pdf`

**Cross-refs:**

- `data/eval/methodology_references.md` — formal BibTeX + applicability notes for the three IR-eval papers

---

## 3. Methodology

### 3.1 Task and dataset

**Purpose:** describe what the system does, what it operates over, and how the corpus was prepared.

**Writing prompts:**

- Task formulation: given a natural-language query, return the top-K most relevant MTG cards
- Corpus: Scryfall `oracle-cards` bulk data, 2026-09-11 snapshot
- Corpus preparation and filtering rules (six filter categories, retention rate, final counts)
- Multi-face card handling (`(oracle_id, face_index)` composite key)

**Journal cross-refs:**

- `docs/journal/2026-09-11-*.md` §4 (Corpus filter cleanup) + §8 (Notes for final report — Corpus preparation) — has ready-to-adapt methodology prose

**Notes for filling:**

- Final corpus stats table: 38,740 raw → 31,972 face rows (31,124 unique oracle_ids), 80.3% retention
- Skip-reason breakdown (digital_only: 2058, non_card_layout: 3739, funny_set: 998, silver_bordered: 458, memorabilia: 350, token_set: 13)

### 3.2 Three-stage retrieval cascade

**Purpose:** describe the architecture in detail — this is the core of the paper.

**Writing prompts:**

- Cascade concept (Wang et al. 2011 tradition) — sequential stages, not parallel towers
- Design invariant: semantic search always runs INSIDE the SQL-pre-filtered candidate set (pre-filter, not post-filter)
- Rationale for pre-filter over post-filter on top-K (recall preservation)
- Cross-reference: CLAUDE.md §2 has the tight architectural summary

#### 3.2.1 Stage 1: HyDE query rewriter

**Purpose:** describe the query-rewriting stage in enough detail for reproducibility.

**Writing prompts:**

- LLM used: `meta-llama/Llama-3.1-8B-Instruct` (or whichever wins M4 candidate eval — task #27)
- Local inference: Apple Silicon M3 MacBook Pro, MPS backend, ~5-6 tokens/sec typical
- Output format: JSON with `filters` (structured attributes for Stage 2) + `hypothetical_card` (ability text for Stage 3)
- Few-shot examples: 5-8, covering the six query categories (natural, jargon, fragmented, hybrid, constrained, mechanical)
- Prompt versioning: `prompts/hyde_v1.yaml`, iterated as `v2`, `v3` — each version logged in `experiment_runs`

**Sources to cite:**

- Gao et al. 2022 (HyDE) — primary technique citation
- Ouyang et al. 2022 (InstructGPT / RLHF) — foundational for instruction-following capability

**Notes for filling:**

- Include the final HyDE prompt in an Appendix
- Table of prompt-version deltas: what changed between v1 → v2 → v3 and why
- Latency measurement of the HyDE call (target: <2s p95 per `docs/roadmap/phase-4-hyde-and-prefilter.md`)

#### 3.2.2 Stage 2: SQL pre-filter

**Purpose:** describe how HyDE's extracted attributes narrow the candidate set.

**Writing prompts:**

- Structured attributes filtered on: colors, color_identity, cmc (numeric), type_line, keywords, layout, legalities
- Postgres schema and indexes (GIN on array columns, B-tree on numeric)
- Fallback behavior when HyDE doesn't extract certain attributes (leave null; don't over-constrain)
- Handling of ambiguous queries (color prefixes, mana-cost ranges vs. exact)

**Sources to cite:**

- 2026 Filter-Agnostic Vector Search on Postgres — **primary citation** for the pre-filter methodology
- 2025 Attribute Filtering in ANN Search — comparative context
- Wang et al. 2011 (cascade ranking origin) — architectural precedent

**Journal cross-refs:**

- `src/db/migrations/0002_cards.sql` — schema definition (GIN indexes on colors, color_identity, keywords)

**Notes for filling:**

- Query-plan analysis for a representative pre-filter query (EXPLAIN output showing GIN index usage)
- Selectivity measurements: on average, how much does SQL narrow the candidate set for the eval-set queries?

#### 3.2.3 Stage 3: Semantic vector search

**Purpose:** describe the encoding + retrieval step.

**Writing prompts:**

- Encoder: `nomic-ai/nomic-embed-text-v1.5`, 137M parameters, 768-dim output
- Bi-encoder architecture in the SBERT lineage (Reimers & Gurevych 2019)
- Task-specific prefixes: `search_document:` (corpus side) and `search_query:` (query side)
- Cosine similarity retrieval; pgvector's `<=>` operator
- Encoding infrastructure: MPS backend on Apple Silicon; batch encoding at ingest time; per-query embedding at search time

**Sources to cite:**

- Reimers & Gurevych 2019 (SBERT paradigm)
- Nussbaum et al. 2024 (Nomic Embed) — specific encoder
- Karpukhin et al. 2020 (DPR) — dense retrieval framing
- Manning et al. 2008 Ch 8 — cosine similarity and metric background

**Notes for filling:**

- Encoding throughput measurements (140 rows/sec on MPS in our runs)
- Full corpus encoding time (229 seconds for 31,972 face rows)

### 3.3 Corpus text preparation and reminder-text augmentation

**Purpose:** the encoder side of the query-document asymmetry — describe how you condition the corpus.

**Writing prompts:**

- Oracle text is embedded, not the whole card record (mana cost, colors, type stay in SQL)
- Reminder-text augmentation: for each keyword the card has but doesn't explain inline, append canonical Wizards-authored reminder text
- Reminder-text dictionary construction: harvested by scanning all Scryfall printings for parenthetical patterns
- Manual override file for keywords Wizards has never printed reminders for
- Why NOT to hand-build a keyword definition dictionary (see CLAUDE.md §5)

**Journal cross-refs:**

- `docs/journal/2026-05-17-keyword-augmentation.md` — original design of the reminder-text pipeline

### 3.4 Evaluation setup

**Purpose:** describe the eval set, metrics, and comparators.

**Writing prompts:**

- Eval set: 26 hand-curated queries with tri-state relevance judgments
- Query categories: natural, jargon, fragmented, hybrid, constrained, mechanical
- Metrics: recall@10 (correct results in top 10), MRR (rank of first correct match)
- Comparators: (1) Scryfall search with LLM-crafted expert queries — primary; (2) –SQL ablation (HyDE + semantic without pre-filter) — targeted ablation

**Sources to cite:**

- Järvelin & Kekäläinen 2002 (graded relevance)
- Sormunen 2002 (tri-state analogue)
- Voorhees 2000 (single-curator eval defense)
- Manning et al. 2008 Ch 8 (metrics)

**Journal cross-refs:**

- `data/eval/queries_v1_draft.yaml` — the eval set itself
- `data/eval/methodology_references.md` — formal citations
- `docs/journal/2026-05-18-eval-set-construction.md` — eval-set design notes

**Notes for filling:**

- Table: per-category query count + relevance-label distribution
- Description of the LLM-crafted Scryfall query methodology (which LLM, how prompted; task #20 material)

---

## 4. Experiments

### 4.1 Experimental configurations

**Purpose:** enumerate the configurations tested.

**Writing prompts:**

- Config 1: Full cascade (HyDE + SQL + Nomic Embed v1.5)
- Config 2: –SQL ablation (HyDE + Nomic Embed v1.5, no pre-filter)
- Config 3: Scryfall search with LLM-crafted expert queries — reference comparator

**Notes for filling:**

- Configuration YAML files: `configs/cascade_v1.yaml`, `configs/no_sql_ablation.yaml` (create when experiments run)

### 4.2 Baselines and reference points

**Purpose:** be honest about what's NOT included in the experimental matrix and why.

**Writing prompts:**

- Naive dense retrieval baseline: not run as a formal comparator (paper is outcome-anchored, not diff-anchored). Qualitative failure-mode observations from `scripts/test_search.py` reference this configuration for illustration only.
- –HyDE ablation: not run. Evidence for HyDE's contribution comes from (a) the Scryfall comparator (SQL-heavy no-HyDE) and (b) argument-from-literature (Gao et al. 2022 established HyDE's contribution on standard benchmarks).
- –Semantic ablation: not run. A HyDE-to-SQL pipeline without semantic ranking would be a hobbled version of what Scryfall does with purpose-built tuning. Scryfall serves as the fairer comparator.

**Journal cross-refs:**

- `docs/journal/2026-09-11-*.md` §5 — full defense of the scoping decisions

### 4.3 Implementation and hardware

**Purpose:** reproducibility.

**Writing prompts:**

- MacBook Pro M3, 36GB unified memory
- Python 3.13, PostgreSQL 16 with pgvector, sentence-transformers, transformers
- Model checkpoints: exact tags on Hugging Face
- All code + configs versioned in the public repo

---

## 5. Results

*Fill after M4/M5 measurements complete.*

### 5.1 Overall retrieval performance

**Purpose:** headline numbers table.

**Notes for filling:**

- Table: config × metric (recall@10, MRR) × [aggregate, per-category]
- Statistical significance if applicable (paired bootstrap or similar)

### 5.2 Per-query-category breakdown

**Purpose:** where the cascade wins and loses, by query type.

**Notes for filling:**

- Table: query category × config × recall@10
- Discussion of where each config dominates

### 5.3 Ablation: –SQL pre-filter

**Purpose:** isolate the SQL pre-filter's contribution.

**Notes for filling:**

- Table: full cascade vs. –SQL, per-category
- Attribution of the pre-filter's contribution to specific query categories (structural queries expected to benefit most)

### 5.4 Comparator: Scryfall (expert LLM-crafted queries)

**Purpose:** headline outcome comparison.

**Notes for filling:**

- Table: cascade vs. Scryfall, per-category
- Cases where Scryfall wins (expert-friendly structural queries) vs. where cascade wins (jargon, compositional queries)

### 5.5 Latency

**Purpose:** practical viability.

**Notes for filling:**

- Latency table: p50, p95 per configuration
- HyDE LLM call breakdown from total retrieval latency (per `docs/roadmap/phase-4-hyde-and-prefilter.md` line 41 budget of <2s p95)

---

## 6. Discussion

### 6.1 Accessibility vs. capability

**Purpose:** the paper's core interpretive framing.

**Writing prompts:**

- Scryfall wins on precision, latency, and expert-crafted queries — a purpose-built structured tool operated by an expert always wins on the exact task it was built for
- The cascade's contribution is orthogonal: accessibility for users who cannot construct expert queries, while remaining accurate enough for regular use
- Result works in all three outcome scenarios: cascade wins → strong claim; cascade ties → "brings expert-level results to non-experts"; cascade loses → "approaches expert-level without requiring expert query construction"

**Journal cross-refs:**

- `docs/journal/2026-09-11-*.md` §2 — the framing was developed in this session

### 6.2 Failure modes

**Purpose:** honest analysis of where the cascade struggles.

**Writing prompts:**

- Reference `scripts/test_search.py` qualitative observations
- Categories where naive dense retrieval fails (jargon, structural, fragmented)
- Categories where HyDE helps most (jargon → HyDE bridges to formal card text)
- Categories where SQL pre-filter is essential (structural, constrained)
- Compositional queries: sometimes the LLM misunderstands intent

### 6.3 Generalization to other catalog domains

**Purpose:** why MTG isn't special.

**Writing prompts:**

- The query-document asymmetry problem generalizes to any consumer catalog domain with structured attributes + formal documentation vocabulary
- Analogous domains: e-commerce (Amazon), media libraries (movies, music, books), technical documentation search
- The cascade design is not MTG-specific — swap the schema and reminder-text harvest procedure, rest of the pipeline transfers
- Universes Beyond and other domain-crossover cards in MTG are already a test bed for cross-domain generalization within one catalog

**Sources to cite:**

- Nigam et al. 2020 (Amazon) — direct analogue
- 2026 semantic product search review — broader landscape

### 6.4 Limitations

**Purpose:** be up-front about what the paper doesn't do.

**Writing prompts:**

- Naive dense retrieval baseline not run as formal comparator — decision documented; qualitative reference only
- –HyDE ablation not run — HyDE's contribution defended via literature (Gao et al. 2022) and Scryfall comparator, not empirical isolation on MTG
- Single-curator eval set (though defended by Voorhees 2000)
- English-only corpus (Scryfall oracle_cards bulk is English-only; multilingual eval is future work)
- Modern encoders (E5-Mistral, BGE, Qwen3-Embedding) not compared — held encoder variable constant to isolate cascade contribution; multi-encoder comparison is future work
- HyDE LLM choice not tuned via full grid search — one candidate model evaluated
- Small corpus (~30k cards) — the pre-filter cost/benefit story may differ at web-scale

**Sources to cite:**

- 2025 Rethinking-LLM-QE — engage with the LLM-knowledge-leakage critique of HyDE; explain why MTG's bounded formal domain is less susceptible

### 6.5 Ethical and reproducibility considerations

**Purpose:** standard section in modern IR papers.

**Writing prompts:**

- All code, configs, and eval set publicly versioned in the repo
- Model weights come from HuggingFace (public); Postgres + pgvector are open-source
- Scryfall data terms of use (attribution, no scraping — bulk data is the sanctioned access path)

---

## 7. Conclusion and Future Work

### 7.1 Conclusion

**Purpose:** brief restatement of contribution and result. Standard boilerplate.

**Writing prompts:**

- One paragraph: restate the accessibility framing, restate the primary outcome (vs. Scryfall), restate the generalization claim

### 7.2 Future work

**Purpose:** point to obvious next steps.

**Writing prompts:**

- Fine-tuning the encoder with synthetic (query, card) training pairs (M6 territory; deferred)
- NL-to-SQL alternate architecture (see Sept 11 discussion — could bypass the semantic stage entirely for structured queries)
- Scryfall tags integration (`docs/roadmap/phase-5-systematic-eval.md`; task #17)
- Multi-encoder comparison (BGE, E5-Mistral, Qwen3-Embedding)
- Cross-domain generalization test — apply the pipeline to a non-MTG catalog

---

## References

*Full bibliography compiled from `docs/sources/README.md`. When exporting the final paper, generate BibTeX from the source annotations and let LaTeX or Word manage formatting.*

Primary references (in order of expected first cite):

1. **Manning, C.D., Raghavan, P., Schütze, H. (2008).** *Introduction to Information Retrieval*, Chapter 8. Cambridge University Press. [`docs/sources/2008_manning-raghavan-schutze_ir-book-ch08-evaluation.pdf`]
2. **Reimers, N., Gurevych, I. (2019).** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP.* [`docs/sources/2019_reimers-gurevych_sentence-bert.pdf`]
3. **Karpukhin, V., et al. (2020).** Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP.* [`docs/sources/2020_karpukhin_dpr.pdf`]
4. **Nigam, P., et al. (2020).** Semantic Product Search for Matching Structured Product Catalogs in E-Commerce. [`docs/sources/2020_nigam-etal_semantic-product-search-structured-catalogs.pdf`]
5. **Thakur, N., et al. (2021).** BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. *NeurIPS.* [`docs/sources/2021_thakur_beir-benchmark.pdf`]
6. **Ouyang, L., et al. (2022).** Training language models to follow instructions with human feedback. *NeurIPS.* (InstructGPT / RLHF)
7. **Gao, L., et al. (2022).** Precise Zero-Shot Dense Retrieval without Relevance Labels. (HyDE) [`docs/sources/2022_gao_hyde.pdf`]
8. **Wang, L., et al. (2023).** Query2doc: Query Expansion with Large Language Models. *ACL.* [`docs/sources/2023_wang_query2doc.pdf`]
9. **Nussbaum, Z., et al. (2024).** Nomic Embed: Training a Reproducible Long Context Text Embedder. [`docs/sources/2024_nussbaum_nomic-embed.pdf`]
10. **[Filter-Agnostic Postgres authors] (2026).** An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System. [`docs/sources/2026_filter-agnostic-vector-search-postgres.pdf`]
11. **[Attribute Filtering authors] (2025).** Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study. [`docs/sources/2025_attribute-filtering-ann-study.pdf`]

Foundational (paywalled, cited via `data/eval/methodology_references.md`):

- **Voorhees, E.M. (2000).** Variations in relevance judgments and the measurement of retrieval effectiveness. *IPM.*
- **Järvelin, K., Kekäläinen, J. (2002).** Cumulated gain-based evaluation of IR techniques. *ACM TOIS.*
- **Sormunen, E. (2002).** Liberal relevance criteria of TREC: counting on negligible documents? *SIGIR.*

Additional (cite as needed):

- 2024 MTEB Survey, 2024 LLM Query Expansion Best Practices, 2025 Rethinking LLM Query Expansion, 2025 Query Expansion Survey, 2025 LLM Relevance Judgment papers, 2026 semantic retrieval review, Wang et al. 2011 cascade ranking.

---

## Appendices

### Appendix A: HyDE prompt (final version)

Include the exact system prompt + few-shot examples used for the final results. Source of truth lives in `prompts/hyde_v1.yaml` (or the winning iteration).

### Appendix B: Full evaluation set

The 26-query eval set with all relevance judgments. Source: `data/eval/queries_v1_draft.yaml`.

### Appendix C: Per-query results

Complete recall@10 and MRR values for every (query, configuration) pair. Source: `experiment_runs` table dump.

### Appendix D: Corpus preparation details

Full filter-rule specification and rejection-count breakdown. Source: `src/data_processing/scryfall_classify.py`, `src/data_processing/ingest_transform.py`, corpus survey outputs.
