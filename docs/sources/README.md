# Sources — Annotated Bibliography

This folder holds PDFs of the academic sources supporting the thesis. The PDFs themselves are `.gitignore`d (see policy at bottom); this README is the version-controlled index.

**Cross-reference:** for the three IR-evaluation methodology papers that back the tri-state relevance schema, formal BibTeX blocks and applicability notes already live in [`../../data/eval/methodology_references.md`](../../data/eval/methodology_references.md). This README is the broader source index.

---

## Reading priority order

Read in this order for maximum efficiency:

**Tier 1 — Must read before M4 begins:**
1. **Gao et al. 2022 (HyDE)** — primary architecture citation. Sections 1–4 minimum.
2. **Reimers & Gurevych 2019 (SBERT)** — background for the model family you're using.
3. **Rethinking LLM-based Query Expansion (2025)** — directly critiques HyDE; if reviewers know one recent paper in this space, it's likely this one.
4. **Wang et al. 2023 (Query2doc)** — parallel technique to HyDE; your paper needs to distinguish itself from it.

**Tier 2 — Read before M5 (systematic eval):**
5. **Manning/Raghavan/Schütze Ch 8** — foundational for the metrics (recall@K, MRR, evaluation protocol).
6. **Thakur et al. 2021 (BEIR)** — standard eval benchmark; frames how dense retrieval systems get compared.
7. **Karpukhin et al. 2020 (DPR)** — foundational modern dense retrieval; ubiquitous citation.
8. **Choi et al. 2020 (Semantic Product Search for Structured Catalogs)** — direct analogue to your problem (structured catalog + semantic retrieval). Home Depot's approach; cites Nigam et al. (Amazon) as prior work.

**Tier 2b — Read before M6 fine-tuning starts (added 2026-09-18):**
- **Nussbaum et al. 2024 (Nomic Embed)** §4.2 — the base model's own supervised recipe; every hyperparameter default starts here.
- **Tamber et al. 2025** and **Murtaza et al. 2026** — the two papers showing naive fine-tuning of a strong small encoder can go backwards, and how to stop it.
- **Lan et al. 2026 (BioHiCL)** — hierarchical community tags as contrastive supervision; closest analogue to Scryfall oracle tags.
- **CoHyDE (2026)** — why HyDE and a fine-tuned encoder are complementary, not substitutes.

**Tier 3 — Read while writing (Weeks 9+):**
9. Query Expansion Survey (2025), MTEB Survey (2024) — background context for Related Work section.
10. Everything else — as needed.

---

## Query rewriting / expansion (the HyDE family)

### Gao et al. 2022 — HyDE (Precise Zero-Shot Dense Retrieval without Relevance Labels)
- **File:** `2022_gao_hyde.pdf`
- **arXiv:** [2212.10496](https://arxiv.org/abs/2212.10496)
- **Why cite:** Primary source for the query-rewriter stage of your cascade. Introduces the technique of embedding an LLM-generated hypothetical document instead of the raw query.
- **How to use:** Cite in Methodology when introducing Stage 1. Cite in Related Work as the origin technique.

### Wang et al. 2023 — Query2doc
- **File:** `2023_wang_query2doc.pdf`
- **arXiv:** [2303.07678](https://arxiv.org/abs/2303.07678)
- **Why cite:** Parallel technique to HyDE (published within months). Uses LLM to generate pseudo-documents for query expansion. Your paper needs to distinguish HyDE-based approach from Query2doc-style expansion.
- **How to use:** Related Work — position as "concurrent work"; note the technical difference (Query2doc concatenates the pseudo-document with the query; HyDE embeds it standalone).

### Wang 2024 — Exploring the Best Practices of Query Expansion with Large Language Models
- **File:** `2024_wang_query-expansion-best-practices.pdf`
- **arXiv:** [2401.06311](https://arxiv.org/abs/2401.06311)
- **Why cite:** Empirical study of best practices in LLM query expansion. Useful for defending prompt-engineering decisions in your HyDE implementation.
- **How to use:** Methodology — reference for prompt design choices. Related Work — cite alongside HyDE/Query2doc.

### Rethinking LLM-based Query Expansion (2025) — Hypothetical Documents or Knowledge Leakage?
- **File:** `2025_rethinking-llm-query-expansion.pdf`
- **arXiv:** [2504.14175](https://arxiv.org/abs/2504.14175)
- **Why cite:** Directly critiques HyDE. Argues that observed HyDE gains may be from LLM knowledge leakage (LLM already knows relevant documents) rather than pure query rewriting. **This is the paper a hostile reviewer will bring up.** Read carefully and be prepared to defend that your MTG setup doesn't suffer this failure mode (MTG cards are not in the LLM's training data in the same way general documents are).
- **How to use:** Related Work + Discussion. Your defense: MTG's Oracle text is highly formalized game content, and the "leakage" concern applies less when the corpus is a bounded, structured, well-known domain.

### Query Expansion Survey (2025) — Query Expansion in the Age of Pre-trained and Large Language Models
- **File:** `2025_query-expansion-survey.pdf`
- **arXiv:** [2509.07794](https://arxiv.org/abs/2509.07794)
- **Why cite:** Comprehensive survey. Best single reference for positioning your work in the broader LLM-query-expansion landscape.
- **How to use:** Related Work — one citation covers a lot of context.

---

## Embedding models & benchmarks

### Reimers & Gurevych 2019 — Sentence-BERT (SBERT)
- **File:** `2019_reimers-gurevych_sentence-bert.pdf`
- **arXiv:** [1908.10084](https://arxiv.org/abs/1908.10084)
- **Why cite:** Canonical reference for the sentence-transformers family. You use `multi-qa-distilbert-cos-v1`, which is a direct descendant of this work.
- **How to use:** Methodology — must cite when introducing your embedding model.

### Karpukhin et al. 2020 — DPR (Dense Passage Retrieval for Open-Domain QA)
- **File:** `2020_karpukhin_dpr.pdf`
- **arXiv:** [2004.04906](https://arxiv.org/abs/2004.04906)
- **Why cite:** Foundational paper for modern dense retrieval. Nearly every 2020+ dense-retrieval paper cites this.
- **How to use:** Related Work — one of the standard citations when framing "dense retrieval" as a paradigm.

### Thakur et al. 2021 — BEIR (A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models)
- **File:** `2021_thakur_beir-benchmark.pdf`
- **arXiv:** [2104.08663](https://arxiv.org/abs/2104.08663)
- **Why cite:** Standard benchmark for zero-shot dense retrieval evaluation. Frames how the field evaluates and compares retrieval systems.
- **How to use:** Related Work — evaluation-methodology context. Discussion — if you compare your eval methodology to BEIR's.

### MTEB Survey (2024) — Recent advances in text embedding
- **File:** `2024_mteb-survey.pdf`
- **arXiv:** [2406.01607](https://arxiv.org/abs/2406.01607)
- **Why cite:** Reviews top-performing methods on the MTEB benchmark; useful for defending your choice of embedding model against reviewer questions like "why not BGE / E5 / GTE?"
- **How to use:** Discussion or Limitations — acknowledge SOTA models exist; defend your choice as one of practical accessibility / already-established baseline.

### Nussbaum et al. 2024 — Nomic Embed
- **File:** `2024_nussbaum_nomic-embed.pdf`
- **arXiv:** [2402.01613](https://arxiv.org/abs/2402.01613)
- **Why cite:** Fully open-source SOTA embedding model. Useful reference point for "modern alternative to DistilBERT-family."
- **How to use:** Discussion — alternative model consideration, or Future Work as a proposed upgrade path.

---

## Hybrid / multi-stage retrieval

### Hybrid Retrieval and Multi-stage Text Ranking Solution at TREC 2022 (2023)
- **File:** `2023_hybrid-multi-stage-trec2022.pdf`
- **arXiv:** [2308.12039](https://arxiv.org/abs/2308.12039)
- **Why cite:** Concrete example of a hybrid (dense + sparse) multi-stage retrieval pipeline in a competitive setting. Related to your cascade architecture.
- **How to use:** Related Work — precedent for multi-stage retrieval architectures.

---

## HyDE latency and query-rewriter efficiency

### Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support (2025)
- **File:** `2025_never-come-up-empty-adaptive-hyde.pdf`
- **arXiv:** [2507.16754](https://arxiv.org/abs/2507.16754)
- **Why cite:** **Direct HyDE optimization paper (2025).** Proposes adaptive HyDE — a runtime decision layer that skips or shortens hypothetical-document generation when the query already has sufficient signal, reducing latency without accuracy loss. The most current published work on the specific problem of "HyDE is slow, how do we fix it."
- **How to use:** Methodology (justifies your choice of single-shot HyDE over the original 8-sample ensemble). Discussion/Future Work (an adaptive-skip layer is a natural M6 optimization if base HyDE latency turns out to bottleneck end-to-end performance).

### Retrieval-Feedback-Driven Distillation and Preference Alignment for Efficient LLM-based Query Expansion (2026)
- **File:** `2026_retrieval-feedback-distillation-query-expansion.pdf`
- **arXiv:** [2603.13776](https://arxiv.org/abs/2603.13776)
- **Why cite:** Shows that a distilled small model (Qwen3-4B) reaches ~97% of a 685B teacher model's performance on query-expansion retrieval tasks. Direct evidence that a specialized small HyDE model can match a much larger general model — supports Future Work argument for a distilled query-rewriter as an M6 optimization.
- **How to use:** Future Work (distillation-based HyDE for tighter latency budgets). Related Work (framing for smaller-model choices in the design space).

---

## Filtered vector search (SQL pre-filter methodology)

### Filter-Agnostic Vector Search on a PostgreSQL Database System (2026)
- **File:** `2026_filter-agnostic-vector-search-postgres.pdf`
- **arXiv:** [2603.23710](https://arxiv.org/abs/2603.23710)
- **Why cite:** **Directly relevant** — this paper studies filtered ANN search *in PostgreSQL*, which is exactly your pgvector setup. Compares pre-filter vs. post-filter strategies on Postgres and quantifies when each wins.
- **How to use:** Related Work + Methodology — primary citation for your SQL pre-filter design decision. Cite alongside your CLAUDE.md §2 statement "pre-filter is critical or recall collapses on constrained queries."

### Attribute Filtering in Approximate Nearest Neighbor Search: An In-depth Experimental Study (2025)
- **File:** `2025_attribute-filtering-ann-study.pdf`
- **arXiv:** [2508.16263](https://arxiv.org/abs/2508.16263)
- **Why cite:** Comprehensive experimental study comparing filtered ANN algorithms across pre-filter, post-filter, and hybrid strategies. Frames the design-space tradeoffs your cascade navigates.
- **How to use:** Related Work — background for the SQL pre-filter design decision. Discussion — reference for defending pre-filter choice over post-filter.

### Benchmarking Filtered ANN Search Algorithms on Transformer-based Embedding Vectors (2025)
- **File:** `2025_filtered-ann-transformer-embeddings-benchmark.pdf`
- **arXiv:** [2507.21989](https://arxiv.org/abs/2507.21989)
- **Why cite:** Benchmarks filtered ANN specifically on transformer-based (BERT-family) embeddings — the same class as your Nomic Embed encoder. Empirical baseline for filtered-search performance on modern dense embeddings.
- **How to use:** Related Work + Discussion — grounding for filtered vector search performance expectations with modern encoders.

---

## Product / catalog search (generalization anchor)

### Choi, Kallumadi, Mitra, Agichtein & Javed 2020 — Semantic Product Search for Matching Structured Product Catalogs in E-Commerce
- **File:** `2020_choi-etal_semantic-product-search-structured-catalogs.pdf`
- **arXiv:** [2008.08180](https://arxiv.org/abs/2008.08180)
- **Correction (2026-09-18):** previously mislabelled as Nigam et al. (Amazon). This is the **Home Depot** paper; Nigam et al. 2019 (KDD) is cited inside it as ref [9] and is not in this folder.
- **Why cite:** **THE closest published analogue to your problem.** Combines structured catalog data with a siamese DistilBERT retriever. Builds training pairs from click logs binarised at a threshold (§4) — the weak-label precedent for tag-derived pairs. Error analysis (Table 3) shows exact-token attributes (brand, colour, units) got *worse* with the semantic module — supports keeping those in the SQL pre-filter.
- **How to use:** Related Work + Discussion. This is the paper that lets you claim your work isn't domain-idiosyncratic. Methodology — weak-label training-pair precedent.

### Kothari et al. 2026 — Semantic Retrieval for Product Search in E-Commerce (Flipkart)
- **File:** `2026_semantic-retrieval-product-search-ecommerce.pdf`
- **arXiv:** [2606.01504](https://arxiv.org/abs/2606.01504)
- **Why cite:** Production fine-tuning of a Qwen3-Embedding backbone with LoRA on behavioural + synthetic pairs. Two directly reusable ideas: (1) a **false-negative margin mask** (δ=0.1) that drops in-batch documents scoring above the labelled positive — the same problem as functional near-duplicates in MTG; (2) the ablation (Table 1) shows the first contrastive stage delivers the bulk of all gain (+12.4 pts), everything after is ≤1.7. Gains concentrate on rare/long-tail queries (Table 3).
- **How to use:** Related Work — e-commerce generalisation anchor. Methodology — margin-mask citation if we adopt it.

---

## Embedder fine-tuning (M6 pivot — added 2026-09-18)

Backing for `docs/journal/2026-09-18-fine-tuning-pivot-oracle-tags-and-recipe.md` §3–4. Ranked by fit to our setup: 137M encoder, ~30k-doc bounded corpus, no relevance labels, community tags as distant supervision, single Apple Silicon machine.

### Paull 2025 — CustomIR: Unsupervised Fine-Tuning of Dense Embeddings for Known Document Corpora
- **File:** `2025_paull_customir-unsupervised-finetuning-known-corpora.pdf`
- **arXiv:** [2510.21729](https://arxiv.org/abs/2510.21729)
- **Why cite:** Closest published recipe — fixed known corpus (1.7k–10k docs), LLM-synthesised queries under multiple personas, BM25-mined negatives, LLM verification of each negative. Measured **20–32% of mined negatives were actually relevant** before verification. Full-parameter InfoNCE, lr 1e-5, early stopping on val loss.
- **How to use:** Methodology — the false-negative-rate number is the argument for tag-aware negative filtering.

### Tamber, Kazi, Sourabh & Lin 2025 — Conventional Contrastive Learning Often Falls Short
- **File:** `2025_tamber_contrastive-falls-short-listwise-distillation-synthetic.pdf`
- **arXiv:** [2505.19274](https://arxiv.org/abs/2505.19274)
- **Why cite:** Fine-tunes 110–137M encoders (incl. GTE-base-v1.5, same size class as Nomic) on LLM-synthesised queries of three types. **Plain InfoNCE degraded strong base models** (BGE-base SciFact 74.1→72.1); listwise cross-encoder distillation recovered it. 56k synthetic queries matched 56k human queries.
- **How to use:** Discussion — why naive MNRL can go backwards; justifies mixing query types to match the eval categories.

### Murtaza, Nie, Soni, Wen & Frydenlund 2026 — When Synthetic Data Hurts: Catastrophic Forgetting in Skill Retrieval (EMNLP 2026 Industry)
- **File:** `2026_murtaza_synthetic-data-catastrophic-forgetting-skill-retrieval.pdf`
- **arXiv:** [2609.10750](https://arxiv.org/abs/2609.10750)
- **Why cite:** Near-identical setup: 0.6B retriever, ~15k synthetic pairs, ~34k-item catalog. Aggressive fine-tuning dropped OOD recall 0.85→0.65; an **embedding-anchor regulariser** (L2 to the frozen encoder's outputs) restored OOD with ~14% ID gain intact. EWC / LwF / L2-init performed the same.
- **How to use:** Methodology — the forgetting mitigation and the reason the NanoBEIR probe is mandatory.

### Lan, Zheng & Kilicoglu 2026 — BioHiCL: Hierarchical Multi-Label Contrastive Learning with MeSH Labels (ACL 2026)
- **File:** `2026_lan_biohicl-hierarchical-multilabel-contrastive-mesh.pdf`
- **arXiv:** [2604.15591](https://arxiv.org/abs/2604.15591)
- **Why cite:** Turns a **hierarchical community tag taxonomy** into contrastive supervision — positives by label overlap, negatives by zero overlap, depth-weighted. LoRA ≈ full on bge-base; 0.1B model beats a 1B domain retriever. This is our situation with MeSH in place of Scryfall oracle tags.
- **How to use:** Methodology — primary precedent for tag-derived pairs and the zero-overlap negative rule.

### Shiraee Kasmaee et al. 2025 — ChEmbed: Domain-Specific Text Embeddings for Chemical Literature
- **File:** `2025_chembed-domain-embeddings-nomic-finetune.pdf`
- **arXiv:** [2508.01643](https://arxiv.org/abs/2508.01643)
- **Why cite:** The only paper fine-tuning the **Nomic Embed lineage**. Uses Nomic's own lr 2e-5. Key negative result: on a jargon-heavy corpus, **in-batch-only negatives beat mined hard negatives** in every configuration tried. Tokenizer extension gave +0.9 pts.
- **How to use:** Methodology — hyperparameter baseline; Discussion — why the –hard-negative ablation is mandatory.

### Senthil, Hathidara & Schreiber 2026 — CoHyDE: Co-Training LLM Rewriter & Dense Encoder (REALM @ EMNLP 2026)
- **File:** `2026_senthil_cohyde-cotraining-rewriter-encoder.pdf`
- **arXiv:** [2605.29271](https://arxiv.org/abs/2605.29271)
- **Why cite:** Directly answers "does HyDE still help after in-domain fine-tuning?" — yes for vague/informal queries (+6.3 pp nDCG@5), while the fine-tuned encoder wins on well-formed ones. Removing either component costs up to −8 pp.
- **How to use:** Discussion — the 2×2 (HyDE on/off × fine-tuned on/off) ablation table; defends keeping Stage 1 after the pivot.

### Gwon, Jedidi & Lin 2025 — Study on LLMs for Promptagator-Style Dense Retriever Training (CIKM 2025)
- **File:** `2025_gwon_promptagator-style-open-llms.pdf`
- **arXiv:** [2510.02241](https://arxiv.org/abs/2510.02241)
- **Why cite:** Ten open LLMs (1B–14B) as synthetic-query generators; Llama-3.2-3B ≈ Llama-3.1-8B ≈ proprietary Promptagator. Licenses using our local Llama 3.1 8B for query generation.
- **How to use:** Methodology — synthetic-query generator choice.

### Moreira et al. 2024/25 — NV-Retriever: Effective Hard-Negative Mining
- **File:** `2024_moreira_nv-retriever-hard-negative-mining.pdf`
- **arXiv:** [2407.15831](https://arxiv.org/abs/2407.15831)
- **Why cite:** Positive-aware mining — drop negatives scoring within a margin of the positive (TopK-MarginPos 0.05 / PercPos 95%). +8–11% over naive top-k. This is `mine_hard_negatives(relative_margin=0.05)` in Sentence Transformers.
- **How to use:** Methodology — the mining rule, if the hard-negative ablation wins.

### Wang, Tang, Zhang, Guo & Bi 2026 — Training Dense Retrievers with Multiple Positive Passages (KDD 2026)
- **File:** `2026_wang_multiple-positive-passages-dense-retrievers.pdf`
- **arXiv:** [2602.12727](https://arxiv.org/abs/2602.12727)
- **Why cite:** When a query has many positives (a tag with hundreds of cards), sampling one positive per step is a robust baseline; LSEPair is the upgrade if positives are noisy.
- **How to use:** Methodology — justifies the one-positive-per-step design.

### Khattab et al. 2026 — Less Finetuning, Better Retrieval: Synthesize-Train-Merge (Findings of EMNLP 2026)
- **File:** `2026_khattab_synthesize-train-merge-biomedical-retrievers.pdf`
- **arXiv:** [2602.04731](https://arxiv.org/abs/2602.04731)
- **Why cite:** Parameter-space merging of a fine-tuned specialist with the base model keeps general-domain performance while retaining in-domain gains. Cheap forgetting hedge (weight interpolation).
- **How to use:** Methodology — fallback if the NanoBEIR probe drops.

### Pande, Kumar & Damle 2025 — When Fine-Tuning Fails: Lessons from MS MARCO
- **File:** `2025_pande_when-finetuning-fails-msmarco.pdf`
- **arXiv:** [2506.18535](https://arxiv.org/abs/2506.18535)
- **Why cite:** Cautionary — every fine-tuning variant (full and LoRA) of all-MiniLM-L6 fell below the untouched baseline. The frozen base stays as a permanent ablation row.
- **How to use:** Discussion — why we report the base model alongside every tuned checkpoint.

### Wischounig, Abdallah & Jatowt 2026 — Negative Sampling Techniques in IR: A Survey (Findings of EACL 2026)
- **File:** `2026_wischounig_negative-sampling-survey.pdf`
- **arXiv:** [2603.18005](https://arxiv.org/abs/2603.18005)
- **Why cite:** Taxonomy of random / static-mined / dynamic / LLM-synthetic negatives.
- **How to use:** Related Work — one citation for the negatives paragraph.

### Shuttleworth, Andreas, Torralba & Sharma 2024/25 — LoRA vs Full Fine-tuning: An Illusion of Equivalence
- **File:** `2024_shuttleworth_lora-vs-full-finetuning-illusion.pdf`
- **arXiv:** [2410.21228](https://arxiv.org/abs/2410.21228)
- **Why cite:** LoRA forgets less but through "intruder dimensions"; the two methods are not interchangeable. Background for the full-vs-LoRA decision and for the HyDE-model LoRA step if it happens.
- **How to use:** Discussion — full-vs-LoRA rationale.

### Yuksel & Kamps 2025 — On Correlating Factors for Domain Adaptation Performance
- **File:** `2025_yuksel-kamps_domain-adaptation-correlating-factors.pdf`
- **arXiv:** [2501.14466](https://arxiv.org/abs/2501.14466)
- **Why cite:** The **type distribution** of generated training queries must match the test queries — the dominant factor in domain-adaptation gains. Justifies forcing synthetic queries across our six eval categories.
- **How to use:** Methodology — synthetic-query design.

---

## Recent IR-evaluation methodology (LLM-based judgment)

### Arabzadeh et al. 2025 — Benchmarking LLM-based Relevance Judgment Methods
- **File:** `2025_arabzadeh_llm-relevance-judgment-benchmark.pdf`
- **arXiv:** [2504.12558](https://arxiv.org/abs/2504.12558)
- **Why cite:** Directly relevant to task #25 (LLM-based evaluator with meta-evaluation). Benchmarks how well LLM-generated relevance judgments correlate with human judgments.
- **How to use:** Methodology — if you use LLM-generated relevance labels or the LLM-expert-Scryfall-query methodology, cite here.

### TRUE Framework (2025) — Reproducible Framework for LLM-Driven Relevance Judgment
- **File:** `2025_true-llm-relevance-judgment-framework.pdf`
- **arXiv:** [2509.25602](https://arxiv.org/abs/2509.25602)
- **Why cite:** Framework for LLM-driven relevance judgment. Useful methodological reference for the LLM-eval side of task #25.
- **How to use:** Methodology — if you go with LLM-based evaluation, cite alongside Arabzadeh 2025.

---

## Foundational IR-evaluation methodology (older but still standard)

### Manning, Raghavan, Schütze 2008 — Introduction to Information Retrieval (Ch. 8: Evaluation)
- **File:** `2008_manning-raghavan-schutze_ir-book-ch08-evaluation.pdf`
- **URL:** [nlp.stanford.edu/IR-book](https://nlp.stanford.edu/IR-book/pdf/08eval.pdf)
- **Why cite:** THE standard textbook citation for IR evaluation metrics (recall@K, MRR, precision, nDCG).
- **How to use:** Methodology — cite when defining evaluation metrics.

### Järvelin & Kekäläinen 2002 — Cumulated Gain-Based Evaluation of IR Techniques
- **File:** *not downloaded — paywalled (ACM Digital Library)*
- **DOI:** [10.1145/582415.582418](https://dl.acm.org/doi/10.1145/582415.582418)
- **Access:** CSUSB library subscription or ACM Digital Library
- **Why cite:** Introduced graded relevance and cumulated-gain metrics (nDCG). Foundational.
- **How to use:** Methodology — justifies graded relevance judgments over binary. Formal BibTeX and applicability notes in [`../../data/eval/methodology_references.md`](../../data/eval/methodology_references.md).

### Voorhees 2000 — Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness
- **File:** *not downloaded — paywalled (Elsevier IPM)*
- **DOI:** [10.1016/S0306-4573(00)00010-8](https://doi.org/10.1016/S0306-4573(00)00010-8)
- **Access:** CSUSB library subscription or NIST publication portal
- **Why cite:** Establishes that single-curator eval sets produce stable comparative rankings — defends your solo-curator methodology.
- **How to use:** Methodology — justification for solo-curator eval set. Formal notes in `methodology_references.md`.

### Sormunen 2002 — Liberal Relevance Criteria of TREC
- **File:** *not downloaded — paywalled (ACM Digital Library)*
- **DOI:** [10.1145/564376.564433](https://dl.acm.org/doi/10.1145/564376.564433)
- **Access:** CSUSB library subscription or Tampere University research portal
- **Why cite:** Direct empirical analogue for tri-state relevance schema.
- **How to use:** Methodology — closest published precedent for your tri-state approach. Formal notes in `methodology_references.md`.

---

## Gaps / follow-ups

**Papers not yet in this folder that may still be worth adding:**
- **Robertson et al. 2009 (BM25)** — sparse retrieval baseline; foundational. Consider adding if your paper cites BM25.
- **Wang et al. 2011 (A Cascade Ranking Model for Efficient Ranked Retrieval)** — origin of "cascade" terminology used in your architecture name.
- **Nogueira & Cho 2019 (Passage Re-ranking with BERT)** — cross-encoder re-ranking; may be relevant if you add a re-ranker stage.
- **Lewis et al. 2020 (Retrieval-Augmented Generation)** — RAG framing; useful if you position your work in the broader retrieval-for-generation landscape.

---

## File storage policy

- **PDFs are `.gitignore`d.** They're large binaries and freely re-downloadable from the URLs above. This README is the version-controlled index.
- **This README is committed.** It's the record of which sources support the project; the PDFs themselves are a local cache.
- **If a PDF goes missing:** re-download from the arXiv URL listed under each entry.
- **Naming convention:** `YYYY_lead-author_short-title.pdf`. Keep consistent when adding new files.
