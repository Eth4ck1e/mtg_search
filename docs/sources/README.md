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
8. **Nigam et al. 2020 (Semantic Product Search for Structured Catalogs)** — direct analogue to your problem (structured catalog + semantic retrieval). Amazon's approach.

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

## Product / catalog search (generalization anchor)

### Nigam et al. 2020 — Semantic Product Search for Matching Structured Product Catalogs in E-Commerce
- **File:** `2020_nigam-etal_semantic-product-search-structured-catalogs.pdf`
- **arXiv:** [2008.08180](https://arxiv.org/abs/2008.08180)
- **Why cite:** **THE closest published analogue to your problem.** Amazon's approach to combining structured catalog data with semantic retrieval. Directly supports your "generalizes beyond MTG" contribution claim.
- **How to use:** Related Work + Discussion. This is the paper that lets you claim your work isn't domain-idiosyncratic.

### Semantic Retrieval for Product Search in E-Commerce (2026)
- **File:** `2026_semantic-retrieval-product-search-ecommerce.pdf`
- **arXiv:** [2606.01504](https://arxiv.org/abs/2606.01504)
- **Why cite:** Recent review of semantic product search techniques.
- **How to use:** Related Work — background context for the e-commerce generalization claim.

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
