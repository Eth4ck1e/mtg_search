# Academic references for eval-set methodology

This file collects the academic precedent backing the eval-set construction methodology used in `queries_v1.yaml`. Source material for the methodology section of the final paper and for the eval-set journal entry.

## Tri-state / graded-relevance background

Graded relevance has been the IR-evaluation mainstream since the early 2000s. Voorhees (2000) established that comparative system rankings remain stable across assessor disagreement, legitimizing collections built on a single curator's judgments. Järvelin & Kekäläinen (2002) introduced cumulated-gain metrics specifically to credit systems for retrieving highly-relevant documents, normalizing the use of multi-level relevance scales. Sormunen (2002) showed empirically that ~50% of "relevant" documents in TREC are in fact marginal, motivating the very partition that our tri-state methodology codifies.

## Primary references

### 1. Järvelin & Kekäläinen 2002 — Cumulated Gain-Based Evaluation

**Citation:**
```bibtex
@article{jarvelin2002cumulated,
  author  = {J{\"a}rvelin, Kalervo and Kek{\"a}l{\"a}inen, Jaana},
  title   = {Cumulated gain-based evaluation of IR techniques},
  journal = {ACM Transactions on Information Systems},
  volume  = {20}, number = {4}, pages = {422--446}, year = {2002},
  doi     = {10.1145/582415.582418}
}
```

**Direct applicability.** The canonical citation for moving beyond binary relevance. The authors explicitly motivate the work by arguing IR evaluation must be "extended from binary relevance judgments to graded relevance judgments" so that systems are credited for surfacing the *most* relevant documents rather than treated identically across a flat "relevant" bin. Our three-bucket scheme is a graded-relevance scheme that simply zeroes out the partial-credit weight on borderline items — fully within the family of methods this paper inaugurates.

**Caveat.** The paper advocates *using* graded judgments via DCG/nDCG weighting; we instead *discard* the middle bucket from recall@K/MRR. Cite for the principle, not the exact metric form.

### 2. Voorhees 2000 — Variations in Relevance Judgments

**Citation:**
```bibtex
@article{voorhees2000variations,
  author  = {Voorhees, Ellen M.},
  title   = {Variations in relevance judgments and the measurement of retrieval effectiveness},
  journal = {Information Processing \& Management},
  volume  = {36}, number = {5}, pages = {697--716}, year = {2000},
  doi     = {10.1016/S0306-4573(00)00010-8}
}
```

**Direct applicability.** Voorhees demonstrates that "comparative evaluation of retrieval performance is stable despite substantial differences in relevance judgments" — the empirical foundation that makes single-curator evaluation sets like ours scientifically defensible. The paper directly justifies why our tri-state judgments (made by one curator under documented criteria) can produce reliable comparative rankings of retrieval variants without requiring multi-assessor adjudication.

**Caveat.** Voorhees studies binary judgments with overlap variation, not explicit tri-state schemes; the support is methodological (single-assessor judgments yield stable rankings), not specific to a borderline exclusion rule.

### 3. Sormunen 2002 — Liberal Relevance Criteria of TREC

**Citation:**
```bibtex
@inproceedings{sormunen2002liberal,
  author    = {Sormunen, Eero},
  title     = {Liberal relevance criteria of {TREC}: counting on negligible documents?},
  booktitle = {Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {324--330}, year = {2002},
  doi       = {10.1145/564376.564433}
}
```

**Direct applicability.** The closest direct analogue to our methodology. Sormunen reassessed TREC-7/8 pools on a four-point scale (irrelevant / marginally relevant / fairly relevant / highly relevant) and found that ~50% of TREC-"relevant" documents are merely *marginal* — exactly the population our `borderline` bucket isolates. The paper argues that binary pooling conflates topically-rich and topically-poor "relevant" documents in ways that distort evaluation, which is precisely the failure mode our tri-state scheme prevents.

**Caveat.** Sormunen's four-level scale is collapsed into binary via thresholding for metric computation, rather than excluding the middle from scoring; cite for the *diagnosis* (marginal documents dominate "relevant" pools and need separate treatment), then justify our exclusion choice as a methodological refinement.

## Source URLs (verification trail)

- [Järvelin & Kekäläinen 2002 — ACM TOIS](https://dl.acm.org/doi/10.1145/582415.582418)
- [Voorhees 2000 — IPM (dblp record)](https://dblp.org/rec/journals/ipm/Voorhees00a.html)
- [Voorhees 2000 — NIST publication record](https://www.nist.gov/publications/variations-relevance-judgments-and-measurement-retrieval-effectiveness)
- [Sormunen 2002 — SIGIR proceedings](https://dl.acm.org/doi/10.1145/564376.564433)
- [Sormunen 2002 — Tampere University research portal](https://researchportal.tuni.fi/en/publications/liberal-relevance-criteria-of-trec-counting-on-negligible-documen)

## How each paper enters our argument

| Paper | Cited for |
|---|---|
| Järvelin & Kekäläinen 2002 | Establishes graded relevance as standard IR practice — frames the principle |
| Voorhees 2000 | Single-curator eval sets produce stable comparative rankings — defends our team-of-one annotation |
| Sormunen 2002 | Documents the empirical prevalence of marginal-relevance cases — motivates the borderline bucket directly |

Together these three give us: (1) principle, (2) defense of methodology, (3) direct empirical analogue. They will anchor the "Methodology — Evaluation Design" subsection of the final paper.
