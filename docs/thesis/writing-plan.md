# Writing plan — how the paper gets written

Set by Mitchell on 2026-10-03. Mitchell writes the paper. Claude prompts.

## The process

1. **Project review first.** A step-by-step walk through the whole project: what was done, why, and what the research said. No writing yet. The point is that Mitchell can explain every decision in his own words before putting any of it on the page.
2. **Then the paper, in stages, paragraph by paragraph, in logical order.**
3. For each paragraph, Claude gives **a prompt and simplified bullet points of the facts** needed. Mitchell writes the paragraph.
4. Claude helps **revise each piece before moving on** — by asking questions or giving simplified feedback, **not by rewriting it**.
5. Figures, tables, diagrams, and screenshots are generated or captured as they come up.
6. Expected to take a few days for a full draft.

**The rule that protects the process:** Claude does not draft or rewrite prose, so Mitchell is not steered toward Claude's wording. Sections of `paper-draft.md` marked *[DRAFTED — REVIEW]* are raw material to be replaced, not text to polish. Claude's own work is the mechanical layer: numbers, tables, figures, references, appendices, and fact-checking.

## Stage 1 — Project review (in the order things happened)

Each step is one sitting-sized chunk. The source is where the record lives; the review is a conversation, not a reading assignment.

| # | Topic | What to be able to explain | Source |
|---|---|---|---|
| 1 | The problem and the original proof of concept | Why short informal queries miss formal card text (query–document asymmetry); what the POC got wrong | `docs/archive/2025-11-03-original-proposal.md`, `archive/poc_v1/` |
| 2 | Corpus and storage | What was filtered out and why; one row per card face; why Postgres + pgvector rather than a separate vector store | CLAUDE.md §3–4, journal 2026-05-17 |
| 3 | What gets embedded | Oracle text only; reminder-text augmentation and why it exists | CLAUDE.md §5 |
| 4 | The evaluation set | 26 queries, six categories, tri-state judgments, and the literature behind single-curator graded relevance | `data/eval/methodology_references.md` |
| 5 | The September pivot | Why the baseline was abandoned; why the encoder changed; why the comparison became "versus Scryfall" | journal 2026-09-11 |
| 6 | HyDE and the first prompt | What HyDE is (Gao et al.); the three jargon shapes; what the 10-query test series showed on 8B and 27B | journal 2026-09-18 (design notes), paper §5.1–5.2 |
| 7 | Stage 2 and 3 | The filter compiler's rules (colour, types, keywords off); pre-filter versus post-filter | `src/search.py`, journal 2026-09-18 §5a |
| 8 | Why fine-tune, and how | The knowledge-ceiling argument; oracle tags as training signal; the recipe and the papers behind each choice | journal 2026-09-18 §1–4a |
| 9 | What fine-tuning did | Held-out tags; the forgetting probe; which configurations improved and which got worse | journal §5b–5c, report Table 3 |
| 10 | The measurement problem | Annotation holes; why top-10 precision was the wrong instrument; R-precision, reachable, depth-to-90% | journal §5d–5g |
| 11 | The second prompt | Concepts, explicit-only filters, omit-nulls; the 2×2 result | journal §5h–5i, report Table 1 |
| 12 | The Scryfall comparison | How expert queries were written and fetched; parity; query complexity; why the comparator is imperfect | journal §5j, report Tables 4–5 |
| 13 | Limits and what comes next | "Destroy all creatures"; the small test set; intersection training and LLM-refined subtags | journal §5k–5l |

## Stage 2 — The paper, section by section

Order is the order to *write* in, not the order of the document: results and method first, framing last.

| Order | Section | Facts come from | Figures / tables available |
|---|---|---|---|
| 1 | 3 Methodology (task, cascade, corpus text, evaluation setup) | CLAUDE.md §2–5, `src/search.py`, review steps 2–4 and 7 | Stage diagram (deck slide 3); Appendix A (prompt), B (eval set) |
| 2 | 5 Results | `docs/reports/<date>/report.md` and `tables/*.csv` only | Report Figures 1–3; Tables 1–6; Appendix C |
| 3 | 6 Discussion | journal §5c–5l; Mitchell's dashboard observations | Dashboard screenshots |
| 4 | 2 Related work | `docs/sources/README.md`, journal §3; `references.bib` | — |
| 5 | 1 Introduction | Review steps 1 and 5; the accessibility framing | Problem table (deck slide 2) |
| 6 | 7 Conclusion and future work | journal §5l | — |
| 7 | Abstract | Written last, from the finished sections | — |

## Standing rules while writing

- Every number comes from `docs/reports/`. If a number is needed that is not in a report table, add it to `scripts/generate_report.py` and regenerate — do not copy from a terminal.
- After changing citations, run `scripts/build_references.py`; the References section and `references.bib` are generated.
- **What a source says comes from the source.** Any claim attributed to a paper, any quotation, and any page number is looked up in the source library (`scripts/source_search.py`, or ask for the citation-librarian agent) before it goes in a paragraph or in the fact bullets for a paragraph. Quotations must pass `--verify` against the cited source; the tool reports the PDF page. If the library does not have it, the answer is "not found", not a best guess. Seven references have no PDF in `docs/sources/` and cannot be checked this way until one is added: `hu2021lora`, `muennighoff2022mteb`, `ouyang2022training`, `jarvelin2002cumulated`, `voorhees2000variations`, `sormunen2002liberal`, `wang2011cascade`.
- The final source review happens after the draft is written, over the sources the paper actually cites (decided 2026-10-05): the citation-librarian audits each citation in the draft against its source.
- After new experiment rows, run `scripts/generate_report.py --paper`; Appendices A–C are generated.
- One unresolved citation remains: "Guo et al., 2016" in Section 2.3 has no identified source. Resolve or remove it when that paragraph is written.
- Appendix D (the 10-query failure-mode series with full rewriter output for the 8B and 27B models) is still a placeholder: those outputs were reviewed in conversation but not logged to a file. Re-running the ten queries on both models would take a few minutes with the servers up.
