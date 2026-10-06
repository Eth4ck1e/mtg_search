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

Every review step and every writing prompt opens with a one-line reminder of where we are: the stage, the step, and the paper section it feeds (asked for by Mitchell, 2026-10-05).

| # | Topic | What to be able to explain | Source | Feeds paper section |
|---|---|---|---|---|
| 1 | The problem and the original proof of concept | Why short informal queries miss formal card text (query–document asymmetry); what the POC got wrong | `docs/archive/2025-11-03-original-proposal.md`, `archive/poc_v1/` | §1.1 Problem statement (the POC itself is not in the draft yet — decide whether it belongs) |
| 2 | Corpus and storage | What was filtered out and why; one row per card face; why Postgres + pgvector rather than a separate vector store | CLAUDE.md §3–4, journal 2026-05-17 | §3.1 Task and dataset; §2.3 Filtered vector search |
| 3 | What gets embedded | Oracle text only; reminder-text augmentation and why it exists | CLAUDE.md §5 | §3.3 Corpus text preparation |
| 4 | The evaluation set | 26 queries, six categories, tri-state judgments, and the literature behind single-curator graded relevance | `data/eval/methodology_references.md` | §3.4 Evaluation setup; §2.6 IR evaluation methodology |
| 5 | The September pivot | Why the baseline was abandoned; why the encoder changed; why the comparison became "versus Scryfall" | journal 2026-09-11 | §1.1–1.2 framing; §3.4 Evaluation setup |
| 6 | HyDE and the first prompt | What HyDE is (Gao et al.); the three jargon shapes; what the 10-query test series showed on 8B and 27B | journal 2026-09-18 (design notes), paper §5.1–5.2 | §2.2 Query rewriting; §3.2 (Stage 1); §4.2; §5.1–5.2 |
| 7 | Stage 2 and 3 | The filter compiler's rules (colour, types, keywords off); pre-filter versus post-filter | `src/search.py`, journal 2026-09-18 §5a | §3.2 (Stages 2–3); §2.3; §6.2 |
| 8 | Why fine-tune, and how | The knowledge-ceiling argument; oracle tags as training signal; the recipe and the papers behind each choice | journal 2026-09-18 §1–4a | §2.5 Domain adaptation; §6.3; §5.4 (setup) |
| 9 | What fine-tuning did | Held-out tags; the forgetting probe; which configurations improved and which got worse | journal §5b–5c, report Table 3 | §5.4 Embedder fine-tuning |
| 10 | The measurement problem | Annotation holes; why top-10 precision was the wrong instrument; R-precision, reachable, depth-to-90% | journal §5d–5g | §5.5 Set retrieval; §3.4 |
| 11 | The second prompt | Concepts, explicit-only filters, omit-nulls; the 2×2 result | journal §5h–5i, report Table 1 | §5.6 The v2 rewriter; §4.1 Configurations |
| 12 | The Scryfall comparison | How expert queries were written and fetched; parity; query complexity; why the comparator is imperfect | journal §5j, report Tables 4–5 | §5.7 Comparison against expert Scryfall queries; §3.4 |
| 13 | Limits and what comes next | "Destroy all creatures"; the small test set; intersection training and LLM-refined subtags | journal §5k–5l | §6.6–6.7; §7.2 Future work |

## Review notes — decisions and facts to carry into the writing

Recorded during the Stage 1 review (started 2026-10-05). Mitchell's decisions are marked **[M]**. Facts were checked against code, logs, the database or the source library on the date shown. These are notes, not paper text.

**Step 1 — POC (§1.1)**
- **[M]** The POC is inspiration only and is not described in the paper.
- "Flicker" appears in 7 card names and 0 rules texts of 31,124 cards: training on card text alone cannot link the word to the mechanic.
- No POC results were saved; nothing about its performance can be reported as measured.

**Step 2 — Corpus and storage (§3.1, §2.3)**
- Corpus size is **31,124 cards**; 31,972 is the number of faces (vectors), from 847 multi-face cards. Every metric counts cards.
- 38,740 bulk entries − 7,616 filtered (ingest log 2026-09-11).
- Vector search is **exact**, not ANN: there is no vector index. Do not write "ANN".

**Step 3 — Embedded text (§3.3)**
- Reminder-text dictionary: 240 definitions, 143 harvested, 97 hand-written (58 replacing a harvested one, 39 new). Covers 79% of keyword occurrences. 9,929 of 31,972 faces augmented.
- Augmentation was **never ablated**; the tuned embedder was trained on augmented text. **[M]** Discuss as an oversight / limitation.

**Step 4 — Evaluation set (§3.4, §2.6)**
- Judgments were LLM-drafted and audited by Mitchell at batch level (skim), not judged card by card. Describe it that way.
- 13 of 26 queries are jargon; other categories have 1–4 queries each. **[M]** Jargon is the point of the project.
- Headline numbers (Scryfall parity, R-precision vs tag pools) do not depend on these judgments; P@10 and MRR do.
- Voorhees 2000, Järvelin & Kekäläinen 2002, Sormunen 2002 have no PDF in the library: claims about them are unverified.

**Step 5 — September pivot (§1.1–1.2, §3.4)**
- **[M]** The fabricated baseline is left out of the paper entirely.
- **[M]** The paper's question: can plain language do what Scryfall's search syntax does, without the user knowing that syntax?
- Scryfall is the **reference**, not a "baseline"; users write Scryfall syntax, not SQL. The logged floor is the raw query on the base embedder (row 88).
- The Nomic paper describes v1; "v1.5" does not appear in it. "137 million parameter" (p. 1) and the task prefixes (p. 7) are verified.

**Step 6 — HyDE and prompt v1 (§3.2, §5.1–5.2)**
- In the flicker failure the **rewriter** lacked the knowledge; the embedder did its job.
- The 10-query outputs (8B and 27B) are not logged; Appendix D is a placeholder.
- Llama 3.1 8B vs Gemma 3 27B are different model families: "a larger model fixed it" is supported, "size alone" is not.
- **[M]** Design premise: acceptable accuracy at a per-query cost close to a database lookup; a frontier LLM doing everything is assumed better but too costly per query. The assumption is unmeasured. Corpus is about 6.4M characters (~1.6M tokens) of card text.

**Step 7 — Filter and ranking (§3.2, §6.2)**
- Example of a reasonable filter losing correct cards: the counterspell tag has 513 cards — 376 instants, 104 creatures, 33 other.
- "Selective strictness" as a ranking boost for inferred filters was never built; what exists is keywords-off (v1) and explicit-only filters (v2).

**Step 8 — Fine-tuning (§2.5, §5.4, §6.3, §7.2)**
- Only the embedder was trained. The rewriter is stock; LoRA on it was never run.
- Not done: synthetic user queries as training text; mined hard negatives.
- CustomIR "20–32% of mined negatives were relevant" (journal) is unverified.
- The sweeper tag has 655 directly tagged cards and one child tag (`sweeper-one-sided`, 216). Nothing separates damage from destroy, or creature sweepers from land sweepers.
- **[M]** Distinguishing cards *within* a tag group is the main problem for future work (§7.2), with the sweeper case as the example.

**Step 9 — What fine-tuning did (§5.4)**
- Seen tags ndcg@10 0.19 → 0.54; held-out tags 0.114 → 0.242; general benchmarks 0.514 → 0.495.
- Tags were held out, not cards: 29,355 of 31,124 cards appear in training under some tag.
- Training was one-directional (short text → card). The tuned model helps most when the query side is short text, least when it is a long hypothetical card.

**Step 10 — The measurement problem (§5.5, §3.4)**
- **[M]** The reframing to complete-set retrieval changes how earlier results read; §3.4 states which measure is primary and why before any table.
- **[M] (2026-10-05)** The judged-list metrics (P@10, MRR against `queries_v1_draft.yaml` relevance lists) are **out of the paper entirely** — not in tables, not mentioned as an abandoned instrument. The paper presents the methods and measures used, not the ones replaced along the way. Consequence: §3.4, the tri-state methodology material and Appendix B reduce to how the 26 queries were chosen; primary measures are R-precision / reachable / depth-to-90% vs tag pools (21 queries) and Scryfall parity (26 queries). Claude's recommendation to mention the replaced instrument in one sentence was declined.
- Set metrics are over the 21 queries that map to a tag, not 26.
- "Reachable" also charges *correct* narrowing (the tag pool is the wrong target for a query like "instants that draw cards"); the planned `target_filter` fix was not built.
- **Correction to the record (measured 2026-10-05):** the journal and draft say 20 of 21 target tags were seen in training and only `burn` was held out. Three target tag names were held out: `burn`, `flicker`, `cast-trigger-you`. For `flicker` and `burn`, child tags were trained and cover 93% and 51% of the pool; `cast-trigger-you` was effectively unseen (5 of 1,175 cards).
- For trained target tags, the share of the pool actually paired with that tag name in training runs from 5% (`removal`, 297 of 5,968) to 100% (tags under ~150 cards); 19%–100% when child tags are counted. Pairs were capped at 150 cards per anchor text.
- **[M]** The system is not handed the tag: the rewriter has to get from plain language to the concept. In-domain training is the intervention, not a flaw.
- The answer key is still tag membership. Evidence not graded on trained tags: held-out tags (0.114 → 0.242); Scryfall expert queries without tags (control 0.351 → headline 0.434, versus 0.343 → 0.565 with tags allowed).

**Step 11 — The second prompt (§5.6, §4.1)**
- v1 → v2: 11 → 8 rules, 7 → 8 examples, 2,079 → 1,414 request tokens (−32%), 42 → 28 output tokens, ~1.1 → ~0.85 s. The time saving comes from output tokens (server caches the prompt prefix); do not write that the shorter prompt made it faster.
- 2×2 (R-prec vs tag pools, filters on, n=21): v1/base 0.215, v1/tuned 0.326, v2/base 0.169, v2/tuned 0.510. Central mechanism: fine-tuning makes the lighter rewriter viable; neither half alone.
- One query = ~0.048 of a 21-query average. Differences under ~0.05 are one query, not a finding; lead with the large effects and point to Appendix C.

**Step 12 — Scryfall comparison (§5.7)**
- Expert queries (`data/eval/scryfall_expert_queries_v1.yaml`) were drafted by Claude and are **not yet reviewed by Mitchell** — gates §5.7. (The file header cites a `scripts/craft_scryfall_queries.py` that does not exist.)
- Parity (26 queries): control 0.343 / 0.351 (tags allowed / no tags); headline 0.565 / 0.434; depth-to-90% 0.95×; 9 of 26 ≥ 0.85, 7 below 0.30.
- Complexity: 3.7 words vs 20 chars + 1.7 operators (tags) vs 52 chars + 3.2 operators (no tags); 81% of expert queries need `otag:`.
- Claude's recommendation (Mitchell undecided): lead with tags-allowed, always show no-tags beside it, never tags-allowed alone.
- "Destroy all creatures" 0.08: partly unfair (damage wipes), partly fair (land/artifact wipes); the rewriter replaced a specific phrase with the category.
- 81% lets you claim the expert's set requires knowing a tag vocabulary; it does not let you claim users find this easier (no user study).

**Step 13 — Limits and next (§6.6–6.7, §7.2)**
- Draft §6.7 items now wrong: "no naive dense baseline" (raw-query row exists), "no –HyDE ablation" (raw/pass-through rows are it), "single-curator eval set / Voorhees" (out).
- **[M]** First fix with another month: within-tag sub-categories; v3 idea = an LLM curates the full tag list into sub-groups over a long task, possibly with manual review of the custom labels.
- **[M] Conclusion in Mitchell's words:** for less ambiguous card pools (cleaner ability profiles) the system performs very well and can compete with an expert Scryfall user; too many holes remain for it to be reliable in many cases. Claude's refinement: the pattern in the per-query data is "query maps onto one tag or one filter" (good) vs "query is more specific than any tag, or a long description the 8B rewriter misreads" (bad) — pool size is not the divider (tutor, 1,117 cards, scores 0.96).

**Parked until after the writing sessions**
- Fix "20 of 21 / only burn held out" in `paper-draft.md` and the 2026-09-18 journal (see Step 10 correction).
- No-augmentation ablation on the base embedder.
- Re-run and log the 10-query series on both rewriter models (Appendix D).
- A frontier model as the rewriter on the 26 queries, to measure the cost/accuracy assumption.

## Second-pass notes (catch on the next read-through, not now)

Rough-draft mode (Mitchell, 2026-10-05 evening): he writes, Claude inserts the text into `paper-draft.md` fixing only typos and very minor grammar, pushes back only where something is wrong, and logs everything else here for later passes. Goal: a working draft tonight.

- §3.1 ¶1: sentences 2 and 3 overlap ("success is determined by comparing the two" / "The two resulting sets are compared… comparing"); "from a corpus of 31,124 cards" attaches to "match" rather than to what the system is given. Scryfall and Magic must be introduced in §1 before this paragraph uses them.

- §3.1 ¶2: "for limited use including research" — Scryfall's bulk-data terms were not checked; verify against Scryfall's documentation before the final pass and cite it.
- §3.1 ¶3: the tag paragraph no longer says the tags were used for training until its last sentence; fine, but check flow. Tag snapshot date (2026-09-18) is not stated.

- §3.2 overview: "Original work focused only on Stage 3…" refers to the POC, which Mitchell decided to leave out of the paper; Mitchell keeps it for now as the one permitted mention; remove on a later pass if it does no work. §3.2.2 keyword-filter sentence was reworded by Claude at Mitchell's request (factual fix). The design invariant (ranking always runs inside the filtered set, never post-filter) is not stated anywhere in §3.2.
- §3.2.1: "three fields to instruct the model" — the fields are the output contract, not instructions; the JSON goes to Stage 2 *and* Stage 3 (concepts), the text says only Stage 2. The Nomic prefixes (`search_query:` / `search_document:`) are not mentioned in §3.2.3.
- §3.2 sub-headings were renamed by Claude for structure only ("Query rewriter", "SQL pre-filter", "Semantic ranking"); Mitchell wrote "Stage 1/2/3".

- §3.3: does not yet say that only rules text is embedded (no name, colour, cost, type line) or that augmentation covers both embedder checkpoints; Nomic prefix sentence written by Claude at Mitchell's request (end of §3.3). Skip reason fixed in Mitchell's words.

- §3.4 ¶1 starts with a numeral ("26 plain-language…"). Items 1 and 2 reworded by Claude at Mitchell's request (2026-10-05). Claude changed "SQL squeries" → "Scryfall queries" and "automatically generated" → "generated" (factual).

- **Voice (Mitchell, 2026-10-05):** the paper is first-person singular (I / my) — independent study, one researcher. Existing "we/our" throughout the draft to be converted on a later pass. "Reviewed by me" (§3.4) is accurate: a loose review has been done; a thorough one is still owed.

- §3.4 measures: "1/21 (0.048)" holds for the 21-query tag-pool averages; the 26-query parity averages are 1/26 (0.038). Claude changed "for this section" → "for the results section" (tables are in §5). "we/our" slips left for the voice pass.

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
