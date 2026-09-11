# Abstract — Rough Draft Assembly Kit

**Purpose:** Raw material for Mitchell to draft the rough abstract due **Fri Sept 4, 2026**. This document is a workbench, not a template to fill in — read the content bank, then write the abstract from scratch in your own voice.

**Why this exists as a kit and not a pre-written draft:** the abstract is load-bearing writing that appears in the department-honors report and the thesis paper. It has to be defensible as your own — see [feedback_code_ownership.md](../../.claude/memory/feedback_code_ownership.md) equivalent principle for load-bearing artifacts. This kit compiles the facts, references, and shape; the prose is yours.

---

## 1. Target shape (default research-abstract conventions)

- **Length:** 200–300 words. Rough drafts commonly land shorter (150–200) because Preliminary Results are placeholder-heavy at this stage. Aim ~220 for the rough.
- **Structure:** single paragraph, no headings, no bullets, no citations.
- **Five moves, in order:**
  1. **Problem** — the gap the work addresses (1–2 sentences)
  2. **Approach** — the architectural response (1–2 sentences)
  3. **Method** — how the approach is being evaluated (1 sentence)
  4. **Preliminary results** — baseline established; measurements in-progress (1–2 sentences)
  5. **Contribution / significance** — why this matters beyond the specific dataset (1 sentence)
- **Voice:** third person, present tense for the system as it exists ("The system uses..."), past tense for completed work ("A baseline was measured..."), future tense sparingly for in-progress work ("Subsequent milestones will measure...").
- **No first person.** No "we," no "I."
- **No citations.** Author names and years go in Related Work, not the abstract. HyDE and tri-state can be referenced by concept name without citation.
- **No undefined jargon.** "HyDE" is fine if immediately clarified ("HyDE-style query rewriting, in which the query is transformed into a hypothetical target document before embedding"). "pgvector" and `multi-qa-distilbert-cos-v1` should not appear — they are implementation details.

---

## 2. The five moves — what each one does

| # | Move | What it does | Length |
|---|---|---|---|
| 1 | Problem | Names the specific gap the work addresses. Not "search is hard" but the query-document asymmetry framing. | 1–2 sentences |
| 2 | Approach | The architectural response as a claim. Names the three-tower design at conceptual level. | 1–2 sentences |
| 3 | Method | How the approach's contribution is measured. Names the eval set + relevance model + comparison structure. | 1 sentence |
| 4 | Preliminary results | Baseline number + what is being measured next. For the rough draft, this is largely a placeholder pointing at in-progress work. | 1–2 sentences |
| 5 | Contribution | Why this generalizes beyond MTG. The commercial/accessibility angle grounded in the broader search problem. | 1 sentence |

---

## 3. Content bank

Each move below has:
- **Facts / claims / numbers** — the raw material to draw from.
- **Example synthesis** — one candidate sentence showing how the material could combine. Treat as a *shape reference*, not a template. Rewrite in your own words.

### Move 1 — Problem

**Facts / claims to draw from:**
- Trading card games, product search, and other consumer-facing catalog domains still use keyword or attribute-filter search as the dominant retrieval mode.
- Natural-language search over these catalogs is limited by a **query-document asymmetry**: short informal user queries embed too far from formal domain text for direct vector search to work.
- The asymmetry has both a *distributional* component (queries and documents come from different registers) and an *encoder-quality* component (small encoders learn only surface fingerprints of the training distribution).
- Concrete failure example: querying `multi-qa-distilbert-cos-v1` over Oracle text with informal MTG queries produces recall@10 = 0.019 (baseline, `experiment_runs.id = 13`).
- Traditional keyword and Boolean filter interfaces work for expert users but exclude the broader user population that thinks in natural language rather than in the domain's structured vocabulary.

**Example synthesis** *(shape reference — rewrite):*
> Consumer-facing catalog search remains dominated by keyword and attribute-filter interfaces, which work well for expert users but exclude the majority who query in natural language. Applying off-the-shelf dense retrieval directly to this setting fails: short informal queries embed too far from the formal prose of catalog documents to produce useful matches, an asymmetry that produces a baseline recall@10 of 0.019 on a corpus of Magic: The Gathering cards.

### Move 2 — Approach

**Facts / claims to draw from:**
- The architectural response is a **three-tower retrieval pipeline**.
- Tower 1: SQL pre-filter over structured attributes (color, mana value, type line, legality). Exact-match facts stay in relational storage, not the vector space.
- Tower 2: HyDE-style query rewriting — a small local language model transforms the user's natural-language query into a hypothetical target document, and it is the hypothetical (not the query) that is embedded.
- Tower 3: dense vector search over card ability text, executed *inside* the candidate set already filtered by tower 1 (pre-filter, not post-filter on top-K).
- Ordering matters: SQL first, then vector search. Post-filtering top-K vector results collapses recall on constrained queries.
- Storage is a single Postgres instance with a vector extension, chosen because the corpus size (~30k documents) fits comfortably in relational vector storage and avoids operating a second database.

**Example synthesis** *(shape reference — rewrite):*
> This work proposes a three-tower retrieval architecture that addresses the asymmetry directly. A SQL pre-filter narrows the candidate set on structured attributes such as color and mana value; a query rewriter transforms the informal query into a hypothetical target document; and dense vector search over card text runs inside the filtered candidate set rather than after it.

### Move 3 — Method

**Facts / claims to draw from:**
- Evaluation harness: a hand-curated 26-query evaluation set with **tri-state relevance judgments** (relevant / partially relevant / not relevant), following Järvelin & Kekäläinen (2002) and Sormunen (2002).
- Each architectural change is measured against the same eval set so per-component contribution is quantified rather than asserted.
- Comparators include the domain-standard Scryfall search interface, allowing the paper to claim (or not claim) improvement against the tool the target user population already uses.
- Ablations planned: HyDE-with-SQL-only (semantic tower removed) to establish whether the semantic component earns its complexity.

**Example synthesis** *(shape reference — rewrite):*
> The architecture is evaluated against a hand-curated 26-query benchmark with tri-state relevance judgments, and each component's contribution is measured through ablations against a fixed baseline and against the domain-standard Scryfall search tool.

### Move 4 — Preliminary results

**Facts / claims to draw from:**
- Baseline (naive dense retrieval, `multi-qa-distilbert-cos-v1` over Oracle text, no SQL, no HyDE): recall@10 = 0.019, MRR = 0.154, recorded as `experiment_runs.id = 13`.
- The low baseline recall confirms the query-document asymmetry as the primary failure mode rather than a marginal effect.
- The three-tower components (SQL pre-filter, HyDE, ablations) are under active implementation and measurement; results are expected in the Sept 18 revision of the abstract.
- Systematic evaluation and the full experimental matrix land in the M5 milestone (Weeks 5–7 of the fall term).

**Example synthesis** *(shape reference — rewrite):*
> An initial baseline confirms the severity of the asymmetry: naive dense retrieval yields recall@10 = 0.019 and MRR = 0.154 on the evaluation set. Measurements of the three-tower architecture against this baseline are in progress and will be reported in subsequent revisions.

### Move 5 — Contribution / significance

**Facts / claims to draw from:**
- The contribution is both architectural (a three-tower design that addresses query-document asymmetry) and methodological (a measurement discipline that quantifies each component's role).
- Magic: The Gathering serves as a test bed; the asymmetry problem generalizes to other consumer catalog domains — e-commerce product search, media libraries, technical documentation search — where a small vocabulary of expert users controls the existing search interface.
- The commercial motivation: bridging the asymmetry lets users who cannot construct expert queries reach the same results as those who can, expanding the accessible user population for these systems.

**Example synthesis** *(shape reference — rewrite):*
> While the corpus is domain-specific, the query-document asymmetry it exposes generalizes to consumer catalog search broadly, where the barrier of expert-vocabulary interfaces excludes the natural-language user population that products increasingly need to serve.

---

## 4. Open questions Mitchell needs to answer

These fill gaps I cannot derive. Answer before you draft:

- [ ] Does the thesis class syllabus have an implied audience (committee members from other CS subfields? industry reviewers?) that shifts the register?
- [ ] Do you want to name-drop HyDE in the abstract or only describe it conceptually? (Recommend: describe conceptually — the acronym earns its place in Related Work, not the abstract.)
- [ ] Does the rough draft need a working title above the abstract? If yes, draft the title alongside — a title change late is cheaper than the wrong framing settling into your head.
- [ ] Is there anything from the Sept 4 timeline document you want to explicitly cross-reference in the abstract? (Recommend: no. The abstract stands alone.)

---

## 5. After-you-draft checklist

Before submitting, verify:

- [ ] Word count between 200 and 300.
- [ ] Every sentence advances one of the five moves — no throat-clearing openers, no restatements of the title.
- [ ] The baseline number appears exactly once, with its metric name.
- [ ] The word "asymmetry" (or your chosen substitute for it) is defined the first time it appears.
- [ ] Zero first-person pronouns. Zero citations. Zero library/tool names below the level of "Postgres" and "language model."
- [ ] The contribution sentence generalizes beyond MTG. If a reader from an unrelated CS subfield reads only the last sentence, they should understand why the work is not trivially domain-specific.
- [ ] Read it aloud. If a sentence needs a breath midway, split it.

---

## 6. References (in-repo)

- `docs/process/timeline.md` §1 — one-paragraph project summary (closest existing prose to an abstract; useful shape reference, but *do not* copy from it directly — the timeline audience is different).
- `CLAUDE.md` §1–§3 — mission, architecture, storage rationale.
- `CLAUDE.md` §6 — evaluate-before-optimize principle (feeds Move 3).
- `data/eval/queries_v1_draft.yaml` — the 26-query eval set.
- `data/eval/methodology_references.md` — the three IR-evaluation papers backing tri-state relevance.
- `experiment_runs.id = 13` — the baseline row in Postgres.
- `docs/journal/2026-05-18-baseline-results.md` — narrative around the baseline measurement.
