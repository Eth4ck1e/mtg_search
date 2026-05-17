# 2026-05-17 — POC retrospective and what it taught us

## Context

The POC (preserved at git tag `v0.1-poc`, code archived at `archive/poc_v1/`) was an exploratory effort with no formal written plan — a "wing it" exercise to see what a vector search over MTG cards would even look like. It served as the catalyst for turning the project into a research independent study by surfacing problems that aren't obvious from the outside but are central once you start measuring. This entry captures what the POC taught us before the new architecture eclipses it. The lessons here are the foundation of the research paper's Background / Motivation section.

## What the POC did

In one sentence: it tokenized card text with raw `distilbert-base-uncased`, took the `[CLS]` token of the last hidden layer as a 768-d vector, indexed everything in FAISS (`IndexFlatL2`), and queried by embedding the user's query through the same encoder and searching the index. It also attempted continued-pretraining (MLM) on card text and a fine-tune on rulings data plus a planned knowledge-distillation into `distilgpt2`. Concretely:

- `archive/poc_v1/src/compile_initial_training_data.py` — preprocessing and a 14-entry hand-built keyword definition dictionary that injected reminder text inline.
- `archive/poc_v1/src/training/train_initial_model_mps.py` — MLM continued pretraining at `MAX_LENGTH = 64`.
- `archive/poc_v1/src/training/fine_tune_with_rulings.py` — fine-tune on rulings, using `AutoModelForCausalLM` (a causal LM, not a sentence encoder).
- `archive/poc_v1/src/vector_db/build_vector_db.py` — FAISS index over the `[CLS]` token embeddings.
- `archive/poc_v1/src/vector_db/query_vector_db.py` — query path using the same encoder, same truncation, same `[CLS]` token.

## What broke (and what each failure taught us)

### 1. Query-document asymmetry — the central problem

Short informal queries ("a blue counterspell that costs 2", "ramp", "cards that flicker") landed in regions of embedding space that the dense, prose-y card text never occupies. Cosine similarity between a 4-token query embedding and a 60-token oracle-text embedding is dominated by the structural difference between the two text genres, not by topical relevance. The POC's search results often looked plausible at first glance (right colors, right card types) but consistently missed the cards a player would name when asked the same question.

**Lesson:** the retrieval problem in this domain is fundamentally about bridging two different text distributions (informal user query vs. formal card text). It can't be solved by training a better encoder alone — you need either query rewriting that brings the query into the card-text distribution, or document rewriting that brings cards into a query-like distribution, or both. The redesign chooses query rewriting via HyDE.

### 2. `MAX_LENGTH = 64` was undersized

The POC tokenized to 64 tokens. Many MTG cards have oracle text longer than that, especially modal cards, sagas, and anything with reminder text. Aggressive truncation hid information the model needed and is one of the simplest, most actionable failure modes the POC exposed.

**Lesson:** measure the token-length distribution of the corpus before choosing a truncation budget. The new pipeline uses `multi-qa-distilbert-cos-v1`'s native 512-token window.

### 3. Hand-maintained keyword dictionary

The 14-entry dictionary in `compile_initial_training_data.py` covered evergreen keywords (Flying, Trample, Haste, ...) but missed every set-mechanic released in the last decade (Adventure, Surveil, Investigate, Adapt, Mutate, ...) and every player-jargon term that doesn't have a literal Wizards keyword (flicker, ramp, mill, blink, ETB).

**Lesson:** any process that asks a human to maintain a dictionary of MTG mechanics will go stale within a single set rotation. The redesign harvests reminder text from the corpus itself — Wizards has, somewhere across all printings, written canonical reminder text for nearly every keyword. The dictionary is built once by parser and self-maintains on re-ingestion.

### 4. Raw `AutoModel` `[CLS]` is not optimized for similarity

`distilbert-base-uncased` is trained for masked language modeling. Its `[CLS]` token isn't shaped by any contrastive or similarity objective. Sentence-transformer variants (e.g., `multi-qa-distilbert-cos-v1`) are explicitly trained so that cosine similarity between encoded sentences reflects semantic similarity. The POC was extracting an embedding the model was never trained to produce in a useful form.

**Lesson:** when the downstream task is retrieval, use a model trained for retrieval. Don't pre-emptively fine-tune until the right base model has been measured.

### 5. No structured pre-filter, no recall on constrained queries

For "cheap red removal under 3 mana": the right answers are filtered first by color (red), CMC (≤3), and effect type (removal), and only then ranked by the semantic specifics. The POC tried to do all of this in the embedding space, which means asking the encoder to learn "redness", "low mana cost", and "removal-ness" simultaneously from oracle text alone. Encoders can sort of pick up the last one. The first two are facts, not semantics.

**Lesson:** push categorical and numeric constraints out of the embedding into SQL, then run vector search inside the filtered candidate set. Pre-filter, not post-filter — post-filtering on top-K throws away recall when the constraints are tight.

### 6. Architecture choices that were premature optimization

- **FAISS** was chosen because it's the canonical vector DB. At ~30k cards, it's overkill and adds a second data store. pgvector handles this corpus trivially and keeps everything in one place.
- **Knowledge distillation** into `distilgpt2` was on the roadmap before any retrieval quality was measured.
- **Fine-tuning** was treated as a phase you do *before* evaluation, not a remediation you do *because of* evaluation.

**Lesson:** measure first. The new architecture's principle is **evaluation before optimization** — a 30–50 query hand-curated evaluation set comes before any decision to fine-tune, swap models, or invest in optimization.

## Why none of this was wasted

The POC produced the failure cases that justify the redesign. Without it, the new architecture would be a list of plausible-sounding choices; with it, every choice in the new architecture maps to a concrete failure mode that was actually observed. That mapping is what makes a research paper rather than an engineering project, and it's what the Background / Motivation section will be built from.

## Notes for the final report

- **Background section:** describe the POC's architecture in two paragraphs (encoder choice, index choice, training plan).
- **Motivation section:** present each of the five failure modes above as concrete observations from the POC, with quotes or example queries where possible. Each motivates a specific design choice in the new system.
- **Methodology / design rationale:** for every component of the new architecture (HyDE, pgvector pre-filter, sentence-transformer base, reminder-text augmentation, evaluation-first principle), point back to the specific POC failure it addresses. This is the single highest-value structural connection in the paper.

## Open follow-ups

- [ ] Decide whether to include any actual POC search results (good and bad examples) in the appendix. They make the failure modes concrete, but reproducing them requires rebuilding the POC from `v0.1-poc`.
- [ ] During Phase 5's per-query analysis, look for queries where the POC failure modes still partially apply to the new system — those are the most interesting cases for the Discussion.

## Related

- `archive/poc_v1/README.md` — index of archived modules with one-line "why archived" for each
- `docs/archive/2025-11-03-original-proposal.md` — the proposal that bracketed the POC era
- `CLAUDE.md` §2 — the new architecture as a direct response to these failures
