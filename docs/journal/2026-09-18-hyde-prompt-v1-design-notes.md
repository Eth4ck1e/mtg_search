# 2026-09-18 — HyDE prompt v1: design notes and expected failure modes

**Author:** Mitchell Trafford
**Context:** M4 kickoff — first HyDE prompt drafted in `prompts/hyde_v1.yaml`, six few-shot examples pending refinement. Two design decisions and one expected failure mode identified during example selection that are worth capturing before empirical measurements start, so the paper's Methodology and Discussion sections can trace back to their reasoning.

---

## 1. Design decision: three jargon shapes, three HyDE output shapes

> **Correction (2026-09-18, later in the same session):** an earlier draft of this section grouped "cantrips", "ramp", and "removal" together as "broad-category jargon" that should all be filter-only. Empirical testing surfaced that this was wrong. "Ramp" and "removal" are ability-text jargon (they refer to what a card DOES via its rules text), so they warrant populated hypotheticals — that's HyDE's core value-add. "Cantrip" (bare) is the outlier: it refers primarily to structural properties (cheap spell that draws a card), so filter-only is correct there. The corrected framework below distinguishes structural from ability-text jargon.

**Setup.** The eval set contains queries at multiple abstraction levels. Jargon queries in particular vary in what they refer to — some name a structural class of card, most name an ability-text pattern, some combine both. HyDE's correct output shape differs for each.

**The three jargon shapes (corrected):**

| Jargon sub-shape | Refers to | Examples | HyDE output |
|---|---|---|---|
| **Structural jargon** | Structural properties — cost, type, or structural absence of rules text | "cantrips" (bare), "vanilla creatures" | Filter-only. `hypothetical_card = null`. Populating a hypothetical would narrow the search to one specific ability profile within the broader structural class. |
| **Ability-text jargon** *(most common)* | What the card DOES via its rules text | "ramp", "removal", "flicker", "wheels", "tutors", "card draw", "board wipes", "recursion", "graveyard hate", "sac outlets", "mana rocks" | Populated `hypothetical_card` with canonical Wizards-authored rules text for that mechanic. Filters minimal or absent (whatever the query does specify structurally). **This is HyDE's core value-add.** |
| **Compound jargon** | Both structural AND ability | "cantrips that dig", "cheap red removal" | Both populated. Filters carry the structural half; `hypothetical_card` translates the mechanical half into canonical rules text. |

**Concrete cases that surfaced this framework.**

1. **Cantrips edge case.** The initial `jargon` few-shot used bare `"cantrips"` with Opt's rules text as the hypothetical. Two problems: (a) Opt-style text is *one specific* cantrip (library-digging), narrowing semantic search away from the broader cantrip class; (b) making it filter-only produces the same shape as a fragmented example. Resolution: replaced the few-shot with `"cantrips that dig"` (compound jargon) which demonstrates HyDE's translation capability. Bare "cantrips" behavior (filter-only) is expressed via a system-prompt rule, not a dedicated few-shot.

2. **Ramp keyword hallucination.** When tested against `"ramp spells"` (an ability-text jargon query), HyDE correctly populated `types: [Instant, Sorcery]` and a canonical hypothetical (`"Add 2 mana of any one color."`), but ALSO invented `keywords: ["Ramp"]` — a non-canonical entry that would produce zero SQL matches downstream. This surfaced a distinct-but-related bug: the model treats broad player-language mechanic names as if they were canonical Scryfall keywords. Resolution: added an explicit "canonical keywords only" rule to the system prompt with a list of common non-keyword jargon terms flagged as forbidden values for the `keywords` field.

**Longer-term intervention.** Both classes of failure (jargon misclassification, non-canonical keyword hallucination) would be more cleanly resolved by fine-tuning or retrieval-augmenting the HyDE model on **Scryfall tag data** — Scryfall assigns tags like `card-draw`, `ramp`, `removal`, `tutor`, `combo-piece` to cards, and these tags provide a canonical mapping from player-language jargon to card-level anchors. Task #17 (Scryfall tags investigation) and task #27 (M6 fine-tuning consideration) both intersect here. Prompt engineering can push accuracy to a point; specialized training on Scryfall tags is the natural next intervention.

**Notes for final report — Methodology (Prompt Design subsection):**

> HyDE's output structure is designed to reflect three distinct jargon-query shapes observed in the evaluation set. Ability-text jargon queries — the most common shape — name a mechanic by its player-language term (e.g., "ramp", "removal", "flicker", "wheels", "tutors") and refer to what a card does through its rules text; for these, the query rewriter populates `hypothetical_card` with canonical Wizards-authored rules text, and filter fields carry whatever structural attributes the query separately specifies. Structural jargon — a narrower class exemplified by "cantrips" (bare) and "vanilla creatures" — refers primarily to structural properties (mana cost, absence of rules text); for these, the rewriter produces filter attributes only and leaves the hypothetical_card null, so the SQL pre-filter defines the search space without over-narrowing to a specific ability profile within the broader structural class. Compound jargon (e.g., "cantrips that dig", "cheap red removal") combines both dimensions; both filters and a hypothetical_card are produced. The system prompt enforces a hard constraint that the `keywords` filter field accepts only Scryfall's canonical parsed keyword list; non-canonical player-language jargon must go in `hypothetical_card` and cannot be filtered on directly at this stage. Longer-term, Scryfall's card-level tags (card-draw, ramp, removal, tutor, combo-piece) provide a canonical anchor for player-language jargon that would enable either fine-tuning or retrieval-augmenting the HyDE model on domain-labeled data — a natural extension left as future work.

---

## 2. Expected failure mode: lexical variation in semantically-equivalent mechanics

**The pattern.** MTG rules text uses one canonical phrasing for a mechanic; player-facing queries use another. The two phrasings are semantically equivalent but lexically distant. An off-the-shelf sentence encoder (like `nomic-ai/nomic-embed-text-v1.5`) has no MTG-specific knowledge to bridge that gap in embedding space.

**Concrete case.** The Opt-style cantrip text — `"Look at the top two cards of your library. Put one into your hand and the other on the bottom of your library."` — never uses the word "draw" or "card draw," yet the effect is *functionally equivalent* to drawing a card: the user starts and ends with one more card in their hand. A player query like *"spells that draw cards"* semantically includes Opt-family cards, but Opt's rules text and the query "draw cards" embed far apart because the canonical MTG wording for scry-and-put-in-hand doesn't share vocabulary with "draw." Similar cases: *"return to hand"* ≠ *"unsummon"*, *"put into the battlefield"* ≠ *"cheat into play"*, *"proliferate"* ≠ *"add more counters"*.

**Where the cascade helps and where it doesn't.**

- **HyDE helps** on the query side by rewriting player jargon into canonical rules text (jargon→rules-text is what HyDE is designed for).
- **Reminder-text augmentation helps** on the corpus side by injecting canonical rules text where Wizards has printed it inline.
- **Neither addresses** the case where two *canonical* rules-text formulations are semantically equivalent but lexically distant. Fine-tuning the encoder on synthetic (jargon-query, real-card) pairs would help but is deferred to M6.

**Related sub-failure: cost-pattern queries with variable payoffs.** A separate flavor of the same underlying problem shows up for queries whose intent is *"any card with cost pattern X, regardless of what the effect is"* — the canonical example being **"sac outlet"** (any permanent whose activated ability has "sacrifice a creature" as its cost). The cost pattern is a lexical template repeated across many cards; the effect varies wildly (scry, mana, damage, life gain, tokens, draw). HyDE can produce a single hypothetical with one specific payoff, but that hypothetical will embed close to sac-outlets sharing *that specific payoff* and further from sac-outlets with different payoffs. Vector search has no wildcard operator to express "match the cost prefix, don't care about the effect." Practical mitigations that could help but are out of scope for M4: (a) HyDE ensembles with N hypotheticals covering payoff diversity (~N× latency); (b) hybrid retrieval combining sparse BM25 (which matches the "Sacrifice a creature:" token pattern lexically) with dense retrieval; (c) targeted fine-tuning on cost-pattern queries. This failure sub-class should surface distinctly in the M5 per-query analysis and be discussed separately in the paper's failure-mode section.

**Where the whole stack is expected to struggle most.** Queries whose intent covers multiple canonical mechanic wordings (e.g., "cards that get me more cards" — encompasses draw, tutor, scry-into-hand, cascade, top-of-library manipulation, cost-reducing spells that let you cast more) will show retrieval gaps that no single stage fixes. Predicted worst-case categories in the eval set: broad natural-language queries and cross-mechanic queries.

**Notes for final report — Discussion (Failure modes subsection):**

> A predicted failure mode of the retrieval cascade concerns lexical variation across semantically-equivalent card mechanics. MTG's card text uses one canonical phrasing for a mechanic (e.g., "Look at the top two cards of your library. Put one into your hand..."), while player-facing queries may use another (e.g., "spells that draw cards"). Off-the-shelf sentence encoders have no domain-specific knowledge to bridge such phrasings, and the two representations sit at low cosine similarity despite carrying equivalent game semantics. Neither the HyDE query rewriter (which addresses the query-side jargon-to-canonical-text asymmetry) nor the reminder-text corpus augmentation (which addresses inline keyword definitions) fully mitigates this class of failure. Fine-tuning the embedding model on synthetic query-to-card pairs — deferred as future work per Section [X] — is the natural next intervention.

---

## 3. Implications for eval-set interpretation

Both design decisions above have testable predictions that will show up in the M4/M5 measurements:

- **Broad-jargon queries** ("cantrips", "removal", "board wipes", "card draw engines", "graveyard recursion") should show relatively small deltas between the full cascade and the –SQL ablation on aggregate recall, because for these queries the semantic stage contributes little.
- **Mechanic-specific jargon queries** ("flicker effects", "ETB triggers", "extra turns") should show large deltas between naive dense retrieval and the full cascade, because HyDE's canonical-text rewriting is doing meaningful work.
- **Cross-mechanic natural-language queries** ("cards that get me more cards", "cards that mess with the top of my library") should be the *hardest* category for the cascade as a whole — even with all three stages firing, lexical-variation gaps in the encoder will cap recall.

These predictions feed the pre-M4 predictions journal entry (task #11) once written.

---

## 4. Current few-shot content (v1)

Seven examples chosen to teach seven distinct HyDE output shapes. The set does not map one-to-one to the six eval-set categories — "keywords" is a separate few-shot category we introduced because it teaches use of the structured `keywords` filter field, which is a distinct output shape not covered by the "fragmented" example.

| # | Category | Query | Filter output | hypothetical_card |
|---|---|---|---|---|
| 1 | natural | "spells that draw cards in white or red" | colors + types | "Draw two cards." |
| 2 | jargon (compound) | "cantrips that dig" | cmc + types | Opt-style rules text |
| 3 | fragmented (cost-pattern) | "sac outlet" | null | "Sacrifice a creature: Scry 1." |
| 4 | keywords | "trample creatures" | types + keywords | null |
| 5 | hybrid | "creatures that are blue and/or white with etb effects that cost 2" | colors + types + cmc | Generic ETB trigger |
| 6 | constrained | "green creatures with 5 or more power" | colors + types + power | null |
| 7 | mechanical | "put a +1/+1 counter on target creature" | null | Literal rules text |

**Coverage of eval-set categories** (many-to-many, since few-shot categories are output-shape-oriented and eval categories are query-shape-oriented):

- **Eval "natural"** — covered by #1 (natural)
- **Eval "jargon"** — covered by #2 (compound-jargon translation) + the broad-category behavior described in §1's three-shape framework (produces filter-only output at inference time even without a dedicated few-shot for it)
- **Eval "fragmented"** — covered by #3 (cost-pattern via sac outlet) and #4 (keyword-based fragments)
- **Eval "hybrid"** — covered by #5
- **Eval "constrained"** — covered by #6
- **Eval "mechanical"** — covered by #7

**None of these queries appear in `data/eval/queries_v1_draft.yaml`** — verified before selection to avoid data leakage between few-shot demonstrations and eval-time inputs. Prompt iteration will land in future versions (`v2`, `v3`, ...) each with its own `experiment_runs` row.

---

## 5. Future work — interactive filter refinement

Design observation from M4 kickoff testing (this session): the HyDE output shape has a natural production extension — **user-adjustable filters after initial display**.

In a production UI, the flow would be:

1. User submits natural-language query → HyDE runs → filters + hypothetical_card produced
2. Filters displayed as editable UI controls (color pickers, mana-value sliders, type checkboxes, etc.)
3. User adjusts filters as needed (e.g., broadens colors from `contains_any: [R]` to `contains_any: [R, B]`, tightens `cmc <= 2` to `cmc = 1`)
4. Re-run uses the ADJUSTED filters + **original hypothetical_card** — no HyDE re-inference

The key insight is that HyDE inference is the expensive stage (~500ms–2s per call on local hardware); SQL filter adjustment is cheap (a single Postgres query). Separating them lets users interactively refine retrieval without repeated LLM calls. Architecturally, `query_rewriter.py` produces a `HyDEResult` object whose `filters` field can be modified in place before being passed to `search.py` — no code change needed to support this pattern; it's a UI-layer concern.

Out of scope for this thesis (no production frontend planned; the PHP frontend was dropped from term scope on 2026-08-30). Worth noting in the paper's Future Work section because the cascade architecture naturally supports this interactive refinement pattern, which is one of the accessibility-story angles: non-expert users benefit from LLM-driven initial-filter proposal + expert-mode manual refinement.

---

## References

- `prompts/hyde_v1.yaml` — the prompt being iterated
- `prompts/hyde_v1_outline.md` — scaffold + iteration guidance
- `docs/journal/2026-09-11-pivot-baseline-abandonment-and-encoder-switch.md` — the research reframing that grounds the outcome-vs-Scryfall framing this design serves
- `docs/sources/2022_gao_hyde.pdf` — HyDE primary source (Gao et al. 2022)
- `docs/sources/2025_never-come-up-empty-adaptive-hyde.pdf` — adaptive HyDE, most current work on when to skip HyDE generation (relevant to the "leave hypothetical_card null" design choice)
