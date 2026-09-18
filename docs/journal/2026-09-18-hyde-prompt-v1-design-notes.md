# 2026-09-18 — HyDE prompt v1: design notes and expected failure modes

**Author:** Mitchell Trafford
**Context:** M4 kickoff — first HyDE prompt drafted in `prompts/hyde_v1.yaml`, six few-shot examples pending refinement. Two design decisions and one expected failure mode identified during example selection that are worth capturing before empirical measurements start, so the paper's Methodology and Discussion sections can trace back to their reasoning.

---

## 1. Design decision: three jargon shapes, three HyDE output shapes

**Setup.** The eval set contains queries at multiple abstraction levels. Some are specific mechanics ("ETB triggers"); some are broad category names ("cantrips", "removal", "ramp"); some are compound player-language phrases ("cantrips that dig", "cheap red removal"). HyDE's correct output shape differs for each — and the difference matters enough that it should be encoded in the few-shot examples so the model learns it.

**The three shapes:**

| Query shape | Example | HyDE output shape | Rationale |
|---|---|---|---|
| **Broad-category jargon** | "cantrips", "ramp", "removal" | filters only, `hypothetical_card = null` | Names a class defined primarily by structural properties (mana cost, types, keywords). SQL pre-filter defines the search space; forcing a specific hypothetical_card narrows the search to a subset of the true class. |
| **Compound jargon** | "cantrips that dig", "cheap red removal" | filters + populated `hypothetical_card` | Combines structural constraints with a specific mechanical component. Filters carry the structural half; hypothetical_card translates the mechanical half into canonical rules text. |
| **Mechanic-specific jargon** | "flicker effects", "ETB triggers" | filters (thin or null) + populated `hypothetical_card` | Names a specific rules-text pattern. Hypothetical_card is the dominant signal; filters are minimal or absent. |

**Concrete case that surfaced this.** The initial `jargon` few-shot draft used the bare query `"cantrips"` with a specific hypothetical_card (Opt's rules text: `"Look at the top two cards of your library. Put one into your hand and the other on the bottom of your library."`). Two problems surfaced:

1. **Over-narrowing** — Opt-style text is *one kind* of cantrip (library-digging). "Cantrip" as a class includes anything cheap that draws or replaces itself. The specific hypothetical would drag semantic search toward Opt-family cards and miss the broader class.
2. **Fragmented-shape collision** — dropping the hypothetical to null (filter-only) fixes the over-narrowing problem, but a single-word query with filter-only output has the same structural shape as a fragmented-category example ("trample creatures", "creatures with flying"). Two examples teaching the model the same shape wastes a few-shot slot.

**Resolution.** The `jargon` few-shot uses `"cantrips that dig"` — a compound query where the structural half ("cantrips" → cheap spell) becomes filters and the mechanical half ("dig" → library manipulation) becomes canonical rules text. This teaches the model HyDE's actual value-add: translating player-language mechanics into Wizards-authored voice. The broad-category behavior (filter-only for bare "cantrips") is described in the system prompt's rules and is expected as an inference-time output shape — even without an explicit few-shot for it.

**Notes for final report — Methodology (Prompt Design subsection):**

> HyDE's output structure is designed to reflect three distinct query shapes in the evaluation set. Broad-category jargon queries (e.g., "cantrips", "ramp", "removal") name a class of cards defined primarily by structural properties; for these, the query rewriter produces filter attributes only, leaving the hypothetical_card field null so the SQL pre-filter defines the search space without over-narrowing. Compound jargon queries (e.g., "cantrips that dig", "cheap red removal") combine structural and mechanical components; both filters and a hypothetical_card are produced. Mechanic-specific jargon (e.g., "flicker effects", "ETB triggers") names a specific rules-text pattern rather than a structural class; the hypothetical_card carries the dominant signal, with filters minimal or absent. The distinction is taught to the query-rewriter model through the choice of few-shot demonstrations and reinforced in the system-prompt rules.

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

## References

- `prompts/hyde_v1.yaml` — the prompt being iterated
- `prompts/hyde_v1_outline.md` — scaffold + iteration guidance
- `docs/journal/2026-09-11-pivot-baseline-abandonment-and-encoder-switch.md` — the research reframing that grounds the outcome-vs-Scryfall framing this design serves
- `docs/sources/2022_gao_hyde.pdf` — HyDE primary source (Gao et al. 2022)
- `docs/sources/2025_never-come-up-empty-adaptive-hyde.pdf` — adaptive HyDE, most current work on when to skip HyDE generation (relevant to the "leave hypothetical_card null" design choice)
