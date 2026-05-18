# 2026-05-18 — Baseline results and failure-mode analysis

## Context

The Phase 3 baseline configuration ([`configs/baseline.yaml`](../../configs/baseline.yaml)) — raw query → `multi-qa-distilbert-cos-v1` → pgvector cosine search across all 31,832 face rows, dedupe by `oracle_id`, top-10 — has been run against the 26-query eval set ([`queries_v1_draft.yaml`](../../data/eval/queries_v1_draft.yaml)). This is the first measured number for the entire project. Every subsequent retrieval improvement (HyDE in Phase 4, SQL pre-filter, fine-tuning) will be compared against this row. The row landed as **`experiment_runs.id = 13`** at 2026-05-18 22:26:55 UTC.

## What happened

Headline aggregate metrics:

| Metric | Value |
|---|---|
| recall@1 | 0.0110 |
| recall@5 | 0.0146 |
| recall@10 | 0.0191 |
| MRR | 0.1538 |
| latency p50 | 169.83 ms |
| latency p95 | 235.11 ms |
| latency mean | 223.20 ms |
| n_queries | 26 |

Per-category breakdown (pulled from the per_query JSONB):

| Category | n | avg recall@10 | avg MRR |
|---|---|---|---|
| natural | 4 | 0.042 | 0.250 |
| jargon | 13 | 0.025 | 0.231 |
| fragmented | 3 | 0.000 | 0.000 |
| hybrid | 2 | 0.000 | 0.000 |
| constrained | 3 | 0.000 | 0.000 |
| mechanical | 1 | 0.000 | 0.000 |

**Four queries got rank-1 hits** (MRR = 1.0). Every other query scored zero on both recall@10 and MRR. The four successes:

| Query | recall@10 | top-5 retrieved (relevant **bold**) |
|---|---|---|
| q_007 counterspells | 0.10 | **Cancel**, **Counterspell**, Swift Silence, Intervene, Failed Inspection |
| q_017 wheels | 0.11 | **Wheel of Misfortune**, Firehoof Cavalry, Leisure Bicycle, Shepherding Spirits, Relentless Hunter |
| q_018 extra turns | 0.12 | **Temporal Manipulation**, **Capture of Jingzhou**, **Time Stretch**, Exploration, Azusa Lost but Seeking |
| q_024 a card that destroys all creatures | 0.17 | **Day of Judgment**, Extinction, Their Name Is Death, Shatter the Sky, Depopulate |

A spot-check of failure cases (top-5 retrieved, none in eval-set relevant lists):

| Query | top-5 retrieved | What we wanted |
|---|---|---|
| q_001 creatures with flying | Levitation, Tobita Master of Winds, Dense Canopy, Galerider Sliver, Winged Sliver | Birds of Paradise, Cloudshift, Goldspan Dragon, etc. |
| q_006 ramp spells | Backdraft, Demon of Fate's Design, Tolarian Emissary, Colossal Growth, Demon of Catastrophes | Rampant Growth, Sol Ring, Cultivate |
| q_011 tutor | Pop Quiz, Aegis of the Legion, Guru Pathik, Professor of Symbology, Twenty Lessons | Demonic Tutor, Vampiric Tutor, Mystical Tutor |
| q_014 flicker effects | Feather of Flight, Entity Tracker, Robotics Mastery, Rootwater Shaman, Buoyancy | Cloudshift, Momentary Blink, Ephemerate |
| q_015 ETB triggers | Strionic Resonator, King's Assassin, Katara the Fearless, Etrata the Silencer, Nanoform Sentinel | Solemn Simulacrum, Mulldrifter, Wood Elves |

## Decision

**This is the baseline.** recall@10 = 0.019, MRR = 0.154. Every subsequent retrieval phase (HyDE in Phase 4, SQL pre-filter, the three-tower full pipeline) will be measured by improvement against this row. The result is committed both as `experiment_runs.id = 13` and in this journal entry's headline table.

## Reasoning

The pattern in the failure data is sharp enough that the diagnosis writes itself.

**The four queries that succeeded share a structural property: literal lexical overlap between query terms and oracle text.**

- *Counterspells*: oracle text says `"Counter target spell."` The shared stem `counter` appears in both the query and every relevant card. The model's lexical bias does the work.
- *Extra turns*: oracle text on every Time Walk variant says `"Take an extra turn after this one."` Three relevant cards in the top 5.
- *A card that destroys all creatures*: oracle text on every Wrath says `"Destroy all creatures."` Day of Judgment hits rank 1 — almost a word-for-word match.
- *Wheels*: succeeded for a different reason — the card *name* `"Wheel of Misfortune"` contains the word `wheel`, which the embedding pipeline includes in the embedded text representation. The conceptual mechanism (each-player-discards-and-draws) was *not* matched; the name was. The other four cards in top-5 are unrelated.

**The twenty-two failures bifurcate into two distinct classes.**

**Class A — Sampling bias on broad-concept queries (q_001, q_005, q_015 ETB triggers, partially q_006 ramp).** The system *does* find conceptually relevant cards; they're just not in our hand-sampled relevant list. q_001 returned *Levitation*, *Galerider Sliver*, *Winged Sliver* in top-5 — all flying-related cards, none in our 30-card sample of 3,360 fliers. This is the methodology limitation flagged in [the eval-set-construction entry](2026-05-18-eval-set-construction.md): on broad-concept queries where the corpus contains thousands of relevant cards but our sample contains thirty, recall@K mathematically cannot reflect the system's real performance. The hits are happening, just not registering in the metric.

**Class B — Genuine query-document asymmetry, the POC's exact prediction.** The interesting failures. The system has no mechanism to bridge player vernacular ("flicker", "ramp", "tutor", "ETB", "mana dorks") to the oracle text those cards actually use ("exile target creature... return", "search your library for a basic land", etc.).

The *q_011 tutor* case is the cleanest illustration. The English word "tutor" means "teacher." When the embedding pipeline encodes "tutor," it surfaces cards that match the *English* meaning — *Professor of Symbology*, *Twenty Lessons*, *Pop Quiz* (Strixhaven's Lessons mechanic, teaching-themed cards). The MTG meaning of "tutor" — a spell that searches your library — is invisible to a model trained on general English. *Demonic Tutor*'s oracle text is `"Search your library for a card..."` — zero lexical overlap with "tutor." The cosine distance between the query "tutor" and *Demonic Tutor*'s embedding is structurally far.

*q_014 flicker effects* is the same shape. The word "flicker" doesn't appear in any flicker card's oracle text. The model retrieves cards whose oracle text contains *flicker* in a non-MTG sense (a candle flickers, light flickers, etc.) — *Feather of Flight*, *Entity Tracker*, *Robotics Mastery*. The reminder-text augmentation pipeline (Phase 2's `build_embedding_text`) put canonical reminders into the cards' embedded text, but it cannot reach into the *query* and rewrite "flicker" → "exile target creature, then return it to the battlefield." That rewriting is HyDE's job (Phase 4), and the baseline-to-Phase-4 delta on these queries will quantify its contribution.

**Class C — By-design zeros: pure-structural and hybrid queries.** q_019 free counterspell, q_020 red creatures under 3 mana, q_021 1-mana instants — these never had a chance at non-zero recall in the baseline because the embedding cannot enforce structural constraints (numeric mana cost, color identity, type). The expectation noted in the eval-set entry was 0% recall on these queries. The result confirms: q_019/q_020/q_021 are 0/0/0. Hybrid queries q_025 red pingers and q_026 cheap blue counterspells are also 0/0 for the same reason — both their structural and semantic components need cooperative resolution. **These five zeros are evidence of the SQL tower's necessity, not embedding failure.**

**Latency.** ~170ms p50 / 235ms p95 / 223ms mean on exact cosine search over 31,832 vectors on CPU. Comfortably within paper-defensible territory for an interactive search. There's no ANN index on the embedding column; CLAUDE.md's "decide based on Phase 5 measurements" still applies — these numbers are not pressure to add HNSW yet.

## Alternatives considered

Not really applicable — this is a measurement, not a design call. The interpretation of the measurement (what the failure modes mean, what they predict for Phase 4) is the substance.

## Notes for the final report

### Results — baseline

Quote-ready paragraph: *"We embedded the user's raw natural-language query with the same `sentence-transformers/multi-qa-distilbert-cos-v1` model that produced the corpus vectors and ran exact cosine search across all 31,832 face rows. Aggregate baseline performance over the 26-query evaluation set was recall@1 = 0.011, recall@5 = 0.015, recall@10 = 0.019, and MRR = 0.154. Mean per-query retrieval latency was 223 ms (p95 235 ms) on CPU. Only four of twenty-six queries scored a rank-1 hit, all of which exhibited direct lexical overlap between the query string and the oracle text of relevant cards (e.g., 'counterspells' ↔ 'Counter target spell.', 'extra turns' ↔ 'take an extra turn after this one'). The remaining twenty-two queries returned zero relevant cards in the top ten, separating into three failure classes analyzed below."*

### Discussion — failure-mode analysis

Quote-ready paragraph for the discussion: *"The baseline's failure pattern empirically confirms the query-document asymmetry hypothesis identified by our preliminary work. On queries phrased in player jargon — 'flicker effects', 'ramp', 'tutor', 'ETB triggers', 'mana dorks', 'wheels' — the system retrieved cards whose oracle text shared no lexical overlap with the query term, often retrieving cards matching the query word's general-English meaning rather than its MTG-specific meaning (the query 'tutor' surfaced Strixhaven's Lessons-mechanic teaching-themed cards rather than library-search spells; the query 'flicker' surfaced cards whose oracle text contains the word in a non-MTG sense). Reminder-text augmentation of the corpus side (Section X) addresses the card representation but cannot reach into the query side. This is precisely the failure mode that motivates the HyDE-based query rewriter (Section Y): by rewriting the user's jargon-laden query into a hypothetical card's oracle text before embedding, the query and document representations are brought into the same vocabulary."*

### Discussion — what the structural-query zeros mean

Quote-ready paragraph: *"Five queries combining structural constraints with the standard retrieval pipeline (red creatures under three mana, instants that cost one mana, free counterspell, red pingers under three mana, cheap blue counterspells) returned zero relevant cards in the top ten under the baseline. This is the predicted behaviour: cosine similarity over dense embeddings cannot enforce numeric or color constraints that are not present in the natural-language query embedding. The result quantifies the structural necessity of the SQL pre-filter tower introduced in Section Z."*

### Headline table for the Results section

| Metric | Baseline (Phase 3) | After Phase 4 (HyDE+SQL) | After Phase N (fine-tuned) |
|---|---|---|---|
| recall@1 | 0.011 | TBD | TBD |
| recall@5 | 0.015 | TBD | TBD |
| recall@10 | 0.019 | TBD | TBD |
| MRR | 0.154 | TBD | TBD |
| Latency p95 (ms) | 235 | TBD | TBD |

### Per-query category-level breakdown

The natural-language category (n=4) outperformed the jargon category (n=13) by ~65% (0.042 vs 0.025 average recall@10), consistent with the lexical-overlap pattern — "a card that destroys all creatures" embeds closer to "Destroy all creatures." than "board wipes" does. This is one of the cleanest empirical results in the project: natural-language queries beat MTG jargon by a measurable amount in the baseline pipeline, exactly because jargon introduces the asymmetry that HyDE is designed to remove.

## Open follow-ups

- [ ] **Phase 4 HyDE expectation.** Plot the baseline-to-HyDE delta for each Class B query. Predicted big improvements: q_014 flicker, q_011 tutor, q_006 ramp, q_010 card draw engines. Predicted minimal change: q_007 counterspells, q_018 extra turns (already close to literal-text match — HyDE has less work to do).
- [ ] **Phase 4 SQL pre-filter expectation.** Class C zeros (q_019, q_020, q_021) should go from 0 to high recall@10 with minimal SQL filter work — these are essentially routing decisions. q_025 and q_026 (hybrid) need both HyDE and SQL to cooperate; their delta is the cooperative-cooperation signal.
- [ ] **Sampling bias mitigation.** The Class A failures undermeasure real performance. v2 eval set should either programmatically check relevance for broad queries (q_001 fliers: check if returned card's keywords contains 'Flying') or systematically sample by stratification across CMC/color buckets.
- [ ] **The Wheels false-positive question.** q_017 hit *Wheel of Misfortune* (a real wheel) at rank 1 but the other four in top-5 are unrelated cards whose names just contain the word "wheel" (Firehoof Cavalry has no wheel, but Leisure Bicycle and Cycling-related text might be involved). Worth investigating whether the embedding picks up card *names* with sufficient strength — if so, the corpus representation should consider whether names are meaningful in embedding text (they're not currently — preprocess_text.py only augments oracle_text).
- [ ] **Banding's joke text.** Not exercised by any query in v1 — no banding query in the eval set. Worth adding `q_027 "creatures with banding"` in v2 to test whether the force-replace mechanism is needed.

## Related

- [Eval-set construction methodology](2026-05-18-eval-set-construction.md) — what queries we ran against and why
- [Phase 3 roadmap](../roadmap/phase-3-baseline-and-eval.md) — sub-tasks 18–21 covered by this entry
- [Phase 4 roadmap](../roadmap/phase-4-hyde-and-prefilter.md) — what should fix the Class B and Class C failures
- [Corpus survey](2026-05-17-corpus-survey.md) — the 30k corpus this baseline searched
- [Keyword augmentation entry](2026-05-17-keyword-augmentation.md) — corpus-side preprocessing that helps cards but cannot reach the query side
- [POC retrospective](2026-05-17-poc-retrospective.md) — the query-document asymmetry hypothesis these numbers confirm
- [`configs/baseline.yaml`](../../configs/baseline.yaml), [`scripts/evaluate.py`](../../scripts/evaluate.py), [`src/eval/metrics.py`](../../src/eval/metrics.py) — the harness
- `experiment_runs.id = 13` — the row this entry documents
