# 2026-05-17 — Keyword reminder-text augmentation and manual overrides

## Context

Phase 2 sub-tasks 20-22: build the `build_embedding_text` function that produces the canonical string fed to the encoder, and curate the keyword reminder-text dictionary it consumes. The corpus survey ([corpus-survey entry](2026-05-17-corpus-survey.md)) confirmed 30k+ cards with keyword-bearing oracle text, the embedding tokenizer has zero truncation pressure at 512 tokens, and the auto-extracted dict from `scripts/build_keyword_dict.py` produced 205 reminders from ~709 keywords Scryfall labels in the post-filter corpus. The remaining work: validate the dict against real cards, fix what was wrong, and decide what to do about the 504 keywords for which no parenthetical reminder existed in the corpus.

## What happened

Implemented [`src/preprocess_text.py:build_embedding_text`](../../src/preprocess_text.py). The function is deliberately small: for each keyword in a card's `keywords[]`, if the keyword has a dict entry **and** the card's `oracle_text` doesn't already contain an inline parenthetical for that keyword (checked via [`extract_reminder_texts`](../../src/data_processing/keyword_extract.py)), append `<Keyword> (<reminder>)` on a new line. Augmentation is **always additive** — the function never modifies existing oracle text. Thirteen unit tests in [`tests/test_preprocess_text.py`](../../tests/test_preprocess_text.py) pin down the four sub-task-22 cases (passthrough, no inline reminder, inline already present, keyword not in dict) plus edge cases (cost-keyword inline detection, empty oracle text, None handling, the merge precedence of `load_keyword_dict`).

A first demo run against the live corpus surfaced three failure classes I hadn't anticipated:

1. **Cross-talk false positives.** Keyword Enchant's auto-extracted reminder was `"Battlefield, command, exile, and stack are shared zones..."` — an unrelated parenthetical from some other card that happened to be preceded by "Enchant" at start-of-line. Same for Food (Crew-text bled onto a Food-themed Vehicle) and Machina (Equip-text on a `Machina — Equip {4} (...)` card). Three keywords, impact 1,223 + 177 + 1 cards respectively.
2. **Variable-instance wording.** Protection's reminder was `"...by anything blue"`, baked in from the most recent printing's specific from-color. Cycling included `"{2}, Discard..."` (a specific cost). Ward, Kicker, Equip, Bushido, Crew, Affinity, ~30 others all had the same shape — the *category* of mechanic was right but the *instance* was wrong.
3. **Em-dash-format ability words missing entirely.** Landfall, Threshold, Delirium, Constellation, Heroic, Magecraft, Morbid — none had parens to extract because Wizards prints these with em-dash separators (`Landfall — Whenever a land enters...`). High impact: Landfall alone has 172 cards.

Audited all 205 extracted entries in three batches (top 60 by impact, 61-120, 121-205), proposed generic wording for variable-instance and cross-talk cases, and worked through the 504 no-reminder list to surface ~39 high-impact ability words and evergreen verbs (Mill at 612 cards, Scry at 444, Surveil at 219, etc.) that the auto-extractor structurally couldn't capture.

Two real-data fact-checks during the audit:

- Verified **Amass varies by creature type** across the corpus — `Amass Zombies` (Dreadhorde Invasion), `Amass Orcs` (Lord of the Rings set), `Amass Slivers` (Lazotep Sliver). The proposed override removed the Zombie-baked wording.
- Verified **Threshold is always 7** — spot-checked 10 cards across Odyssey (2002) through current prints. The "seven or more" wording is canonical and stable.

Final dictionary state:

| Source | Count | Notes |
|---|---|---|
| Auto-extracted (`reminder_text.json`) | 205 | Phase 1 sub-task 15-18 deliverable; regenerable from the corpus |
| Manual overrides — FIX | 96 | Corrections of variable-instance or cross-talk; new generic reminders for em-dash mechanics and evergreen verbs |
| Manual overrides — SKIP | 4 | Empty-string sentinel = "do not augment" (Enchant, Food, Machina, Gift) |
| **Effective dict size** | **~239 distinct keywords** | (overrides win over auto when both exist) |

Augmentation coverage after curation:

| Metric | Value |
|---|---|
| Face rows with non-empty `keywords[]` | 15,942 |
| Face rows actually augmented | 9,677 (60.7%) |
| Mean augmentations per augmented row | 1.28 |
| Max augmentations on a single row (Odric, Blood-Cursed) | 10 |

## Decision

The embedding text for a card row is the card's `oracle_text` with `<Keyword> (<canonical reminder>)` lines appended for each keyword the card has a dict entry for but doesn't already explain inline. Dict construction is a **two-stage corpus-driven process**: auto-extract parenthetical reminders from the corpus, then hand-curate via `data/keywords/manual_overrides.json` for the three failure classes above. The overrides file is committed as a reproducibility artifact for the paper. The function is pure; versioning lives in `settings.preprocess_version` and is bumped when the curation logic meaningfully changes.

The auto-extractor produces `reminder_text.json` (regenerable). The overrides file is hand-maintained. Both are committed. Phase 3's embed.py reads the merged dict via `load_keyword_dict()`.

## Reasoning

**Why corpus-driven extraction in the first place.** CLAUDE.md §5 anti-suggests hand-building a keyword dictionary from comprehensive-rules text. The argument: somewhere across all printings of every keyword Wizards has actually printed canonical reminder text. Harvesting from the corpus is self-maintaining — every new set's keywords arrive with their parentheticals already attached and get picked up on the next ingest. The corpus survey confirmed the volume is real (709 keywords appear across 15,942 cards in the post-filter corpus), and the auto-extractor caught 205 of them with no manual intervention.

**Why three failure classes need different treatment.** Cross-talk false positives have to be *removed* (SKIP, empty string) — they actively inject wrong information into the embedding. Variable-instance wording has to be *generalized* — a Protection card with the right concept but the wrong specific color is only marginally degraded, but generalizing costs nothing. Em-dash ability words have to be *added* — the auto-extractor can't see them at all and they're high-impact (Landfall: 172 cards, Mill: 612).

**Why the inline-check is the load-bearing design choice.** Without it, every augmentation would duplicate the inline reminder text on cards that already have it — Storm Crow would have `Flying (This creature can't be blocked...)` twice in its embedding text. With it, augmentation fires *only* on cards that need it: vanilla creatures with `keywords[]` of just `[Flying]` and no inline paren get the boost; old cards with the full parenthetical inline are left untouched. This is what makes the function safely additive — it can never replace canonical Wizards-written text with a genericized substitute, by construction.

**Why "starting conservatively" is the right curation policy.** Augmentation is additive: an over-aggressive override only adds noise, never removes signal. The cost of a too-specific generic ("Cycling means specifically pay {2} and discard") is bounded — the embedding model trained on natural English language will still match queries for "cycling" against any cycling card. The cost of a wrong cross-talk reminder is also bounded — the inline-check skips augmentation when there's already a parenthetical. So the dominant risk isn't "did we get the wording exactly right" but "did we miss high-impact mechanics that need *some* reminder." We aimed at the second.

## Alternatives considered

- **Hand-build the keyword dict from MTG comprehensive rules text.** Rejected. Forces ongoing manual maintenance every time Wizards prints a new set, mismatches Scryfall keyword spellings (CR uses different capitalization in places), and contradicts CLAUDE.md §5. The corpus already has every keyword Wizards has ever printed reminder text for — there's no need to duplicate that.
- **Replace-mode for "bad inline reminders" like Banding's joke text.** Banding's auto-extracted reminder is `"Just ask around until you find someone who knows."` — Wizards' actual printed joke on those cards. The manual override has a real mechanic-explanation but never fires, because all 19 Banding cards have the joke inline and the inline-check skips augmentation. A force-replace mode would require modifying `build_embedding_text` to strip the existing parenthetical for tagged keywords. Deferred — engineering cost is high for 19 cards, and the joke text isn't semantically *wrong*, just unhelpful. Revisit if Phase 3 eval shows banding queries failing.
- **Stricter regex anchoring to avoid cross-talk at extraction time.** Could add `—` (em-dash) to the regex exclude set, which would catch the Machina case at the auto-extract stage. But it would also break Awaken extraction (`Awaken N—{cost} (...)` is the standard pattern). Trade-off rejected — Machina has impact 1, Awaken has 15.
- **Generate the dict from Scryfall's keyword API directly.** Scryfall has a `catalog/keyword-abilities` endpoint that lists keywords but doesn't include reminder text. Same coverage problem as the comprehensive rules approach — would still need corpus-driven harvesting for the actual reminder strings.
- **Per-instance reminders (one entry per Protection-from-color, one per Cycling cost).** Combinatorial explosion. The dict would balloon to thousands of entries and still miss combinations. The generic wording captures the concept, which is what semantic search needs.

## Notes for the final report

### Methodology — embedding text construction

Quote-ready paragraph: *"For each card face we construct an embedding-text representation by augmenting the face's `oracle_text` with canonical reminder text for any keyword the face has but doesn't already explain inline. The keyword dictionary is built by two-stage corpus-driven curation: an automatic extraction pass identifies parenthetical reminder text following keyword tokens in the existing oracle texts of all printed cards (205 keywords extracted with no manual intervention), and a manual-override layer corrects three classes of failure surfaced by hand audit (cross-talk false positives where a generic English word incidentally precedes an unrelated parenthetical; instance-specific wording where the most recent printing's specific cost or color is baked into a reminder that should be generic; and em-dash-format ability words like Landfall and Threshold whose Wizards-prescribed reminders never appear in parentheticals anywhere in the corpus). The final dictionary has 100 manual override entries (4 suppressions, 96 corrections or additions) on top of the 205 auto-extracted base, covering an effective ~239 distinct keywords."*

Quote-ready paragraph on the inline-check: *"The augmentation function is structurally additive: it never modifies an existing inline reminder, only appends new lines to the end of the oracle text. This is enforced by checking, for each keyword, whether the card's existing oracle text already contains a parenthetical attributable to that keyword via the same regex used for extraction; if so, augmentation is suppressed. By construction the function cannot replace canonical Wizards-written text with a genericized substitute."*

### Results — augmentation coverage

Direct table material: 15,942 face rows with non-empty keyword lists, of which 9,677 (60.7%) received at least one augmentation. Mean 1.28 augmentations per augmented row, max 10 (*Odric, Blood-Cursed* — Deathtouch, Lifelink, Reach, Indestructible, Hexproof, First strike, Haste, Trample, Menace, Double strike).

### Discussion

- The three failure classes are *empirically observed*, not anticipated from the literature. They're the kind of finding that justifies the corpus-driven approach over a rules-document approach: only running against real cards surfaced them.
- The "Bashful Beastie / Manifest dread" edge case (period before paren) is a small but real limitation of the inline-check regex worth documenting as future work.

## Open follow-ups

- [ ] **Bashful Beastie / Manifest dread edge case.** Card text ending in `"manifest dread. (Look at...)"` fails the inline-check because the period is in the regex exclude set. Augmentation fires and duplicates the explanation. Flag for Phase 3 eval — if "manifest dread" queries show no degradation, leave as-is.
- [ ] **Suspend / Foretell / Plot search-term coverage.** My generic wordings preserve the mechanic ("exile it... cast it later without paying its mana cost") but don't make `"cast from exile"` or `"cast not from hand"` lexically explicit. If queries for those phrases miss suspend/foretell/plot cards, tighten the wording.
- [ ] **Banding's joke text on 19 old cards.** The manual override is dormant because the inline-check sees the printed joke as a valid paren and skips. If Phase 3 shows banding queries failing, build a force-replace mechanism.
- [ ] **Remaining ~465 keywords in the 504-no-reminder list.** Mostly card-specific named abilities ("Vicious Mockery", "Wake Up!", "Tunnel Snakes Rule!") that don't have generic semantic content. Defer until Phase 3 eval surfaces specific gaps.
- [ ] Phase 3 sub-task: `embed.py` will read `load_keyword_dict()` and pass results to `build_embedding_text`. The `embedding_text_hash` column on each face row will commit to the exact text fed to the encoder, so a dict change forces re-embedding on next run.

## Related

- [Phase 2 roadmap](../roadmap/phase-2-ingestion-and-schema.md) — sub-tasks 15-22 covered by this entry
- [Corpus survey](2026-05-17-corpus-survey.md) — the 709-keyword universe and 512-token headroom finding
- [Cards schema entry](2026-05-17-cards-schema.md) — the `embedding_text_hash` column that pairs with `build_embedding_text`'s output
- [`CLAUDE.md`](../../CLAUDE.md) §5 — the keywords/reminder-text philosophy this implements
- [`src/preprocess_text.py`](../../src/preprocess_text.py), [`src/data_processing/keyword_extract.py`](../../src/data_processing/keyword_extract.py), [`scripts/build_keyword_dict.py`](../../scripts/build_keyword_dict.py)
- [`data/keywords/reminder_text.json`](../../data/keywords/reminder_text.json), [`data/keywords/manual_overrides.json`](../../data/keywords/manual_overrides.json)
