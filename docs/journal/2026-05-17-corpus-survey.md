# 2026-05-17 — Corpus survey and Phase 2 filter rules

## Context

Phase 1 deliverable 13 ([roadmap](../roadmap/phase-1-foundation-and-logging.md)): the Scryfall `oracle_cards` bulk file has been downloaded (`data/raw/oracle-cards-2026-05-17.json`, 172,955,737 bytes, sha256 `da0f03ad…`), and we need to characterize it before Phase 2's ingest script codifies its filter rules. The survey doubles as the source-of-truth answer to "how big is the corpus, really?" and to the embedding-truncation question raised by the POC retrospective.

## What happened

Ran `scripts/survey_corpus.py` (one PipelineRun, 22.7s, log at `logs/survey_corpus/2026-05-17.jsonl`, structured findings at `data/processed/corpus-survey-2026-05-17.json`). Headline numbers:

| Metric | Value | Source field |
|---|---|---|
| Total records | **37,442** | bulk file iteration |
| Non-card layouts | 3,705 | `NON_CARD_LAYOUTS` (`scryfall_classify.py:24`) |
| `digital: true` | 2,113 | `is_digital_only` |
| `border_color: "silver"` | 462 | `is_silver_bordered` |
| `set_type: "memorabilia"` | 2,585 | survey `by_set_type` |
| `set_type: "funny"` | 1,499 | survey `by_set_type` |
| Multi-faced cards | 3,133 (944 real, after art_series) | `has_multiple_faces` |
| Empty/missing `oracle_text` | 698 | survey loop |

The layout distribution surfaced one unfamiliar value, **`prepare` (47 records)** — Scryfall's new layout for the Secrets of Strixhaven "Prepare" mechanic. Spot-checked one (`Adventurous Eater // Have a Bite`): `digital: false`, `legalities.modern: legal`, two faces. These are real paper cards; the existing `has_multiple_faces` and the absence of `prepare` from `NON_CARD_LAYOUTS` already classify them correctly.

`reversible_card` layout is absent from the `oracle_cards` bulk. Scryfall presumably folds reversibles into their canonical layout when deduping to oracle_id; only the `default_cards` bulk would surface them. Not our concern.

The token-length distribution (using `sentence-transformers/multi-qa-distilbert-cos-v1`, batched 256 at a time, one entry per *embedding row* — i.e. per face for multi-faced cards):

| stat | tokens |
|---|---|
| count | 40,580 |
| min / p50 / p90 / p99 | 2 / 34 / 70 / 101 |
| max | 359 |
| **over 512 (`settings.max_length`)** | **0** |

Character-length p99 on single-faced cards is 396, max 1,489. Per-face on multi-faced p99 is 457, max 1,232. Combined `" // "`-joined p99 is 644.

## Decision

Phase 2's ingest script will exclude a record if **any** of:

1. `layout in NON_CARD_LAYOUTS` (already enforced by `is_non_card_layout`)
2. `digital == true`
3. `border_color == "silver"`
4. `set_type == "memorabilia"`

It will **not** filter on `set_type == "funny"` (the silver-border filter already catches the un-fun ones), on empty `oracle_text` (vanilla creatures stay in), or on `legalities` (format legality is a SQL pre-filter at query time, not an ingest decision).

Expected post-filter corpus: **~28,000–29,000** oracle records producing **~29,000–30,000** embedding rows after the per-face expansion. Matches the "~30k unique cards" framing in [`CLAUDE.md`](../../CLAUDE.md) §1.

The `max_length=512` setting in `src/config.py` is confirmed safe: zero records exceed it. Reminder-text augmentation (Phase 2) has tokens of headroom to add without forcing truncation. The POC's documented failure at `max_length=64` (POC retrospective) is fully resolved by the architecture decision, with measurement to back it.

## Reasoning

- **Non-card layouts** (rule 1) are not cards — tokens, emblems, art-series printings, schemes and planes for one-off formats, vanguard avatars. They have no semantic identity worth searching against. 3,705 records.
- **Digital-only** (rule 2): the project's stated scope is paper Magic, and matching against established baselines (EDHREC, Scryfall search behavior, MTGGoldfish) only makes sense over paper cards. Alchemy rebalances would also pollute precision metrics by introducing near-duplicates of paper cards under different oracle_ids.
- **Silver-bordered** (rule 3): joke un-set cards, none tournament-legal in eternal formats. Filtered by border color specifically, *not* by `set_type: funny`, because Unfinity introduced black-bordered legal cards under the `funny` set_type that should stay in the corpus.
- **Memorabilia** (rule 4): physical swag — oversized commander cards, World Championship decks display copies, art prints. Not cards you cast.

Filters compose at the call site (Phase 2 ingest), not inside the classifier. `scryfall_classify.py` reports facts; the script that uses it owns policy. Keeps the rule set inspectable and easy to ablate later if a filter turns out to be wrong.

The token-length finding is the load-bearing measurement: 0 over 512. Without this number, max_length=512 would be a hopeful guess; with it, we can claim the embedding pipeline has zero truncation loss against this corpus and prove it. Anything paper Magic adds going forward would have to be extraordinarily verbose to change this.

## Alternatives considered

- **Filter `set_type == "funny"` wholesale (1,499 records).** Would catch Unfinity legal cards as collateral damage. The silver-border filter is the right granularity.
- **Filter empty `oracle_text` (698 records).** A vanilla 2/2 for `{1}{G}` has real semantic identity carried by mana cost + type + P/T — the SQL tower covers it. Embedding for empty text is degenerate but harmless if nothing queries by ability text for vanilla cards. Phase 3's eval set will tell us if this is wrong; cheap to revisit.
- **Filter `legalities`-based at ingest time.** Rejected: legality is what the SQL pre-filter is for (`format=commander` should be a query-time predicate, not an ingest decision). Including all legalities at ingest lets us search across formats.
- **Apply rules as positive include-list rather than negative excludes.** Considered for explicitness; rejected because the layout taxonomy will grow over time (Scryfall just added `prepare`) and a positive list would silently drop new layouts until updated.

## Notes for the final report

- **Methodology — data section:** the four exclusion rules above, with per-rule counts, go directly into the paper. The framing "we report rules and counts" is the kind of rigor the methodology section needs.
- **Methodology — embedding pipeline:** the zero-over-512 number defends the `max_length=512` configuration. Paragraph in methodology: "We measured token-length distribution against the embedding tokenizer over the full filtered corpus. The 99th percentile is 101 tokens; the maximum is 359. With `max_length=512`, no record is truncated, so the embedding pipeline preserves the full ability text of every card in scope."
- **Background — corpus stats:** 37,442 → ~29,000 after filters, 944 real multi-faced cards, 40,580 embedding rows total. Table material.
- **Reproducibility:** the survey output is regenerable from the input file's sha256 and the script's git SHA (both logged). Cite the JSONL run log.

## Open follow-ups

- [ ] Phase 2: codify the four rules in `scripts/ingest.py`. The PipelineRun should `run.skip(reason)` for each, so the production ingest produces a finish line with per-rule exclusion counts that should match this survey within ±epsilon. Mismatches signal Scryfall corpus drift between download and ingest.
- [ ] Phase 2: confirm `oracle_id` is present on every record we want to keep (composite key for face rows depends on it). Survey did not measure this — quick to add as a `null_oracle_id_count` field.
- [ ] Phase 3: revisit the "keep empty oracle_text" call once the eval set runs. If queries for vanilla creatures consistently miss, the SQL tower needs to pick up the slack and we may want to embed `type_line + P/T` as a fallback (despite [`CLAUDE.md`](../../CLAUDE.md) §11's anti-suggestion against it — would need explicit justification).
- [ ] Consider whether the corpus survey should run automatically after every `download_scryfall` run. Currently manual.

## Related

- [Phase 1 roadmap](../roadmap/phase-1-foundation-and-logging.md) — tasks 13 (download), 14 (survey), 15 (this writeup)
- [Phase 2 roadmap](../roadmap/phase-2-ingestion-and-schema.md) — where these rules get implemented
- [Redesign kickoff](2026-05-17-redesign-kickoff.md)
- [POC retrospective](2026-05-17-poc-retrospective.md) — the `max_length=64` failure that the token-length finding resolves
- `scripts/survey_corpus.py`, `src/data_processing/scryfall_classify.py`
- `data/processed/corpus-survey-2026-05-17.json` (gitignored, regenerable)
