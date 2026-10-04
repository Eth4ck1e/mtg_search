# 2026-10-03 — Report generator (M5 deliverable) and a clean grid

**Status:** done. `scripts/generate_report.py` exists and runs; first report at `docs/reports/2026-10-03/`.

## What was built

- **`scripts/generate_report.py`** — reads `experiment_runs`, selects the newest row per (configuration, embedder) since `--since`, joins Scryfall-comparator rows through `cascade_row`, and writes `report.md`, one CSV per table, and three SVG figures. No new dependencies (figures are hand-emitted SVG). Logged as a pipeline run.
- **Tables:** (1) configuration grid, (2) by query category, control vs headline, (3) fine-tuning probes incl. NanoBEIR base → tuned, (4) Scryfall parity for every cell and both expert variants, per-query parity with the expert query text, (5) query-complexity comparison, plus a provenance appendix (selected row, date, prompt, prompt hash, superseded rows).
- **Figures:** grouped bars for rewriter × embedder (two categorical slots, validated for colour-vision separation in light and dark); emphasis bars for Scryfall parity with the headline configuration highlighted; a ranked single-hue bar list of parity per query. Light-mode colours are written as SVG attributes so Word/pandoc/slide tools render them; dark mode is a CSS override. Titles are descriptive, not conclusions — the argument belongs in the text.

## The clean grid (rows 88–101, comparators 102–129)

All seven configurations were re-run on **both** embedders in one session (possible without re-embedding since migration 0004), then the comparator on each row for both expert variants. Every earlier headline number reproduced exactly — control parity 0.343, v2+tuned+keywords parity 0.565, v2 tuned R-precision 0.510, v2 base 0.169 — so the pipeline is deterministic at temperature 0 and the earlier ad-hoc rows were sound. The grid adds cells that were missing: raw-dense and no-filter rows on the base embedder with set metrics, and comparator rows for every cell (the base rows range 0.15–0.34 parity; the tuned rows 0.32–0.57).

## Fixes made along the way

- **Prompt content hash.** `hyde_v2.yaml` was edited in place (omit-nulls, the empty-colours rule) under the same `version: v2`, so rows 51–72 cannot be attributed to a prompt state from the row alone. Eval rows now record `hyde_prompt_sha` (first 12 hex of the file's SHA-256), shown in the report appendix. Going forward: bump the version *or* rely on the hash, but never compare rows across differing hashes without saying so.
- **Tag-pool queries.** `evaluate.py` and the comparator each looped 21 tags through the recursive closure view (~1.5 s per tag). One `ANY()` query now: the comparator went from 33 s to 2 s per run with identical output.
- A regex edit left `src/search.py` with a syntax error for a few minutes and the first grid attempt started against it; it wrote no rows (verified) and was restarted after the fix.

## Observations from the full grid

- Stage 1 latency in the grid varies run to run (v1: 1.05–1.72 s; v2: 0.76–1.22 s) because each run is a separate process and the first calls pay model warm-up; the per-query token counts are stable (v1 42, v2 28 output tokens). Report latency as a range and tokens as the stable cost measure.
- On the full-pool measure the unfiltered tuned rows score highest (raw query 0.598, v2 concepts 0.623) because filters cut the reachable pool to ~0.7; on Scryfall parity — where the expert's set is also filtered — the filtered rows win (0.565 vs 0.40–0.42). The two measures answer different questions and both tables are needed.

## Correction to the 2026-09-22 comparator write-up

Exporting slide data from the report CSVs exposed numbers in journal §5j (2026-09-18 entry) and paper §5.7 that were copied by hand from the **21-query** comparator rows (68–71) but labelled "26 queries". The 26-query values, from `06_query_complexity.csv` and `05_parity_per_query.csv`:

| Quantity | Written on 09-22 | Correct (n = 26) |
|---|---|---|
| Plain query length | 3.8 words | 3.7 words |
| Expert query, no tags | 59 chars, 3.3 operators | 52 chars, 3.2 operators |
| Expert query, tags | 20 chars, 1.5 operators | 20 chars, 1.7 operators |
| Queries needing `otag:` | "100% of jargon queries" | 81% of all queries |
| Expert sets vs judgments (tags) | 11% precision / 90% recall | 9% / 92% |
| Expert sets vs judgments (no tags) | 17% / 72% | 14% / 77% |
| Queries at ≥ 0.85 parity | "ten of 26" | nine of 26 |

Paper §5.7 is corrected and now cites the report folder. The parity headline numbers (0.343 → 0.565) were always from 26-query rows and are unchanged. This is the failure the report generator exists to prevent: **numbers go from `experiment_runs` to the report to the paper and slides, never from a terminal to prose.**

## Presentation deck (thesis class, 5–10 minute slot)

`docs/thesis/presentation/`: `build_deck.js` (pptxgenjs; `npm install && npm run build`), `deck_data.json` (exported from the report CSVs), `presentation-notes.md` (timing plans for 5 and 10 minutes, demo script with one-click links, expected questions, rebuild steps). The `.pptx` and the demo screenshots are gitignored (regenerable; screenshots contain card art).

- Eight slides + five backup: problem (plain words vs Scryfall syntax), the three stages, the "tutor" before/after, rewriter × embedder, Scryfall parity, live demo, limits and next; backups are per-query parity and four demo screenshots.
- **Demo links.** The dashboard now accepts `?q=<query>&emb=tuned|base&k=N` and runs the search on load, so the three demo searches are pre-loaded browser tabs rather than live typing. Best demo found by auditioning queries: **"tutor"** — the base embedder returns Mentor-mechanic creatures (Barging Sergeant, Blade Instructor, Proud Mentor: a tutor is a teacher); the tuned embedder returns Grim Tutor, Rhystic Tutor, Demonic Bargain.
- **Charts are drawn as shapes, not chart objects.** Both QuickLook and Keynote dropped pptxgenjs's native charts on this machine; shape-drawn bars render identically in Keynote and PowerPoint. Verified by exporting the deck through Keynote to PDF and inspecting every chart slide.
- Display rounding: values exported to three decimals are rounded half-up in the deck (0.565 → 0.57), matching the report.

## Next

Presentation prep (window opens 2026-10-19); expert-query review and the hole-judging pass remain open on Mitchell's side.
