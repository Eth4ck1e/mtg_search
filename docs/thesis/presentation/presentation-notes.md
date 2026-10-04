# Presentation notes — thesis class, Oct 19 – Nov 15 window

Deck: `mtg-search-presentation.pptx` (this folder; gitignored, rebuild below). Eight slides plus five backup. Speaker notes with per-slide timings are inside the file (View → Notes Page in PowerPoint).

## The one sentence

A newcomer can type what they mean and reach most of what an expert reaches with Scryfall's query syntax — and getting there took teaching the embedding model the game's vocabulary, which then let the query rewriter get simpler.

## Timing plans

The slot is expected to be 5–10 minutes including questions. Slide timings in the notes add up to about 8 minutes with the demo.

| Slide | Content | 10-min plan | 5-min plan |
|---|---|---|---|
| 1 | Title | 20 s | 15 s |
| 2 | The problem: plain words vs syntax | 60 s | 45 s |
| 3 | Three stages | 60 s | **skip** — say it in one sentence on slide 2 |
| 4 | Teaching the embedder ("tutor") | 75 s | 60 s |
| 5 | Result: rewriter × embedder | 60 s | 45 s |
| 6 | Result: parity with expert Scryfall | 60 s | 45 s |
| 7 | Live demo | 120 s | 60 s — run only "tutor", base then tuned |
| 8 | Limits and next | 45 s | 30 s |
| | **Total** | **≈ 8 min 20 s** | **≈ 5 min** |

If the slot turns out to be a strict 5 minutes, hide slide 3 (right-click → Hide Slide) rather than rushing it.

## Demo script

**Before the talk** (5 minutes of setup; do it before you walk in):

```bash
cd "<repo>"
docker compose up -d                                   # Postgres
PYTHONPATH="$PWD" .venv/bin/python -m mlx_lm server \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --port 8080 --log-level WARNING   # terminal 1
PYTHONPATH="$PWD" .venv/bin/python scripts/dashboard.py                                    # terminal 2
```

Open these three links in browser tabs, in order, and leave them loaded. Each one runs the search on load, so the results are already on screen when you switch tabs — no typing, no waiting on the model in front of the room.

1. <http://localhost:8765/?q=is%20a%20planeswalker&k=12&emb=tuned>
2. <http://localhost:8765/?q=tutor&k=12&emb=base>
3. <http://localhost:8765/?q=blue%20counterspells%20that%20cost%202%20or%20less&k=12&emb=tuned>

**During the demo** (about two minutes):

1. **"is a planeswalker"** — point at the Filters box: the rewriter produced `types: ["Planeswalker"]`, 319 candidates, every result a planeswalker. Say: *"Typed in plain words. On Scryfall the same words return nothing; you have to know `t:planeswalker`."*
2. **"tutor"** on the base embedder — Barging Sergeant, Blade Instructor, Proud Mentor. Say: *"A general-purpose model reads 'tutor' as 'teacher' — these all have the Mentor mechanic."* Then switch the **Embedder** dropdown to `models/nomic-mtg-v1`. The grid re-searches: Grim Tutor, Rhystic Tutor, Demonic Bargain. Say: *"Same query, fine-tuned model. Now it knows what a player means."* This is the moment of the talk; give it a beat.
3. **"blue counterspells that cost 2 or less"** — point at the filter (blue, cost ≤ 2) and the candidate count, then the results: all counterspells. Say: *"Colour and cost became database filters; the meaning did the ranking inside them."*

**If anything fails** (server down, projector trouble): go to the backup slides — the same three searches as screenshots, in the same order.

**Do not** type a new query live unless asked; the rewriter takes about a second and an unlucky query can return a confusing first page. If someone asks for one, good candidates are `extra turns`, `fetch lands`, `counterspells`, `ramp`. Avoid `destroy all creatures` unless you want to show the limitation on purpose (it returns every board wipe).

## Numbers on the slides and where they come from

Every number is exported from `docs/reports/2026-10-03/tables/*.csv` into `deck_data.json`; none were typed by hand.

| Slide | Number | Source |
|---|---|---|
| 2 | 3.7 words / 20 chars / 52 chars | Table 5 of the report (query complexity, 26 queries) |
| 3 | 31,124 → 3,364 cards; Miscalculation, Censor, Lofty Denial | live result for "cheap blue counterspells", tuned embedder |
| 4 | 207k pairs, 71 min, 0.114 → 0.242 | Table 3 (fine-tuning), `experiment_runs` row 19 |
| 5 | 0.215 / 0.326 / 0.169 / 0.510; 28 vs 42 tokens | Table 1 (grid), rows 90, 97, 92, 99 |
| 6 | 0.343 → 0.565; 9 of 26; 0.95× | Table 4 and the per-query table, rows 90 and 101 |

## Questions to expect

- **"Does it beat Scryfall?"** No, and that isn't the claim. An expert with the right syntax gets an exact set. The claim is that someone without the syntax gets most of that set (0.57 overlap on average, 0.85 or better on 9 of 26 queries) by typing what they mean.
- **"Did you train on the test?"** The embedder was trained on Scryfall's community tags, and most test queries correspond to tags it saw. Two checks: 482 tags were held out of training entirely and retrieval on them still doubled; and one test query ("burn spell…") maps to a held-out tag.
- **"Why not just use a bigger LLM?"** A 27B model did fix most of the jargon failures in the rewriter. But it costs more per search. Fine-tuning a 137M embedder once (71 minutes on a laptop) moved the knowledge into the cheap component and let the rewriter get *smaller*: 28 output tokens per query instead of 42.
- **"What is R-precision?"** If the true answer set has N cards, look at our first N results and ask what fraction are in the set. 1.0 means the first N results are exactly the right N cards.
- **"What does it get wrong?"** Requests more specific than any community tag — "destroy all creatures" returns every board wipe, including land destruction — and long descriptive queries the small rewriter misreads. Backup slide 9 shows every query.
- **"How big is the evaluation?"** 26 hand-written queries across six styles. Small; it is the main limitation, and fixing one query often costs another at that size.
- **"Does this generalise beyond Magic?"** The pattern should: any catalog with structured attributes, free text, and community vocabulary (products, recipes, parts). It has only been measured on this one.

## Rebuilding the deck

```bash
# 1. regenerate the report tables (only if new experiment rows exist)
PYTHONPATH="$PWD" .venv/bin/python scripts/generate_report.py --since 2026-09-01 --out docs/reports/

# 2. demo screenshots for the backup slides (dashboard + rewriter server running)
#    headless Chrome, 1600x1000, one per demo link above (plus tutor on the tuned embedder),
#    saved as JPEG in docs/thesis/presentation/assets/:
#      demo_planeswalker.jpg  demo_tutor_base.jpg  demo_tutor_tuned.jpg  demo_counterspells.jpg

# 3. build
cd docs/thesis/presentation && npm install && npm run build
```

`deck_data.json` is the exported subset of the report tables the slides read. If the report changes, update it from the new `tables/*.csv` before rebuilding.

## What was checked

The deck passes the PowerPoint file validator and was exported through Keynote to PDF and inspected slide by slide (layout, fonts, all three charts, the screenshot slides). Charts are drawn as shapes rather than PowerPoint chart objects because Keynote and QuickLook both dropped native charts; they therefore look the same in either app, but you edit them by changing `deck_data.json` and rebuilding, not by editing chart data in PowerPoint. It has **not** been opened in PowerPoint itself — worth a one-minute look if you present from PowerPoint.
