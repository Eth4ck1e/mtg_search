"""Render an eval-set draft YAML as visual HTML for review.

For each query in the draft, the HTML shows two side-by-side card grids
— green-tinted for "picked as relevant" and red-tinted for "rejected".
Each card cell shows the Scryfall image, the card name, and the
one-line reasoning the picker left. Clicking an image opens the
Scryfall card page so you can read the full oracle text.

Draft YAML format::

    version: "v1-draft"
    queries:
      - id: q_014
        query: "flicker effects"
        difficulty: hard
        category: jargon
        notes: "Tests reminder-text augmentation for blink/exile-and-return."
        batch: 1
        scryfall_queries:
          - 'o:"exile target creature you control"'
        relevant:
          - id: 6879f5ce-7a1b-4606-bad1-885779b0d456
            why: "Canonical 1-mana flicker"
        rejected:
          - id: 0a094c5f-cefb-4ba1-90c8-dd78ae8efe95
            why: "Exile without return — removal, not flicker"

Usage::

    python scripts/render_review.py --draft data/eval/queries_v1_draft.yaml --out data/eval/review_batch_1.html --batch 1
"""

from __future__ import annotations

import argparse
import sys
import time
from html import escape
from pathlib import Path

import requests
import yaml

SCRYFALL_COLLECTION_URL = "https://api.scryfall.com/cards/collection"
USER_AGENT = "mtg_search/0.2.0 (+https://github.com/Eth4ck1e/mtg_search)"
ACCEPT = "application/json;q=0.9,*/*;q=0.8"
INTER_REQUEST_DELAY_S = 0.1


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": ACCEPT}


def _fetch_cards(oracle_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch Scryfall card data keyed by oracle_id."""
    out: dict[str, dict] = {}
    for start in range(0, len(oracle_ids), 75):
        chunk = oracle_ids[start : start + 75]
        body = {"identifiers": [{"oracle_id": oid} for oid in chunk]}
        resp = requests.post(SCRYFALL_COLLECTION_URL, json=body, headers=_headers(), timeout=30)
        resp.raise_for_status()
        for card in resp.json().get("data", []):
            out[card["oracle_id"]] = card
        if start + 75 < len(oracle_ids):
            time.sleep(INTER_REQUEST_DELAY_S)
    return out


def _image_uri(card: dict) -> str:
    """Pull the best front-face image URL from a Scryfall card record."""
    # Prefer 'large' (672x936) — oracle text is readable. Fall back to png, then normal.
    for size in ("large", "png", "normal"):
        if "image_uris" in card and size in card["image_uris"]:
            return card["image_uris"][size]
        faces = card.get("card_faces") or []
        if faces and "image_uris" in faces[0] and size in faces[0]["image_uris"]:
            return faces[0]["image_uris"][size]
    return ""


def _card_block(card: dict | None, pick: dict, bucket: str) -> str:
    """One card cell in the grid. Falls back gracefully if Scryfall couldn't find it.

    bucket: one of "relevant", "borderline", "rejected".
    """
    cls = f"card {bucket}"
    name = escape(card.get("name", "Unknown card") if card else f"oracle_id: {pick['id']}")
    why = escape(pick.get("why", ""))
    img = _image_uri(card) if card else ""
    scry_url = escape(card.get("scryfall_uri", "#") if card else "#")
    img_html = (
        f'<img src="{escape(img)}" alt="{name}" loading="lazy">'
        if img
        else '<div class="no-image">no image</div>'
    )
    return f"""
      <div class="{cls}">
        <a href="{scry_url}" target="_blank" rel="noopener">{img_html}</a>
        <div class="card-name">{name}</div>
        <div class="card-why">{why}</div>
      </div>
    """


def _query_block(query: dict, cards: dict[str, dict]) -> str:
    relevant = query.get("relevant") or []
    borderline = query.get("borderline") or []
    rejected = query.get("rejected") or []
    scryfall_qs = query.get("scryfall_queries") or []

    relevant_html = "\n".join(_card_block(cards.get(p["id"]), p, "relevant") for p in relevant)
    borderline_html = "\n".join(
        _card_block(cards.get(p["id"]), p, "borderline") for p in borderline
    )
    rejected_html = "\n".join(_card_block(cards.get(p["id"]), p, "rejected") for p in rejected)
    scry_list = "".join(f"<li><code>{escape(q)}</code></li>" for q in scryfall_qs)

    borderline_section = (
        f"""
      <h3 class="header-borderline">Borderline — unjudged ({len(borderline)})</h3>
      <p class="bucket-explainer">These cards are excluded from metric calculation. They neither count for nor against the system's recall@K or MRR if surfaced.</p>
      <div class="grid">{borderline_html or '<p class="empty">— none —</p>'}</div>
        """
        if borderline
        else ""
    )

    return f"""
    <section class="query-block">
      <h2>{escape(query["id"])}: &ldquo;{escape(query.get("query", ""))}&rdquo;</h2>
      <div class="meta">
        difficulty: <b>{escape(query.get("difficulty", "?"))}</b>
        &nbsp;|&nbsp; category: <b>{escape(query.get("category", "?"))}</b>
        &nbsp;|&nbsp; relevant: <b>{len(relevant)}</b>
        &nbsp;|&nbsp; borderline: <b>{len(borderline)}</b>
        &nbsp;|&nbsp; rejected: <b>{len(rejected)}</b>
      </div>
      {f'<p class="notes">{escape(query["notes"])}</p>' if query.get("notes") else ""}
      {f"<details><summary>Scryfall queries used</summary><ul>{scry_list}</ul></details>" if scry_list else ""}
      <h3 class="header-relevant">Picked as relevant ({len(relevant)})</h3>
      <div class="grid">{relevant_html or '<p class="empty">— none —</p>'}</div>
      {borderline_section}
      <h3 class="header-rejected">Rejected ({len(rejected)})</h3>
      <div class="grid">{rejected_html or '<p class="empty">— none —</p>'}</div>
    </section>
    """


_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0 auto;
      padding: 24px;
      max-width: 1500px;
      background: #fafafa;
      color: #222;
    }}
    h1 {{ margin-top: 0; }}
    .query-block {{
      margin: 32px 0 48px;
      padding: 20px 24px;
      background: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .query-block h2 {{ margin: 0 0 6px; font-size: 1.4em; }}
    .meta {{ color: #666; font-size: 0.92em; margin-bottom: 6px; }}
    .notes {{ color: #555; font-style: italic; margin: 8px 0; }}
    details {{ margin: 8px 0 16px; font-size: 0.9em; }}
    summary {{ cursor: pointer; color: #555; }}
    details code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
    .header-relevant {{
      background: #d4edda;
      padding: 8px 12px;
      border-radius: 4px;
      margin-top: 24px;
      font-size: 1em;
    }}
    .header-borderline {{
      background: #fff3cd;
      padding: 8px 12px;
      border-radius: 4px;
      margin-top: 24px;
      font-size: 1em;
    }}
    .header-rejected {{
      background: #f8d7da;
      padding: 8px 12px;
      border-radius: 4px;
      margin-top: 24px;
      font-size: 1em;
    }}
    .bucket-explainer {{
      font-size: 0.85em;
      color: #666;
      font-style: italic;
      margin: 4px 0 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 20px;
      margin-top: 12px;
    }}
    .card {{
      padding: 10px;
      border-radius: 6px;
      transition: transform 0.1s;
    }}
    .card:hover {{ transform: translateY(-2px); }}
    .card.relevant {{ background: #e9f7ec; border: 1px solid #b6dfbe; }}
    .card.borderline {{ background: #fff8e1; border: 1px solid #ffe082; }}
    .card.rejected {{ background: #fbeaec; border: 1px solid #e9bcc0; opacity: 0.85; }}
    .card img {{
      width: 100%;
      height: auto;
      border-radius: 4.75% / 3.5%;
      display: block;
    }}
    .no-image {{
      width: 100%;
      aspect-ratio: 5 / 7;
      background: #eee;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #888;
      font-size: 0.9em;
      border-radius: 4%;
    }}
    .card-name {{ font-weight: 600; margin-top: 8px; font-size: 0.95em; }}
    .card-why {{ font-size: 0.82em; color: #444; margin-top: 4px; line-height: 1.3; }}
    .empty {{ color: #aaa; font-style: italic; padding: 8px 0; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Click any card image to open its Scryfall page.</p>
  {body}
</body>
</html>
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft",
        type=Path,
        required=True,
        help="Path to the draft YAML (e.g. data/eval/queries_v1_draft.yaml).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="HTML file to write.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Only render queries whose `batch:` field matches this number.",
    )
    args = parser.parse_args()

    draft = yaml.safe_load(args.draft.read_text(encoding="utf-8"))
    queries = draft.get("queries", [])
    if args.batch is not None:
        queries = [q for q in queries if q.get("batch") == args.batch]
    if not queries:
        print(f"No queries to render (batch filter: {args.batch}).")
        return 1

    oracle_ids = sorted(
        {
            p["id"]
            for q in queries
            for p in (q.get("relevant") or [])
            + (q.get("borderline") or [])
            + (q.get("rejected") or [])
        }
    )
    print(f"Fetching {len(oracle_ids)} unique card records from Scryfall...")
    cards = _fetch_cards(oracle_ids)
    missing = set(oracle_ids) - cards.keys()
    if missing:
        print(f"  Warning: Scryfall did not return data for {len(missing)} oracle_ids.")

    body = "\n".join(_query_block(q, cards) for q in queries)
    title = f"Eval Set Review — Batch {args.batch}" if args.batch else "Eval Set Review"
    html = _HTML_PAGE.format(title=escape(title), body=body)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  Queries rendered: {len(queries)}")
    print(f"  Cards rendered:   {len(cards)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
