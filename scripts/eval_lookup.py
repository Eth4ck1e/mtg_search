"""Look up Scryfall candidates for an eval-set query.

Hits the Scryfall `/cards/search` API with the given query string,
deduplicates to one entry per oracle_id, cross-references against the
local ``cards`` table, and prints a structured list with full oracle
text. The output is shaped for relevance judgment — for each natural-
language eval query, run a Scryfall syntax search, scan the results,
pick the genuine matches, and copy their oracle_ids into
``data/eval/queries_v1.yaml``.

Cards Scryfall returns that aren't in our local table (filtered out by
ingest — digital-only, memorabilia, silver-bordered, non-card layouts)
are hidden by default. Pass ``--include-missing`` to surface them too.

Scryfall query syntax cheat-sheet:
    o:"text"             oracle text contains "text"
    t:type               type-line contains "type"
    c:U / c<=UB          color identity
    cmc:N / cmc<=N       mana value
    legal:modern         format-legal
    (combine with AND or just stacking terms)

Usage::

    python scripts/eval_lookup.py 'o:"exile target creature" o:"return"'
    python scripts/eval_lookup.py 'o:"counter target spell" cmc<=2'
    python scripts/eval_lookup.py 't:land o:"search your library"' --limit 30
"""

from __future__ import annotations

import argparse
import sys
import time

import psycopg
import requests

from src.config import settings

SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/search"
USER_AGENT = "mtg_search/0.2.0 (+https://github.com/Eth4ck1e/mtg_search)"
ACCEPT = "application/json;q=0.9,*/*;q=0.8"
INTER_REQUEST_DELAY_S = 0.1  # Scryfall asks for 50-100ms between requests.


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept": ACCEPT}


def _fetch_all(query: str, limit: int) -> list[dict]:
    """Page through Scryfall's search API, return up to `limit` cards."""
    cards: list[dict] = []
    params = {"q": query, "unique": "cards"}
    url: str | None = SCRYFALL_SEARCH_URL
    next_params: dict | None = params

    while url and len(cards) < limit:
        resp = requests.get(url, params=next_params, headers=_headers(), timeout=30)
        if resp.status_code == 404:
            # Scryfall returns 404 when a search yields zero results.
            return []
        resp.raise_for_status()
        payload = resp.json()
        cards.extend(payload.get("data", []))
        if payload.get("has_more"):
            url = payload.get("next_page")
            next_params = None  # The full URL already has the cursor.
            time.sleep(INTER_REQUEST_DELAY_S)
        else:
            url = None
    return cards[:limit]


def _local_oracle_ids(conn: psycopg.Connection, oracle_ids: list[str]) -> set[str]:
    if not oracle_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT oracle_id::text FROM cards WHERE oracle_id::text = ANY(%s)",
            (oracle_ids,),
        )
        return {row[0] for row in cur.fetchall()}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        help='Scryfall search syntax, e.g. \'o:"exile target creature" o:"return"\'',
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Cap on results to fetch (default 50, max 175 per Scryfall page).",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Also show cards Scryfall returned but our local DB does not contain.",
    )
    args = parser.parse_args()

    cards = _fetch_all(args.query, args.limit)
    if not cards:
        print(f"No results from Scryfall for query: {args.query}")
        return 0

    oracle_ids = [c.get("oracle_id") for c in cards if c.get("oracle_id")]
    with psycopg.connect(settings.database_url) as conn:
        local = _local_oracle_ids(conn, oracle_ids)

    present = [c for c in cards if c.get("oracle_id") in local]
    missing = [c for c in cards if c.get("oracle_id") not in local]

    print(f"Query: {args.query}")
    print(
        f"Scryfall returned {len(cards):,} cards "
        f"({len(present):,} in local DB, {len(missing):,} filtered out)"
    )
    print()

    for card in present:
        print(f"  - oracle_id: {card['oracle_id']}")
        print(f"    name:        {card.get('name')}")
        print(f"    mana_cost:   {card.get('mana_cost') or '—'}")
        print(f"    type_line:   {card.get('type_line')}")
        # Multi-face cards lack a top-level oracle_text; fall back to faces.
        oracle = card.get("oracle_text")
        if not oracle and card.get("card_faces"):
            oracle = " // ".join(f.get("oracle_text", "") for f in card["card_faces"])
        oracle = (oracle or "").replace("\n", " | ")
        print(f"    oracle_text: {oracle}")
        print()

    if args.include_missing and missing:
        print(f"--- {len(missing)} cards NOT in local DB (filtered by ingest) ---")
        for card in missing:
            print(
                f"  - {card.get('name')} ({card.get('set_name')}) oracle_id={card.get('oracle_id')}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
