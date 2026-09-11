"""Interactive semantic-search test tool.

**Naive dense retrieval only.** No HyDE query rewriting, no SQL pre-filter —
this is the baseline configuration for hand-testing the fresh corpus with
Nomic Embed v1.5. Useful for getting a qualitative feel for where semantic
search alone succeeds and where it fails (jargon queries, structural queries,
etc.).

The M4 cascade — HyDE query rewriter + SQL pre-filter + semantic search —
will run through a proper ``src/search.py``. This script is scratch.

Usage::

    python scripts/test_search.py "cheap red removal"
    python scripts/test_search.py --top 20 "cards that flicker creatures"
    python scripts/test_search.py --dedupe "counterspells"
"""

from __future__ import annotations

import argparse
import sys

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

import src.utils.quiet  # noqa: F401  — side-effect: silence known-benign upstream warnings
from src.config import settings
from src.preprocess_text import format_for_nomic_query
from src.utils.device import select_device

_SEARCH_SQL = """
    SELECT name, type_line, mana_cost, oracle_text,
           1 - (embedding <=> %s::vector) AS similarity
    FROM cards
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""

_SEARCH_SQL_DEDUPED = """
    WITH ranked AS (
        SELECT name, type_line, mana_cost, oracle_text, oracle_id,
               embedding <=> %s::vector AS distance,
               ROW_NUMBER() OVER (
                   PARTITION BY oracle_id ORDER BY embedding <=> %s::vector
               ) AS rn
        FROM cards
        WHERE embedding IS NOT NULL
    )
    SELECT name, type_line, mana_cost, oracle_text, 1 - distance AS similarity
    FROM ranked
    WHERE rn = 1
    ORDER BY distance
    LIMIT %s
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", type=str, help="Natural-language query.")
    parser.add_argument(
        "--top", type=int, default=10, help="Number of results to return (default: 10)."
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate results by oracle_id (best face per card only).",
    )
    args = parser.parse_args()

    device = select_device()
    print(f"[loading {settings.embedding_model} on {device}]", file=sys.stderr)
    model = SentenceTransformer(
        settings.embedding_model,
        device=str(device),
        trust_remote_code=True,
    )

    prefixed = format_for_nomic_query(args.query)
    query_vec = model.encode(prefixed, normalize_embeddings=True)

    sql = _SEARCH_SQL_DEDUPED if args.dedupe else _SEARCH_SQL
    with psycopg.connect(settings.database_url) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(sql, (query_vec, query_vec, args.top))
            rows = cur.fetchall()

    print()
    print(f'Query: "{args.query}"')
    print(f"Top {len(rows)} results (naive dense retrieval, no HyDE / no SQL pre-filter):")
    print("-" * 100)
    for i, (name, type_line, mana_cost, oracle_text, similarity) in enumerate(rows, 1):
        cost = mana_cost or ""
        text = (oracle_text or "").replace("\n", " ")
        if len(text) > 90:
            text = text[:87] + "..."
        print(f"{i:2}. [{similarity:.3f}] {name:28} {cost:12} {type_line}")
        if text:
            print(f"     {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
