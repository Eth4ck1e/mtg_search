"""Search the paper's source PDFs, or verify a quotation against them.

    python scripts/source_search.py "why do dense retrievers fail on specialised vocabulary"
    python scripts/source_search.py "hard negatives false negatives" --exact --source moreira2024nvretriever
    python scripts/source_search.py --verify "Training for multiple epochs hurts performance" --source nussbaum2024nomic
    python scripts/source_search.py --page nussbaum2024nomic 5     # one page's text, original case
    python scripts/source_search.py --list            # indexed sources and their BibTeX keys

Every result carries the BibTeX key (docs/thesis/references.bib), the PDF file
and the page. ``--verify`` exits 0 only on a verbatim match, so it can gate a
quotation: a paraphrase, or a quote with a changed word, fails.
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import psycopg

import src.utils.quiet  # noqa: F401
from scripts.build_references import load_entries
from src.config import settings
from src.source_library import page_text, search, verify_quote


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("query", nargs="?", help="Question (semantic) or words (--exact).")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--source", help="Restrict to one BibTeX key.")
    parser.add_argument(
        "--exact", action="store_true", help="Full-text word match instead of semantic."
    )
    parser.add_argument("--verify", metavar="QUOTE", help="Check a quotation appears verbatim.")
    parser.add_argument(
        "--page", nargs=2, metavar=("KEY", "N"), help="Print one page of a source as extracted."
    )
    parser.add_argument("--list", action="store_true", help="List indexed sources.")
    args = parser.parse_args()

    if args.list:
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT source_key, file, COUNT(*) FROM source_pages GROUP BY 1, 2 ORDER BY 1"
            )
            rows = cur.fetchall()
        for key, file, pages in rows:
            print(f"  {key:34s} {pages:3d} pages   {file}")
        missing = sorted({e["key"] for e in load_entries()[0]} - {r[0] for r in rows})
        if missing:
            print(
                "\n  In references.bib but NO PDF indexed (cannot be searched or quoted from here):"
            )
            for key in missing:
                print(f"    {key}")
        return 0

    if args.page:
        text = page_text(args.page[0], int(args.page[1]))
        if text is None:
            print(f"  No page {args.page[1]} indexed for {args.page[0]!r}.")
            return 1
        print(text)
        return 0

    if args.verify:
        r = verify_quote(args.verify, source_key=args.source)
        if r.found:
            print(f"  VERIFIED  {r.source_key}, p. {r.page}  ({r.file})  — {r.note}")
            print(
                textwrap.fill(
                    f"…{r.context}…", width=110, initial_indent="    ", subsequent_indent="    "
                )
            )
            return 0
        print(f"  {r.note}")
        return 1

    if not args.query:
        parser.error("give a query, --verify QUOTE, or --list")
    for i, p in enumerate(
        search(args.query, k=args.k, source_key=args.source, exact=args.exact), 1
    ):
        print(f"\n  [{i}] {p.source_key}, p. {p.page}   score {p.score:.3f}   ({p.file})")
        print(textwrap.fill(p.text, width=110, initial_indent="      ", subsequent_indent="      "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
