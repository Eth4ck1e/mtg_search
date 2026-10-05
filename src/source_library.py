"""Source library — find passages in the paper's source PDFs and verify quotations.

The rule this module exists to enforce: **nothing is attributed to a source
unless it was retrieved from that source's text.** Two operations:

* :func:`search` — passages relevant to a question (semantic, via pgvector) or
  containing given words (full-text), each with its BibTeX key and page.
* :func:`verify_quote` — does this exact wording appear in that source, and on
  which page? Deterministic string matching after normalising the things PDF
  extraction mangles (line breaks, hyphenation at line ends, ligatures,
  curly quotes). A paraphrase does not verify. Neither does a quote with a
  word changed.

Index built by ``scripts/index_sources.py``; CLI in ``scripts/source_search.py``.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from src.config import settings

SOURCE_EMBEDDER = "nomic-ai/nomic-embed-text-v1.5"  # stock model: academic prose, not card text
# PDF typography folded to ASCII before comparison (code points, so the source stays unambiguous).
_LIGATURES = {
    0xFB01: "fi",
    0xFB02: "fl",
    0xFB00: "ff",
    0xFB03: "ffi",
    0xFB04: "ffl",
    0x2019: "'",  # right single quote
    0x2018: "'",  # left single quote
    0x201C: '"',
    0x201D: '"',
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2212: "-",  # minus sign
    0x00A0: " ",  # no-break space
}


def normalize(text: str) -> str:
    """Canonical form for verbatim comparison: what survives PDF extraction.

    Joins words hyphenated across line ends, folds ligatures and typographic
    quotes/dashes, collapses all whitespace, and lower-cases. Punctuation and
    word order are preserved — a changed word still fails to match.
    """
    text = unicodedata.normalize("NFKC", text).translate(_LIGATURES)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # hyphen-ation at a line end
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Passage:
    source_key: str
    file: str
    page: int
    text: str
    score: float


@dataclass
class QuoteCheck:
    found: bool
    source_key: str | None
    file: str | None
    page: int | None
    context: str | None  # the surrounding source text, as extracted
    note: str


def _connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def load_embedder() -> Any:
    """The stock Nomic model (its loader prints a status line; silenced so CLI output stays clean)."""
    import contextlib
    import io

    from sentence_transformers import SentenceTransformer

    from src.utils.device import select_device

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return SentenceTransformer(
            SOURCE_EMBEDDER, device=str(select_device()), trust_remote_code=True
        )


def page_text(source_key: str, page: int) -> str | None:
    """The extracted text of one page, exactly as stored (original case and line breaks)."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT text FROM source_pages WHERE source_key = %s AND page = %s", (source_key, page)
        )
        row = cur.fetchone()
    return row[0] if row else None


def search(
    query: str,
    *,
    k: int = 5,
    source_key: str | None = None,
    exact: bool = False,
    model: Any = None,
) -> list[Passage]:
    """Top-k passages. ``exact=True`` uses full-text matching on the words given
    (ranked by ts_rank); otherwise cosine similarity on the embedded question."""
    where, params = ["TRUE"], {"k": k}
    if source_key:
        where.append("source_key = %(key)s")
        params["key"] = source_key
    with _connect() as conn, conn.cursor() as cur:
        if exact:
            params["q"] = query
            cur.execute(
                f"""SELECT source_key, file, page, text,
                           ts_rank(text_tsv, websearch_to_tsquery('english', %(q)s)) AS score
                    FROM source_chunks
                    WHERE {" AND ".join(where)} AND text_tsv @@ websearch_to_tsquery('english', %(q)s)
                    ORDER BY score DESC LIMIT %(k)s""",
                params,
            )
        else:
            if model is None:
                model = load_embedder()
            params["vec"] = model.encode([f"search_query: {query}"], normalize_embeddings=True)[0]
            cur.execute(
                f"""SELECT source_key, file, page, text, 1 - (embedding <=> %(vec)s) AS score
                    FROM source_chunks WHERE {" AND ".join(where)}
                    ORDER BY embedding <=> %(vec)s LIMIT %(k)s""",
                params,
            )
        return [Passage(*row) for row in cur.fetchall()]


def verify_quote(quote: str, *, source_key: str | None = None) -> QuoteCheck:
    """Is ``quote`` verbatim in a source (optionally a specific one)? Reports the page."""
    with _connect() as conn, conn.cursor() as cur:
        if source_key:
            cur.execute(
                "SELECT file, page, source_key, text FROM source_pages WHERE source_key = %s ORDER BY file, page",
                (source_key,),
            )
        else:
            cur.execute("SELECT file, page, source_key, text FROM source_pages ORDER BY file, page")
        return match_quote(quote, cur.fetchall(), source_key=source_key)


def match_quote(
    quote: str, rows: list[tuple[str, int, str, str]], *, source_key: str | None = None
) -> QuoteCheck:
    """Find ``quote`` in ``rows`` of (file, page, source_key, page text), pages in order.

    Matching is on :func:`normalize`-d text over each file's pages joined in
    order, so a quotation that runs across a page break still verifies (the
    reported page is where it starts).
    """
    needle = normalize(quote)
    if len(needle) < 12:
        return QuoteCheck(
            False,
            None,
            None,
            None,
            None,
            "Quote too short to verify meaningfully (< 12 characters).",
        )
    if not rows:
        return QuoteCheck(
            False, source_key, None, None, None, f"No indexed text for source {source_key!r}."
        )
    by_file: dict[str, list[tuple[int, str, str]]] = {}
    for file, page, key, text in rows:
        by_file.setdefault(file, []).append((page, key, text))
    # Pass 1 is strict. Pass 2 ignores hyphens only: PDF extraction sometimes drops or keeps a
    # hyphen where a compound breaks across lines, which is typesetting, not wording.
    for loose in (False, True):
        target = needle.replace("-", "") if loose else needle
        for file, pages in by_file.items():
            joined, starts = "", []
            for page, _key, text in pages:
                starts.append((len(joined), page))
                norm = normalize(text)
                joined += (norm.replace("-", "") if loose else norm) + " "
            at = joined.find(target)
            if at >= 0:
                page = max(p for off, p in starts if off <= at)
                ctx = joined[max(0, at - 160) : at + len(target) + 160]
                note = (
                    "Match ignoring hyphenation only — copy the hyphens from the PDF page before quoting."
                    if loose
                    else "Verbatim match (after whitespace/line-break/ligature normalisation)."
                )
                return QuoteCheck(True, pages[0][1], file, page, ctx, note)
    scope = f"source {source_key!r}" if source_key else "any indexed source"
    return QuoteCheck(
        False,
        source_key,
        None,
        None,
        None,
        f"NOT FOUND verbatim in {scope}. Do not use this wording as a quotation.",
    )
