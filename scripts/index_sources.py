"""Index the source PDFs in ``docs/sources/`` into the source library.

For each PDF: extract text page by page (``pdftotext``, reading order, so
two-column papers come out column by column), store every page verbatim in
``source_pages``, split pages into paragraph-sized passages, embed them with
the stock Nomic model, and store them in ``source_chunks``. Full replace.

Each file is mapped to its BibTeX key (``docs/thesis/references.bib``) through
the arXiv id recorded for it in ``docs/sources/README.md``; a file with no
resolvable key is skipped and reported, so nothing unciteable is searchable.

Usage::

    python scripts/index_sources.py
    python scripts/index_sources.py --dry-run     # extract + chunk, no DB writes
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

import src.utils.quiet  # noqa: F401
from scripts.build_references import load_entries
from src.config import settings
from src.logging_utils import PipelineRun
from src.source_library import SOURCE_EMBEDDER, load_embedder

SOURCES = settings.repo_root / "docs" / "sources"
TARGET_WORDS, MAX_WORDS = 160, 260
# Files whose key cannot come from an arXiv id.
MANUAL_KEYS = {
    "2008_manning-raghavan-schutze_ir-book-ch08-evaluation.pdf": "manning2008introduction"
}


def file_to_key() -> dict[str, str]:
    """PDF filename -> BibTeX key, via the arXiv id listed beside it in the README."""
    readme = (SOURCES / "README.md").read_text(encoding="utf-8")
    by_arxiv = {e["arxiv"]: e["key"] for e in load_entries()[0] if e.get("arxiv")}
    out = dict(MANUAL_KEYS)
    for m in re.finditer(
        r"\*\*File:\*\* `([^`]+\.pdf)`\s*\n- \*\*arXiv:\*\* \[([0-9.]+)\]", readme
    ):
        if m.group(2) in by_arxiv:
            out[m.group(1)] = by_arxiv[m.group(2)]
    return out


def extract_pages(pdf: Path) -> list[str]:
    raw = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf), "-"], capture_output=True, check=True
    ).stdout
    pages = raw.decode("utf-8", errors="replace").split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


def chunk_page(text: str) -> list[str]:
    """Paragraph-ish passages of ~TARGET_WORDS words, never above MAX_WORDS."""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for para in paras:
        words = para.split()
        while len(words) > MAX_WORDS:  # a wall of text with no blank lines
            chunks.append(" ".join(words[:MAX_WORDS]))
            words = words[MAX_WORDS:]
        if n + len(words) > MAX_WORDS and cur:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(" ".join(words))
        n += len(words)
        if n >= TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= 8]  # drop page furniture


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = file_to_key()
    pdfs = sorted(SOURCES.glob("*.pdf"))
    version = f"{SOURCE_EMBEDDER}|chunk=para{TARGET_WORDS}"
    with PipelineRun(
        "index_sources",
        inputs={"pdfs": len(pdfs), "embedding_version": version, "dry_run": args.dry_run},
    ) as run:
        pages_rows, chunk_rows = [], []
        for pdf in pdfs:
            key = keys.get(pdf.name)
            if not key:
                run.skip("no_bibtex_key")
                print(f"  SKIP (no BibTeX key): {pdf.name}")
                continue
            pages = extract_pages(pdf)
            n_chunks = 0
            for pno, text in enumerate(pages, 1):
                if not text.strip():
                    continue
                pages_rows.append((pdf.name, pno, key, text))
                for ci, chunk in enumerate(chunk_page(text)):
                    chunk_rows.append((pdf.name, pno, ci, key, chunk))
                    n_chunks += 1
            run.processed()
            print(f"  {key:34s} {len(pages):3d} pages  {n_chunks:4d} passages   {pdf.name}")
        run.note(pages=len(pages_rows), passages=len(chunk_rows))
        print(
            f"\n  {len(pages_rows)} pages, {len(chunk_rows)} passages from {run.processed_count} sources"
        )
        if args.dry_run:
            print("  Dry run — nothing written.")
            return 0

        model = load_embedder()
        vectors = model.encode(
            [f"search_document: {c[4]}" for c in chunk_rows],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        with psycopg.connect(settings.database_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("TRUNCATE source_chunks, source_pages")
                cur.executemany(
                    "INSERT INTO source_pages (file, page, source_key, text) VALUES (%s, %s, %s, %s)",
                    pages_rows,
                )
                cur.executemany(
                    "INSERT INTO source_chunks (file, page, chunk_index, source_key, text, embedding, embedding_version) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [(*c, v, version) for c, v in zip(chunk_rows, vectors, strict=True)],
                )
            conn.commit()
        print("  Indexed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
