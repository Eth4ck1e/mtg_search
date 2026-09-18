"""Ingest the Scryfall ``oracle_tags`` bulk file into ``oracle_tags`` + ``card_tags``.

Streams the gzipped JSON-Lines file (one Tag object per line — see
https://scryfall.com/docs/api/tags), keeps ``type == "oracle"`` tags, and
loads them with a **full replace inside one transaction**: ``card_tags``
and ``oracle_tags`` are truncated, then repopulated. Tags are derived data
with no local edits, so replace-all is the simplest idempotent strategy
and it naturally drops tags/taggings the community has since removed.

After loading, the script measures coverage against the ingested corpus
(how many taggings reference an ``oracle_id`` present in ``cards``) and
records it in the run log — that ratio is what tells us how much of the
tag signal is actually usable as training data.

Usage::

    python scripts/ingest_tags.py
    python scripts/ingest_tags.py --bulk data/raw/oracle-tags-2026-09-18.jsonl.gz
    python scripts/ingest_tags.py --dry-run     # parse + count, no DB writes
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from tqdm import tqdm

from src.config import settings
from src.logging_utils import PipelineRun

_FILENAME_DATE = re.compile(r"oracle-tags-(\d{4}-\d{2}-\d{2})\.jsonl\.gz$")

_INSERT_TAG = """
    INSERT INTO oracle_tags
        (id, slug, label, description, aliases, parent_ids, tagging_count, bulk_updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_TAGGING = """
    INSERT INTO card_tags (tag_id, oracle_id, weight, annotation)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (tag_id, oracle_id) DO NOTHING
"""

_COVERAGE_SQL = """
    SELECT
        COUNT(*)                                        AS taggings_total,
        COUNT(*) FILTER (WHERE c.oracle_id IS NOT NULL) AS taggings_in_corpus,
        COUNT(DISTINCT ct.oracle_id)                    AS tagged_cards_total,
        COUNT(DISTINCT c.oracle_id)                     AS tagged_cards_in_corpus,
        (SELECT COUNT(DISTINCT oracle_id) FROM cards)   AS corpus_cards
    FROM card_tags ct
    LEFT JOIN (SELECT DISTINCT oracle_id FROM cards) c USING (oracle_id)
"""


def _find_latest_bulk(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("oracle-tags-*.jsonl.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No oracle-tags-*.jsonl.gz in {raw_dir}. "
            "Run scripts/download_scryfall.py --dataset oracle-tags first."
        )
    return candidates[-1]


def _bulk_date(bulk_path: Path) -> datetime:
    """Date the bulk file was fetched, from the download script's filename convention.

    The Tag objects themselves carry no timestamp; the exact Scryfall
    ``updated_at`` lives in the download run log. The fetch date is
    sufficient for citation purposes (the file is regenerated daily).
    """
    m = _FILENAME_DATE.search(bulk_path.name)
    if not m:
        raise ValueError(
            f"Cannot infer bulk date from {bulk_path.name}; expected oracle-tags-YYYY-MM-DD.jsonl.gz"
        )
    return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=UTC)


def _tag_params(tag: dict[str, Any], bulk_date: datetime) -> tuple[Any, ...]:
    return (
        tag["id"],
        tag["slug"],
        tag["label"],
        tag.get("description") or None,
        list(tag.get("aliases") or []),
        list(tag.get("parent_ids") or []),
        len(tag.get("taggings") or []),
        bulk_date,
    )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bulk",
        type=Path,
        default=None,
        help="Path to oracle-tags-*.jsonl.gz. Defaults to newest under data/raw/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count without touching the database.",
    )
    args = parser.parse_args()

    bulk_path = args.bulk or _find_latest_bulk(settings.raw_data_dir)
    bulk_date = _bulk_date(bulk_path)

    with PipelineRun(
        "ingest_tags",
        inputs={"bulk_path": str(bulk_path), "dry_run": args.dry_run},
    ) as run:
        tags_loaded = taggings_loaded = 0
        tags_without_taggings = 0

        # ---- Parse (in memory: ~4.5k tags / ~240k taggings, ~6 MB gzipped) ----
        tags: list[dict[str, Any]] = []
        with gzip.open(bulk_path, "rt", encoding="utf-8") as fh:
            for line in tqdm(fh, desc="parse", unit="tag"):
                tag = json.loads(line)
                if tag.get("object") != "tag":
                    run.skip("not_tag_object")
                    continue
                if tag.get("type") != "oracle":
                    run.skip(f"type_{tag.get('type')}")
                    continue
                run.processed()
                tags.append(tag)
                if not tag.get("taggings"):
                    tags_without_taggings += 1

        # Referential integrity for parent_ids is not enforced by the schema
        # (parents load in arbitrary order); verify it here instead.
        ids = {t["id"] for t in tags}
        dangling = sum(1 for t in tags for p in t.get("parent_ids") or [] if p not in ids)

        coverage: dict[str, Any] = {}
        if not args.dry_run:
            with psycopg.connect(settings.database_url, autocommit=False) as conn:
                with conn.cursor() as cur:
                    # card_tags has ON DELETE CASCADE, but be explicit.
                    cur.execute("TRUNCATE card_tags, oracle_tags")
                    cur.executemany(_INSERT_TAG, [_tag_params(t, bulk_date) for t in tags])
                    tags_loaded = len(tags)
                    tagging_rows = [
                        (
                            t["id"],
                            tg["oracle_id"],
                            tg.get("weight") or "median",
                            tg.get("annotation") or None,
                        )
                        for t in tqdm(tags, desc="taggings", unit="tag")
                        for tg in t.get("taggings") or []
                    ]
                    cur.executemany(_INSERT_TAGGING, tagging_rows)
                    cur.execute("SELECT COUNT(*) FROM card_tags")
                    taggings_loaded = cur.fetchone()[0]
                    cur.execute(_COVERAGE_SQL)
                    row = cur.fetchone()
                    coverage = dict(
                        zip(
                            (
                                "taggings_total",
                                "taggings_in_corpus",
                                "tagged_cards_total",
                                "tagged_cards_in_corpus",
                                "corpus_cards",
                            ),
                            row,
                            strict=True,
                        )
                    )
                conn.commit()

        run.note(
            bulk_path=str(bulk_path),
            bulk_date=bulk_date.date().isoformat(),
            tags_parsed=len(tags),
            tags_without_taggings=tags_without_taggings,
            dangling_parent_refs=dangling,
            tags_loaded=tags_loaded,
            taggings_loaded=taggings_loaded,
            **coverage,
        )

        print()
        print(f"  Bulk file:               {bulk_path.name}")
        print(f"  Oracle tags parsed:      {len(tags):,}")
        print(f"  Tags with no taggings:   {tags_without_taggings:,}")
        print(f"  Dangling parent refs:    {dangling:,}")
        print(f"  Skipped per rule:        {run.skipped_counts}")
        if args.dry_run:
            print("  Dry run — no writes.")
            return 0
        print(f"  Tags loaded:             {tags_loaded:,}")
        print(f"  Taggings loaded:         {taggings_loaded:,}")
        if coverage:
            in_corpus = coverage["taggings_in_corpus"]
            total = coverage["taggings_total"]
            cards_in = coverage["tagged_cards_in_corpus"]
            corpus = coverage["corpus_cards"]
            print(f"  Taggings in corpus:      {in_corpus:,} / {total:,} ({in_corpus / total:.1%})")
            print(
                f"  Corpus cards with ≥1 tag: {cards_in:,} / {corpus:,} ({cards_in / corpus:.1%})"
            )
        return 0


if __name__ == "__main__":
    sys.exit(main())
