"""Embed the cards corpus.

Selects face rows whose ``embedding`` is NULL or whose
``embedding_version`` is stale, runs each through
``build_embedding_text`` to produce the canonical text, encodes the
text with ``sentence-transformers/multi-qa-distilbert-cos-v1``, and
writes back the vector along with the version string and a SHA-256
hash of the exact text that was encoded.

The cards schema enforces a paired invariant on
``(embedding, embedding_version, embedding_text_hash)`` — all three
are NULL together or all three NOT NULL together. This script always
sets all three in one UPDATE, so the constraint is satisfied by
construction.

Idempotent: re-running with the same ``settings.embedding_version``
selects zero rows because the WHERE clause filters them out.

Usage::

    python scripts/embed.py
    python scripts/embed.py --limit 100        # smoke test
    python scripts/embed.py --batch-size 64
    python scripts/embed.py --dry-run          # parse + encode counts, no DB writes
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import settings
from src.logging_utils import PipelineRun
from src.preprocess_text import build_embedding_text, load_keyword_dict
from src.utils.device import select_device

_SELECT_SQL = """
    SELECT oracle_id, face_index, oracle_text, keywords
    FROM cards
    WHERE embedding IS NULL OR embedding_version IS DISTINCT FROM %s
"""

_UPDATE_SQL = """
    UPDATE cards
    SET embedding = %s,
        embedding_version = %s,
        embedding_text_hash = %s,
        updated_at = NOW()
    WHERE oracle_id = %s AND face_index = %s
"""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on rows to embed. Useful for smoke tests before a full run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.embed_batch_size,
        help="Rows per encode/UPDATE batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, build embedding texts, encode — but do not UPDATE the database.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (cuda/mps/cpu). Default: select_device() cascade.",
    )
    args = parser.parse_args()

    version = settings.embedding_version
    device = select_device(prefer=args.device)
    keyword_dict = load_keyword_dict()

    with PipelineRun(
        "embed",
        inputs={
            "model": settings.embedding_model,
            "embedding_version": version,
            "device": str(device),
            "batch_size": args.batch_size,
            "limit": args.limit,
            "dry_run": args.dry_run,
        },
    ) as run:
        # 1. Identify rows to embed.
        with psycopg.connect(settings.database_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(_SELECT_SQL, (version,))
                rows: list[tuple[str, int, str, list[str]]] = cur.fetchall()
        if args.limit:
            rows = rows[: args.limit]
        run.event("rows_selected", count=len(rows))
        if not rows:
            print("\n  Nothing to embed — every row is already at the current version.")
            return 0

        # 2. Build embedding texts (preprocessing happens once, here).
        texts: list[str] = []
        hashes: list[str] = []
        for _oracle_id, _face_index, oracle_text, keywords in rows:
            text = build_embedding_text(oracle_text or "", keywords or [], keyword_dict)
            texts.append(text)
            hashes.append(_hash_text(text))
        run.event("texts_built", count=len(texts))

        # 3. Load the model once.
        load_start = time.perf_counter()
        model = SentenceTransformer(settings.embedding_model, device=str(device))
        run.event(
            "model_loaded",
            elapsed_s=round(time.perf_counter() - load_start, 3),
            model=settings.embedding_model,
        )

        # 4. Encode + UPDATE in batches.
        updated = 0
        n_batches = (len(rows) + args.batch_size - 1) // args.batch_size
        sample_interval = max(1, n_batches // 6)

        if args.dry_run:
            # In dry-run we still encode (to confirm the pipeline runs) but skip DB writes.
            for batch_idx, start in enumerate(
                tqdm(range(0, len(rows), args.batch_size), desc="encode", unit="batch")
            ):
                end = min(start + args.batch_size, len(rows))
                _ = model.encode(
                    texts[start:end],
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                if batch_idx % sample_interval == 0:
                    run.event(
                        "sample_text",
                        oracle_id=str(rows[start][0]),
                        face_index=rows[start][1],
                        text_preview=texts[start][:300],
                    )
            run.note(rows_selected=len(rows), updated=0, dry_run=True)
            print(f"\n  Dry run: {len(rows):,} rows would be embedded. No writes.")
            return 0

        with psycopg.connect(settings.database_url, autocommit=False) as conn:
            register_vector(conn)
            for batch_idx, start in enumerate(
                tqdm(range(0, len(rows), args.batch_size), desc="encode", unit="batch")
            ):
                end = min(start + args.batch_size, len(rows))
                batch_texts = texts[start:end]
                batch_hashes = hashes[start:end]
                batch_rows = rows[start:end]

                vectors = model.encode(
                    batch_texts,
                    batch_size=args.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                with conn.cursor() as cur:
                    for (oracle_id, face_index, _oracle_text, _keywords), vec, h in zip(
                        batch_rows, vectors, batch_hashes, strict=True
                    ):
                        cur.execute(
                            _UPDATE_SQL,
                            (vec, version, h, oracle_id, face_index),
                        )
                conn.commit()
                updated += end - start

                if batch_idx % sample_interval == 0:
                    run.event(
                        "sample_text",
                        oracle_id=str(rows[start][0]),
                        face_index=rows[start][1],
                        text_preview=texts[start][:300],
                    )

        run.processed(updated)
        run.note(
            rows_selected=len(rows),
            updated=updated,
            embedding_version=version,
            device=str(device),
        )

        print()
        print(f"  Rows selected:      {len(rows):,}")
        print(f"  Rows embedded:      {updated:,}")
        print(f"  Embedding version:  {version}")
        print(f"  Device:             {device}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
