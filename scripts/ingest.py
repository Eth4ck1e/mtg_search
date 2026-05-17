"""Ingest the Scryfall oracle-cards bulk into the ``cards`` table.

Streams the bulk JSON, applies the four Phase 1 filter rules
(see [[2026-05-17-corpus-survey]]), explodes multi-face records into
face rows, and UPSERTs them into ``cards``. Idempotent: a row is
updated only when at least one mutable column actually differs
(``IS DISTINCT FROM`` on every non-key column inside the
``ON CONFLICT DO UPDATE WHERE`` clause). ``RETURNING (xmax = 0)`` is
how we distinguish inserts from updates in the result; rows the WHERE
filters out are counted as unchanged.

PipelineRun captures per-rule skip counters that should match the
corpus survey within ±epsilon. The whole ingest runs in one
transaction — a mid-flight crash leaves the schema in its prior state
and a rerun catches up cheaply.

Usage::

    python scripts/ingest.py
    python scripts/ingest.py --bulk data/raw/oracle-cards-2026-05-17.json
    python scripts/ingest.py --dry-run     # parse + filter, no DB writes
"""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from functools import partial
from pathlib import Path
from typing import Any

import ijson
import psycopg
from psycopg.types.json import Jsonb
from tqdm import tqdm

from src.config import settings
from src.data_processing.ingest_transform import iter_face_rows, should_include
from src.logging_utils import PipelineRun

# Column order for the prepared INSERT. Must align with _build_upsert_sql below
# and with the row dicts produced by iter_face_rows.
_COLUMNS = (
    "oracle_id",
    "face_index",
    "name",
    "mana_cost",
    "colors",
    "type_line",
    "oracle_text",
    "power",
    "toughness",
    "loyalty",
    "cmc",
    "color_identity",
    "keywords",
    "layout",
    "released_at",
    "legalities",
    "raw",
)

# Columns compared by IS DISTINCT FROM in the ON CONFLICT WHERE clause.
# Anything not in this set (the PK plus audit/embedding columns) is not part
# of "did the row change."
_CHANGE_COLUMNS = tuple(c for c in _COLUMNS if c not in ("oracle_id", "face_index"))

# JSONB columns — wrapped in Jsonb() before binding.
_JSONB_COLUMNS = frozenset({"legalities", "raw"})


def _json_default(obj: Any) -> Any:
    """json.dumps fallback for ijson-produced Decimal values inside ``raw``."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Unserializable type for JSONB: {type(obj).__name__}")


_dumps_with_decimal = partial(json.dumps, default=_json_default)


def _build_upsert_sql() -> str:
    cols = ", ".join(_COLUMNS)
    placeholders = ", ".join(["%s"] * len(_COLUMNS))
    set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in _CHANGE_COLUMNS)
    where_clause = " OR ".join(f"cards.{c} IS DISTINCT FROM EXCLUDED.{c}" for c in _CHANGE_COLUMNS)
    return f"""
        INSERT INTO cards ({cols})
        VALUES ({placeholders})
        ON CONFLICT (oracle_id, face_index) DO UPDATE SET
            {set_clause},
            updated_at = NOW()
        WHERE {where_clause}
        RETURNING (xmax = 0) AS inserted
    """


def _row_to_params(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        Jsonb(row[c], dumps=_dumps_with_decimal) if c in _JSONB_COLUMNS else row[c]
        for c in _COLUMNS
    )


def _find_latest_bulk(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("oracle-cards-*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No oracle-cards-*.json in {raw_dir}. Run scripts/download_scryfall.py first."
        )
    return candidates[-1]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bulk",
        type=Path,
        default=None,
        help="Path to oracle-cards-*.json. Defaults to newest under data/raw/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and filter without touching the database.",
    )
    args = parser.parse_args()

    bulk_path = args.bulk or _find_latest_bulk(settings.raw_data_dir)
    sql = _build_upsert_sql()

    with PipelineRun(
        "ingest",
        inputs={"bulk_path": str(bulk_path), "dry_run": args.dry_run},
    ) as run:
        inserted = updated = unchanged = 0
        face_rows = 0

        conn_ctx: Any = (
            _NullCM() if args.dry_run else psycopg.connect(settings.database_url, autocommit=False)
        )

        with conn_ctx as conn:
            cur = None if args.dry_run else conn.cursor()
            with bulk_path.open("rb") as fh:
                stream = tqdm(ijson.items(fh, "item"), desc="ingest", unit="card")
                for card in stream:
                    reason = should_include(card)
                    if reason is not None:
                        run.skip(reason)
                        continue
                    run.processed()
                    for _face_index, row in iter_face_rows(card):
                        face_rows += 1
                        if args.dry_run:
                            continue
                        cur.execute(sql, _row_to_params(row))
                        result = cur.fetchone()
                        if result is None:
                            unchanged += 1
                        elif result[0]:
                            inserted += 1
                        else:
                            updated += 1
            if not args.dry_run:
                conn.commit()

        run.note(
            bulk_path=str(bulk_path),
            face_rows_total=face_rows,
            inserted=inserted,
            updated=updated,
            unchanged=unchanged,
        )

        print()
        print(f"  Cards processed:    {run.processed_count:,}")
        print(f"  Face rows produced: {face_rows:,}")
        if args.dry_run:
            print("  Dry run — no writes.")
        else:
            print(f"  Inserted:           {inserted:,}")
            print(f"  Updated:            {updated:,}")
            print(f"  Unchanged:          {unchanged:,}")
        print(f"  Skipped per rule:   {run.skipped_counts}")
        return 0


class _NullCM:
    """Context manager that yields ``None`` — used to skip DB connection in dry-run."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: Any) -> None:
        return None


if __name__ == "__main__":
    sys.exit(main())
