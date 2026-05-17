"""Build the reminder-text dictionary used by the embedding pipeline.

Scans the ``cards`` table for keyword-parenthetical patterns and picks
the most recently printed canonical wording per keyword. The output
JSON is the dictionary that ``src/preprocess_text.py:build_embedding_text``
will consume in Phase 3.

Pipeline:
    1. SELECT (oracle_id, name, keywords, oracle_text, released_at) for every
       face row whose keywords list is non-empty.
    2. For each (face row, keyword) pair, call
       ``extract_reminder_texts`` from src/data_processing/keyword_extract.py.
       The function returns 0 or 1+ reminder strings.
    3. Aggregate candidates per keyword across all face rows. Tiebreak by
       released_at DESC — the most recent canonical printing wins.
    4. Keywords that appear in any card's keywords list but never produce
       a reminder-text candidate are reported as "no_reminder" — these
       are manual-override territory.
    5. Write reminder_text.json (consumer format) and create
       manual_overrides.json if it doesn't already exist.

Usage::

    python scripts/build_keyword_dict.py
    python scripts/build_keyword_dict.py --version v2
    python scripts/build_keyword_dict.py --out-dir /tmp
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import psycopg

from src.config import settings
from src.data_processing.keyword_extract import extract_reminder_texts
from src.logging_utils import PipelineRun

_SELECT_SQL = """
    SELECT oracle_id, name, keywords, oracle_text, released_at
    FROM cards
    WHERE keywords <> '{}'
"""


def _iter_cards(
    conn: psycopg.Connection,
) -> Iterator[tuple[str, str, list[str], str, date | None]]:
    with conn.cursor() as cur:
        cur.execute(_SELECT_SQL)
        yield from cur


def _build_dict(
    cards: Iterator[tuple[str, str, list[str], str, date | None]],
    *,
    run: PipelineRun,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Aggregate candidates, pick newest-wins, return (reminder_dict, audit).

    audit shape:
        {
            "keywords_seen": int,
            "keywords_with_reminder": int,
            "keywords_without_reminder": list[str],
            "keywords_with_conflicts": dict[str, {winning_card, winning_released_at, distinct_reminder_count}],
        }
    """
    # candidates[keyword] -> list of (released_at, reminder_text, oracle_id, source_card_name)
    candidates: dict[str, list[tuple[date | None, str, str, str]]] = defaultdict(list)
    all_keywords_seen: set[str] = set()

    for oracle_id, name, keywords, oracle_text, released_at in cards:
        for kw in keywords:
            all_keywords_seen.add(kw)
            for reminder in extract_reminder_texts(oracle_text or "", kw):
                candidates[kw].append((released_at, reminder, oracle_id, name))

    reminder_dict: dict[str, str] = {}
    conflicts: dict[str, dict[str, Any]] = {}
    epoch = date(1900, 1, 1)

    for kw, cands in candidates.items():
        cands.sort(key=lambda c: c[0] or epoch, reverse=True)
        winner = cands[0]
        reminder_dict[kw] = winner[1]
        distinct_reminders = {c[1].strip() for c in cands}
        if len(distinct_reminders) > 1:
            conflicts[kw] = {
                "winning_card": winner[3],
                "winning_released_at": str(winner[0]) if winner[0] else None,
                "distinct_reminder_count": len(distinct_reminders),
                "total_candidates": len(cands),
            }

    no_reminder = sorted(all_keywords_seen - reminder_dict.keys())

    run.note(
        keywords_seen=len(all_keywords_seen),
        keywords_with_reminder=len(reminder_dict),
        keywords_without_reminder_count=len(no_reminder),
        keywords_without_reminder=no_reminder,
        keywords_with_conflicts_count=len(conflicts),
        keywords_with_conflicts=conflicts,
    )

    audit = {
        "keywords_seen": len(all_keywords_seen),
        "keywords_with_reminder": len(reminder_dict),
        "keywords_without_reminder": no_reminder,
        "keywords_with_conflicts": conflicts,
    }
    return reminder_dict, audit


def _write_outputs(
    out_dir: Path,
    reminder_dict: dict[str, str],
    *,
    version: str,
    card_count: int,
) -> tuple[Path, Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reminder_path = out_dir / "reminder_text.json"
    overrides_path = out_dir / "manual_overrides.json"

    payload = {
        "version": version,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "card_count_scanned": card_count,
        "keywords": dict(sorted(reminder_dict.items())),
    }
    reminder_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    overrides_created = False
    if not overrides_path.exists():
        overrides_path.write_text(
            json.dumps(
                {
                    "version": version,
                    "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "keywords": {},
                    "_note": (
                        "Hand-maintained overrides for keywords that never received "
                        "reminder text in any printing. Keys must match Scryfall's "
                        "canonical keyword capitalisation."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        overrides_created = True

    return reminder_path, overrides_path, overrides_created


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="v1",
        help="Version string embedded in the dict's metadata. Bump on logic changes.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=settings.keywords_dir,
        help="Where reminder_text.json and manual_overrides.json land (default: data/keywords/).",
    )
    args = parser.parse_args()

    with PipelineRun(
        "build_keyword_dict",
        inputs={"version": args.version, "out_dir": str(args.out_dir)},
    ) as run:
        with psycopg.connect(settings.database_url) as conn:
            card_iter = _iter_cards(conn)
            # Materialise — generator is consumed once, and we want a count.
            cards = list(card_iter)

        reminder_dict, audit = _build_dict(iter(cards), run=run)

        reminder_path, overrides_path, overrides_created = _write_outputs(
            args.out_dir,
            reminder_dict,
            version=args.version,
            card_count=len(cards),
        )
        run.note(
            reminder_path=str(reminder_path),
            overrides_path=str(overrides_path),
            overrides_created=overrides_created,
        )
        run.processed(len(reminder_dict))

        print()
        print(f"  Cards scanned:                 {len(cards):,}")
        print(f"  Distinct keywords seen:        {audit['keywords_seen']:,}")
        print(f"  Keywords with reminder text:   {audit['keywords_with_reminder']:,}")
        print(f"  Keywords without reminder:     {len(audit['keywords_without_reminder']):,}")
        print(f"  Keywords with conflicts:       {len(audit['keywords_with_conflicts']):,}")
        print()
        print(f"  Wrote {reminder_path}")
        if overrides_created:
            print(f"  Created {overrides_path} (empty — hand-edit to add overrides)")
        else:
            print(f"  Preserved {overrides_path}")
        if audit["keywords_without_reminder"]:
            print()
            print("  Keywords needing manual override:")
            for kw in audit["keywords_without_reminder"]:
                print(f"    - {kw}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
