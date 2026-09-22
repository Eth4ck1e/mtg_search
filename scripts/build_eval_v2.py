"""Fold dashboard judgments into a new eval-set version (annotation-hole repair).

Reads ``data/eval/judgments_pending.jsonl`` (appended by the dashboard's
judgment buttons), keeps the LATEST label per (eval query, card), and writes a
new eval YAML in which each query's ``relevant`` / ``borderline`` lists are the
union of the previous version's judgments and the new ones (``not_relevant``
removes a card from both). Cards already judged in the base set are never
downgraded silently: conflicts are printed for review.

The base YAML is not modified. Judge in the dashboard, run this, review the
diff, then point configs at the new file.

Usage::

    python scripts/build_eval_v2.py                       # -> data/eval/queries_v2.yaml
    python scripts/build_eval_v2.py --out data/eval/queries_v2_draft.yaml --version v2-draft
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import psycopg
import yaml

from src.config import settings

JUDGMENTS = settings.eval_dir / "judgments_pending.jsonl"
BASE = settings.eval_dir / "queries_v1_draft.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--out", type=Path, default=settings.eval_dir / "queries_v2.yaml")
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()

    if not JUDGMENTS.exists():
        print(f"No judgments yet at {JUDGMENTS}; judge some cards in the dashboard first.")
        return 1
    latest: dict[tuple[str, str], dict] = {}
    for line in JUDGMENTS.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("eval_id"):
            latest[(r["eval_id"], r["oracle_id"])] = r  # file order == time order
    names: dict[str, str] = {}
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT ON (oracle_id) oracle_id::text, name FROM cards ORDER BY oracle_id, face_index"
        )
        names = dict(cur.fetchall())

    ev = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    stats: Counter[str] = Counter()
    conflicts: list[str] = []
    for q in ev["queries"]:
        rel = {r["id"]: r for r in q.get("relevant") or []}
        bord = {r["id"]: r for r in q.get("borderline") or []}
        for (qid, oid), r in latest.items():
            if qid != q["id"]:
                continue
            label = r["label"]
            was = "relevant" if oid in rel else "borderline" if oid in bord else None
            if was and was != label:
                conflicts.append(
                    f"{qid} {names.get(oid, oid)}: base={was} dashboard={label} (dashboard wins)"
                )
            rel.pop(oid, None)
            bord.pop(oid, None)
            entry = {"id": oid, "why": f"{names.get(oid, '?')} — dashboard judgment {r['ts'][:10]}"}
            if label == "relevant":
                rel[oid] = entry
            elif label == "borderline":
                bord[oid] = entry
            stats[f"{label}{' (changed)' if was else ' (new)'}"] += 1
        q["relevant"] = list(rel.values())
        q["borderline"] = list(bord.values())
    ev["version"] = args.version
    ev["derived_from"] = {
        "base": str(args.base),
        "judgments": str(JUDGMENTS),
        "n_judgments": len(latest),
    }
    args.out.write_text(
        yaml.safe_dump(ev, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8"
    )
    print(f"  Judgments applied: {dict(stats)}")
    if conflicts:
        print("  Conflicts with the base set (dashboard label kept):")
        for c in conflicts:
            print(f"    {c}")
    print(f"  Wrote {args.out}  (version={args.version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
