"""Scryfall comparator — result parity and query complexity vs expert Scryfall queries.

The paper's primary claim (CLAUDE.md §1): plain language through the cascade
reaches the result set an expert gets from Scryfall's structured syntax. This
script measures that in two parts:

1. **Fetch** each expert query in ``data/eval/scryfall_expert_queries_v1.yaml``
   (variants ``expert`` = otag allowed, ``no_tag`` = Oracle text/attributes
   only) through ``GET api.scryfall.com/cards/search``, paging every page,
   and cache the resulting oracle_ids to ``data/eval/scryfall_results_<version>.json``
   (gitignored — ~2 MB; a small ``.manifest.json`` with queries, counts and
   fetch dates is committed, and ``fetch`` reproduces the cache in ~5 min).
   Rate limit: Scryfall's hard limit on /cards/search is 2 req/s; this script
   sends **1 req/s** with the required User-Agent/Accept headers and stops on
   any 429. A missing tag or empty result is a 404 — recorded as an empty set.
2. **Compare** a logged ``experiment_runs`` row (one produced with
   ``set_metrics: true``, which stores the top-2000 ranking per query) against
   each Scryfall set restricted to our corpus:

   * result parity — R-precision of the cascade's ranking with the Scryfall
     set as target (fraction of the first |S| results that are in S), plus
     recall@100/@500 of S and depth to 90 % of S;
   * Scryfall's own quality — the expert set's precision/recall against the
     eval judgments and the tag pool, so "parity" is read against a
     comparator that is itself imperfect;
   * query complexity — the syntax a user had to type: operator count,
     distinct operator kinds, boolean/negation/grouping count, quoted phrases,
     whether ``otag:`` was needed, versus the word count of the plain query.

   One ``experiment_runs`` row of kind ``scryfall_comparator`` per run.

Usage::

    python scripts/scryfall_comparator.py fetch                 # populate the cache (1 req/s, ~3 min)
    python scripts/scryfall_comparator.py compare --row 61      # score a logged cascade row
    python scripts/scryfall_comparator.py compare --row 61 --variant no_tag
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
import requests
import yaml

from src.config import settings
from src.db.experiment_log import log_experiment
from src.eval.metrics import aggregate_set_metrics, compute_set_metrics
from src.logging_utils import PipelineRun

SEARCH_URL = "https://api.scryfall.com/cards/search"
REQUEST_INTERVAL_S = 1.0  # Scryfall hard limit is 0.5 s on /cards/search; stay well under
HTTP_TIMEOUT_S = 30
EXPERT_QUERIES = settings.eval_dir / "scryfall_expert_queries_v1.yaml"
EVAL_SET = settings.eval_dir / "queries_v1_draft.yaml"
ORACLE = settings.eval_dir / "tag_label_oracle_v1.yaml"

# Scryfall syntax pieces, for the complexity proxy.
_OPERATOR = re.compile(r"(?<![\w\"])(-?)([a-z]+)(:|=|!=|<=|>=|<|>)", re.I)
_QUOTED = re.compile(r'"[^"]*"')
_BOOL = re.compile(r"\b(or|and)\b|[()]", re.I)


# ---- fetch --------------------------------------------------------------


def _headers() -> dict[str, str]:
    return {
        "User-Agent": settings.scryfall_user_agent,
        "Accept": "application/json;q=0.9,*/*;q=0.8",
    }


def fetch_query(q: str, *, log: PipelineRun) -> tuple[list[str], int, int]:
    """All oracle_ids for a Scryfall search (deduped), plus total_cards and pages fetched."""
    ids: list[str] = []
    seen: set[str] = set()
    url: str | None = SEARCH_URL
    params: dict[str, Any] | None = {"q": q, "unique": "cards", "order": "name"}
    total = pages = 0
    while url:
        time.sleep(REQUEST_INTERVAL_S)
        resp = requests.get(url, params=params, headers=_headers(), timeout=HTTP_TIMEOUT_S)
        pages += 1
        if resp.status_code == 404:  # no matches (or unknown tag) — Scryfall's "empty" answer
            log.event("empty", query=q)
            return [], 0, pages
        if resp.status_code == 429:
            raise RuntimeError(f"Scryfall rate-limited us (429) on {q!r}; stop and wait.")
        resp.raise_for_status()
        body = resp.json()
        total = body.get("total_cards", total)
        for card in body.get("data", []):
            oid = card.get("oracle_id") or (card.get("card_faces") or [{}])[0].get("oracle_id")
            if oid and oid not in seen:
                seen.add(oid)
                ids.append(oid)
        url = body.get("next_page") if body.get("has_more") else None
        params = None  # next_page already carries the query string
    return ids, total, pages


def _compact_json(cache: dict[str, Any]) -> str:
    """One entry per line: diff-friendly, and under the repo's 500 KB file cap."""
    items = list(cache.items())
    body = [
        f"  {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}"
        + ("," if i < len(items) - 1 else "")
        for i, (k, v) in enumerate(items)
    ]
    return "\n".join(["{", *body, "}"]) + "\n"


def _write_manifest(cache: dict[str, Any], out_path: Path) -> None:
    """Small committed record of what was fetched (the id lists themselves are
    ~2 MB and gitignored; re-fetch reproduces them from this file's queries)."""
    manifest = {
        k: {
            "query": v["query"],
            "total_cards": v["total_cards"],
            "n_oracle_ids": len(v["oracle_ids"]),
            "fetched_at": v["fetched_at"],
        }
        for k, v in cache.items()
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def cmd_fetch(args: argparse.Namespace) -> int:
    spec = yaml.safe_load(EXPERT_QUERIES.read_text(encoding="utf-8"))
    out_path = settings.eval_dir / f"scryfall_results_{spec['version']}.json"
    cache: dict[str, Any] = json.loads(out_path.read_text()) if out_path.exists() else {}
    with PipelineRun("scryfall_comparator_fetch", inputs={"queries": str(EXPERT_QUERIES)}) as run:
        for qid, variants in spec["queries"].items():
            for variant, q in variants.items():
                key = f"{qid}|{variant}"
                if key in cache and cache[key].get("query") == q and not args.force:
                    continue
                ids, total, pages = fetch_query(q, log=run)
                cache[key] = {
                    "query": q,
                    "total_cards": total,
                    "oracle_ids": ids,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                run.processed()
                run.event("fetched", key=key, total=total, pages=pages)
                print(f"  {key:18s} {total:6d} cards  {pages:3d} pages   {q}")
                out_path.write_text(_compact_json(cache), encoding="utf-8")
                _write_manifest(cache, out_path)
        run.note(out_path=str(out_path), entries=len(cache))
    print(f"\n  Cached {len(cache)} result sets → {out_path}")
    return 0


# ---- compare ------------------------------------------------------------


def complexity(q: str) -> dict[str, Any]:
    ops = _OPERATOR.findall(q)
    kinds = {op.lower() for _neg, op, _cmp in ops}
    return {
        "operators": len(ops),
        "operator_kinds": sorted(kinds),
        "n_kinds": len(kinds),
        "negations": sum(1 for neg, _o, _c in ops if neg) + q.count(" -"),
        "boolean_or_grouping": len(_BOOL.findall(q)),
        "quoted_phrases": len(_QUOTED.findall(q)),
        "uses_otag": any(k in ("otag", "oracletag", "function") for k in kinds),
        "chars": len(q),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    spec = yaml.safe_load(EXPERT_QUERIES.read_text(encoding="utf-8"))
    cache = json.loads((settings.eval_dir / f"scryfall_results_{spec['version']}.json").read_text())
    eval_set = {q["id"]: q for q in yaml.safe_load(EVAL_SET.read_text())["queries"]}
    oracle = yaml.safe_load(ORACLE.read_text())["queries"] if ORACLE.exists() else {}

    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT config, per_query FROM experiment_runs WHERE id = %s", (args.row,))
        got = cur.fetchone()
        if not got:
            raise SystemExit(f"experiment_runs id={args.row} not found")
        cfg, per_query = got
        cur.execute("SELECT DISTINCT oracle_id::text FROM cards")
        corpus = {r[0] for r in cur.fetchall()}
        slug_by_qid = {qid: e["tags"][0] for qid, e in oracle.items() if e.get("tags")}
        cur.execute(
            """
    SELECT t.slug, array_agg(DISTINCT ct.oracle_id::text)
    FROM oracle_tags t
    JOIN oracle_tag_closure cl ON cl.ancestor_id = t.id
    JOIN card_tags ct ON ct.tag_id = cl.descendant_id
    JOIN (SELECT DISTINCT oracle_id FROM cards) c ON c.oracle_id = ct.oracle_id
    WHERE t.slug = ANY(%s)
    GROUP BY t.slug
""",
            (sorted(set(slug_by_qid.values())),),
        )
        by_slug = {slug: set(ids) for slug, ids in cur.fetchall()}
        pools: dict[str, set[str]] = {q: by_slug.get(sl, set()) for q, sl in slug_by_qid.items()}

    rows_out: list[dict[str, Any]] = []
    sms = []
    for rec in per_query:
        qid = rec["id"]
        key = f"{qid}|{args.variant}"
        if key not in cache or "ranking_top" not in rec:
            continue
        q = cache[key]["query"]
        scry = set(cache[key]["oracle_ids"]) & corpus
        ranking = rec["ranking_top"]
        sm = compute_set_metrics(ranking, scry) if scry else None
        if sm:
            sms.append(sm)
        relevant = {r["id"] for r in eval_set[qid].get("relevant") or []}
        pool = pools.get(qid)
        top_n = set(ranking[: max(len(scry), 1)])
        rows_out.append(
            {
                "id": qid,
                "query": rec["query"],
                "plain_words": len(rec["query"].split()),
                "scryfall_query": q,
                "complexity": complexity(q),
                "scryfall_set_size": len(scry),
                "scryfall_total_cards": cache[key]["total_cards"],
                "parity": (
                    {
                        "r_precision": sm.r_precision,
                        "recall_at_100": round(
                            sum(1 for o in ranking[:100] if o in scry) / len(scry), 4
                        ),
                        "recall_at_500": sm.recall_at_500,
                        "depth_to_90": sm.depth_to_90,
                        "reachable": sm.reachable,
                        "jaccard_at_n": round(len(top_n & scry) / len(top_n | scry), 4),
                    }
                    if sm
                    else None
                ),
                "scryfall_vs_judgments": (
                    {
                        "precision": round(len(scry & relevant) / len(scry), 4),
                        "recall": round(len(scry & relevant) / len(relevant), 4),
                    }
                    if scry and relevant
                    else None
                ),
                "scryfall_vs_tag_pool": (
                    {
                        "precision": round(len(scry & pool) / len(scry), 4),
                        "recall": round(len(scry & pool) / len(pool), 4),
                    }
                    if scry and pool
                    else None
                ),
            }
        )

    agg = aggregate_set_metrics(sms)
    n = len(rows_out)
    cx = [r["complexity"] for r in rows_out]
    summary = {
        "n_queries": n,
        "n_with_scryfall_results": len(sms),
        "parity_r_precision": agg.get("set_r_precision", 0.0),
        "parity_recall_at_500": agg.get("set_recall_at_500", 0.0),
        "parity_depth90_x_set_median": agg.get("set_depth90_x_target_median", 0.0),
        "parity_reachable_frac": agg.get("set_reachable_frac", 0.0),
        "parity_jaccard_at_n_mean": sum(
            r["parity"]["jaccard_at_n"] for r in rows_out if r["parity"]
        )
        / max(len(sms), 1),
        "scryfall_precision_vs_judgments_mean": sum(
            r["scryfall_vs_judgments"]["precision"] for r in rows_out if r["scryfall_vs_judgments"]
        )
        / max(sum(1 for r in rows_out if r["scryfall_vs_judgments"]), 1),
        "scryfall_recall_vs_judgments_mean": sum(
            r["scryfall_vs_judgments"]["recall"] for r in rows_out if r["scryfall_vs_judgments"]
        )
        / max(sum(1 for r in rows_out if r["scryfall_vs_judgments"]), 1),
        "complexity_operators_mean": sum(c["operators"] for c in cx) / max(n, 1),
        "complexity_kinds_mean": sum(c["n_kinds"] for c in cx) / max(n, 1),
        "complexity_chars_mean": sum(c["chars"] for c in cx) / max(n, 1),
        "complexity_uses_otag_frac": sum(c["uses_otag"] for c in cx) / max(n, 1),
        "plain_words_mean": sum(r["plain_words"] for r in rows_out) / max(n, 1),
    }

    print(
        f"\n  Cascade row {args.row} ({cfg.get('config_name')}) vs Scryfall [{args.variant}]  n={n}"
    )
    print("  === Result parity (cascade ranking vs Scryfall set) ===")
    for k_ in (
        "parity_r_precision",
        "parity_recall_at_500",
        "parity_jaccard_at_n_mean",
        "parity_reachable_frac",
        "parity_depth90_x_set_median",
    ):
        print(f"    {k_:36s} {summary[k_]:.4f}")
    print("  === Scryfall expert set vs eval judgments ===")
    print(f"    {'precision':36s} {summary['scryfall_precision_vs_judgments_mean']:.4f}")
    print(f"    {'recall':36s} {summary['scryfall_recall_vs_judgments_mean']:.4f}")
    print("  === Query complexity (what the user would have typed) ===")
    print(f"    {'operators / query':36s} {summary['complexity_operators_mean']:.2f}")
    print(f"    {'operator kinds / query':36s} {summary['complexity_kinds_mean']:.2f}")
    print(
        f"    {'chars / query':36s} {summary['complexity_chars_mean']:.1f}   (plain: {summary['plain_words_mean']:.1f} words)"
    )
    print(f"    {'needs otag:':36s} {summary['complexity_uses_otag_frac']:.0%}")
    print()
    print(
        f"  {'id':6s} {'set':>5s} {'R-prec':>6s} {'R@500':>6s} {'jacc':>5s}  {'ops':>3s} {'otag':>4s}  {'scry P/R vs judg':>16s}  query"
    )
    for r in sorted(rows_out, key=lambda r: -(r["parity"]["r_precision"] if r["parity"] else -1)):
        p = r["parity"]
        j = r["scryfall_vs_judgments"]
        pr = f"{p['r_precision']:.2f}" if p else "  —"
        r5 = f"{p['recall_at_500']:.2f}" if p else "  —"
        jc = f"{p['jaccard_at_n']:.2f}" if p else "  —"
        sj = f"{j['precision']:.2f}/{j['recall']:.2f}" if j else "—"
        print(
            f"  {r['id']:6s} {r['scryfall_set_size']:5d} {pr:>6s} {r5:>6s} {jc:>5s}  {r['complexity']['operators']:3d} {'y' if r['complexity']['uses_otag'] else '':>4s}  {sj:>16s}  {r['query'][:34]}"
        )

    if args.dry_run:
        print("\n  Dry run — no experiment_runs row written.")
        return 0
    record = log_experiment(
        eval_set_version=f"scryfall-expert-{spec['version']}|{args.variant}",
        config={
            "kind": "scryfall_comparator",
            "cascade_row": args.row,
            "cascade_config": cfg,
            "variant": args.variant,
            "expert_queries": str(EXPERT_QUERIES),
        },
        metrics=summary,
        per_query=rows_out,
        notes=args.notes or f"Scryfall comparator ({args.variant}) vs cascade row {args.row}",
    )
    print(f"\n  Logged as experiment_runs id={record.id}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="Fetch expert-query result sets from Scryfall (cached).")
    f.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    c = sub.add_parser("compare", help="Score a logged cascade row against the cached sets.")
    c.add_argument("--row", type=int, required=True, help="experiment_runs id with ranking_top.")
    c.add_argument("--variant", choices=["expert", "no_tag"], default="expert")
    c.add_argument("--notes", default="")
    c.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return cmd_fetch(args) if args.cmd == "fetch" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
