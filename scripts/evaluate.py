"""Run an evaluation configuration against the eval set, write a row.

Loads a configuration YAML and the eval set, runs every query through
:class:`src.search.Searcher` (Stage 1 mode, SQL on/off, and filter policy
come from the config), scores the top-K under the tri-state relevance
scheme (borderline excluded from numerator and denominator), and writes one
row to the ``experiment_runs`` table.

Config contract (``retrieval`` block)::

    retrieval:
      type: cascade            # or "embedding_only" (legacy alias for stage1: raw)
      stage1: hyde             # hyde | passthrough | raw | concepts | query_plus_concepts
      prompt: prompts/hyde_v2.yaml   # optional; default prompts/hyde_v1.yaml
      sql: true                # false = minus-SQL ablation
      keywords_filter: false   # FilterPolicy.keywords
      k: 10
      label_oracle: data/eval/tag_label_oracle_v1.yaml   # optional: embed tag LABELS instead of query text
      set_metrics: true        # also rank the WHOLE candidate set and score it against
                               # each query's tag pool (R-precision, P@100, depth-to-90%);
                               # target mapping comes from data/eval/tag_label_oracle_v1.yaml

Every run logs the Stage 1 output, the compiled WHERE clause, the candidate
count, and per-stage timings for each query, so the grid in the 2026-09-18
journal can be rebuilt from ``experiment_runs`` alone.

Usage::

    python scripts/evaluate.py --config configs/cascade_hyde_v1.yaml
    python scripts/evaluate.py --config configs/cascade_hyde_v1.yaml --notes "control row"
    python scripts/evaluate.py --config configs/raw_dense.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
import yaml
from tqdm import tqdm

from src.config import settings
from src.db.experiment_log import log_experiment
from src.eval.metrics import (
    QueryMetrics,
    SetMetrics,
    aggregate_metrics,
    aggregate_set_metrics,
    compute_query_metrics,
    compute_set_metrics,
)
from src.logging_utils import PipelineRun
from src.search import FilterPolicy, Searcher, SearchResult, Stage1Mode

RANKING_KEEP = 2000  # oracle_ids of the full ranking stored per query (set_metrics rows)


def _build_per_query_record(
    query: dict[str, Any],
    result: SearchResult,
    metrics: QueryMetrics,
    latency_ms: float,
    relevant_ids: set[str],
    borderline_ids: set[str],
) -> dict[str, Any]:
    """Per-query trace for the experiment_runs.per_query JSONB column."""
    annotated_top = []
    for rank, hit in enumerate(result.hits, 1):
        annotated_top.append(
            {
                "rank": rank,
                "oracle_id": hit.oracle_id,
                "name": hit.name,
                "distance": round(hit.distance, 6),
                "is_relevant": hit.oracle_id in relevant_ids,
                "is_borderline": hit.oracle_id in borderline_ids,
            }
        )
    hyde = result.hyde.model_dump(exclude_none=True) if result.hyde is not None else None
    return {
        "id": query["id"],
        "query": query.get("query"),
        "difficulty": query.get("difficulty"),
        "category": query.get("category"),
        "relevant_count": len(relevant_ids),
        "borderline_count": len(borderline_ids),
        "stage1": hyde,
        "stage1_completion_tokens": result.hyde.completion_tokens if result.hyde else None,
        "query_text": result.query_text,
        "where_sql": result.where_sql,
        "candidate_count": result.candidate_count,
        "warnings": result.warnings,
        "timings_ms": result.timings_ms,
        "top_k": annotated_top,
        "hit_rank": metrics.hit_rank,
        "recall_at_1": round(metrics.recall_at_1, 6),
        "recall_at_5": round(metrics.recall_at_5, 6),
        "recall_at_10": round(metrics.recall_at_10, 6),
        "precision_at_10": round(metrics.precision_at_10, 6),
        "reciprocal_rank": round(metrics.reciprocal_rank, 6),
        "relevant_in_top_k": metrics.relevant_in_top_k,
        "latency_ms": round(latency_ms, 3),
    }


def _ids(query: dict[str, Any], key: str) -> set[str]:
    return {p["id"] for p in (query.get(key) or [])}


def _load_label_oracle(path: Path) -> dict[str, str]:
    """query_id -> text to embed (tag labels joined with ', '); queries with no tags omitted."""
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    wanted: dict[str, list[str]] = {
        qid: list(entry.get("tags") or []) for qid, entry in spec["queries"].items()
    }
    slugs = sorted({s for tags in wanted.values() for s in tags})
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT slug, label FROM oracle_tags WHERE slug = ANY(%s)", (slugs,))
        label_by_slug = dict(cur.fetchall())
    missing = [s for s in slugs if s not in label_by_slug]
    if missing:
        raise ValueError(f"label_oracle references unknown tag slugs: {missing}")
    return {qid: ", ".join(label_by_slug[s] for s in tags) for qid, tags in wanted.items() if tags}


def _load_tag_pools(path: Path) -> dict[str, tuple[str, set[str]]]:
    """query_id -> (tag slug, closure pool of in-corpus oracle_ids) for mapped queries.

    One query for all tags: the closure view is a recursive CTE and is
    re-evaluated per statement, so a per-tag loop cost ~1.5 s x 21 tags.
    """
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]
    slug_by_qid = {qid: e["tags"][0] for qid, e in spec.items() if e.get("tags")}
    with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
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
        pools = {slug: set(ids) for slug, ids in cur.fetchall()}
    return {qid: (slug, pools.get(slug, set())) for qid, slug in slug_by_qid.items()}


def _parse_retrieval(cfg: dict[str, Any]) -> tuple[Stage1Mode, bool, FilterPolicy, int]:
    retrieval = cfg.get("retrieval", {})
    rtype = retrieval.get("type", "cascade")
    if rtype == "embedding_only":
        mode, use_sql = Stage1Mode.RAW, False
    elif rtype == "cascade":
        mode = Stage1Mode(retrieval.get("stage1", "hyde"))
        use_sql = bool(retrieval.get("sql", True))
    else:
        raise ValueError(f"retrieval.type={rtype!r} not recognised (cascade | embedding_only)")
    policy = FilterPolicy(keywords=bool(retrieval.get("keywords_filter", False)))
    return mode, use_sql, policy, int(retrieval.get("k", 10))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Evaluation configuration YAML (e.g. configs/cascade_hyde_v1.yaml).",
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=None,
        help="Override the eval_set path specified in the config.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free-text annotation written to the experiment_runs.notes column.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but don't write a row to experiment_runs.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    eval_set_path = args.eval_set or Path(cfg.get("eval_set", "data/eval/queries_v1_draft.yaml"))
    eval_set = yaml.safe_load(eval_set_path.read_text(encoding="utf-8"))
    queries = eval_set.get("queries", [])

    mode, use_sql, policy, k = _parse_retrieval(cfg)
    config_name = cfg.get("name", args.config.stem)
    oracle_path = cfg.get("retrieval", {}).get("label_oracle")
    label_oracle = _load_label_oracle(Path(oracle_path)) if oracle_path else {}
    want_set = bool(cfg.get("retrieval", {}).get("set_metrics", False))
    tag_pools = _load_tag_pools(Path("data/eval/tag_label_oracle_v1.yaml")) if want_set else {}

    with PipelineRun(
        "evaluate",
        inputs={
            "config_name": config_name,
            "config_path": str(args.config),
            "eval_set_path": str(eval_set_path),
            "eval_set_version": eval_set.get("version", "v1-draft"),
            "stage1": mode.value,
            "sql": use_sql,
            "keywords_filter": policy.keywords,
            "k": k,
            "label_oracle": oracle_path,
            "set_metrics": want_set,
            "dry_run": args.dry_run,
        },
    ) as run:
        print(f"\n  Config:        {config_name}")
        print(f"  Eval set:      {eval_set_path}  (version={eval_set.get('version')})")
        print(f"  Stage 1:       {mode.value}   SQL: {'on' if use_sql else 'off'}   k={k}")
        print(f"  Embedder:      {settings.embedding_version}")
        if mode is not Stage1Mode.RAW:
            print(f"  HyDE model:    {settings.hyde_model}")
        print(f"  Queries:       {len(queries)}")
        if label_oracle:
            print(f"  Label oracle:  {oracle_path}  ({len(label_oracle)} queries overridden)")
        print()

        per_query: list[dict[str, Any]] = []
        per_query_metrics: list[QueryMetrics] = []
        set_metrics_list: list[SetMetrics] = []
        latencies_ms: list[float] = []

        prompt_path = cfg.get("retrieval", {}).get("prompt")
        with Searcher(
            policy=policy, prompt_path=Path(prompt_path) if prompt_path else None
        ) as searcher:
            run.event("model_loaded", model=settings.embedding_model)
            if mode is not Stage1Mode.RAW:
                print(f"  Prompt:        {searcher.prompt_version}\n")
            for q in tqdm(queries, desc="evaluate", unit="query"):
                relevant_ids = _ids(q, "relevant")
                borderline_ids = _ids(q, "borderline")

                t0 = time.perf_counter()
                result = searcher.search(
                    q["query"],
                    k=k,
                    mode=mode,
                    use_sql=use_sql,
                    embed_text=label_oracle.get(q["id"]),
                )
                latency_ms = (time.perf_counter() - t0) * 1000

                top_k_ids = [hit.oracle_id for hit in result.hits]
                metrics = compute_query_metrics(top_k_ids, relevant_ids, borderline_ids)

                latencies_ms.append(latency_ms)
                per_query_metrics.append(metrics)
                record = _build_per_query_record(
                    q, result, metrics, latency_ms, relevant_ids, borderline_ids
                )
                if want_set:
                    filters = result.hyde.filters if (use_sql and result.hyde) else None
                    ranking = searcher.rank_prepared(result.query_text, filters)
                    # Top of the full ranking for EVERY query, so comparators
                    # (Scryfall) can score this row later without re-running.
                    record["ranking_top"] = ranking[:RANKING_KEEP]
                    if q["id"] in tag_pools:
                        slug, pool = tag_pools[q["id"]]
                        sm = compute_set_metrics(ranking, pool)
                        set_metrics_list.append(sm)
                        record["set_metrics"] = {"tag": slug, **sm.__dict__}
                per_query.append(record)
                run.processed()

        aggregate = aggregate_metrics(per_query_metrics, latencies_ms)
        if want_set:
            aggregate.update(aggregate_set_metrics(set_metrics_list))
        toks = [
            r["stage1_completion_tokens"] for r in per_query if r.get("stage1_completion_tokens")
        ]
        if toks:
            aggregate["stage1_completion_tokens_mean"] = sum(toks) / len(toks)
        s1 = [
            r["timings_ms"].get("stage1_ms") for r in per_query if r["timings_ms"].get("stage1_ms")
        ]
        if s1:
            aggregate["stage1_ms_mean"] = sum(s1) / len(s1)
        run.note(**{f"agg_{key}": val for key, val in aggregate.items()})

        # ----- Stdout summary -----
        print()
        print("  === Aggregate metrics ===")
        for label, value in [
            ("recall@1", aggregate["recall_at_1"]),
            ("recall@5", aggregate["recall_at_5"]),
            ("recall@10", aggregate["recall_at_10"]),
            ("precision@10", aggregate["precision_at_10"]),
            ("MRR", aggregate["mrr"]),
            ("latency p50 (ms)", aggregate["latency_p50"]),
            ("latency p95 (ms)", aggregate["latency_p95"]),
            ("latency mean (ms)", aggregate["latency_mean"]),
            ("stage1 mean (ms)", aggregate.get("stage1_ms_mean", 0.0)),
            ("stage1 out tokens", aggregate.get("stage1_completion_tokens_mean", 0.0)),
        ]:
            print(f"    {label:20s} {value:.4f}")
        if want_set:
            print()
            print(
                f"  === Set retrieval vs tag pools (n={aggregate['set_n_queries']}, inside Stage 2 set) ==="
            )
            for label, key in [
                ("R-precision", "set_r_precision"),
                ("R-prec (reachable)", "set_r_precision_reachable"),
                ("P@100", "set_precision_at_100"),
                ("R@500", "set_recall_at_500"),
                ("reachable frac", "set_reachable_frac"),
                ("depth90 / target", "set_depth90_x_target_median"),
            ]:
                print(f"    {label:20s} {aggregate[key]:.4f}")

        print()
        print("  === Per-query (sorted by precision@10) ===")
        for r in sorted(per_query, key=lambda x: x["precision_at_10"], reverse=True):
            flag = " !" if r["warnings"] else ""
            print(
                f"    {r['id']:6s}  P@10={r['precision_at_10']:.2f}  R@10={r['recall_at_10']:.3f}"
                f"  MRR={r['reciprocal_rank']:.3f}"
                f"  cand={r['candidate_count']:>6,}  {(r['query'] or '')[:50]}{flag}"
            )
        warned = [r for r in per_query if r["warnings"]]
        if warned:
            print()
            print("  === Warnings ===")
            for r in warned:
                for w in r["warnings"]:
                    print(f"    {r['id']:6s}  {w}")

        # ----- Write experiment_runs row -----
        if args.dry_run:
            print("\n  Dry run — no experiment_runs row written.")
            return 0

        record = log_experiment(
            eval_set_version=eval_set.get("version", "v1-draft"),
            config={
                "config_name": config_name,
                "config_path": str(args.config),
                "stage1": mode.value,
                "sql": use_sql,
                "filter_policy": policy.__dict__,
                "k": k,
                "embedding_model": settings.embedding_model,
                "embedding_version": settings.embedding_version,
                "preprocess_version": settings.preprocess_version,
                "hyde_model": settings.hyde_model if mode is not Stage1Mode.RAW else None,
                "hyde_prompt": searcher.prompt_version if mode is not Stage1Mode.RAW else None,
                "label_oracle": oracle_path,
                "hyde_prompt_sha": searcher.prompt_sha if mode is not Stage1Mode.RAW else None,
                "eval_set_path": str(eval_set_path),
                "description": cfg.get("description"),
            },
            metrics=aggregate,
            per_query=per_query,
            notes=args.notes or cfg.get("notes"),
        )
        run.note(experiment_run_id=record.id)
        print()
        print(f"  Logged as experiment_runs id={record.id}  (at {record.created_at})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
