"""Run an evaluation configuration against the eval set, write a row.

Loads a configuration YAML and the eval set, embeds each query with
the same model that produced the corpus embeddings, retrieves top-K
oracle_ids via cosine search, scores under the tri-state relevance
scheme (borderline excluded from numerator and denominator), and
writes one row to the ``experiment_runs`` table.

Phase 3 baseline retrieval is ``embedding_only`` — raw query → encode
→ pgvector cosine search → dedupe by oracle_id (closest face wins).
Future configs will plug in HyDE and SQL pre-filter.

Usage::

    python scripts/evaluate.py --config configs/baseline.yaml
    python scripts/evaluate.py --config configs/baseline.yaml --notes "first run"
    python scripts/evaluate.py --config configs/baseline.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
import yaml
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import settings
from src.db.experiment_log import log_experiment
from src.eval.metrics import (
    QueryMetrics,
    aggregate_metrics,
    compute_query_metrics,
)
from src.logging_utils import PipelineRun
from src.utils.device import select_device

_SEARCH_SQL = """
    SELECT oracle_id::text, name, embedding <=> %s::vector AS distance
    FROM cards
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""


def _retrieve(
    conn: psycopg.Connection,
    query_vec,
    k: int,
    oversample: int,
) -> list[dict[str, Any]]:
    """Cosine-search the corpus, dedupe by oracle_id (closest face wins), take top K."""
    with conn.cursor() as cur:
        cur.execute(_SEARCH_SQL, (query_vec, query_vec, oversample))
        rows = cur.fetchall()

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for oracle_id, name, distance in rows:
        if oracle_id in seen:
            continue
        seen.add(oracle_id)
        out.append({"oracle_id": oracle_id, "name": name, "distance": float(distance)})
        if len(out) >= k:
            break
    return out


def _build_per_query_record(
    query: dict[str, Any],
    top_k: list[dict[str, Any]],
    metrics: QueryMetrics,
    latency_ms: float,
    relevant_ids: set[str],
    borderline_ids: set[str],
) -> dict[str, Any]:
    """Per-query trace for the experiment_runs.per_query JSONB column."""
    annotated_top = []
    for rank, hit in enumerate(top_k, 1):
        oid = hit["oracle_id"]
        annotated_top.append(
            {
                "rank": rank,
                "oracle_id": oid,
                "name": hit["name"],
                "distance": round(hit["distance"], 6),
                "is_relevant": oid in relevant_ids,
                "is_borderline": oid in borderline_ids,
            }
        )
    return {
        "id": query["id"],
        "query": query.get("query"),
        "difficulty": query.get("difficulty"),
        "category": query.get("category"),
        "relevant_count": len(relevant_ids),
        "borderline_count": len(borderline_ids),
        "top_k": annotated_top,
        "hit_rank": metrics.hit_rank,
        "recall_at_1": round(metrics.recall_at_1, 6),
        "recall_at_5": round(metrics.recall_at_5, 6),
        "recall_at_10": round(metrics.recall_at_10, 6),
        "reciprocal_rank": round(metrics.reciprocal_rank, 6),
        "relevant_in_top_k": metrics.relevant_in_top_k,
        "latency_ms": round(latency_ms, 3),
    }


def _ids(query: dict[str, Any], key: str) -> set[str]:
    return {p["id"] for p in (query.get(key) or [])}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Evaluation configuration YAML (e.g. configs/baseline.yaml).",
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

    retrieval = cfg.get("retrieval", {})
    k = int(retrieval.get("k", 10))
    oversample = int(retrieval.get("oversample", 50))
    retrieval_type = retrieval.get("type", "embedding_only")

    if retrieval_type != "embedding_only":
        raise NotImplementedError(
            f"retrieval.type={retrieval_type!r} not implemented yet; "
            "Phase 3 supports only 'embedding_only'."
        )

    device = select_device()

    with PipelineRun(
        "evaluate",
        inputs={
            "config_name": cfg.get("name", args.config.stem),
            "config_path": str(args.config),
            "eval_set_path": str(eval_set_path),
            "eval_set_version": eval_set.get("version", "v1-draft"),
            "retrieval_type": retrieval_type,
            "k": k,
            "device": str(device),
            "dry_run": args.dry_run,
        },
    ) as run:
        print(f"\n  Config:        {cfg.get('name', args.config.stem)}")
        print(f"  Eval set:      {eval_set_path}  (version={eval_set.get('version')})")
        print(f"  Retrieval:     {retrieval_type}, k={k}")
        print(f"  Device:        {device}")
        print(f"  Queries:       {len(queries)}")
        print()

        model = SentenceTransformer(settings.embedding_model, device=str(device))
        run.event("model_loaded", model=settings.embedding_model)

        per_query: list[dict[str, Any]] = []
        per_query_metrics: list[QueryMetrics] = []
        latencies_ms: list[float] = []

        with psycopg.connect(settings.database_url) as conn:
            register_vector(conn)

            for q in tqdm(queries, desc="evaluate", unit="query"):
                relevant_ids = _ids(q, "relevant")
                borderline_ids = _ids(q, "borderline")

                t0 = time.perf_counter()
                query_vec = model.encode(
                    [q["query"]], normalize_embeddings=True, show_progress_bar=False
                )[0]
                top_k = _retrieve(conn, query_vec, k=k, oversample=oversample)
                latency_ms = (time.perf_counter() - t0) * 1000

                top_k_ids = [hit["oracle_id"] for hit in top_k]
                metrics = compute_query_metrics(top_k_ids, relevant_ids, borderline_ids)

                latencies_ms.append(latency_ms)
                per_query_metrics.append(metrics)
                per_query.append(
                    _build_per_query_record(
                        q, top_k, metrics, latency_ms, relevant_ids, borderline_ids
                    )
                )
                run.processed()

        aggregate = aggregate_metrics(per_query_metrics, latencies_ms)
        run.note(**{f"agg_{key}": val for key, val in aggregate.items()})

        # ----- Stdout summary -----
        print()
        print("  === Aggregate metrics ===")
        for label, value in [
            ("recall@1", aggregate["recall_at_1"]),
            ("recall@5", aggregate["recall_at_5"]),
            ("recall@10", aggregate["recall_at_10"]),
            ("MRR", aggregate["mrr"]),
            ("latency p50 (ms)", aggregate["latency_p50"]),
            ("latency p95 (ms)", aggregate["latency_p95"]),
            ("latency mean (ms)", aggregate["latency_mean"]),
        ]:
            print(f"    {label:20s} {value:.4f}")

        print()
        print("  === Per-query recall@10 ===")
        for r in sorted(per_query, key=lambda x: x["recall_at_10"], reverse=True):
            rec = r["recall_at_10"]
            mrr_q = r["reciprocal_rank"]
            qtext = (r["query"] or "")[:60]
            print(f"    {r['id']:6s}  R@10={rec:.3f}  MRR={mrr_q:.3f}  {qtext}")

        # ----- Write experiment_runs row -----
        if args.dry_run:
            print("\n  Dry run — no experiment_runs row written.")
            return 0

        record = log_experiment(
            eval_set_version=eval_set.get("version", "v1-draft"),
            config={
                "config_name": cfg.get("name", args.config.stem),
                "config_path": str(args.config),
                "retrieval_type": retrieval_type,
                "k": k,
                "oversample": oversample,
                "embedding_model": settings.embedding_model,
                "embedding_version": settings.embedding_version,
                "preprocess_version": settings.preprocess_version,
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
