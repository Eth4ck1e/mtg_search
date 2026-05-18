"""Pure-Python metric calculation for retrieval evaluation.

No DB, no model, no I/O — every function here takes already-retrieved
oracle_ids and a relevance judgment set and returns the metrics. This
isolation makes the metric math unit-testable without a Postgres
fixture and without loading the embedding model.

The tri-state relevance scheme (see [[2026-05-17-keyword-augmentation]]
and the eval-set README for rationale) is enforced here: cards in the
``borderline`` set are removed from both numerator and denominator of
recall@K and from MRR consideration. They neither help nor hurt the
score if the system surfaces them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryMetrics:
    """Metrics for a single query against the eval set."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    hit_rank: int | None
    relevant_in_top_k: dict[int, int]


def compute_query_metrics(
    top_k_oracle_ids: list[str],
    relevant_ids: set[str],
    borderline_ids: set[str],
    k_levels: tuple[int, ...] = (1, 5, 10),
) -> QueryMetrics:
    """Compute recall@K and MRR for one query.

    Args:
        top_k_oracle_ids: System's retrieved oracle_ids in rank order
            (rank 1 first). Length should be at least ``max(k_levels)``
            after borderline filtering, otherwise lower-rank metrics
            will be computed over the truncated list.
        relevant_ids: Cards the eval set deems clearly relevant. recall
            denominator. Should not overlap with borderline_ids.
        borderline_ids: Cards excluded from scoring. Removed from
            ``top_k_oracle_ids`` before computing recall and MRR.
        k_levels: K values to report recall@K for. (1, 5, 10) by default.

    Returns:
        QueryMetrics with per-K recall, the reciprocal rank of the
        first relevant hit, and the count of relevant cards retrieved
        at each K threshold.
    """
    scored = [oid for oid in top_k_oracle_ids if oid not in borderline_ids]
    max_k = max(k_levels)

    relevant_in_top_k: dict[int, int] = {}
    recall_at: dict[int, float] = {}
    for k in k_levels:
        count = sum(1 for oid in scored[:k] if oid in relevant_ids)
        relevant_in_top_k[k] = count
        recall_at[k] = count / len(relevant_ids) if relevant_ids else 0.0

    hit_rank: int | None = None
    for i, oid in enumerate(scored[:max_k], 1):
        if oid in relevant_ids:
            hit_rank = i
            break
    rr = (1.0 / hit_rank) if hit_rank else 0.0

    return QueryMetrics(
        recall_at_1=recall_at.get(1, 0.0),
        recall_at_5=recall_at.get(5, 0.0),
        recall_at_10=recall_at.get(10, 0.0),
        reciprocal_rank=rr,
        hit_rank=hit_rank,
        relevant_in_top_k=relevant_in_top_k,
    )


def aggregate_metrics(
    query_metrics: list[QueryMetrics],
    latencies_ms: list[float],
) -> dict[str, float]:
    """Mean recall@K, MRR, latency p50/p95/mean across a list of queries.

    The aggregate is an unweighted mean over queries (each query
    counts equally regardless of how many relevant cards it has).
    This is the standard IR-eval interpretation — query-level macro
    average rather than micro-averaging across all judgements.
    """
    if not query_metrics:
        return {
            "n_queries": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr": 0.0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_mean": 0.0,
        }

    n = len(query_metrics)
    sorted_latencies = sorted(latencies_ms)
    return {
        "n_queries": n,
        "recall_at_1": sum(qm.recall_at_1 for qm in query_metrics) / n,
        "recall_at_5": sum(qm.recall_at_5 for qm in query_metrics) / n,
        "recall_at_10": sum(qm.recall_at_10 for qm in query_metrics) / n,
        "mrr": sum(qm.reciprocal_rank for qm in query_metrics) / n,
        "latency_p50": _percentile(sorted_latencies, 50),
        "latency_p95": _percentile(sorted_latencies, 95),
        "latency_mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)
