"""Tests for the retrieval metric calculations.

Pure-Python unit tests — no DB, no model. The point is to prove the
metric math is correct (the tri-state borderline filtering is the
load-bearing piece) before pointing the harness at the real corpus.
"""

from __future__ import annotations

from src.eval.metrics import (
    QueryMetrics,
    _percentile,
    aggregate_metrics,
    compute_query_metrics,
)


def test_all_relevant_in_top_1_gives_perfect_metrics() -> None:
    """If the first hit is relevant, recall@1 = 1/|relevant|, MRR = 1.0."""
    m = compute_query_metrics(
        top_k_oracle_ids=["a", "b", "c"],
        relevant_ids={"a"},
        borderline_ids=set(),
    )
    assert m.recall_at_1 == 1.0
    assert m.recall_at_5 == 1.0
    assert m.recall_at_10 == 1.0
    assert m.reciprocal_rank == 1.0
    assert m.hit_rank == 1


def test_no_relevant_in_top_k_gives_zero() -> None:
    m = compute_query_metrics(
        top_k_oracle_ids=["a", "b", "c"],
        relevant_ids={"x", "y"},
        borderline_ids=set(),
    )
    assert m.recall_at_1 == 0.0
    assert m.recall_at_10 == 0.0
    assert m.reciprocal_rank == 0.0
    assert m.hit_rank is None


def test_recall_is_per_total_relevant() -> None:
    """recall@K = (relevant retrieved in K) / (total relevant). Not divided by K."""
    m = compute_query_metrics(
        top_k_oracle_ids=["a", "b", "c", "x", "y"],
        relevant_ids={"a", "b", "z"},
        borderline_ids=set(),
    )
    assert m.recall_at_5 == 2 / 3
    assert m.relevant_in_top_k[5] == 2


def test_hit_rank_is_one_indexed_first_relevant() -> None:
    m = compute_query_metrics(
        top_k_oracle_ids=["x", "y", "a", "b"],
        relevant_ids={"a", "b"},
        borderline_ids=set(),
    )
    assert m.hit_rank == 3
    assert m.reciprocal_rank == 1 / 3


def test_borderline_cards_are_removed_before_scoring() -> None:
    """The load-bearing invariant: borderline cards are neutral.

    If the system surfaces a borderline card, it should not push relevant
    cards further down the rank — borderline is filtered out, then ranks
    are reconsidered on the filtered list.
    """
    m = compute_query_metrics(
        top_k_oracle_ids=["bord1", "a", "bord2", "b"],
        relevant_ids={"a", "b"},
        borderline_ids={"bord1", "bord2"},
    )
    # After filtering: [a, b]. a at rank 1, b at rank 2.
    assert m.hit_rank == 1
    assert m.recall_at_1 == 0.5
    assert m.recall_at_5 == 1.0


def test_borderline_does_not_count_toward_denominator() -> None:
    """recall denominator is |relevant|, not |relevant| + |borderline|."""
    m = compute_query_metrics(
        top_k_oracle_ids=["a"],
        relevant_ids={"a"},
        borderline_ids={"b", "c", "d"},
    )
    assert m.recall_at_1 == 1.0


def test_empty_relevant_set_returns_zero_recall() -> None:
    """Defensive: a query with no relevant cards returns 0 recall (not div-by-zero)."""
    m = compute_query_metrics(
        top_k_oracle_ids=["a", "b"],
        relevant_ids=set(),
        borderline_ids=set(),
    )
    assert m.recall_at_10 == 0.0
    assert m.reciprocal_rank == 0.0
    assert m.hit_rank is None


def test_partial_recall_with_many_relevant() -> None:
    """recall@K caps at retrieved/total even if more relevant exist than K."""
    m = compute_query_metrics(
        top_k_oracle_ids=["a", "b", "c"],
        relevant_ids={"a", "b", "c", "d", "e"},
        borderline_ids=set(),
    )
    assert m.recall_at_5 == 3 / 5


def test_aggregate_metrics_macro_averages_per_query() -> None:
    """Aggregate is per-query mean, not per-judgement micro-average."""
    q1 = QueryMetrics(
        recall_at_1=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        reciprocal_rank=1.0,
        hit_rank=1,
        relevant_in_top_k={1: 1, 5: 1, 10: 1},
    )
    q2 = QueryMetrics(
        recall_at_1=0.0,
        recall_at_5=0.5,
        recall_at_10=1.0,
        reciprocal_rank=0.25,
        hit_rank=4,
        relevant_in_top_k={1: 0, 5: 1, 10: 2},
    )
    agg = aggregate_metrics([q1, q2], latencies_ms=[10.0, 20.0])
    assert agg["n_queries"] == 2
    assert agg["recall_at_1"] == 0.5
    assert agg["recall_at_5"] == 0.75
    assert agg["recall_at_10"] == 1.0
    assert agg["mrr"] == 0.625
    assert agg["latency_p50"] == 15.0
    assert agg["latency_mean"] == 15.0


def test_aggregate_handles_empty_input() -> None:
    agg = aggregate_metrics([], latencies_ms=[])
    assert agg["n_queries"] == 0
    assert agg["recall_at_10"] == 0.0
    assert agg["mrr"] == 0.0


def test_percentile_handles_single_value() -> None:
    assert _percentile([42.0], 50) == 42.0
    assert _percentile([42.0], 95) == 42.0


def test_percentile_interpolates_correctly() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert _percentile(values, 50) == 30.0
    # 95th percentile of 5 values: interp between index 3 (40) and 4 (50)
    assert _percentile(values, 95) == 48.0
