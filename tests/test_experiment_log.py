"""Integration test for the experiment_runs writer.

Hits the real local Postgres. Skipped automatically when the database
is not reachable so CI (which currently has no Postgres service) still
passes.
"""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip("psycopg")

from src.config import settings  # noqa: E402
from src.db.experiment_log import log_experiment  # noqa: E402


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2):
            return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(
    not _db_reachable(),
    reason="Postgres not reachable at settings.database_url",
)


@requires_db
def test_log_experiment_round_trip() -> None:
    """A logged row reads back with JSONB fields fully preserved."""
    record = log_experiment(
        eval_set_version="test-v0",
        config={"model": "stub", "k": 10, "filters": {"colors": ["U"]}},
        metrics={"recall@5": 0.42, "mrr": 0.31},
        per_query=[
            {"query": "blue counterspell", "hit_rank": 1, "latency_ms": 12.3},
            {"query": "green ramp", "hit_rank": None, "latency_ms": 9.8},
        ],
        notes="pytest synthetic row",
    )

    assert record.id > 0
    assert record.created_at is not None

    try:
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT eval_set_version, config, metrics, per_query, notes "
                "FROM experiment_runs WHERE id = %s",
                (record.id,),
            )
            row = cur.fetchone()

        assert row is not None
        eval_set_version, config, metrics, per_query, notes = row
        assert eval_set_version == "test-v0"
        assert config == {"model": "stub", "k": 10, "filters": {"colors": ["U"]}}
        assert metrics == {"recall@5": 0.42, "mrr": 0.31}
        assert per_query == [
            {"query": "blue counterspell", "hit_rank": 1, "latency_ms": 12.3},
            {"query": "green ramp", "hit_rank": None, "latency_ms": 9.8},
        ]
        assert notes == "pytest synthetic row"
    finally:
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM experiment_runs WHERE id = %s", (record.id,))


@requires_db
def test_log_experiment_omits_optional_fields() -> None:
    """per_query and notes default to NULL when not provided."""
    record = log_experiment(
        eval_set_version="test-v0",
        config={"model": "stub"},
        metrics={"recall@5": 0.0},
    )

    try:
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT per_query, notes FROM experiment_runs WHERE id = %s",
                (record.id,),
            )
            row = cur.fetchone()
        assert row == (None, None)
    finally:
        with psycopg.connect(settings.database_url) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM experiment_runs WHERE id = %s", (record.id,))
