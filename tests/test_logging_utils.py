"""Tests for the JSONL pipeline-run logger.

All tests redirect the log directory to a pytest tmp_path, so no real
``logs/`` files are touched.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.logging_utils import PipelineRun


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _log_path_for(tmp_path: Path, script: str) -> Path:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return tmp_path / script / f"{today}.jsonl"


def test_writes_start_and_finish(tmp_path: Path) -> None:
    with PipelineRun("demo", inputs={"file": "x.json"}, log_dir=tmp_path) as run:
        run.processed(3)

    lines = _read_jsonl(_log_path_for(tmp_path, "demo"))
    assert len(lines) == 2

    start = lines[0]
    assert start["event"] == "start"
    assert start["script"] == "demo"
    assert start["inputs"] == {"file": "x.json"}
    assert "ts" in start
    assert "version" in start

    finish = lines[1]
    assert finish["event"] == "finish"
    assert finish["success"] is True
    assert finish["processed"] == 3
    assert finish["skipped"] == {}
    assert finish["duration_s"] >= 0


def test_skip_counters_aggregate_by_reason(tmp_path: Path) -> None:
    with PipelineRun("demo", log_dir=tmp_path) as run:
        run.skip("token")
        run.skip("token")
        run.skip("digital_only")
        run.skip("token", n=5)

    finish = _read_jsonl(_log_path_for(tmp_path, "demo"))[-1]
    assert finish["skipped"] == {"token": 7, "digital_only": 1}


def test_notes_merge_into_finish_line(tmp_path: Path) -> None:
    with PipelineRun("demo", log_dir=tmp_path) as run:
        run.note(preprocess_version="v1", model="multi-qa-distilbert-cos-v1")
        run.note(model="other")  # later wins

    finish = _read_jsonl(_log_path_for(tmp_path, "demo"))[-1]
    assert finish["notes"] == {"preprocess_version": "v1", "model": "other"}


def test_event_lines_appear_between_start_and_finish(tmp_path: Path) -> None:
    with PipelineRun("demo", log_dir=tmp_path) as run:
        run.event("model_loaded", model="x", device="cpu")
        run.event("checkpoint_saved", path="/tmp/x")

    lines = _read_jsonl(_log_path_for(tmp_path, "demo"))
    assert [line["event"] for line in lines] == [
        "start",
        "model_loaded",
        "checkpoint_saved",
        "finish",
    ]
    assert lines[1] == {**lines[1], "model": "x", "device": "cpu"}


def test_exception_writes_failure_finish_and_propagates(tmp_path: Path) -> None:
    with (
        pytest.raises(ValueError, match="boom"),
        PipelineRun("demo", log_dir=tmp_path) as run,
    ):
        run.processed(2)
        raise ValueError("boom")

    finish = _read_jsonl(_log_path_for(tmp_path, "demo"))[-1]
    assert finish["success"] is False
    assert finish["processed"] == 2
    assert finish["error_type"] == "ValueError"
    assert finish["error_message"] == "boom"
    assert "ValueError: boom" in finish["traceback"]


def test_event_outside_context_raises(tmp_path: Path) -> None:
    run = PipelineRun("demo", log_dir=tmp_path)
    with pytest.raises(RuntimeError, match="outside"):
        run.event("never_emitted")


def test_multiple_runs_same_day_append(tmp_path: Path) -> None:
    with PipelineRun("demo", log_dir=tmp_path) as run:
        run.processed(1)
    with PipelineRun("demo", log_dir=tmp_path) as run:
        run.processed(2)

    lines = _read_jsonl(_log_path_for(tmp_path, "demo"))
    finishes = [line for line in lines if line["event"] == "finish"]
    assert len(finishes) == 2
    assert [f["processed"] for f in finishes] == [1, 2]
