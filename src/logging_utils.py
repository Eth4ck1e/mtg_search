"""Pipeline-run logger.

Every state-mutating script (ingest, embed, build_keyword_dict, ...) wraps
its main body in :class:`PipelineRun`. The context manager writes one
``start`` line, any ad-hoc ``event`` lines the script emits, and one
``finish`` line — all to ``logs/<script_name>/<YYYY-MM-DD>.jsonl``. The
finish line captures duration, counts, skip reasons, and any notes
accumulated during the run.

The JSONL contract: every line is a self-contained JSON object that means
something on its own. ``grep`` a single line and you should still know
which script, which run, and what happened.

Used by callers like::

    with PipelineRun("ingest", inputs={"file": str(path)}) as run:
        for record in stream(path):
            if record["layout"] == "token":
                run.skip("token")
                continue
            insert(record)
            run.processed()
        run.note(preprocess_version=settings.preprocess_version)
"""

from __future__ import annotations

import json
import subprocess
import traceback
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from src.config import settings


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _script_version() -> str:
    """Return git short SHA, suffixed ``-dirty`` if the working tree is dirty.

    Returns ``"unknown"`` if git is unavailable or the repo cannot be located.
    The dirty flag is load-bearing: a metric logged from a dirty tree cannot
    be reproduced from the SHA alone.
    """
    repo_root = settings.repo_root
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{sha}-dirty" if porcelain else sha
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


class PipelineRun:
    """Context manager that writes a JSONL log for one pipeline invocation."""

    def __init__(
        self,
        script_name: str,
        *,
        inputs: dict[str, Any] | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.script_name = script_name
        self.inputs = inputs or {}
        self._log_dir = (log_dir or settings.logs_dir) / script_name
        self._file: TextIO | None = None
        self._start_perf: float | None = None
        self._processed = 0
        self._skipped: Counter[str] = Counter()
        self._notes: dict[str, Any] = {}

    # ----- lifecycle ------------------------------------------------------

    def __enter__(self) -> PipelineRun:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        path = self._log_dir / f"{today}.jsonl"
        self._file = path.open("a", encoding="utf-8")
        import time

        self._start_perf = time.perf_counter()
        self._emit(
            {
                "event": "start",
                "script": self.script_name,
                "version": _script_version(),
                "inputs": self.inputs,
            }
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        import time

        assert self._file is not None and self._start_perf is not None
        duration_s = round(time.perf_counter() - self._start_perf, 6)
        finish: dict[str, Any] = {
            "event": "finish",
            "script": self.script_name,
            "duration_s": duration_s,
            "success": exc is None,
            "processed": self._processed,
            "skipped": dict(self._skipped),
            "notes": self._notes,
        }
        if exc is not None:
            finish["error_type"] = exc_type.__name__ if exc_type else "Unknown"
            finish["error_message"] = str(exc)
            finish["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))[-4000:]
        try:
            self._emit(finish)
        finally:
            self._file.close()
            self._file = None
        # Returning None / False re-raises any in-flight exception, which is
        # what we want — logging the failure does not swallow it.

    # ----- counters & metadata --------------------------------------------

    def processed(self, n: int = 1) -> None:
        """Record that ``n`` records were successfully processed."""
        self._processed += n

    def skip(self, reason: str, n: int = 1) -> None:
        """Record that ``n`` records were skipped, grouped by ``reason``."""
        self._skipped[reason] += n

    def note(self, **kwargs: Any) -> None:
        """Merge key/values into the eventual finish line's ``notes`` field."""
        self._notes.update(kwargs)

    def event(self, name: str, **payload: Any) -> None:
        """Emit one JSONL line immediately. Use for in-run milestones."""
        if self._file is None:
            raise RuntimeError("PipelineRun.event() called outside `with` block")
        self._emit({"event": name, "script": self.script_name, **payload})

    # ----- internals ------------------------------------------------------

    def _emit(self, record: dict[str, Any]) -> None:
        assert self._file is not None
        line = {"ts": _utc_now_iso(), **record}
        self._file.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._file.flush()
