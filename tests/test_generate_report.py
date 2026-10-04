"""Pure-function tests for the report generator (no DB)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from scripts.generate_report import BASE_MODEL, embedder_label, hbar_chart, md_table, select


def _row(id_: int, config: dict, metrics: dict | None = None) -> dict:
    return {
        "id": id_,
        "created_at": datetime(2026, 10, 3, tzinfo=UTC),
        "eval_set_version": "v1-draft",
        "config": config,
        "metrics": metrics or {},
        "per_query": [],
        "notes": None,
    }


def test_select_newest_row_wins_and_superseded_are_tracked() -> None:
    rows = [
        _row(1, {"config_name": "a", "embedding_model": BASE_MODEL}),
        _row(2, {"config_name": "a", "embedding_model": "models/tuned"}),
        _row(3, {"config_name": "a", "embedding_model": BASE_MODEL}),
        _row(4, {"kind": "finetune_embedder", "run_name": "tuned"}),
        _row(5, {"kind": "scryfall_comparator", "cascade_row": 3, "variant": "expert"}),
        _row(6, {"kind": "scryfall_comparator", "cascade_row": "3", "variant": "no_tag"}),
    ]
    evals, superseded, finetunes, comparators = select(rows)
    assert evals[("a", BASE_MODEL)]["id"] == 3
    assert evals[("a", "models/tuned")]["id"] == 2
    assert superseded[("a", BASE_MODEL)] == [1]
    assert [r["id"] for r in finetunes] == [4]
    assert set(comparators[3]) == {"expert", "no_tag"}  # str and int cascade_row both join


def test_embedder_label() -> None:
    assert embedder_label(BASE_MODEL) == "base"
    assert embedder_label("models/nomic-mtg-v1") == "tuned (nomic-mtg-v1)"
    assert embedder_label(None) == "—"


def test_md_table_shape() -> None:
    out = md_table(["a", "b"], [[1, 2], [3, 4]]).splitlines()
    assert out[0] == "| a | b |" and out[1] == "|---|---|" and len(out) == 4


def test_hbar_chart_is_well_formed_and_escapes_text() -> None:
    svg = hbar_chart(
        title="A & B <test>",
        subtitle="sub",
        groups=[
            ("row <1>", [("s1", 0.5, 'tip "x" & y'), ("s2", None, "not run")]),
            ("row 2", [("s1", 0.0, "zero"), ("s2", 1.0, "full")]),
        ],
        series=[("s1", "base"), ("s2", "tuned")],
    )
    root = ET.fromstring(svg)  # raises on malformed XML / unescaped text
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}path")) == 3  # the None bar is replaced by a 'not run' label
    assert any(t.text == "not run" for t in root.iter(f"{ns}text"))
    assert root.findall(f"{ns}path")[0].get("fill") == "#2a78d6"  # light colour as an attribute
