"""Tests for Stage 2 filter compilation (src.search.build_where).

Pure-function tests: no DB, no model. These guard the semantics decided in
the 2026-09-18 test series — colourless allowance, keywords off by default,
whitelisted operators, loud failure on rewriter hallucinations.
"""

from __future__ import annotations

import pytest

from src.query_rewriter import HyDEFilters
from src.search import FilterError, FilterPolicy, build_where


def test_none_filters_yield_no_clauses() -> None:
    assert build_where(None) == ([], {})


def test_empty_filters_yield_no_clauses() -> None:
    assert build_where(HyDEFilters()) == ([], {})


def test_contains_any_admits_colourless() -> None:
    clauses, params = build_where(HyDEFilters(colors=["R"]))
    assert clauses == ["(colors && %(colors_v)s OR cardinality(colors) = 0)"]
    assert params["colors_v"] == ["R"]


def test_exactly_excludes_colourless() -> None:
    clauses, _ = build_where(HyDEFilters(colors=["R"], colors_op="exactly"))
    assert "cardinality" not in clauses[0]
    assert "<@" in clauses[0] and "@>" in clauses[0]


def test_colours_normalised_and_deduped() -> None:
    _, params = build_where(HyDEFilters(colors=["r", "R", " u "]))
    assert params["colors_v"] == ["R", "U"]


def test_invalid_colour_symbol_raises() -> None:
    with pytest.raises(FilterError):
        build_where(HyDEFilters(colors=["red"]))


def test_empty_colour_list_means_colourless() -> None:
    clauses, _ = build_where(HyDEFilters(colors=[]))
    assert clauses == ["cardinality(colors) = 0"]


def test_identity_mirrors_colour_op() -> None:
    clauses, params = build_where(
        HyDEFilters(colors=["W", "U"], color_identity=["W", "U"], colors_op="subset_of")
    )
    assert clauses == ["colors <@ %(colors_v)s", "color_identity <@ %(color_identity_v)s"]
    assert params["color_identity_v"] == ["U", "W"]


def test_types_and_subtypes_are_whole_word_regexes() -> None:
    clauses, params = build_where(HyDEFilters(types=["Creature"], subtypes=["Elf"]))
    assert clauses == ["type_line ~* %(type_0)s", "type_line ~* %(subtype_0)s"]
    assert params["type_0"] == r"\mCreature\M"
    assert params["subtype_0"] == r"\mElf\M"


def test_multiple_types_are_ored_and_subtypes_anded() -> None:
    """'instants or sorceries' must not AND to an empty set (smoke-test regression)."""
    clauses, _ = build_where(HyDEFilters(types=["Instant", "Sorcery"], subtypes=["Elf", "Warrior"]))
    assert clauses == [
        "(type_line ~* %(type_0)s OR type_line ~* %(type_1)s)",
        "(type_line ~* %(subtype_0)s AND type_line ~* %(subtype_1)s)",
    ]


def test_cmc_between() -> None:
    clauses, params = build_where(HyDEFilters(cmc={"op": "between", "value": [1, 3]}))
    assert clauses == ["(cmc BETWEEN %(cmc_lo)s AND %(cmc_hi)s)"]
    assert (params["cmc_lo"], params["cmc_hi"]) == (1.0, 3.0)


def test_cmc_bad_op_raises() -> None:
    with pytest.raises(FilterError):
        build_where(HyDEFilters(cmc={"op": "DROP TABLE", "value": 1}))


def test_power_guards_non_numeric_text() -> None:
    clauses, params = build_where(HyDEFilters(power={"op": ">=", "value": 4}))
    assert clauses == ["(power ~ '^-?[0-9]+$' AND power::numeric >= %(power_v)s)"]
    assert params["power_v"] == 4.0


def test_keywords_off_by_default() -> None:
    clauses, _ = build_where(HyDEFilters(keywords=["Flying"]))
    assert clauses == []


def test_keywords_on_when_policy_enables() -> None:
    clauses, params = build_where(
        HyDEFilters(keywords=["Flying", "Trample"]), FilterPolicy(keywords=True)
    )
    assert clauses == ["keywords @> %(keywords_v)s"]
    assert params["keywords_v"] == ["Flying", "Trample"]


def test_format_legality() -> None:
    clauses, params = build_where(
        HyDEFilters(format_legality={"format": "commander", "status": "legal"})
    )
    assert clauses == ["legalities->>%(legal_fmt)s = %(legal_status)s"]
    assert (params["legal_fmt"], params["legal_status"]) == ("commander", "legal")


def test_unknown_format_raises() -> None:
    with pytest.raises(FilterError):
        build_where(HyDEFilters(format_legality={"format": "kitchen_table", "status": "legal"}))


def test_no_user_values_interpolated_into_sql_text() -> None:
    """Every user-supplied value must travel as a bound parameter."""
    f = HyDEFilters(
        colors=["R"],
        types=["Creature'; --"],
        subtypes=["Elf"],
        cmc={"op": "<=", "value": 2},
        format_legality={"format": "modern", "status": "legal"},
    )
    clauses, params = build_where(f)
    joined = " AND ".join(clauses)
    assert "Creature" not in joined
    assert "modern" not in joined
    assert all("%(" in c for c in clauses)
    assert set(params) == {"colors_v", "type_0", "subtype_0", "cmc_v", "legal_fmt", "legal_status"}
