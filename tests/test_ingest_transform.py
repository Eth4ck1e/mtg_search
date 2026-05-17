"""Tests for the ingest transform layer.

Pure dict-to-dict transforms; no DB. Covers the four filter rules and
the per-face row construction for every multi-face layout flagged by
the corpus survey (transform, modal_dfc, split, adventure, flip, meld,
prepare).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.data_processing.ingest_transform import (
    iter_face_rows,
    should_include,
)


# --------------------------------------------------------------------- filter
@pytest.mark.parametrize(
    ("card", "expected"),
    [
        ({"layout": "token"}, "non_card_layout"),
        ({"layout": "emblem"}, "non_card_layout"),
        ({"layout": "art_series"}, "non_card_layout"),
        ({"layout": "normal", "digital": True}, "digital_only"),
        ({"layout": "normal", "border_color": "silver"}, "silver_bordered"),
        ({"layout": "normal", "set_type": "memorabilia"}, "memorabilia"),
        ({"layout": "normal"}, None),
        ({"layout": "transform"}, None),
    ],
)
def test_should_include(card: dict, expected: str | None) -> None:
    assert should_include(card) == expected


def test_filter_rule_priority_order_is_stable() -> None:
    """A card matching multiple rules returns the first one (non_card_layout)."""
    multi = {"layout": "token", "digital": True, "border_color": "silver"}
    assert should_include(multi) == "non_card_layout"


# ---------------------------------------------------------------- single face
def _normal_card() -> dict:
    return {
        "oracle_id": "00000000-0000-0000-0000-000000000001",
        "name": "Test Creature",
        "mana_cost": "{1}{G}",
        "cmc": Decimal("2.0"),
        "colors": ["G"],
        "color_identity": ["G"],
        "type_line": "Creature — Beast",
        "oracle_text": "Trample",
        "keywords": ["Trample"],
        "power": "2",
        "toughness": "3",
        "loyalty": None,
        "layout": "normal",
        "released_at": "2025-01-01",
        "legalities": {"modern": "legal"},
    }


def test_single_face_yields_one_row_with_face_index_zero() -> None:
    rows = list(iter_face_rows(_normal_card()))
    assert len(rows) == 1
    face_index, row = rows[0]
    assert face_index == 0
    assert row["face_index"] == 0
    assert row["oracle_id"] == "00000000-0000-0000-0000-000000000001"
    assert row["name"] == "Test Creature"
    assert row["mana_cost"] == "{1}{G}"
    assert row["cmc"] == Decimal("2.0")
    assert row["colors"] == ["G"]
    assert row["color_identity"] == ["G"]
    assert row["keywords"] == ["Trample"]
    assert row["layout"] == "normal"
    assert row["released_at"] == "2025-01-01"
    assert row["legalities"] == {"modern": "legal"}


def test_empty_oracle_text_becomes_empty_string() -> None:
    card = _normal_card()
    card["oracle_text"] = None
    _, row = next(iter(iter_face_rows(card)))
    assert row["oracle_text"] == ""


def test_missing_mana_cost_stays_none() -> None:
    card = _normal_card()
    card["mana_cost"] = None
    _, row = next(iter(iter_face_rows(card)))
    assert row["mana_cost"] is None


def test_raw_field_contains_full_original_card() -> None:
    card = _normal_card()
    card["extra_field"] = "something"
    _, row = next(iter(iter_face_rows(card)))
    assert row["raw"] is card  # passthrough — no copy
    assert row["raw"]["extra_field"] == "something"


# -------------------------------------------------------------------- transform
def test_transform_card_produces_two_rows_with_back_face_mana_cost_empty() -> None:
    """Werewolf-style transform: back face has no cast cost."""
    card = {
        "oracle_id": "11111111-1111-1111-1111-111111111111",
        "name": "Ulvenwald Captive // Ulvenwald Abomination",
        "cmc": Decimal("2.0"),
        "color_identity": ["G"],
        "keywords": [],
        "layout": "transform",
        "released_at": "2016-04-08",
        "legalities": {"modern": "legal"},
        "card_faces": [
            {
                "name": "Ulvenwald Captive",
                "mana_cost": "{1}{G}",
                "type_line": "Creature — Werewolf Horror",
                "oracle_text": "{6}{G}: Transform.",
                "colors": ["G"],
                "power": "1",
                "toughness": "2",
            },
            {
                "name": "Ulvenwald Abomination",
                "mana_cost": "",
                "type_line": "Creature — Eldrazi Werewolf",
                "oracle_text": "Trample. At the beginning of each upkeep…",
                "colors": [],
                "power": "4",
                "toughness": "6",
            },
        ],
    }
    rows = list(iter_face_rows(card))
    assert [fi for fi, _ in rows] == [0, 1]

    front = rows[0][1]
    back = rows[1][1]

    assert front["name"] == "Ulvenwald Captive"
    assert back["name"] == "Ulvenwald Abomination"

    # Per-face oracle text isolates each face's abilities.
    assert "Transform" in front["oracle_text"]
    assert "Trample" in back["oracle_text"]

    # Card-level fields are duplicated identically across face rows.
    assert front["cmc"] == back["cmc"] == Decimal("2.0")
    assert front["color_identity"] == back["color_identity"] == ["G"]
    assert front["layout"] == back["layout"] == "transform"


# --------------------------------------------------------------------- split
def test_split_card_carries_individual_face_mana_costs() -> None:
    """Fire // Ice: top-level mana_cost is the joined string, faces carry halves."""
    card = {
        "oracle_id": "22222222-2222-2222-2222-222222222222",
        "name": "Fire // Ice",
        "mana_cost": "{1}{R} // {1}{U}",
        "cmc": Decimal("4.0"),
        "color_identity": ["U", "R"],
        "keywords": [],
        "layout": "split",
        "legalities": {},
        "card_faces": [
            {
                "name": "Fire",
                "mana_cost": "{1}{R}",
                "type_line": "Instant",
                "oracle_text": "Fire deals 2 damage divided…",
                "colors": ["R"],
            },
            {
                "name": "Ice",
                "mana_cost": "{1}{U}",
                "type_line": "Instant",
                "oracle_text": "Tap target permanent. Draw a card.",
                "colors": ["U"],
            },
        ],
    }
    rows = list(iter_face_rows(card))
    assert rows[0][1]["mana_cost"] == "{1}{R}"
    assert rows[1][1]["mana_cost"] == "{1}{U}"
    assert rows[0][1]["colors"] == ["R"]
    assert rows[1][1]["colors"] == ["U"]


# ------------------------------------------------------------------ adventure
def test_adventure_card_yields_two_rows_creature_and_instant() -> None:
    """Bonecrusher Giant: face 0 is the creature, face 1 is the adventure."""
    card = {
        "oracle_id": "33333333-3333-3333-3333-333333333333",
        "name": "Bonecrusher Giant // Stomp",
        "cmc": Decimal("3.0"),
        "color_identity": ["R"],
        "keywords": [],
        "layout": "adventure",
        "legalities": {},
        "card_faces": [
            {
                "name": "Bonecrusher Giant",
                "mana_cost": "{2}{R}",
                "type_line": "Creature — Giant",
                "oracle_text": "Whenever Bonecrusher Giant becomes the target…",
                "colors": ["R"],
                "power": "4",
                "toughness": "3",
            },
            {
                "name": "Stomp",
                "mana_cost": "{1}{R}",
                "type_line": "Instant — Adventure",
                "oracle_text": "Stomp deals 2 damage to any target.",
                "colors": ["R"],
            },
        ],
    }
    rows = list(iter_face_rows(card))
    assert len(rows) == 2
    assert rows[0][1]["type_line"] == "Creature — Giant"
    assert rows[1][1]["type_line"] == "Instant — Adventure"
    # Adventure side has no power/toughness; those fields should be None.
    assert rows[1][1]["power"] is None
    assert rows[1][1]["toughness"] is None


# ----------------------------------------------------------- modal_dfc / flip
@pytest.mark.parametrize("layout", ["modal_dfc", "flip", "meld", "prepare"])
def test_other_multi_face_layouts_produce_one_row_per_face(layout: str) -> None:
    card = {
        "oracle_id": f"44444444-4444-4444-4444-{layout[:8]:_<8s}".replace("_", "4"),
        "name": "Front // Back",
        "cmc": Decimal("3.0"),
        "color_identity": ["W"],
        "keywords": [],
        "layout": layout,
        "legalities": {},
        "card_faces": [
            {"name": "Front", "type_line": "X", "colors": ["W"]},
            {"name": "Back", "type_line": "Y", "colors": ["W"]},
        ],
    }
    rows = list(iter_face_rows(card))
    assert [fi for fi, _ in rows] == [0, 1]
    assert rows[0][1]["layout"] == layout
    assert rows[1][1]["layout"] == layout


# --------------------------------------------------------- card-level dup check
def test_card_level_fields_duplicated_across_face_rows() -> None:
    card = {
        "oracle_id": "55555555-5555-5555-5555-555555555555",
        "name": "Two-Faced",
        "cmc": Decimal("5.0"),
        "color_identity": ["U", "R"],
        "keywords": ["Flying", "Haste"],
        "layout": "transform",
        "released_at": "2024-09-13",
        "legalities": {"modern": "legal", "standard": "not_legal"},
        "card_faces": [
            {"name": "A", "type_line": "Creature", "colors": ["U"]},
            {"name": "B", "type_line": "Creature", "colors": ["R"]},
        ],
    }
    rows = list(iter_face_rows(card))
    front_row, back_row = rows[0][1], rows[1][1]
    for col in ("cmc", "color_identity", "keywords", "layout", "released_at", "legalities"):
        assert front_row[col] == back_row[col], f"{col} should be duplicated identically"
