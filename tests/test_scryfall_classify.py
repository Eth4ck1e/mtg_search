"""Tests for scryfall_classify predicates.

Load-bearing: these predicates govern which cards Phase 2's ingest will
include. A wrong predicate silently corrupts the entire corpus.
"""

from __future__ import annotations

import pytest

from src.data_processing.scryfall_classify import (
    NON_CARD_LAYOUTS,
    card_layout,
    has_multiple_faces,
    is_digital_only,
    is_non_card_layout,
    is_silver_bordered,
    keywords,
    oracle_text_per_face,
)


@pytest.mark.parametrize(
    "layout",
    ["token", "double_faced_token", "emblem", "art_series", "vanguard", "planar", "scheme"],
)
def test_non_card_layouts_are_flagged(layout: str) -> None:
    assert is_non_card_layout({"layout": layout}) is True


@pytest.mark.parametrize(
    "layout",
    ["normal", "split", "flip", "transform", "modal_dfc", "meld", "adventure", "saga"],
)
def test_real_card_layouts_are_not_flagged(layout: str) -> None:
    assert is_non_card_layout({"layout": layout}) is False


def test_missing_layout_defaults_to_unknown_and_is_not_flagged() -> None:
    assert card_layout({}) == "unknown"
    assert is_non_card_layout({}) is False


def test_digital_only_flag() -> None:
    assert is_digital_only({"digital": True}) is True
    assert is_digital_only({"digital": False}) is False
    assert is_digital_only({}) is False  # missing key = not flagged


def test_silver_bordered_flag() -> None:
    assert is_silver_bordered({"border_color": "silver"}) is True
    assert is_silver_bordered({"border_color": "black"}) is False
    assert is_silver_bordered({}) is False


def test_oracle_text_single_face() -> None:
    card = {"oracle_text": "Flying. {T}: Add {W}."}
    assert oracle_text_per_face(card) == ["Flying. {T}: Add {W}."]


def test_oracle_text_multi_face_returns_one_per_face() -> None:
    card = {
        "card_faces": [
            {"oracle_text": "Front face text."},
            {"oracle_text": "Back face text."},
        ],
    }
    assert oracle_text_per_face(card) == ["Front face text.", "Back face text."]


def test_oracle_text_missing_field_returns_empty_string() -> None:
    assert oracle_text_per_face({}) == [""]
    assert oracle_text_per_face({"oracle_text": None}) == [""]


def test_oracle_text_face_with_missing_text() -> None:
    card = {
        "card_faces": [
            {"oracle_text": "Has text."},
            {"name": "Faceless side"},
        ],
    }
    assert oracle_text_per_face(card) == ["Has text.", ""]


def test_single_element_card_faces_still_counts_as_single_face() -> None:
    """Defensive: if Scryfall ever produces a one-element card_faces, treat as single."""
    card = {
        "oracle_text": "Top-level text.",
        "card_faces": [{"oracle_text": "Only face."}],
    }
    assert has_multiple_faces(card) is False
    assert oracle_text_per_face(card) == ["Top-level text."]


def test_has_multiple_faces() -> None:
    assert has_multiple_faces({"card_faces": [{}, {}]}) is True
    assert has_multiple_faces({"card_faces": [{}]}) is False
    assert has_multiple_faces({"card_faces": []}) is False
    assert has_multiple_faces({}) is False


def test_keywords_helper() -> None:
    assert keywords({"keywords": ["Flying", "Vigilance"]}) == ["Flying", "Vigilance"]
    assert keywords({"keywords": []}) == []
    assert keywords({}) == []


def test_non_card_layouts_constant_is_immutable() -> None:
    """NON_CARD_LAYOUTS is a frozenset — guards against accidental in-place mutation."""
    assert isinstance(NON_CARD_LAYOUTS, frozenset)
