"""Pure predicates for classifying Scryfall card records.

Every function takes one Scryfall ``oracle_cards`` JSON record (a dict)
and returns a fact about it — never mutates, never enforces filter
policy. Filter rules are composed at call sites (the Phase 2 ingest
script) so policy stays explicit and inspectable.

Background: Scryfall's ``oracle_cards`` bulk contains entries that are
not playable cards (tokens, emblems, art-only printings, Vanguard
avatars, schemes/planes for one-off formats). For each, ``oracle_id``
may be absent or the record is structurally distinct from real cards.
The :data:`NON_CARD_LAYOUTS` set names those layouts. Phase 1's survey
counts them; Phase 2's ingest filters by them.

Multi-faced cards (transform, modal_dfc, split, adventure, ...) carry
a ``card_faces`` list. The semantic-search schema stores one row per
face, so :func:`oracle_text_per_face` returns a list — length 1 for
single-faced cards, ≥2 for multi-faced.
"""

from __future__ import annotations

from typing import Any

# Scryfall layouts that are not real, individually-cast cards.
# Phase 2 ingest excludes records whose layout is in this set.
NON_CARD_LAYOUTS: frozenset[str] = frozenset(
    {
        "token",
        "double_faced_token",
        "emblem",
        "art_series",
        "vanguard",
        "planar",
        "scheme",
    }
)


def card_layout(card: dict[str, Any]) -> str:
    """Return the ``layout`` field, or ``"unknown"`` if missing."""
    return str(card.get("layout") or "unknown")


def is_non_card_layout(card: dict[str, Any]) -> bool:
    """True if the record's layout marks it as a non-card (token, emblem, ...)."""
    return card_layout(card) in NON_CARD_LAYOUTS


def is_digital_only(card: dict[str, Any]) -> bool:
    """True if the printing exists only on Arena/MTGO (``digital: true``)."""
    return bool(card.get("digital", False))


def is_silver_bordered(card: dict[str, Any]) -> bool:
    """True if ``border_color == "silver"`` (un-set / joke cards)."""
    return card.get("border_color") == "silver"


def set_type(card: dict[str, Any]) -> str:
    """Return ``set_type`` or ``"unknown"`` if missing."""
    return str(card.get("set_type") or "unknown")


def border_color(card: dict[str, Any]) -> str:
    """Return ``border_color`` or ``"unknown"`` if missing."""
    return str(card.get("border_color") or "unknown")


def has_multiple_faces(card: dict[str, Any]) -> bool:
    """True if ``card_faces`` is present with two or more entries."""
    faces = card.get("card_faces")
    return bool(faces) and len(faces) > 1


def oracle_text_per_face(card: dict[str, Any]) -> list[str]:
    """Return oracle text for each face.

    Length 1 for single-faced cards (the top-level ``oracle_text``).
    Length ≥2 for multi-faced cards (one entry per ``card_faces`` element,
    falling back to ``""`` if a face is missing the field).
    """
    faces = card.get("card_faces")
    if faces and len(faces) > 1:
        return [str(f.get("oracle_text") or "") for f in faces]
    return [str(card.get("oracle_text") or "")]


def keywords(card: dict[str, Any]) -> list[str]:
    """Return the ``keywords`` list (Scryfall-parsed), or ``[]`` if missing."""
    kws = card.get("keywords")
    return list(kws) if kws else []
