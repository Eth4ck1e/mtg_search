"""Transform Scryfall records into rows for the ``cards`` table.

Two pure functions, no I/O, no DB:

* :func:`should_include` — applies the four Phase 1 filter rules and
  returns the exclusion reason if any fires, else ``None``. Reasons are
  short stable strings used directly as ``PipelineRun.skip()`` labels so
  the ingest log's per-rule counts can be diffed against the corpus
  survey.
* :func:`iter_face_rows` — yields ``(face_index, row_dict)`` tuples, one
  per face. The row dict keys match the columns of ``0002_cards.sql``.
  Card-level fields (``cmc``, ``color_identity``, ``keywords``, ...)
  are duplicated across face rows because Scryfall does not expose them
  per-face; see [[2026-05-17-cards-schema]] for why.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

from src.data_processing.scryfall_classify import (
    has_multiple_faces,
    is_digital_only,
    is_non_card_layout,
    is_silver_bordered,
)


def should_include(card: dict[str, Any]) -> str | None:
    """Return the exclusion reason if the card is filtered out, else None.

    Reasons (stable strings used as PipelineRun.skip() labels):
        ``"non_card_layout"``, ``"digital_only"``, ``"silver_bordered"``,
        ``"memorabilia"``.
    """
    if is_non_card_layout(card):
        return "non_card_layout"
    if is_digital_only(card):
        return "digital_only"
    if is_silver_bordered(card):
        return "silver_bordered"
    if card.get("set_type") == "memorabilia":
        return "memorabilia"
    return None


def _card_level_fields(card: dict[str, Any]) -> dict[str, Any]:
    """Fields that are identical across all face rows of one oracle_id."""
    return {
        "cmc": card.get("cmc") if card.get("cmc") is not None else Decimal("0"),
        "color_identity": list(card.get("color_identity") or []),
        "keywords": list(card.get("keywords") or []),
        "layout": card["layout"],
        "released_at": card.get("released_at"),
        "legalities": card.get("legalities") or {},
        "raw": card,
    }


def _per_face_fields(face: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": face["name"],
        "mana_cost": face.get("mana_cost"),
        "colors": list(face.get("colors") or []),
        "type_line": face.get("type_line", ""),
        "oracle_text": face.get("oracle_text") or "",
        "power": face.get("power"),
        "toughness": face.get("toughness"),
        "loyalty": face.get("loyalty"),
    }


def iter_face_rows(card: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(face_index, row_dict)`` tuples, one per face.

    Single-faced cards yield exactly one tuple with ``face_index = 0``.
    Multi-faced cards yield one tuple per entry in ``card_faces``.
    """
    oracle_id = card["oracle_id"]
    card_level = _card_level_fields(card)

    if has_multiple_faces(card):
        for face_index, face in enumerate(card["card_faces"]):
            per_face = _per_face_fields(face)
            # Some multi-face layouts (split) have no per-face type_line —
            # fall back to the top-level value so the NOT NULL column holds.
            if not per_face["type_line"]:
                per_face["type_line"] = card.get("type_line", "")
            yield (
                face_index,
                {
                    "oracle_id": oracle_id,
                    "face_index": face_index,
                    **per_face,
                    **card_level,
                },
            )
        return

    yield (
        0,
        {
            "oracle_id": oracle_id,
            "face_index": 0,
            "name": card["name"],
            "mana_cost": card.get("mana_cost"),
            "colors": list(card.get("colors") or []),
            "type_line": card.get("type_line", ""),
            "oracle_text": card.get("oracle_text") or "",
            "power": card.get("power"),
            "toughness": card.get("toughness"),
            "loyalty": card.get("loyalty"),
            **card_level,
        },
    )
