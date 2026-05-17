"""Tests for build_embedding_text.

The four cases from Phase 2 roadmap sub-task 22 plus a few edge cases.
A wrong augmentation here corrupts every embedding it touches, so the
tests are deliberately strict about exact output strings.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.preprocess_text import build_embedding_text, load_keyword_dict

# ----------------------------------------------- build_embedding_text


def test_passthrough_with_no_keywords() -> None:
    assert build_embedding_text("Some text.", [], {}) == "Some text."


def test_passthrough_with_empty_keyword_list() -> None:
    assert build_embedding_text("Some text.", [], {"Flying": "..."}) == "Some text."


def test_keyword_not_in_dict_is_skipped() -> None:
    out = build_embedding_text("Some text.", ["UnknownAbility"], {"Flying": "F"})
    assert out == "Some text."


def test_keyword_in_dict_is_appended_when_no_inline_reminder() -> None:
    out = build_embedding_text(
        "Flying",
        ["Flying"],
        {"Flying": "This creature can't be blocked except by creatures with flying or reach."},
    )
    assert out == (
        "Flying\nFlying (This creature can't be blocked except by creatures with flying or reach.)"
    )


def test_keyword_with_inline_reminder_not_duplicated() -> None:
    """If the card already explains the keyword, leave it alone."""
    inline = "Flying (This creature can't be blocked except by creatures with flying or reach.)"
    out = build_embedding_text(inline, ["Flying"], {"Flying": "OTHER WORDING"})
    assert out == inline


def test_multiple_keywords_each_get_appended() -> None:
    out = build_embedding_text(
        "Flying\nTrample",
        ["Flying", "Trample"],
        {"Flying": "F-reminder.", "Trample": "T-reminder."},
    )
    assert "Flying (F-reminder.)" in out
    assert "Trample (T-reminder.)" in out
    # Original text is preserved at the start.
    assert out.startswith("Flying\nTrample\n")


def test_mixed_some_inline_some_appended() -> None:
    """If one keyword has its reminder inline and another doesn't, only augment the missing one."""
    text = (
        "Flying (This creature can't be blocked except by creatures with flying or reach.)\nTrample"
    )
    out = build_embedding_text(
        text,
        ["Flying", "Trample"],
        {"Flying": "F-dict", "Trample": "T-dict"},
    )
    # Flying already explained — no duplicate.
    assert out.count("Flying") == 1
    # Trample got augmented.
    assert "Trample (T-dict)" in out


def test_empty_oracle_text_with_keyword_returns_just_augmentation() -> None:
    out = build_embedding_text("", ["Flying"], {"Flying": "F"})
    assert out == "Flying (F)"


def test_none_oracle_text_treated_as_empty() -> None:
    out = build_embedding_text(None, ["Flying"], {"Flying": "F"})  # type: ignore[arg-type]
    assert out == "Flying (F)"


def test_none_keywords_treated_as_empty() -> None:
    assert build_embedding_text("Some text.", None, {"Flying": "F"}) == "Some text."  # type: ignore[arg-type]


def test_inline_check_handles_cost_keywords() -> None:
    """A card with 'Flashback {2}{B} (reminder)' should not get Flashback re-appended."""
    text = "Flashback {2}{B} (You may cast this from your graveyard for the flashback cost.)"
    out = build_embedding_text(text, ["Flashback"], {"Flashback": "OTHER"})
    assert out == text


# ----------------------------------------------- load_keyword_dict


def _write_keyword_files(dir_: Path, reminder: dict, overrides: dict) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "reminder_text.json").write_text(
        json.dumps({"version": "test", "keywords": reminder}), encoding="utf-8"
    )
    (dir_ / "manual_overrides.json").write_text(
        json.dumps({"version": "test", "keywords": overrides}), encoding="utf-8"
    )


def test_load_keyword_dict_merges_with_overrides_winning(tmp_path: Path) -> None:
    _write_keyword_files(
        tmp_path,
        reminder={"Flying": "auto-extracted", "Trample": "auto-T"},
        overrides={"Flying": "hand-curated"},
    )
    merged = load_keyword_dict(keywords_dir=tmp_path)
    assert merged["Flying"] == "hand-curated"
    assert merged["Trample"] == "auto-T"


def test_load_keyword_dict_handles_empty_overrides(tmp_path: Path) -> None:
    _write_keyword_files(tmp_path, reminder={"Flying": "F"}, overrides={})
    merged = load_keyword_dict(keywords_dir=tmp_path)
    assert merged == {"Flying": "F"}
