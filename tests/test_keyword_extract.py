"""Tests for the reminder-text extractor.

Load-bearing: a wrong extraction puts a wrong definition into every
embedded card that uses the keyword. The Skyhunter Patrol test is the
one that proves the regex correctly rejects joint-keyword
parentheticals.
"""

from __future__ import annotations

from src.data_processing.keyword_extract import extract_reminder_texts

STORM_CROW = "Flying (This creature can't be blocked except by creatures with flying or reach.)"

# Real Skyhunter Patrol text — the parenthetical describes BOTH keywords jointly.
SKYHUNTER = (
    "Flying, first strike (This creature can't be blocked except by creatures "
    "with flying or reach, and it deals combat damage before creatures without "
    "first strike.)"
)


def test_single_keyword_with_inline_reminder_text() -> None:
    """The clean case — keyword starts the line, parenthetical follows."""
    result = extract_reminder_texts(STORM_CROW, "Flying")
    assert len(result) == 1
    assert result[0].startswith("This creature can't be blocked")


def test_keyword_not_present_returns_empty() -> None:
    assert extract_reminder_texts("Vigilance", "Flying") == []


def test_keyword_present_but_no_parenthetical_returns_empty() -> None:
    assert extract_reminder_texts("Trample", "Trample") == []


def test_joint_keyword_parenthetical_attributed_to_neither() -> None:
    """Skyhunter's text — neither Flying nor First strike should extract."""
    # "Flying" is followed by ", first strike" before the paren — no match.
    assert extract_reminder_texts(SKYHUNTER, "Flying") == []
    # "first strike" sits mid-clause after a comma — no match.
    assert extract_reminder_texts(SKYHUNTER, "First strike") == []


def test_word_boundary_avoids_substring_matches() -> None:
    """Searching for 'Flash' must not match 'Flashback (...)' — different keywords."""
    text = "Flashback {2}{B} (You may cast this from your graveyard for the flashback cost.)"
    assert extract_reminder_texts(text, "Flash") == []
    # The full keyword still extracts cleanly.
    result = extract_reminder_texts(text, "Flashback")
    assert len(result) == 1
    assert "graveyard" in result[0]


def test_multi_word_keyword_extracts() -> None:
    text = "First strike (This creature deals combat damage before creatures without first strike.)"
    result = extract_reminder_texts(text, "First strike")
    assert len(result) == 1
    assert "before creatures" in result[0]


def test_keyword_on_second_line_extracts() -> None:
    """Multi-ability cards have keywords on different lines — each with its own paren."""
    text = (
        "Flying (This creature can't be blocked except by creatures with flying or reach.)\n"
        "Trample (This creature can deal excess combat damage to the player or "
        "planeswalker it's attacking.)"
    )
    flying = extract_reminder_texts(text, "Flying")
    trample = extract_reminder_texts(text, "Trample")
    assert len(flying) == 1
    assert len(trample) == 1
    assert "flying or reach" in flying[0]
    assert "excess combat damage" in trample[0]


def test_case_insensitive_match() -> None:
    """Some cards lowercase keywords mid-text; we still extract from line-start headers."""
    text = "vigilance (Attacking doesn't cause this creature to tap.)"
    result = extract_reminder_texts(text, "Vigilance")
    assert len(result) == 1
    assert "Attacking doesn't" in result[0]


def test_keyword_mid_sentence_with_parenthetical_does_not_match() -> None:
    """The anchor `(?:^|\\n)` prevents mid-sentence false matches."""
    text = "When you have flying (parenthetical), do something."
    assert extract_reminder_texts(text, "Flying") == []


def test_empty_inputs_return_empty() -> None:
    assert extract_reminder_texts("", "Flying") == []
    assert extract_reminder_texts("Flying (...)", "") == []
