"""Extract canonical reminder text for MTG keywords from oracle text.

A *reminder text* is a parenthetical Wizards prints to explain an
ability keyword the first time you see it. The reminder for ``Flying``
is the same on every card it appears on with a reminder; somewhere
across all printings of every keyword, the canonical wording exists.
The corpus survey ([[2026-05-17-corpus-survey]]) is what tells us this
is a viable strategy — keywords are common, parentheticals are
common, and Scryfall already canonicalises keyword spelling in the
``keywords`` field.

This module is a single pure function. The caller (the build script)
iterates the corpus, calls this once per (oracle_text, keyword) pair,
and aggregates candidates across cards. "Most recent printing wins"
is a tiebreak the caller applies — this function doesn't see dates.

Edge case the regex deliberately avoids: joint-keyword parentheticals
like ``"Flying, first strike (one combined explanation)"``. Anchoring
to start-of-line/start-of-string means ``"first strike (...)"`` only
extracts when ``first strike`` is the lead clause, not when it sits
after a comma — so we never mis-attribute a joint reminder to one of
its keywords.
"""

from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=512)
def _pattern_for(keyword: str) -> re.Pattern[str]:
    # Anchored to start-of-string or newline so mid-clause occurrences don't
    # match (handles the joint-keyword parenthetical case like
    # "Flying, first strike (joint reminder)" — neither extracts because the
    # comma is in the exclusion set below).
    #
    # Between the keyword and `(` we allow anything except `(`, newline, and
    # the punctuation that starts a new clause (comma, semicolon, period).
    # This makes the regex tolerant of cost-keyword shapes like
    # "Flashback {2}{B} (...)" or "Cycling {2} (...)" where a mana cost sits
    # between the keyword name and the parenthetical.
    return re.compile(
        r"(?:^|\n)" + re.escape(keyword) + r"(?!\w)[^(\n,;.]*\(([^)]+)\)",
        re.IGNORECASE,
    )


def extract_reminder_texts(oracle_text: str, keyword: str) -> list[str]:
    """Return reminder text strings where ``keyword`` heads a parenthetical clause.

    Returns ``[]`` if the keyword either does not appear in ``oracle_text``
    or appears only in positions where the parenthetical cannot be
    cleanly attributed (mid-line, joint with other keywords, etc.).

    Stripped of leading/trailing whitespace; the surrounding parentheses
    are removed.
    """
    if not oracle_text or not keyword:
        return []
    return [m.group(1).strip() for m in _pattern_for(keyword).finditer(oracle_text)]
