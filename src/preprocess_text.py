"""Build the canonical text that gets fed to the embedding encoder.

The embedding text for a card row is its oracle text augmented with
canonical reminder text for any keyword the card has but doesn't
already explain inline. This is the lever that addresses the
query-document asymmetry the POC retrospective identified: an informal
query like "creature that can't be blocked except by flyers" will
embed closer to *Storm Crow*'s augmented text than to bare "Flying".

The function is pure — same inputs always produce the same string.
Phase 3's ``embed.py`` is responsible for stamping the output's hash
into the ``embedding_text_hash`` column and the model + preprocessing
version into ``embedding_version``. The function does not know its own
version; ``settings.preprocess_version`` is what bumps when the logic
here changes meaningfully.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import settings
from src.data_processing.keyword_extract import extract_reminder_texts


def build_embedding_text(
    oracle_text: str,
    keywords: list[str],
    keyword_dict: dict[str, str],
) -> str:
    """Return ``oracle_text`` with missing-keyword reminder text appended.

    For each keyword in ``keywords``:
      * If it has no entry in ``keyword_dict``, skip — nothing to add.
      * If the card's ``oracle_text`` already has an inline parenthetical
        for this keyword, skip — don't duplicate the explanation.
      * Otherwise append ``"<Keyword> (<reminder>)"`` on its own line.

    The empty-oracle-text-with-keywords case (rare but possible) returns
    just the augmentations, without a leading blank line.
    """
    oracle_text = oracle_text or ""
    augmentations: list[str] = []
    for kw in keywords or []:
        reminder = keyword_dict.get(kw)
        if not reminder:
            continue
        if extract_reminder_texts(oracle_text, kw):
            continue
        augmentations.append(f"{kw} ({reminder})")
    if not augmentations:
        return oracle_text
    if not oracle_text:
        return "\n".join(augmentations)
    return oracle_text + "\n" + "\n".join(augmentations)


def load_keyword_dict(*, keywords_dir: Path | None = None) -> dict[str, str]:
    """Load and merge the auto-extracted dict with manual overrides.

    Manual overrides take precedence — a hand-curated entry replaces
    the auto-extracted one for the same keyword. The combined mapping
    is what callers pass to :func:`build_embedding_text`.
    """
    keywords_dir = keywords_dir or settings.keywords_dir
    reminder_data = json.loads((keywords_dir / "reminder_text.json").read_text(encoding="utf-8"))
    overrides_data = json.loads(
        (keywords_dir / "manual_overrides.json").read_text(encoding="utf-8")
    )
    merged: dict[str, str] = {
        **reminder_data.get("keywords", {}),
        **overrides_data.get("keywords", {}),
    }
    return merged
