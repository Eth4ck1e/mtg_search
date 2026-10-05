"""Quote verification is the guard against invented citations — test the guard."""

from __future__ import annotations

from scripts.index_sources import MAX_WORDS, chunk_page
from src.source_library import match_quote, normalize

PAGES = [
    (
        "a.pdf",
        1,
        "smith2020a",
        "Dense retrieval has been shown effec-\ntive across tasks.\nThe \ufb01rst step is un-\nsupervised.",
    ),
    (
        "a.pdf",
        2,
        "smith2020a",
        "It ends on the next\npage of the paper. Training for multiple epochs hurts performance.",
    ),
    ("b.pdf", 1, "jones2021b", "A different paper says something else entirely about rerankers."),
]


def test_normalize_repairs_extraction_artifacts_only() -> None:
    assert normalize("effec-\ntive  across\n tasks") == "effective across tasks"
    assert normalize("the \ufb01rst \u201cstep\u201d") == 'the first "step"'
    # A changed word is still a different string.
    assert normalize("hurts performance") != normalize("hurt performance")


def test_true_quote_verifies_with_page_and_key() -> None:
    r = match_quote("Dense retrieval has been shown effective across tasks.", PAGES)
    assert (r.found, r.source_key, r.page) == (True, "smith2020a", 1)


def test_quote_spanning_a_page_break_reports_start_page() -> None:
    r = match_quote("The first step is unsupervised. It ends on the next page", PAGES)
    assert (r.found, r.page) == (True, 1)


def test_altered_word_and_paraphrase_do_not_verify() -> None:
    assert not match_quote("Training for multiple epochs helps performance", PAGES).found
    assert not match_quote("Dense retrieval works well on many tasks", PAGES).found


def test_right_words_wrong_source_does_not_verify() -> None:
    only_b = [row for row in PAGES if row[2] == "jones2021b"]
    r = match_quote(
        "Training for multiple epochs hurts performance", only_b, source_key="jones2021b"
    )
    assert not r.found and "NOT FOUND" in r.note


def test_hyphenation_difference_verifies_but_is_flagged() -> None:
    r = match_quote(
        "The first step is unsupervised.",
        PAGES,
    )
    assert r.found and "Verbatim" in r.note
    r = match_quote("The first step is un-supervised.", PAGES)
    assert r.found and "hyphenation" in r.note


def test_short_fragments_are_refused() -> None:
    assert not match_quote("the first", PAGES).found


def test_chunks_are_bounded_and_drop_page_furniture() -> None:
    page = (
        "12\n\n"
        + " ".join(["word"] * 700)
        + "\n\nA short closing paragraph with enough words to keep it."
    )
    chunks = chunk_page(page)
    assert chunks and all(len(c.split()) <= MAX_WORDS for c in chunks)
    assert "12" not in [c.strip() for c in chunks]
