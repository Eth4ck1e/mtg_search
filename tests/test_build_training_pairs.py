"""Pure-function tests for the training-pair builder (no DB)."""

from __future__ import annotations

import random

from scripts.build_training_pairs import Tag, _bucket, _emit_pairs, _split_holdout


def _tag(slug: str, n: int, **kw) -> Tag:
    return Tag(
        id=f"id-{slug}",
        slug=slug,
        label=kw.get("label", slug),
        description=kw.get("description"),
        aliases=kw.get("aliases", []),
        cards=[f"card-{slug}-{i}" for i in range(n)],
    )


def test_anchors_dedupe_label_and_aliases_case_insensitively() -> None:
    t = _tag(
        "sweeper",
        5,
        label="sweeper",
        aliases=["Sweeper", "board wipe", " wrath "],
        description="Kills everything.",
    )
    assert t.anchors() == [
        ("label", "sweeper"),
        ("alias", "board wipe"),
        ("alias", "wrath"),
        ("description", "Kills everything."),
    ]


def test_bucket_edges() -> None:
    assert _bucket(5) == 0
    assert _bucket(19) == 0
    assert _bucket(20) == 1
    assert _bucket(500) == 3
    assert _bucket(10_000) == 3


def test_holdout_is_stratified_and_seeded() -> None:
    tags = [_tag(f"s{i}", 6) for i in range(20)] + [_tag(f"b{i}", 600) for i in range(20)]
    held_a = _split_holdout(tags, 0.25, random.Random(1))
    held_b = _split_holdout(tags, 0.25, random.Random(1))
    assert held_a == held_b
    small = sum(1 for h in held_a if h.startswith("id-s"))
    big = sum(1 for h in held_a if h.startswith("id-b"))
    assert small == 5 and big == 5


def test_emit_pairs_caps_per_anchor_and_skips_textless_cards() -> None:
    t = _tag("ramp", 10, aliases=["mana ramp"])
    faces = {c: [(0, c, f"text for {c}")] for c in t.cards[:8]}  # two cards lack text
    pairs = _emit_pairs(t, faces, max_per_anchor=3, rng=random.Random(0))
    assert len(pairs) == 6  # 2 anchors x 3 sampled cards
    assert {p["anchor"] for p in pairs} == {"ramp", "mana ramp"}
    assert all(p["oracle_id"] in faces for p in pairs)
    assert {p["tag_id"] for p in pairs} == {"id-ramp"}
