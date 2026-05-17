"""Characterize the Scryfall oracle-cards corpus.

Streams the bulk JSON, aggregates counts and length distributions, and
writes:

* ``data/processed/corpus-survey-<UTC-date>.json`` — full structured
  findings (gitignored, regenerable from the input file).
* A human-readable summary to stdout — meant to be eyeballed and pasted
  (or paraphrased) into the journal entry that decides Phase 2's
  ingest-filter rules.

Reports facts, does not enforce policy. The journal entry is where the
human decides what to filter; this script just surfaces the
distributions.

Usage::

    python scripts/survey_corpus.py
    python scripts/survey_corpus.py --input data/raw/oracle-cards-2026-05-17.json
    python scripts/survey_corpus.py --no-tokenize     # skip the embedding-tokenizer step
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ijson
import numpy as np
from tqdm import tqdm

from src.config import settings
from src.data_processing.scryfall_classify import (
    border_color,
    card_layout,
    has_multiple_faces,
    is_digital_only,
    is_non_card_layout,
    is_silver_bordered,
    keywords,
    oracle_text_per_face,
    set_type,
)
from src.logging_utils import PipelineRun

TOKENIZE_BATCH = 256
TRUNCATION_THRESHOLD = settings.max_length  # tokens above this would be truncated at embed time


def _find_latest_bulk(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("oracle-cards-*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No oracle-cards-*.json in {raw_dir}. Run scripts/download_scryfall.py first."
        )
    return candidates[-1]


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.array(values)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "mean": round(float(arr.mean()), 2),
    }


def survey(card_iter: Iterable[dict[str, Any]], *, with_tokenization: bool) -> dict[str, Any]:
    total = 0
    layouts: Counter[str] = Counter()
    set_types: Counter[str] = Counter()
    borders: Counter[str] = Counter()
    digital_only_count = 0
    silver_bordered_count = 0
    non_card_layout_count = 0
    multi_face_count = 0
    multi_face_by_layout: Counter[str] = Counter()
    null_oracle_count = 0
    keyword_freq: Counter[str] = Counter()

    single_face_char_lens: list[int] = []
    per_face_char_lens: list[int] = []
    combined_char_lens: list[int] = []
    texts_to_tokenize: list[str] = []  # one entry per *embedding row* (per-face for multi-face)

    for card in card_iter:
        total += 1
        layouts[card_layout(card)] += 1
        set_types[set_type(card)] += 1
        borders[border_color(card)] += 1
        if is_digital_only(card):
            digital_only_count += 1
        if is_silver_bordered(card):
            silver_bordered_count += 1
        if is_non_card_layout(card):
            non_card_layout_count += 1

        face_texts = oracle_text_per_face(card)
        if has_multiple_faces(card):
            multi_face_count += 1
            multi_face_by_layout[card_layout(card)] += 1
            for text in face_texts:
                per_face_char_lens.append(len(text))
                texts_to_tokenize.append(text)
            combined_char_lens.append(len(" // ".join(face_texts)))
        else:
            text = face_texts[0]
            if not text:
                null_oracle_count += 1
            single_face_char_lens.append(len(text))
            texts_to_tokenize.append(text)

        for kw in keywords(card):
            keyword_freq[kw] += 1

    findings: dict[str, Any] = {
        "total_records": total,
        "by_layout": dict(layouts.most_common()),
        "by_set_type": dict(set_types.most_common()),
        "by_border_color": dict(borders.most_common()),
        "digital_only_count": digital_only_count,
        "silver_bordered_count": silver_bordered_count,
        "non_card_layout_count": non_card_layout_count,
        "multi_face_count": multi_face_count,
        "multi_face_by_layout": dict(multi_face_by_layout.most_common()),
        "null_oracle_text_count": null_oracle_count,
        "keyword_freq": dict(keyword_freq.most_common()),
        "char_length_single_face": _quantiles(single_face_char_lens),
        "char_length_per_face_multi": _quantiles(per_face_char_lens),
        "char_length_combined_multi": _quantiles(combined_char_lens),
    }

    if with_tokenization:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(settings.embedding_model)
        token_lens: list[int] = []
        for i in tqdm(
            range(0, len(texts_to_tokenize), TOKENIZE_BATCH),
            desc="tokenize",
            unit="batch",
        ):
            chunk = texts_to_tokenize[i : i + TOKENIZE_BATCH]
            encoded = tok(chunk, padding=False, truncation=False, return_length=True)
            token_lens.extend(int(n) for n in encoded["length"])

        findings["token_length_per_embedding_row"] = _quantiles(token_lens)
        findings["over_truncation_threshold"] = {
            "threshold": TRUNCATION_THRESHOLD,
            "count": sum(1 for n in token_lens if n > TRUNCATION_THRESHOLD),
        }
        findings["embedding_model_tokenizer"] = settings.embedding_model

    return findings


def _print_summary(findings: dict[str, Any]) -> None:
    print(f"\n  Total records: {findings['total_records']:,}\n")

    print("  Layout distribution (top 15):")
    for layout, count in list(findings["by_layout"].items())[:15]:
        print(f"    {layout:25s} {count:>7,}")

    print("\n  Exclusion candidates:")
    print(f"    non-card layouts        {findings['non_card_layout_count']:>7,}")
    print(f"    digital-only            {findings['digital_only_count']:>7,}")
    print(f"    silver-bordered         {findings['silver_bordered_count']:>7,}")

    print("\n  Multi-faced cards:")
    print(f"    total                   {findings['multi_face_count']:>7,}")
    for layout, count in findings["multi_face_by_layout"].items():
        print(f"    {layout:23s} {count:>7,}")

    print(f"\n  Empty/missing oracle_text: {findings['null_oracle_text_count']:,}")

    print("\n  Character length — single-faced cards:")
    for k, v in findings["char_length_single_face"].items():
        print(f"    {k:6s} {v}")

    print("\n  Character length — per face (multi-faced):")
    for k, v in findings["char_length_per_face_multi"].items():
        print(f"    {k:6s} {v}")

    if "token_length_per_embedding_row" in findings:
        print("\n  Token length — per embedding row (one row per face):")
        for k, v in findings["token_length_per_embedding_row"].items():
            print(f"    {k:6s} {v}")
        over = findings["over_truncation_threshold"]
        print(f"    {'over':6s} {over['count']:,}  (> {over['threshold']} tokens)")
        print(f"    tokenizer: {findings['embedding_model_tokenizer']}")

    print("\n  Top 25 keywords:")
    for kw, count in list(findings["keyword_freq"].items())[:25]:
        print(f"    {kw:30s} {count:>6,}")


def main() -> int:
    # Windows consoles default to a non-UTF-8 codepage; reconfigure so em-dashes
    # and other non-ASCII characters in the summary print correctly.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to oracle-cards-*.json. Defaults to newest under data/raw/.",
    )
    parser.add_argument(
        "--no-tokenize",
        action="store_true",
        help="Skip the embedding-tokenizer length distribution (faster, no model download).",
    )
    args = parser.parse_args()

    input_path = args.input or _find_latest_bulk(settings.raw_data_dir)
    processed_dir = settings.data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    output_path = processed_dir / f"corpus-survey-{today}.json"

    with PipelineRun(
        "survey_corpus",
        inputs={
            "input_path": str(input_path),
            "output_path": str(output_path),
            "tokenize": not args.no_tokenize,
        },
    ) as run:
        with input_path.open("rb") as fh:
            stream = tqdm(ijson.items(fh, "item"), desc="scan", unit="card")
            findings = survey(stream, with_tokenization=not args.no_tokenize)

        output_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

        run.processed(findings["total_records"])
        run.note(
            input_path=str(input_path),
            output_path=str(output_path),
            total_records=findings["total_records"],
            non_card_layout_count=findings["non_card_layout_count"],
            digital_only_count=findings["digital_only_count"],
            multi_face_count=findings["multi_face_count"],
            null_oracle_text_count=findings["null_oracle_text_count"],
        )

        _print_summary(findings)
        print(f"\n  Full report: {output_path}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
