"""Prompt-size probe: rules, examples, words and tokens per prompt version.
The simplification claim (journal §4) is measured here, not asserted."""

import re
from pathlib import Path

import yaml
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
from src.query_rewriter import _build_user_message  # noqa: E402

print(
    f"{'prompt':14s} {'rules':>5s} {'examples':>8s} {'sys words':>9s} {'sys tokens':>10s} {'full request tokens':>19s}"
)
for path in sorted(Path("prompts").glob("hyde_v*.yaml")):
    p = yaml.safe_load(path.read_text())
    rules = len(
        re.findall(r"^\s*- ", p["system"].split("## FILTER SCHEMA")[0].split("## RULES")[-1], re.M)
    )
    full = p["system"] + _build_user_message(p, "cheap red removal")
    print(
        f"{path.stem:14s} {rules:5d} {len(p['examples']):8d} {len(p['system'].split()):9d} {len(tok(p['system']).input_ids):10d} {len(tok(full).input_ids):19d}"
    )
