"""HyDE query rewriter — Stage 1 of the retrieval cascade.

Loads the versioned HyDE prompt (default: ``prompts/hyde_v1.yaml``), formats
the few-shot examples plus the live user query into an OpenAI-compatible
chat-completions request, POSTs to the local MLX server, and returns the
model's structured JSON output as a typed :class:`HyDEResult`.

The return value has two fields:

* ``filters`` — structured attribute filters for the SQL pre-filter (Stage 2).
  May be ``None`` if the query has no structural component.
* ``hypothetical_card`` — canonical MTG rules text for the semantic search
  (Stage 3) to embed. May be ``None`` for purely structural queries.

Usage from another module::

    from src.query_rewriter import rewrite_query
    result = rewrite_query("cheap blue counterspell that also draws a card")
    # → HyDEResult(filters=HyDEFilters(colors=['U'], ...), hypothetical_card="Counter target spell. Draw a card.")

CLI usage (requires ``mlx_lm.server`` running on ``HYDE_SERVER_URL``)::

    PYTHONPATH="$PWD" python -m src.query_rewriter "cheap red removal"
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests
import yaml
from pydantic import BaseModel, ConfigDict

import src.utils.quiet  # noqa: F401  — side-effect: silence known-benign upstream warnings
from src.config import settings

# ---- Response models ---------------------------------------------------


class HyDEFilters(BaseModel):
    """Structured filter attributes HyDE extracts from the user query.

    All fields are optional; any subset may be populated. Fields absent from
    the model output remain ``None``. Extra fields the model might emit are
    silently ignored (``extra="ignore"``) so schema drift on the model side
    does not crash the caller — invalid *values* still raise, though.
    """

    model_config = ConfigDict(extra="ignore")

    colors: list[str] | None = None
    colors_op: str | None = None
    color_identity: list[str] | None = None
    types: list[str] | None = None
    subtypes: list[str] | None = None
    cmc: dict[str, Any] | None = None
    keywords: list[str] | None = None
    power: dict[str, Any] | None = None
    toughness: dict[str, Any] | None = None
    format_legality: dict[str, str] | None = None


class HyDEResult(BaseModel):
    """Full HyDE output — filters for Stage 2, hypothetical_card for Stage 3."""

    model_config = ConfigDict(extra="ignore")

    filters: HyDEFilters | None = None
    hypothetical_card: str | None = None


class HyDEError(RuntimeError):
    """Raised when the HyDE server is unreachable or its response is unusable."""


# ---- Prompt loading + assembly ----------------------------------------


@lru_cache(maxsize=4)
def _load_prompt(prompt_path: Path) -> dict[str, Any]:
    """Load a versioned HyDE prompt YAML. Cached per-path within a process.

    NOTE: cache means the running process does NOT hot-reload edits to the
    YAML — restart the caller to pick up prompt changes. For iterative
    prompt work, run via the CLI (each invocation is a fresh process).
    """
    return yaml.safe_load(prompt_path.read_text(encoding="utf-8"))


def _build_user_message(prompt: dict[str, Any], query: str) -> str:
    """Format the few-shot examples plus the live query into one user turn.

    Alternative design (not chosen for v1): use multi-turn chat where each
    example becomes its own user→assistant exchange. That format can be more
    effective for some chat models but adds token overhead and complicates
    the response_format=json_object contract. Revisit for v2 if v1's
    JSON-validity rate is poor.
    """
    example_blocks = [
        f"Query: {ex['query']}\nOutput: {ex['expected_output'].strip()}"
        for ex in prompt["examples"]
    ]
    return prompt["user_message_template"].format(
        examples_block="\n\n".join(example_blocks),
        query=query,
    )


# ---- Main entry point --------------------------------------------------


def rewrite_query(
    query: str,
    *,
    prompt_path: Path | None = None,
    server_url: str | None = None,
    model: str | None = None,
    timeout_s: float = 30.0,
) -> HyDEResult:
    """Rewrite a natural-language query into structured filters + hypothetical text.

    Arguments:
        query: The user's natural-language query.
        prompt_path: Which prompt version to load. Defaults to
            ``settings.prompts_dir / "hyde_v1.yaml"``.
        server_url: OpenAI-compatible endpoint base URL (must expose
            ``/chat/completions``). Defaults to ``settings.hyde_server_url``.
        model: Model tag to send in the request. Must match what the server
            has loaded. Defaults to ``settings.hyde_model``.
        timeout_s: HTTP request timeout. 30s is generous for local inference
            of a 256-token response at ~57 tok/sec (~4.5s worst case).

    Returns:
        A :class:`HyDEResult` with ``filters`` and ``hypothetical_card`` fields.

    Raises:
        HyDEError: on any failure of the request path — network error, non-2xx
            response, malformed response, or invalid JSON body. Errors do NOT
            retry; failures should surface for investigation rather than be
            silently retried.
    """
    prompt_path = prompt_path or (settings.prompts_dir / "hyde_v1.yaml")
    server_url = server_url or settings.hyde_server_url
    model = model or settings.hyde_model

    prompt = _load_prompt(prompt_path)

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": _build_user_message(prompt, query)},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": settings.hyde_max_tokens,
        "temperature": settings.hyde_temperature,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(
            f"{server_url.rstrip('/')}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise HyDEError(f"MLX server call failed ({server_url}): {exc}") from exc

    body = resp.json()
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HyDEError(f"Unexpected chat-completions response shape: {body}") from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise HyDEError(
            f"HyDE returned non-JSON content (max_tokens truncation?): {content!r}"
        ) from exc

    return HyDEResult.model_validate(parsed)


# ---- CLI ---------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("query", help="Natural-language query to rewrite.")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=None,
        help="Path to a HyDE prompt YAML (default: settings.prompts_dir/hyde_v1.yaml).",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Override HYDE_SERVER_URL for this call.",
    )
    args = parser.parse_args()

    try:
        result = rewrite_query(
            args.query,
            prompt_path=args.prompt,
            server_url=args.server_url,
        )
    except HyDEError as exc:
        print(f"HyDE error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
