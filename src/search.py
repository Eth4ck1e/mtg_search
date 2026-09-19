"""Stage 2 + Stage 3 orchestration — the retrieval cascade's search path.

Given a natural-language query this module (optionally) calls the Stage 1
rewriter, compiles its structured filters into a parameterised SQL WHERE
clause (Stage 2), embeds the query-side text with the Nomic query prefix, and
runs pgvector cosine search **inside** the filtered candidate set (Stage 3).
Results are deduplicated by ``oracle_id`` (closest face wins) and the top-K
returned.

Design invariant (CLAUDE.md §2): the vector search always executes over the
SQL-narrowed set. There is no post-filter path in this module.

Stage 1 modes (the rows of the experimental grid — see
``docs/journal/2026-09-18-fine-tuning-pivot-oracle-tags-and-recipe.md`` §4):

* ``hyde``        — rewriter supplies filters AND the query text is its
                    ``hypothetical_card``. The v1 cascade.
* ``passthrough`` — rewriter supplies filters only; the query text is the
                    user's own words. The post-pivot shape, where the tuned
                    embedder is expected to understand jargon directly.
* ``raw``         — no rewriter call at all: no filters, raw query embedded.
                    Pure dense retrieval, the old ``embedding_only`` config.

Filter semantics follow the decisions logged during the 2026-09-18 test
series: colour filters admit colourless cards unless the op is ``exactly``;
``keywords`` is **off by default** because the rewriter over-narrows on
inferred keywords (selective strictness, journal §2 of the prompt design
notes) — enable it per run via :class:`FilterPolicy`.

The embedding column is pinned to ``settings.embedding_version`` so a search
never silently mixes vectors from different checkpoints. Point
``EMBEDDING_MODEL`` at a fine-tuned checkpoint directory and re-embed; the
version string follows and this module searches only those rows.

CLI (raw mode needs no server; hyde/passthrough need ``mlx_lm.server``)::

    PYTHONPATH="$PWD" python -m src.search "cheap red removal"
    PYTHONPATH="$PWD" python -m src.search "cheap red removal" --mode raw
    PYTHONPATH="$PWD" python -m src.search "cheap red removal" --no-sql --show-sql
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

import src.utils.quiet  # noqa: F401  — side-effect: silence known-benign upstream warnings
from src.config import settings
from src.preprocess_text import format_for_nomic_query
from src.query_rewriter import HyDEFilters, HyDEResult, _load_prompt, rewrite_query
from src.utils.device import select_device

# ---- Stage 2: filter compilation -------------------------------------------

_COLORS = frozenset("WUBRG")
_CMC_OPS = frozenset({"<=", "=", ">=", "<", ">", "between"})
_PT_OPS = frozenset({"<=", "=", ">=", "<", ">"})
_LEGALITY_FORMATS = frozenset(
    {
        "standard",
        "modern",
        "pioneer",
        "legacy",
        "vintage",
        "commander",
        "pauper",
        "historic",
        "alchemy",
        "brawl",
    }
)
_LEGALITY_STATUSES = frozenset({"legal", "not_legal", "banned", "restricted"})


class Stage1Mode(StrEnum):
    HYDE = "hyde"
    PASSTHROUGH = "passthrough"
    RAW = "raw"


@dataclass(frozen=True)
class FilterPolicy:
    """Which rewriter-supplied filters become hard WHERE clauses.

    Default reflects the 2026-09-18 decisions: every structural field is
    strict; ``keywords`` is ignored because the 8B rewriter infers keywords
    from paraphrases and over-narrows. Flip ``keywords=True`` to measure it.
    """

    colors: bool = True
    color_identity: bool = True
    types: bool = True
    subtypes: bool = True
    cmc: bool = True
    power: bool = True
    toughness: bool = True
    format_legality: bool = True
    keywords: bool = False


class FilterError(ValueError):
    """Raised when rewriter output cannot be compiled to SQL safely."""


DEFAULT_POLICY = FilterPolicy()


def _clean_colors(values: list[str], field_name: str) -> list[str]:
    cleaned = sorted({v.strip().upper() for v in values if v and v.strip()})
    bad = [v for v in cleaned if v not in _COLORS]
    if bad:
        raise FilterError(f"{field_name}: invalid colour symbols {bad}")
    return cleaned


def _color_clause(column: str, values: list[str], op: str | None, params: dict[str, Any]) -> str:
    """Compile a colour filter with the colourless allowance.

    ``contains_any`` (default): card shares ≥1 colour with the request, OR is
    colourless — colourless cards fit any colour specification unless the
    query explicitly excludes them, which is what ``exactly`` is for.
    ``subset_of``: every colour on the card is in the request (colourless
    passes naturally). ``contains_all``: card has every requested colour.
    ``exactly``: set equality, no colourless allowance.
    """
    key = f"{column}_v"
    params[key] = values
    op = op or "contains_any"
    if not values:
        # Empty list = "colourless" per the prompt contract.
        return f"cardinality({column}) = 0"
    if op == "contains_any":
        return f"({column} && %({key})s OR cardinality({column}) = 0)"
    if op == "contains_all":
        return f"{column} @> %({key})s"
    if op == "subset_of":
        return f"{column} <@ %({key})s"
    if op == "exactly":
        return f"({column} <@ %({key})s AND {column} @> %({key})s)"
    raise FilterError(f"colors_op: unknown value {op!r}")


def _numeric_clause(
    column: str,
    spec: dict[str, Any],
    allowed_ops: frozenset[str],
    params: dict[str, Any],
    *,
    text_column: bool,
) -> str:
    op = spec.get("op")
    value = spec.get("value")
    if op not in allowed_ops:
        raise FilterError(f"{column}: unsupported op {op!r}")
    # power/toughness are TEXT ('*', '1+*', 'X'); only compare parseable ints.
    expr = f"{column}::numeric" if text_column else column
    guard = f"{column} ~ '^-?[0-9]+$' AND " if text_column else ""
    if op == "between":
        if not (isinstance(value, list | tuple) and len(value) == 2):
            raise FilterError(f"{column}: 'between' needs [low, high], got {value!r}")
        params[f"{column}_lo"], params[f"{column}_hi"] = float(value[0]), float(value[1])
        return f"({guard}{expr} BETWEEN %({column}_lo)s AND %({column}_hi)s)"
    try:
        params[f"{column}_v"] = float(value)
    except (TypeError, ValueError) as exc:
        raise FilterError(f"{column}: non-numeric value {value!r}") from exc
    return f"({guard}{expr} {op} %({column}_v)s)"


def _word_clause(
    column: str, words: list[str], prefix: str, params: dict[str, Any], *, join: str
) -> str | None:
    """Case-insensitive whole-word matches on a text column, combined with ``join``.

    ``types`` are combined with OR — a card has one card type in the common
    case, and "instants or sorceries" is the natural reading of a multi-type
    filter (ANDing them matched zero cards in the first smoke test).
    ``subtypes`` are combined with AND — "elf warriors" means both.
    """
    parts = []
    for i, word in enumerate(w.strip() for w in words if w and w.strip()):
        key = f"{prefix}_{i}"
        params[key] = rf"\m{re.escape(word)}\M"
        parts.append(f"{column} ~* %({key})s")
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else "(" + f" {join} ".join(parts) + ")"


def build_where(
    filters: HyDEFilters | None,
    policy: FilterPolicy = DEFAULT_POLICY,
) -> tuple[list[str], dict[str, Any]]:
    """Compile rewriter filters into SQL clause fragments + named params.

    Returns ``(clauses, params)``. Clauses are ANDed by the caller. All
    values travel as bound parameters; only whitelisted operators and column
    names are interpolated into SQL text.

    Raises :class:`FilterError` on values outside the prompt contract, so a
    rewriter hallucination surfaces as a loud failure rather than a silent
    zero-result search.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if filters is None:
        return clauses, params

    if policy.colors and filters.colors is not None:
        clauses.append(
            _color_clause(
                "colors", _clean_colors(filters.colors, "colors"), filters.colors_op, params
            )
        )
    if policy.color_identity and filters.color_identity is not None:
        clauses.append(
            _color_clause(
                "color_identity",
                _clean_colors(filters.color_identity, "color_identity"),
                filters.colors_op,
                params,
            )
        )
    if policy.types and filters.types:
        clauses.append(_word_clause("type_line", filters.types, "type", params, join="OR"))
    if policy.subtypes and filters.subtypes:
        clauses.append(_word_clause("type_line", filters.subtypes, "subtype", params, join="AND"))
    if policy.cmc and filters.cmc:
        clauses.append(_numeric_clause("cmc", filters.cmc, _CMC_OPS, params, text_column=False))
    if policy.power and filters.power:
        clauses.append(_numeric_clause("power", filters.power, _PT_OPS, params, text_column=True))
    if policy.toughness and filters.toughness:
        clauses.append(
            _numeric_clause("toughness", filters.toughness, _PT_OPS, params, text_column=True)
        )
    if policy.keywords and filters.keywords:
        params["keywords_v"] = [k.strip() for k in filters.keywords if k and k.strip()]
        if params["keywords_v"]:
            clauses.append("keywords @> %(keywords_v)s")
    if policy.format_legality and filters.format_legality:
        fmt = str(filters.format_legality.get("format", "")).lower()
        status = str(filters.format_legality.get("status", "legal")).lower()
        if fmt not in _LEGALITY_FORMATS:
            raise FilterError(f"format_legality.format: unknown format {fmt!r}")
        if status not in _LEGALITY_STATUSES:
            raise FilterError(f"format_legality.status: unknown status {status!r}")
        params["legal_fmt"], params["legal_status"] = fmt, status
        clauses.append("legalities->>%(legal_fmt)s = %(legal_status)s")
    return clauses, params


# ---- Stage 3: search --------------------------------------------------------

_SEARCH_SQL = """
    WITH ranked AS (
        SELECT oracle_id::text AS oracle_id, name, type_line, mana_cost, oracle_text,
               embedding <=> %(vec)s AS distance,
               ROW_NUMBER() OVER (
                   PARTITION BY oracle_id ORDER BY embedding <=> %(vec)s
               ) AS rn
        FROM cards
        WHERE {where}
    )
    SELECT oracle_id, name, type_line, mana_cost, oracle_text, distance
    FROM ranked
    WHERE rn = 1
    ORDER BY distance
    LIMIT %(k)s
"""

_COUNT_SQL = "SELECT COUNT(DISTINCT oracle_id) FROM cards WHERE {where}"


@dataclass
class SearchHit:
    oracle_id: str
    name: str
    type_line: str
    mana_cost: str | None
    oracle_text: str
    distance: float


@dataclass
class SearchResult:
    query: str
    mode: str
    use_sql: bool
    query_text: str  # what was actually embedded (before the Nomic prefix)
    hyde: HyDEResult | None
    where_sql: str
    candidate_count: int
    hits: list[SearchHit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hyde"] = self.hyde.model_dump() if self.hyde is not None else None
        return d


class Searcher:
    """Holds the encoder and a DB connection; call :meth:`search` per query.

    Construct once per process — model load is the expensive part.
    """

    def __init__(
        self,
        *,
        conn: psycopg.Connection | None = None,
        model: SentenceTransformer | None = None,
        policy: FilterPolicy = DEFAULT_POLICY,
        device: str | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self.policy = policy
        self.embedding_version = settings.embedding_version
        self.prompt_path = prompt_path or (settings.prompts_dir / "hyde_v1.yaml")
        self.prompt_version = (
            f"{self.prompt_path.name}:{_load_prompt(self.prompt_path).get('version')}"
        )
        if model is None:
            dev = select_device(prefer=device)
            model = SentenceTransformer(
                settings.embedding_model, device=str(dev), trust_remote_code=True
            )
        self.model = model
        self._owns_conn = conn is None
        self.conn = conn or psycopg.connect(settings.database_url)
        register_vector(self.conn)

    def close(self) -> None:
        if self._owns_conn:
            self.conn.close()

    def __enter__(self) -> Searcher:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- pieces --

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [format_for_nomic_query(text)],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(vec, dtype=np.float32)

    def _run_sql(self, where_clauses: list[str], params: dict[str, Any], vec: np.ndarray, k: int):
        base = ["embedding IS NOT NULL", "embedding_version = %(ver)s", *where_clauses]
        where = " AND ".join(base)
        params = {**params, "ver": self.embedding_version, "vec": vec, "k": k}
        with self.conn.cursor() as cur:
            cur.execute(_COUNT_SQL.format(where=where), params)
            candidate_count = int(cur.fetchone()[0])
            cur.execute(_SEARCH_SQL.format(where=where), params)
            rows = cur.fetchall()
        hits = [
            SearchHit(
                oracle_id=r[0],
                name=r[1],
                type_line=r[2],
                mana_cost=r[3],
                oracle_text=r[4],
                distance=float(r[5]),
            )
            for r in rows
        ]
        return where, candidate_count, hits

    # -- entry point --

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        mode: Stage1Mode | str = Stage1Mode.HYDE,
        use_sql: bool = True,
    ) -> SearchResult:
        """Run the cascade for one query.

        ``use_sql=False`` is the minus-SQL ablation: Stage 1 still runs (in hyde /
        passthrough modes) but its filters are discarded and Stage 3 searches
        the whole corpus.
        """
        mode = Stage1Mode(mode)
        timings: dict[str, float] = {}
        warnings: list[str] = []
        hyde: HyDEResult | None = None

        # ---- Stage 1 ----
        if mode is not Stage1Mode.RAW:
            t0 = time.perf_counter()
            hyde = rewrite_query(query, prompt_path=self.prompt_path)
            timings["stage1_ms"] = (time.perf_counter() - t0) * 1000

        if mode is Stage1Mode.HYDE:
            query_text = (hyde.hypothetical_card or "").strip() if hyde else ""
            if not query_text:
                warnings.append("hyde returned no hypothetical_card; embedding raw query")
                query_text = query
        else:
            query_text = query

        # ---- Stage 2 ----
        t0 = time.perf_counter()
        where_clauses: list[str] = []
        params: dict[str, Any] = {}
        if use_sql and hyde is not None:
            try:
                where_clauses, params = build_where(hyde.filters, self.policy)
            except FilterError as exc:
                warnings.append(f"filters dropped: {exc}")
        timings["stage2_compile_ms"] = (time.perf_counter() - t0) * 1000

        # ---- Stage 3 ----
        t0 = time.perf_counter()
        vec = self.embed_query(query_text)
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        where, candidate_count, hits = self._run_sql(where_clauses, params, vec, k)
        timings["sql_ms"] = (time.perf_counter() - t0) * 1000

        if candidate_count == 0:
            warnings.append("filters matched zero cards")

        return SearchResult(
            query=query,
            mode=mode.value,
            use_sql=use_sql,
            query_text=query_text,
            hyde=hyde,
            where_sql=where,
            candidate_count=candidate_count,
            hits=hits,
            timings_ms={k_: round(v, 2) for k_, v in timings.items()},
            warnings=warnings,
        )


# ---- CLI --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=[m.value for m in Stage1Mode],
        default=Stage1Mode.HYDE.value,
    )
    parser.add_argument("--no-sql", action="store_true", help="Minus-SQL ablation: drop Stage 2.")
    parser.add_argument("--keywords", action="store_true", help="Enable the keywords filter.")
    parser.add_argument("--show-sql", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit the full result as JSON.")
    args = parser.parse_args()

    policy = FilterPolicy(keywords=args.keywords)
    with Searcher(policy=policy) as s:
        result = s.search(args.query, k=args.k, mode=args.mode, use_sql=not args.no_sql)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
        return 0

    print(f"\n  mode={result.mode}  sql={'on' if result.use_sql else 'off'}  k={args.k}")
    if result.hyde is not None:
        print(
            f"  filters:    {json.dumps(result.hyde.filters.model_dump(exclude_none=True) if result.hyde.filters else None)}"
        )
    print(f"  embedded:   {result.query_text!r}")
    if args.show_sql:
        print(f"  where:      {result.where_sql}")
    print(f"  candidates: {result.candidate_count:,}")
    print(f"  timings:    {result.timings_ms}")
    for w in result.warnings:
        print(f"  WARNING:    {w}")
    print()
    for rank, h in enumerate(result.hits, 1):
        text = h.oracle_text.replace("\n", " / ")
        print(f"  {rank:2d}. {h.name}  [{h.type_line}]  {h.mana_cost or ''}  d={h.distance:.4f}")
        print(f"      {text[:160]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
